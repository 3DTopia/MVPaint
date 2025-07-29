import os
import cv2
import tqdm
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from .mesh import Mesh, OrbitCamera, safe_normalize
import nvdiffrast.torch as dr
import imageio

class Renderer:
    def __init__(self, H=512, W=512, radius=3, fovy=50, mode='albedo'):
        self.W = W
        self.H = H
        self.cam = OrbitCamera(W, H, r=radius, fovy=fovy)
        self.bg_color = torch.ones(3, dtype=torch.float32).cuda() # default white bg
        # self.bg_color = torch.zeros(3, dtype=torch.float32).cuda() # black bg

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.alpha = np.zeros((self.W, self.H, 1), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.light_dir = np.array([0, 0])
        self.ambient_ratio = 0.5

        # auto-rotate
        self.auto_rotate_cam = False
        self.auto_rotate_light = False
        
        self.mode = mode
        self.render_modes = ['albedo', 'depth', 'normal', 'lambertian']



    def __del__(self):
        pass
    
    def step(self, mesh, glctx):

        if not self.need_update:
            return
    
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # do MVP for vertices
        pose = torch.from_numpy(self.cam.pose.astype(np.float32)).cuda()
        proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()

        v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (self.H, self.W))

        alpha = (rast[..., 3:] > 0).float()
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
        
        if self.mode == 'depth':
            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            buffer = depth.squeeze(0).detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
        else:
            # use vertex color if exists
            if mesh.vc is not None:
                albedo, _ = dr.interpolate(mesh.vc.unsqueeze(0).contiguous(), rast, mesh.f)
            # use texture image
            else:
                texc, _ = dr.interpolate(mesh.vt.unsqueeze(0).contiguous(), rast, mesh.ft)
                albedo = dr.texture(mesh.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]

            albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background
            albedo = dr.antialias(albedo, rast, v_clip, mesh.f).clamp(0, 1) # [1, H, W, 3]

            if self.mode == 'albedo':
                albedo = albedo * alpha + self.bg_color * (1 - alpha)
                buffer = albedo[0].detach().cpu()
            else:
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal)
                if self.mode == 'normal':
                    normal_image = (normal[0] + 1) / 2
                    normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)) # remove background
                    buffer = normal_image[0].detach().cpu()
                elif self.mode == 'lambertian':
                    light_d = np.deg2rad(self.light_dir)
                    light_d = np.array([
                        np.cos(light_d[0]) * np.sin(light_d[1]),
                        -np.sin(light_d[0]),
                        np.cos(light_d[0]) * np.cos(light_d[1]),
                    ], dtype=np.float32)
                    light_d = torch.from_numpy(light_d).to(albedo.device)
                    lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
                    albedo = (albedo * lambertian.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
                    buffer = albedo[0].detach().cpu()
                    
                elif self.mode == 'pbr':

                    if mesh.metallicRoughness is not None:
                        metallicRoughness = dr.texture(mesh.metallicRoughness.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]
                        metallic = metallicRoughness[..., 2:3] * self.metallic_factor
                        roughness = metallicRoughness[..., 1:2] * self.roughness_factor
                    else:
                        metallic = torch.ones_like(albedo[..., :1]) * self.metallic_factor
                        roughness = torch.ones_like(albedo[..., :1]) * self.roughness_factor

                    xyzs, _ = dr.interpolate(mesh.v.unsqueeze(0), rast, mesh.f) # [1, H, W, 3]
                    viewdir = safe_normalize(xyzs - pose[:3, 3])

                    n_dot_v = (normal * viewdir).sum(-1, keepdim=True) # [1, H, W, 1]
                    reflective = n_dot_v * normal * 2 - viewdir

                    diffuse_albedo = (1 - metallic) * albedo

                    fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1) # [H, W, 2]
                    fg = dr.texture(
                        self.FG_LUT,
                        fg_uv.reshape(1, -1, 1, 2).contiguous(),
                        filter_mode="linear",
                        boundary_mode="clamp",
                    ).reshape(1, self.H, self.W, 2)
                    F0 = (1 - metallic) * 0.04 + metallic * albedo
                    specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]

                    diffuse_light = self.light(normal)
                    specular_light = self.light(reflective, roughness)

                    color = diffuse_albedo * diffuse_light + specular_albedo * specular_light # [H, W, 3]
                    color = color * alpha + self.bg_color * (1 - alpha)

                    buffer = color[0].detach().cpu()
                    

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = False

        if self.auto_rotate_cam:
            self.cam.orbit(5, 0)
            self.need_update = True
        
        if self.auto_rotate_light:
            self.light_dir[1] += 3
            self.need_update = True
        return buffer
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str, help="path to mesh (obj, ply, glb, ...)")
    parser.add_argument('--meshes', action='store', type=str, nargs='+', help="path to mesh (obj, ply, glb, ...)")
    parser.add_argument('--pbr', action='store_true', help="enable PBR material")
    parser.add_argument('--envmap', type=str, default=None, help="hdr env map path for pbr")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--mode', default='normal', type=str, choices=['lambertian', 'albedo', 'normal', 'depth', 'pbr'], help="rendering mode")
    parser.add_argument('--W', type=int, default=512, help="GUI width")
    parser.add_argument('--H', type=int, default=512, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument("--wogui", action='store_true', help="disable all dpg GUI")
    parser.add_argument("--force_cuda_rast", action='store_true', help="force to use RasterizeCudaContext.")
    parser.add_argument('--save', type=str, default=None, help="path to save example rendered images")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--num_azimuth', type=int, default=8, help="number of images to render from different azimuths")
    parser.add_argument('--save_video', type=str, default=None, help="path to save rendered video")

    opt = parser.parse_args()

    renderer = Renderer(opt)
    glctx = dr.RasterizeCudaContext()

    images = []
    elevation = [opt.elevation,]
    azimuth = np.arange(0, 360, 3, dtype=np.int32) # front-->back-->front
    mesh = Mesh.load(opt.mesh, front_dir=opt.front_dir)
    for ele in elevation:
        for azi in azimuth:
            renderer.cam.from_angle(ele, azi)
            renderer.need_update = True
            render_buffer = renderer.step(mesh, glctx)
            image = (render_buffer * 255).astype(np.uint8)
            images.append([image])

    if opt.meshes is not None and len(opt.meshes) >= 1:
        for meshname in opt.meshes:
            mesh = Mesh.load(meshname, front_dir=opt.front_dir)
            for ele in elevation:
                for i, azi in enumerate(azimuth):
                    renderer.cam.from_angle(ele, azi)
                    renderer.need_update = True
                    render_buffer = renderer.step(mesh, glctx)
                    image = (render_buffer * 255).astype(np.uint8)
                    images[i].append(image)
        images = [np.hstack(imgs) for imgs in images]
    else:
        images = [img[0] for img in images]
    imageio.mimwrite(opt.save_video, images, fps=30, quality=8, macro_block_size=1)

