from __future__ import annotations

import argparse
import numpy as np
import torch, os, sys, io
import torch.nn.functional as F
import trimesh, cv2, os, xatlas
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from omegaconf import DictConfig
from torch import Tensor
from typeguard import typechecked as typechecker
from numpy import ndarray
from tqdm.contrib.concurrent import process_map
# import multiprocessing as mp
from tqdm import tqdm, trange
import imageio
from collections import namedtuple
from scipy.spatial.transform import Rotation
from megfile import smart_open, smart_glob, smart_copy
import shutil 

import nvdiffrast.torch as dr
import logging
logging.getLogger('nvdiffrast').setLevel(logging.ERROR)
glctx = dr.RasterizeCudaContext()

# torch / numpy math utils
def dot(x: Union[Tensor, ndarray], y: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """dot product (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        y (Union[Tensor, ndarray]): y, [..., C]

    Returns:
        Union[Tensor, ndarray]: x dot y, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)

def length(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """length of an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: length, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """normalize an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: normalized x, [..., C]
    """

    return x / length(x, eps)

def make_divisible(x: int, m: int = 8):
    """make an int x divisible by m.

    Args:
        x (int): x
        m (int, optional): m. Defaults to 8.

    Returns:
        int: x + (m - x % m)
    """
    return int(x + (m - x % m))

def look_at(campos, target, opengl=True):
    """construct pose rotation matrix by look-at.

    Args:
        campos (np.ndarray): camera position, float [3]
        target (np.ndarray): look at target, float [3]
        opengl (bool, optional): whether use opengl camera convention (forward direction is target --> camera). Defaults to True.

    Returns:
        np.ndarray: the camera pose rotation matrix, float [3, 3], normalized.
    """
   
    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R

def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    """construct a camera pose matrix orbiting a target with elevation & azimuth angle.

    Args:
        elevation (float): elevation in (-90, 90), from +y to -y is (-90, 90)
        azimuth (float): azimuth in (-180, 180), from +z to +x is (0, 90)
        radius (int, optional): camera radius. Defaults to 1.
        is_degree (bool, optional): if the angles are in degree. Defaults to True.
        target (np.ndarray, optional): look at target position. Defaults to None.
        opengl (bool, optional): whether to use OpenGL camera convention. Defaults to True.

    Returns:
        np.ndarray: the camera pose matrix, float [4, 4]
    """
    
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


class OrbitCamera:
    """ An orbital camera class.
    """
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        """init function

        Args:
            W (int): image width
            H (int): image height
            r (int, optional): camera radius. Defaults to 2.
            fovy (int, optional): camera field of view in degree along y-axis. Defaults to 60.
            near (float, optional): near clip plane. Defaults to 0.01.
            far (int, optional): far clip plane. Defaults to 100.
        """
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = Rotation.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        """get the field of view in radians along x-axis

        Returns:
            float: field of view in radians along x-axis
        """
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        """get the camera position

        Returns:
            np.ndarray: camera position, float [3]
        """
        return self.pose[:3, 3]


    @property
    def pose(self):
        """get the camera pose matrix (cam2world)

        Returns:
            np.ndarray: camera pose, float [4, 4]
        """
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    
    @property
    def view(self):
        """get the camera view matrix (world2cam, inverse of cam2world)

        Returns:
            np.ndarray: camera view, float [4, 4]
        """
        return np.linalg.inv(self.pose)

    
    @property
    def perspective(self):
        """get the perspective matrix

        Returns:
            np.ndarray: camera perspective, float [4, 4]
        """
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        # return np.array(
        #     [
        #         [1 / (y * aspect), 0, 0, 0],
        #         [0, -1 / y, 0, 0],
        #         [
        #             0,
        #             0,
        #             -(self.far + self.near) / (self.far - self.near),
        #             -(2 * self.far * self.near) / (self.far - self.near),
        #         ],
        #         [0, 0, -1, 0],
        #     ],
        #     dtype=np.float32,
        # )
        return np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -2/(self.far - self.near), -(self.far + self.near)/(self.far - self.near)],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )


    # intrinsics
    @property
    def intrinsics(self):
        """get the camera intrinsics

        Returns:
            np.ndarray: intrinsics (fx, fy, cx, cy), float [4]
        """
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    
    @property
    def mvp(self):
        """get the MVP (model-view-perspective) matrix.

        Returns:
            np.ndarray: camera MVP, float [4, 4]
        """
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        """ rotate along camera up/side axis!

        Args:
            dx (float): delta step along x (up).
            dy (float): delta step along y (side).
        """
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = Rotation.from_rotvec(rotvec_x) * Rotation.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        """scale the camera.

        Args:
            delta (float): delta step.
        """
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        """pan the camera.

        Args:
            dx (float): delta step along x.
            dy (float): delta step along y.
            dz (float, optional): delta step along x. Defaults to 0.
        """
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])

    def from_angle(self, elevation, azimuth, is_degree=True):
        """set the camera pose from elevation & azimuth angle.

        Args:
            elevation (float): elevation in (-90, 90), from +y to -y is (-90, 90)
            azimuth (float): azimuth in (-180, 180), from +z to +x is (0, 90)
            is_degree (bool, optional): whether the angles are in degree. Defaults to True.
        """
        if is_degree:
            elevation = np.deg2rad(elevation)
            azimuth = np.deg2rad(azimuth)
        x = self.radius * np.cos(elevation) * np.sin(azimuth)
        y = - self.radius * np.sin(elevation)
        z = self.radius * np.cos(elevation) * np.cos(azimuth)
        campos = np.array([x, y, z])  # [N, 3]
        rot_mat = look_at(campos, np.zeros([3], dtype=np.float32))
        self.rot = Rotation.from_matrix(rot_mat)

class Renderer:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
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
        
        self.mode = opt.mode
        self.render_modes = ['albedo', 'depth', 'normal', 'lambertian']

        # load pbr if enabled
        if self.opt.pbr:
            import envlight
            if self.opt.envmap is None:
                hdr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/mnt/share/cq8/wadecheng/thirdparties/kiuikit/kiui/assets/lights/mud_road_puresky_1k.hdr')
            else:
                hdr_path = self.opt.envmap
            self.light = envlight.EnvLight(hdr_path, scale=2, device='cuda')
            self.FG_LUT = torch.from_numpy(np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "/mnt/share/cq8/wadecheng/thirdparties/kiuikit/kiui/assets/lights/bsdf_256_256.bin"), dtype=np.float32).reshape(1, 256, 256, 2)).cuda()

            self.metallic_factor = 1
            self.roughness_factor = 1

            self.render_modes.append('pbr')


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
        # print(v_clip[...,:3].min(), v_clip[...,:3].max(), v_clip.shape)

        rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (self.H, self.W))
        # print(rast.min(), rast.max())

        alpha = (rast[..., 3:] > 0).float()
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
        
        if self.mode == 'depth':
            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            buffer = depth.squeeze(0).detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
        elif self.mode == 'uv':
            texc, _ = dr.interpolate(mesh.vt.unsqueeze(0).contiguous(), rast, mesh.ft)
            texc = torch.where(rast[..., 3:] > 0, texc, torch.tensor(1).to(texc.device)) # remove background
            buffer = texc.squeeze(0).detach().cpu().numpy()
            buffer = np.concatenate([buffer, np.ones_like(buffer[...,:1])], axis=-1)
        elif self.mode == 'inv_uv':
            # texc, _ = dr.interpolate(mesh.vt.unsqueeze(0).contiguous(), rast, mesh.ft)
            # albedo = dr.texture(mesh.albedo.unsqueeze(0), texc, filter_mode='linear')
            # valid_face_idx = torch.unique(rast[...,3]).int()[1:] - 1
            
            # # vt_clip = v_clip.clone()
            # # vt_clip[..., :2] = mesh.vt.unsqueeze(0) * 2 - 1
            # # # v_clip[...,1] = -v_clip[...,1]
            # vt_clip = mesh.vt.unsqueeze(0) * 2 - 1
            # vt_clip = torch.cat(
            #     (
            #         vt_clip,
            #         torch.zeros_like(vt_clip[..., 0:1]),
            #         torch.ones_like(vt_clip[..., 0:1]),
            #     ),
            #     dim=-1,
            # )
            # v_clip[...,:2] = (v_clip[...,:2] + 1)  / 2

            # faces = mesh.f.clone()[valid_face_idx]
            # rast_tex, rast_tex_db = dr.rasterize(glctx, vt_clip, mesh.ft[valid_face_idx], (self.H, self.W))
            # rast_texc, _ = dr.interpolate(v_clip[...,:2].contiguous(), rast_tex, mesh.ft[valid_face_idx])

            # # recon_albedo = dr.texture(albedo, rast_texc, filter_mode='linear')
            # rast_texc = torch.where(rast_tex[..., 3:] > 0, rast_texc, torch.tensor(0).to(texc.device)) # remove background
            # buffer = rast_texc.squeeze(0).detach().cpu().numpy()
            # if buffer.shape[-1] == 2:
            #     buffer = np.concatenate([buffer, np.zeros_like(buffer[...,:1])], axis=-1)        


            valid_face_idx = torch.unique(rast[...,3]).int()[1:] - 1
            vt_clip = mesh.vt.unsqueeze(0) * 2 - 1
            vt_clip = torch.cat(
                (
                    vt_clip,
                    torch.zeros_like(vt_clip[..., 0:1]),
                    torch.ones_like(vt_clip[..., 0:1]),
                ),
                dim=-1,
            )
            v_clip[...,:2] = (v_clip[...,:2] + 1)  / 2
            faces = mesh.f.clone()[valid_face_idx]
            rast_tex, rast_tex_db = dr.rasterize(glctx, vt_clip, mesh.ft[valid_face_idx], (self.H, self.W))
            rast_texc, _ = dr.interpolate(v_clip[...,:2].contiguous(), rast_tex, mesh.f[valid_face_idx])

            rast_texc = torch.where(rast_tex[..., 3:] > 0, rast_texc, torch.tensor(1).to(rast_texc.device)) # remove background
            buffer = rast_texc.squeeze(0).detach().cpu().numpy()
            if buffer.shape[-1] == 2:
                buffer = np.concatenate([buffer, np.ones_like(buffer[...,:1])], axis=-1)       
        else:
            # use vertex color if exists
            if mesh.vc is not None:
                albedo, _ = dr.interpolate(mesh.vc.unsqueeze(0).contiguous(), rast, mesh.f)
            # use texture image
            elif mesh.albedo is not None:
                texc, _ = dr.interpolate(mesh.vt.unsqueeze(0).contiguous(), rast, mesh.ft)
                albedo = dr.texture(mesh.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]
            else:
                mesh.vc = torch.ones_like(mesh.v) * 0.75
                albedo, _ = dr.interpolate(mesh.vc.unsqueeze(0).contiguous(), rast, mesh.f)

            # print(mesh.v.dtype, mesh.vc.dtype, albedo.dtype, rast.dtype)
            albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0.).to(albedo.device)) # remove background
            albedo = dr.antialias(albedo, rast, v_clip, mesh.f).clamp(0, 1) # [1, H, W, 3]

            if self.mode == 'albedo':
                albedo = albedo * alpha + self.bg_color * (1 - alpha)
                buffer = albedo[0].detach().cpu().numpy()
            else:
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal)
                if self.mode == 'normal':
                    normal_image = (normal[0] + 1) / 2
                    normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)) # remove background
                    buffer = normal_image[0].detach().cpu().numpy()
                    # normal_image = (normal[0] + 1) / 2
                    # viewdir = safe_normalize(pose[:3, 3])
                    # # print(viewdir)
                    # nn = torch.where(rast[..., 3:] > 0, normal[0], torch.tensor(0).to(normal.device))
                    # cos_angle = torch.sum(viewdir[None,:] * nn.reshape(-1,3), dim=-1, keepdim=True).reshape([1, normal.shape[1], normal.shape[2], 1])
                    # # print(cos_angle.min(), cos_angle.max())
                    # normal_image = torch.cat([cos_angle]*3, dim=-1).clamp(0, 1)
                    # normal_image[...,0][cos_angle[...,0]<0] = 1
                    # normal_image[...,1:][cos_angle[...,0]<0] = 0
                    # # normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)) # remove background
                    # buffer = normal_image[0].detach().cpu().numpy()
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
                    buffer = albedo[0].detach().cpu().numpy()
                    
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

                    buffer = color[0].detach().cpu().numpy()
                    

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

class MeshLoader:
    """
    A torch-native trimesh class, with support for ``ply/obj/glb`` formats.

    Note:
        This class only supports one mesh with a single texture image (an albedo texture and a metallic-roughness texture).
    """
    def __init__(
        self,
        v: Optional[Tensor] = None,
        f: Optional[Tensor] = None,
        vn: Optional[Tensor] = None,
        fn: Optional[Tensor] = None,
        vt: Optional[Tensor] = None,
        ft: Optional[Tensor] = None,
        vc: Optional[Tensor] = None, # vertex color
        albedo: Optional[Tensor] = None,
        metallicRoughness: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """Init a mesh directly using all attributes.

        Args:
            v (Optional[Tensor]): vertices, float [N, 3]. Defaults to None.
            f (Optional[Tensor]): faces, int [M, 3]. Defaults to None.
            vn (Optional[Tensor]): vertex normals, float [N, 3]. Defaults to None.
            fn (Optional[Tensor]): faces for normals, int [M, 3]. Defaults to None.
            vt (Optional[Tensor]): vertex uv coordinates, float [N, 2]. Defaults to None.
            ft (Optional[Tensor]): faces for uvs, int [M, 3]. Defaults to None.
            vc (Optional[Tensor]): vertex colors, float [N, 3]. Defaults to None.
            albedo (Optional[Tensor]): albedo texture, float [H, W, 3], RGB format. Defaults to None.
            metallicRoughness (Optional[Tensor]): metallic-roughness texture, float [H, W, 3], metallic(Blue) = metallicRoughness[..., 2], roughness(Green) = metallicRoughness[..., 1]. Defaults to None.
            device (Optional[torch.device]): torch device. Defaults to None.
        """
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        # will first see if there is vertex color to use
        self.vc = vc
        # only support a single albedo image
        self.albedo = albedo
        # pbr extension, metallic(Blue) = metallicRoughness[..., 2], roughness(Green) = metallicRoughness[..., 1]
        # ref: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
        self.metallicRoughness = metallicRoughness

        self.ori_center = 0
        self.ori_scale = 1

    @classmethod
    def load(cls, path, glctx, resize=True, clean=False, renormal=True, retex=False, bound=0.9, front_dir='+z', retex_options=None, **kwargs):
        """load mesh from path.

        Args:
            path (str): path to mesh file, supports ply, obj, glb.
            clean (bool, optional): perform mesh cleaning at load (e.g., merge close vertices). Defaults to False.
            resize (bool, optional): auto resize the mesh using ``bound`` into [-bound, bound]^3. Defaults to True.
            renormal (bool, optional): re-calc the vertex normals. Defaults to True.
            retex (bool, optional): re-calc the uv coordinates, will overwrite the existing uv coordinates. Defaults to False.
            bound (float, optional): bound to resize. Defaults to 0.9.
            front_dir (str, optional): front-view direction of the mesh, should be [+-][xyz][ 123]. Defaults to '+z'.
            device (torch.device, optional): torch device. Defaults to None.
        
        Note:
            a ``device`` keyword argument can be provided to specify the torch device. 
            If it's not provided, we will try to use ``'cuda'`` as the device if it's available.

        Returns:
            Mesh: the loaded Mesh object.
        """
        # obj supports face uv
        if path.endswith(".obj"):
            mesh = cls.load_obj(path, **kwargs)
        # trimesh only supports vertex uv, but can load more formats
        else:
            mesh = cls.load_trimesh(path, **kwargs)
        
        # clean
        if clean:
            from kiui.mesh_utils import clean_mesh
            vertices = mesh.v.detach().cpu().numpy()
            triangles = mesh.f.detach().cpu().numpy()
            vertices, triangles = clean_mesh(vertices, triangles, remesh=False)
            mesh.v = torch.from_numpy(vertices).contiguous().float().to(mesh.device)
            mesh.f = torch.from_numpy(triangles).contiguous().int().to(mesh.device)

        # print(f"[INFO] load mesh, v: {mesh.v.shape}, f: {mesh.f.shape}")
        # auto-normalize
        if resize:
            mesh.auto_size(bound=bound)
        # auto-fix normal
        if renormal or mesh.vn is None:
            mesh.auto_normal()
            # print(f"[INFO] load mesh, vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")
        # auto-fix texcoords
        if retex or (mesh.albedo is not None and mesh.vt is None):
            mesh.auto_uv(glctx, cache_path=path, options=retex_options)
            # print(f"[INFO] load mesh, vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")

        # rotate front dir to +z
        if front_dir != "+z":
            # axis switch
            if "-z" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], device=mesh.device, dtype=torch.float32)
            elif "+x" in front_dir:
                T = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "-x" in front_dir:
                T = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "+y" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            elif "-y" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            else:
                T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32)
            # rotation (how many 90 degrees)
            if '1' in front_dir:
                T @= torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            elif '2' in front_dir:
                T @= torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            elif '3' in front_dir:
                T @= torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            mesh.v @= T
            mesh.vn @= T

        return mesh

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None):
        """load an ``obj`` mesh.

        Args:
            path (str): path to mesh.
            albedo_path (str, optional): path to the albedo texture image, will overwrite the existing texture path if specified in mtl. Defaults to None.
            device (torch.device, optional): torch device. Defaults to None.
        
        Note: 
            We will try to read `mtl` path from `obj`, else we assume the file name is the same as `obj` but with `mtl` extension.
            The `usemtl` statement is ignored, and we only use the last material path in `mtl` file.

        Returns:
            Mesh: the loaded Mesh object.
        """
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # load obj
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        mtl_path = None

        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            prefix = split_line[0].lower()
            # mtllib
            if prefix == "mtllib":
                mtl_path = split_line[1]
            # usemtl
            elif prefix == "usemtl":
                pass # ignored
            # v/vn/vt
            elif prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == "vt":
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == "f":
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        if mesh.vt is not None:
            mesh.vt[...,0] = mesh.vt[...,0] - torch.floor(mesh.vt[...,0].mean())
            mesh.vt[...,1] = mesh.vt[...,1] - torch.floor(mesh.vt[...,1].mean())

        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if len(normals) > 0
            else None
        )

        # see if there is vertex color
        use_vertex_color = False
        if mesh.v.shape[1] == 6:
            use_vertex_color = True
            mesh.vc = mesh.v[:, 3:]
            mesh.v = mesh.v[:, :3]
            # print(f"[INFO] load obj mesh: use vertex color: {mesh.vc.shape}")

        # try to load texture image
        if not use_vertex_color:
            # try to retrieve mtl file
            mtl_path_candidates = []
            if mtl_path is not None:
                mtl_path_candidates.append(mtl_path)
                mtl_path_candidates.append(os.path.join(os.path.dirname(path), mtl_path))
            mtl_path_candidates.append(path.replace(".obj", ".mtl"))

            mtl_path = None
            for candidate in mtl_path_candidates:
                if os.path.exists(candidate):
                    mtl_path = candidate
                    break
            
            # if albedo_path is not provided, try retrieve it from mtl
            metallic_path = None
            roughness_path = None
            if mtl_path is not None and albedo_path is None:
                with open(mtl_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0:
                        continue
                    prefix = split_line[0]
                    
                    if "map_Kd" in prefix:
                        # assume relative path!
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        # print(f"[INFO] load obj mesh: use texture from: {albedo_path}")
                    elif "map_Pm" in prefix:
                        metallic_path = os.path.join(os.path.dirname(path), split_line[1])
                    elif "map_Pr" in prefix:
                        roughness_path = os.path.join(os.path.dirname(path), split_line[1])
                    
            # still not found albedo_path, or the path doesn't exist
            ## GSO
            if albedo_path is not None and not os.path.exists(albedo_path):
                albedo_path = os.path.join(os.path.dirname(albedo_path), '../materials/textures/texture.png')
                albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
                albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
                albedo = albedo.astype(np.float32) / 255
                # print(f"[INFO] load obj mesh: load texture: {albedo.shape}")
                mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)
            elif albedo_path is None or not os.path.exists(albedo_path):
                # print(f"[INFO] load obj mesh: failed to load texture!")
                mesh.albedo = None
            else:
                albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
                albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
                albedo = albedo.astype(np.float32) / 255
                # print(f"[INFO] load obj mesh: load texture: {albedo.shape}")
                mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)
            
            # try to load metallic and roughness
            if metallic_path is not None and roughness_path is not None:
                # print(f"[INFO] load obj mesh: load metallicRoughness from: {metallic_path}, {roughness_path}")
                metallic = cv2.imread(metallic_path, cv2.IMREAD_UNCHANGED)
                metallic = metallic.astype(np.float32) / 255
                roughness = cv2.imread(roughness_path, cv2.IMREAD_UNCHANGED)
                roughness = roughness.astype(np.float32) / 255
                metallicRoughness = np.stack([np.zeros_like(metallic), roughness, metallic], axis=-1)

                mesh.metallicRoughness = torch.tensor(metallicRoughness, dtype=torch.float32, device=device).contiguous()

        return mesh

    @classmethod
    def load_trimesh(cls, path, device=None):
        """load a mesh using ``trimesh.load()``.

        Can load various formats like ``glb`` and serves as a fallback.

        Note:
            We will try to merge all meshes if the glb contains more than one, 
            but **this may cause the texture to lose**, since we only support one texture image!

        Args:
            path (str): path to the mesh file.
            device (torch.device, optional): torch device. Defaults to None.

        Returns:
            Mesh: the loaded Mesh object.
        """
        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # use trimesh to load ply/glb
        _data = trimesh.load(path)
        # always convert scene to mesh, and apply all transforms...
        if isinstance(_data, trimesh.Scene):
            # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
            _concat = []
            # loop the scene graph and apply transform to each mesh
            scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
            for k, v in scene_graph.items():
                name = v['geometry']
                if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                    transform = v['transform']
                    _concat.append(_data.geometry[name].apply_transform(transform))
            _mesh = trimesh.util.concatenate(_concat)
        else:
            _mesh = _data
        
        if _mesh.visual.kind == 'vertex':
            vertex_colors = _mesh.visual.vertex_colors
            vertex_colors = np.array(vertex_colors[..., :3]).astype(np.float32) / 255
            mesh.vc = torch.tensor(vertex_colors, dtype=torch.float32, device=device)
            # print(f"[INFO] load trimesh: use vertex color: {mesh.vc.shape}")
        elif _mesh.visual.kind == 'texture':
            _material = _mesh.visual.material
            if isinstance(_material, trimesh.visual.material.PBRMaterial):
                texture = np.array(_material.baseColorTexture).astype(np.float32) / 255
                # load metallicRoughness if present
                if _material.metallicRoughnessTexture is not None:
                    metallicRoughness = np.array(_material.metallicRoughnessTexture).astype(np.float32) / 255
                    mesh.metallicRoughness = torch.tensor(metallicRoughness, dtype=torch.float32, device=device).contiguous()
            elif isinstance(_material, trimesh.visual.material.SimpleMaterial):
                texture = np.array(_material.to_pbr().baseColorTexture).astype(np.float32) / 255
            else:
                raise NotImplementedError(f"material type {type(_material)} not supported!")
            mesh.albedo = torch.tensor(texture[..., :3], dtype=torch.float32, device=device).contiguous()
            # print(f"[INFO] load trimesh: load texture: {texture.shape}")
        else:
            mesh.albedo = None
            # print(f"[INFO] load trimesh: failed to load texture.")

        vertices = _mesh.vertices

        try:
            texcoords = _mesh.visual.uv
            texcoords[:, 1] = 1 - texcoords[:, 1]
        except Exception as e:
            texcoords = None

        try:
            normals = _mesh.vertex_normals
        except Exception as e:
            normals = None

        # trimesh only support vertex uv...
        faces = tfaces = nfaces = _mesh.faces

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if texcoords is not None
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if normals is not None
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if normals is not None
            else None
        )

        return mesh

    # sample surface (using trimesh)
    def sample_surface(self, count: int):
        """sample points on the surface of the mesh.

        Args:
            count (int): number of points to sample.

        Returns:
            torch.Tensor: the sampled points, float [count, 3].
        """
        _mesh = trimesh.Trimesh(vertices=self.v.detach().cpu().numpy(), faces=self.f.detach().cpu().numpy())
        points, face_idx = trimesh.sample.sample_surface(_mesh, count)
        points = torch.from_numpy(points).float().to(self.device)
        return points

    # aabb
    def aabb(self):
        """get the axis-aligned bounding box of the mesh.

        Returns:
            Tuple[torch.Tensor]: the min xyz and max xyz of the mesh.
        """
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self, bound=0.9):
        """auto resize the mesh.

        Args:
            bound (float, optional): resizing into ``[-bound, bound]^3``. Defaults to 0.9.
        """
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 2 * bound / torch.max(vmax - vmin).item()
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal(self):
        """auto calculate the vertex normals.
        """
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        self.vn = vn
        self.fn = self.f

    def auto_uv(self, glctx, cache_path=None, vmap=True, options=None):
        """auto calculate the uv coordinates.

        Args:
            cache_path (str, optional): path to save/load the uv cache as a npz file, this can avoid calculating uv every time when loading the same mesh, which is time-consuming. Defaults to None.
            vmap (bool, optional): remap vertices based on uv coordinates, so each v correspond to a unique vt (necessary for formats like gltf). 
                Usually this will duplicate the vertices on the edge of uv atlas. Defaults to True.
        """
        import xatlas

        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().int().cpu().numpy()
        atlas = xatlas.Atlas()
        if options is not None and options['shuffle']:
            blk_ids = np.arange(min(16, len(f_np)))
            np.random.shuffle(blk_ids)
            print(blk_ids)
            blocks = []
            blk_len = len(f_np) // len(blk_ids)
            for blk_id in blk_ids:
                start = blk_id * blk_len
                end = (blk_id + 1) * blk_len if blk_id != len(blk_ids) -1 else len(f_np)
                blocks.append(f_np[start:end])
            f_np = np.concatenate(blocks, axis=0)
        
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        pack_options = xatlas.PackOptions()
        # chart_options.max_iterations = 4
        if options is not None:
            pack_options.bruteForce = options['bruteForce']
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]
        # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np)
        
        # glctx = dr.RasterizeCudaContext()
        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)
        uv_clip = vt * 2.0 - 1.0
        uv_clip4 = torch.cat(
            (
                uv_clip,
                torch.zeros_like(uv_clip[..., 0:1]),
                torch.ones_like(uv_clip[..., 0:1]),
            ),
            dim=-1,
        )
        # rasterize
        vmapping = torch.from_numpy(vmapping.astype(np.int64)).long().to(self.device)
        rast, _ = dr.rasterize(
            glctx, uv_clip4.unsqueeze(0), ft, (1024, 1024)
        )
        hole_mask = ~(rast[0, :, :, 3] > 0)
        # texc, _ = dr.interpolate(self.vt[vmapping], rast, ft)
        uv_pose, _ = dr.interpolate(self.v[vmapping], rast, ft)
        uv_normal, _ = dr.interpolate(self.vn[vmapping], rast, ft)
        # albedo = dr.texture(self.albedo.unsqueeze(0), texc, filter_mode='linear')
        # print(albedo.shape, self.albedo.shape)
        def uv_padding(image):
            uv_padding_size = 4
            # print(hole_mask.shape, image.shape, rast.shape)
            inpaint_image = (
                cv2.inpaint(
                    (image.detach().cpu().numpy() * 255).astype(np.uint8),
                    (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                    uv_padding_size,
                    cv2.INPAINT_TELEA,
                )
                / 255.0
            )
            return torch.from_numpy(inpaint_image).to(image)
        
        # self.albedo = uv_padding(albedo.squeeze())
        self.uv_pose = uv_pose.squeeze()
        self.uv_normal = uv_normal.squeeze()
        # self.albedo_masked = self.albedo.clone()
        # self.albedo_masked[rast[0, :, :, 3] <= 0] = 1
        self.uv_pose_vis = self.uv_pose.clone()
        self.uv_pose_vis[rast[0, :, :, 3] <= 0] = 1
        self.uv_normal_vis = self.uv_normal.clone()
        self.uv_normal_vis[rast[0, :, :, 3] <= 0] = 1
        
        self.vt = vt
        self.ft = ft

        if vmap:
            self.align_v_to_vt(vmapping)
    
    def align_v_to_vt(self, vmapping=None):
        """ remap v/f and vn/fn to vt/ft.

        Args:
            vmapping (np.ndarray, optional): the mapping relationship from f to ft. Defaults to None.
        """
        if vmapping is None:
            ft = self.ft.view(-1).long()
            f = self.f.view(-1).long()
            vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
            vmapping[ft] = f # scatter, randomly choose one if index is not unique

        self.v = self.v[vmapping]
        self.f = self.ft
        
        if self.vn is not None:
            self.vn = self.vn[vmapping]
            self.fn = self.ft

    def to(self, device):
        """move all tensor attributes to device.

        Args:
            device (torch.device): target device.

        Returns:
            Mesh: self.
        """
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo", "vc", "metallicRoughness"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self
    
    def write(self, path):
        """write the mesh to a path.

        Args:
            path (str): path to write, supports ply, obj and glb.
        """
        if path.endswith(".ply"):
            self.write_ply(path)
        elif path.endswith(".obj"):
            self.write_obj(path)
        elif path.endswith(".glb") or path.endswith(".gltf"):
            self.write_glb(path)
        else:
            raise NotImplementedError(f"format {path} not supported!")
    
    def write_ply(self, path):
        """write the mesh in ply format. Only for geometry!

        Args:
            path (str): path to write.
        """

        if self.albedo is not None:
            print(f'[WARN] ply format does not support exporting texture, will ignore!')

        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        _mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        _mesh.export(path)


    def write_glb(self, path):
        """write the mesh in glb/gltf format.
          This will create a scene with a single mesh.

        Args:
            path (str): path to write.
        """

        # assert self.v.shape[0] == self.vn.shape[0] and self.v.shape[0] == self.vt.shape[0]
        if self.vt is not None and self.v.shape[0] != self.vt.shape[0]:
            self.align_v_to_vt()

        import pygltflib

        f_np = self.f.detach().cpu().numpy().astype(np.uint32)
        f_np_blob = f_np.flatten().tobytes()

        v_np = self.v.detach().cpu().numpy().astype(np.float32)
        v_np_blob = v_np.tobytes()

        blob = f_np_blob + v_np_blob
        byteOffset = len(blob)

        # base mesh
        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[pygltflib.Mesh(primitives=[pygltflib.Primitive(
                # indices to accessors (0 is triangles)
                attributes=pygltflib.Attributes(
                    POSITION=1,
                ),
                indices=0,
            )])],
            buffers=[
                pygltflib.Buffer(byteLength=len(f_np_blob) + len(v_np_blob))
            ],
            # buffer view (based on dtype)
            bufferViews=[
                # triangles; as flatten (element) array
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(f_np_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER, # GL_ELEMENT_ARRAY_BUFFER (34963)
                ),
                # positions; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob),
                    byteLength=len(v_np_blob),
                    byteStride=12, # vec3
                    target=pygltflib.ARRAY_BUFFER, # GL_ARRAY_BUFFER (34962)
                ),
            ],
            accessors=[
                # 0 = triangles
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT, # GL_UNSIGNED_INT (5125)
                    count=f_np.size,
                    type=pygltflib.SCALAR,
                    max=[int(f_np.max())],
                    min=[int(f_np.min())],
                ),
                # 1 = positions
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT, # GL_FLOAT (5126)
                    count=len(v_np),
                    type=pygltflib.VEC3,
                    max=v_np.max(axis=0).tolist(),
                    min=v_np.min(axis=0).tolist(),
                ),
            ],
        )

        # append texture info
        if self.vt is not None:

            vt_np = self.vt.detach().cpu().numpy().astype(np.float32)
            vt_np_blob = vt_np.tobytes()

            albedo = self.albedo.detach().cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
            albedo_blob = cv2.imencode('.png', albedo)[1].tobytes()

            # update primitive
            gltf.meshes[0].primitives[0].attributes.TEXCOORD_0 = 2
            gltf.meshes[0].primitives[0].material = 0

            # update materials
            gltf.materials.append(pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                alphaMode=pygltflib.OPAQUE,
                alphaCutoff=None,
                doubleSided=True,
            ))

            gltf.textures.append(pygltflib.Texture(sampler=0, source=0))
            gltf.samplers.append(pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR_MIPMAP_LINEAR, wrapS=pygltflib.REPEAT, wrapT=pygltflib.REPEAT))
            gltf.images.append(pygltflib.Image(bufferView=3, mimeType="image/png"))

            # update buffers
            gltf.bufferViews.append(
                # index = 2, texcoords; as vec2 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=byteOffset,
                    byteLength=len(vt_np_blob),
                    byteStride=8, # vec2
                    target=pygltflib.ARRAY_BUFFER,
                )
            )

            gltf.accessors.append(
                # 2 = texcoords
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(vt_np),
                    type=pygltflib.VEC2,
                    max=vt_np.max(axis=0).tolist(),
                    min=vt_np.min(axis=0).tolist(),
                )
            )

            blob += vt_np_blob 
            byteOffset += len(vt_np_blob)

            gltf.bufferViews.append(
                # index = 3, albedo texture; as none target
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=byteOffset,
                    byteLength=len(albedo_blob),
                )
            )

            blob += albedo_blob
            byteOffset += len(albedo_blob)

            gltf.buffers[0].byteLength = byteOffset

            # append metllic roughness
            if self.metallicRoughness is not None:
                metallicRoughness = self.metallicRoughness.detach().cpu().numpy()
                metallicRoughness = (metallicRoughness * 255).astype(np.uint8)
                metallicRoughness = cv2.cvtColor(metallicRoughness, cv2.COLOR_RGB2BGR)
                metallicRoughness_blob = cv2.imencode('.png', metallicRoughness)[1].tobytes()

                # update texture definition
                gltf.materials[0].pbrMetallicRoughness.metallicFactor = 1.0
                gltf.materials[0].pbrMetallicRoughness.roughnessFactor = 1.0
                gltf.materials[0].pbrMetallicRoughness.metallicRoughnessTexture = pygltflib.TextureInfo(index=1, texCoord=0)

                gltf.textures.append(pygltflib.Texture(sampler=1, source=1))
                gltf.samplers.append(pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR_MIPMAP_LINEAR, wrapS=pygltflib.REPEAT, wrapT=pygltflib.REPEAT))
                gltf.images.append(pygltflib.Image(bufferView=4, mimeType="image/png"))

                # update buffers
                gltf.bufferViews.append(
                    # index = 4, metallicRoughness texture; as none target
                    pygltflib.BufferView(
                        buffer=0,
                        byteOffset=byteOffset,
                        byteLength=len(metallicRoughness_blob),
                    )
                )

                blob += metallicRoughness_blob
                byteOffset += len(metallicRoughness_blob)

                gltf.buffers[0].byteLength = byteOffset

            
        # set actual data
        gltf.set_binary_blob(blob)

        # glb = b"".join(gltf.save_to_bytes())
        gltf.save(path)


    def write_obj(self, path):
        """write the mesh in obj format. Will also write the texture and mtl files.

        Args:
            path (str): path to write.
        """

        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = path.replace(".obj", "_albedo.png")
        metallic_path = path.replace(".obj", "_metallic.png")
        roughness_path = path.replace(".obj", "_roughness.png")

        v_np = self.v.detach().cpu().numpy()
        vt_np = self.vt.detach().cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.detach().cpu().numpy() if self.vn is not None else None
        f_np = self.f.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.detach().cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f"mtllib {os.path.basename(mtl_path)} \n")

            for v in v_np:
                fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

            if vt_np is not None:
                for v in vt_np:
                    fp.write(f"vt {v[0]} {1 - v[1]} \n")

            if vn_np is not None:
                for v in vn_np:
                    fp.write(f"vn {v[0]} {v[1]} {v[2]} \n")

            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n'
                )

        with open(mtl_path, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")
            if self.albedo is not None:
                fp.write(f"map_Kd {os.path.basename(albedo_path)} \n")
            if self.metallicRoughness is not None:
                # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
                fp.write(f"map_Pm {os.path.basename(metallic_path)} \n")
                fp.write(f"map_Pr {os.path.basename(roughness_path)} \n")

        if self.albedo is not None:
            albedo = self.albedo.detach().cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
        
        if self.metallicRoughness is not None:
            metallicRoughness = self.metallicRoughness.detach().cpu().numpy()
            metallicRoughness = (metallicRoughness * 255).astype(np.uint8)
            cv2.imwrite(metallic_path, metallicRoughness[..., 2])
            cv2.imwrite(roughness_path,  [..., 1])



def remesh(in_mesh_path, out_folder, auto_uv=False):
    options = [
        {'shuffle': False, 'bruteForce': True}
    ]

    obj_name = os.path.basename(out_folder)
    
    glctx = dr.RasterizeCudaContext()
    # mesh = trimesh.load(os.path.join(mesh_folder, 'meshes', 'model.obj'))
    mesh = MeshLoader.load(in_mesh_path, glctx, front_dir='-y', resize=True, clean=False, 
                            renormal=True, bound=0.8, retex=auto_uv, retex_options=options[0])
    mesh.write_obj(f'{out_folder}/{obj_name}.obj')

    # for k, v in vars(mesh).items():
    #     if v is None:
    #         print(k, v, flush=True)
    #     elif isinstance(v, torch.Tensor):
    #         print("=== pt", k, v.size(), flush=True)
    #     else:
    #         print("=== exists", k, type(v), flush=True)
    # print(torch.amin(mesh.vt, dim=0), torch.amax(mesh.vt, dim=0), flush=True)
    
    uv_clip = mesh.vt * 2 - 1
    uv_clip4 = torch.cat(
        (
            uv_clip,
            torch.zeros_like(uv_clip[..., 0:1]),
            torch.ones_like(uv_clip[..., 0:1]),
        ),
        dim=-1,
    )
    rast, _ = dr.rasterize(
        glctx, uv_clip4.unsqueeze(0), mesh.ft, (1024, 1024)
    )
    uv_normal, _ = dr.interpolate(mesh.vn, rast, mesh.ft)
    uv_normal = uv_normal.squeeze()
    uv_normal_vis = uv_normal.clone()
    uv_normal_vis[rast[0, :, :, 3] <= 0] = 1
    uv_normal = (uv_normal_vis.detach().cpu().clamp(-1, 1)).numpy()
    uv_normal = ((uv_normal + 1) / 2 * 255).astype(np.uint8)
    imageio.imwrite(f'{out_folder}/UV_normal.png', uv_normal)
    # resolutions = [256, 32]
    resolutions = [256]
    for res in resolutions:
        render_options = dict(
            pbr=False, envmap=None, front_dir='+z', mode='uv', W=res, H=res, radius=3, fovy=50, elevation=-30
        )
        render_options = namedtuple('render_options', render_options.keys())(*render_options.values())
        renderer = Renderer(render_options)
        for ele in [-30, 0, 30]:
            for start in [0, 23, 45, 68]:
                for mode, ext in zip(['normal', 'albedo', 'uv', 'inv_uv'], ['png', 'png', 'exr', 'exr']):
                    renderer.mode = mode
                    images = []
                    for azi in range(start, 360, 90):
                        renderer.cam.from_angle(-ele, azi)
                        renderer.need_update = True
                        render_buffer = renderer.step(mesh, glctx)
                        if ext == 'png':
                            image = (render_buffer * 255).astype(np.uint8)
                        else:
                            image = render_buffer
                        images.append(image)
                    img = np.vstack([np.hstack(images[:2]), np.hstack(images[2:])])
                    if ext == 'exr':
                        imageio.imwrite(f'{out_folder}/{mode}_{res:03d}_{ele:03d}_{start:03d}.{ext}', img.astype(np.float32))
                        # imageio.imwrite(f'{out_folder}/{mode}_{res:03d}_{ele:03d}_{start:03d}.{ext}', img, \
                        #     flags=imageio.plugins.freeimage.IO_FLAGS.EXR_FLOAT|imageio.plugins.freeimage.IO_FLAGS.EXR_NONE)
                    
                    else:
                        imageio.imwrite(f'{out_folder}/{mode}_{res:03d}_{ele:03d}_{start:03d}.{ext}', img)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--auto_uv", action='store_true', help="if input meshes don't have uv coords")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(args.in_dir, "../results")

    in_root = args.in_dir
    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)

    in_obj_filenames = sorted([x for x in os.listdir(in_root) if x.endswith(".obj")])
    print("==== obj total", len(in_obj_filenames))
    for obj_id, in_obj_fn in tqdm(enumerate(in_obj_filenames)):
        obj_name = in_obj_fn.split(".obj")[0]
        in_mesh_path = os.path.join(in_root, in_obj_fn)
        out_folder = os.path.join(out_root, obj_name)
        os.makedirs(out_folder, exist_ok=True)

        print("=== remeshing", obj_name, flush=True)
        remesh(in_mesh_path, out_folder, args.auto_uv)
        # break 

    print("=== done", flush=True)
