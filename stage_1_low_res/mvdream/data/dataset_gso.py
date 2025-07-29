import json, io, os
import cv2, imageio
import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from mvdream.camera_utils import get_camera
from megfile import smart_open
import tarfile
import imageio.v2 as imageio
import torch.nn.functional as F

def extract_oss_file(oss_url, file_dict, verbose=True):
    with smart_open(oss_url, 'rb') as f:
        tar_content = f.read()

    tar_stream = io.BytesIO(tar_content)
    res = {}
    with tarfile.open(fileobj=tar_stream, mode='r') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name in file_dict.keys():
                filename, key = member.name, file_dict[member.name]
                file_obj = tar.extractfile(member)
                res[key] = imageio.imread(file_obj, os.path.splitext(filename)[-1])
    return res


def cv2_loader(path: str) -> Image.Image:
    img = cv2.imread(path)
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def erode_torch_batch(binary_img_batch, kernel_size):
    pad = (kernel_size - 1) // 2
    bin_img = F.pad(binary_img_batch.unsqueeze(1), pad=[pad, pad, pad, pad], mode='reflect')
    out = -F.max_pool2d(-bin_img, kernel_size=kernel_size, stride=1, padding=0)
    out = out.squeeze(1)
    return out


def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def gather_prompt(prompt):
    out = 'hd, detailed, highly_textured'
    for key in ['Category', 'Texture', 'Characteristics']:
        out += (prompt[key] + ', ') if prompt[key] is not None else ''
    return out

def get_viewdirs(num_frames, elevation, azimuth_start, radius=1, azimuth_span=360):
    angle_gap = azimuth_span / num_frames
    viewdirs = []
    eles, azis = [], []
    for azimuth in np.arange(azimuth_start, azimuth_span+azimuth_start, angle_gap):
        eles.append(elevation); azis.append(azimuth)
        elevation_radius = np.radians(-elevation)
        azimuth_radius = np.radians(azimuth)
        x = radius * np.cos(elevation_radius) * np.sin(azimuth_radius)
        y = - radius * np.sin(elevation_radius)
        z = radius * np.cos(elevation_radius) * np.cos(azimuth_radius)
        viewdirs.append([x, y, z])
    viewdirs = np.array(viewdirs)
    viewdirs = viewdirs / (np.linalg.norm(viewdirs, axis=-1, keepdims=True)+1e-8)
    return viewdirs, eles, azis

class UVDataset(Dataset):
    def __init__(self, root, use_uv, imsize=512, pid=0, pnum=1):
        self.root = root
        self.data = sorted(np.loadtxt('./gso.txt', dtype=np.str_))
        # self.data = ["Glycerin_11_Color_BrllntBluSkydvrSlvrBlckWht_Size_80", "JS_WINGS_20_BLACK_FLAG", "Olive_Kids_Birdie_Pack_n_Snack", "Reebok_SOMERSET_RUN", "Schleich_Hereford_Bull" "Squirrel"]
        self.data = ["Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure"]
        num = int(len(self.data) / pnum)
        start = pid * num
        end = (pid + 1) * num if pid != (pnum - 1) else len(self.data)
        self.data = self.data[start:end]
        # self.data = ['000213ffa75749729c5a69e3d30519a9']
        self.atlas_num = 5
        self.prompt = json.load(open('./gso_annot.json'))
        self.azis = [0, 23, 45, 68]
        self.azis_num = len(self.azis)
        self.use_uv = use_uv
        self.size = imsize


    def __len__(self):
        return len(self.data) #* self.azis_num

    def __getitem__(self, idx):

        # obj_id = idx // self.azis_num
        # azi_id = idx % self.azis_num
        obj_id = idx
        
        obj_id = self.data[obj_id]
        num_frames = 4
        ele1, ele2 = -30, 30
        # azi1, azi2 = self.azis[azi_id], self.azis[(azi_id+self.azis_num//2)%self.azis_num]
        azi1, azi2 = 0, 45
        
        prompt = gather_prompt(self.prompt[obj_id])

        file_dict = {
            f'normal_256_{ele1:03d}_{azi1:03d}.png': 'src1',
            f'albedo_256_{ele1:03d}_{azi1:03d}.png': 'tgt1',
            f'inv_uv_256_{ele1:03d}_{azi1:03d}.exr': 'inv_uv1',
            f'uv_256_{ele1:03d}_{azi1:03d}.exr': 'uv1',
            f'normal_256_{ele2:03d}_{azi2:03d}.png': 'src2',
            f'albedo_256_{ele2:03d}_{azi2:03d}.png': 'tgt2',
            f'inv_uv_256_{ele2:03d}_{azi2:03d}.exr': 'inv_uv2',
            f'uv_256_{ele2:03d}_{azi2:03d}.exr': 'uv2',
            f'{obj_id}_albedo.png': 'uv_albedo',
            f'UV_normal.png': 'uv_normal',
        }
        
        cos_angles_viss = []
        srcs, tgts, coses, uvs, inv_uvs, bound_coses = [], [], [], [], [], []
        for p, ele, azi in zip(["1", "2"], [ele1, ele2], [azi1, azi2]):
            tar_content = extract_oss_file(os.path.join(self.root, f'{obj_id}.tar'), file_dict)

            src = tar_content[f'src{p}'][:,:,:3] #imageio.imread(f'{self.root}/{obj_id}/normal_{ele:03d}_{azi:03d}.png')[:,:,:3]
            src_mask = (np.sum(src == 255, axis=-1, keepdims=True) != 3).astype(np.uint8)
            src = src * src_mask
            tgt = tar_content[f'tgt{p}'][:,:,:3] #imageio.imread(f'{self.root}/{obj_id}/albedo_{ele:03d}_{azi:03d}.png')[:,:,:3]

            inv_uv = tar_content[f'inv_uv{p}'][:,:,:2] #imageio.imread(f'{self.root}/{obj_id}/inv_uv_{ele:03d}_{azi:03d}.exr')[:,:,:2]
            uv = tar_content[f'uv{p}'][:,:,:2] # imageio.imread(f'{self.root}/{obj_id}/uv_{ele:03d}_{azi:03d}.exr')[:,:,:2]
            src, tgt, uv, inv_uv, src_mask = self.grid2batch(src), self.grid2batch(tgt), self.grid2batch(uv, 32), self.grid2batch(inv_uv), self.grid2batch(src_mask[...,0])

            cos_angles = None
            if self.use_uv is not None:
                normals = (np.stack(src).astype(np.float32) / 255. * 2 - 1) * np.stack(src_mask).astype(np.float32)[...,None]
                normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
                viewdirs, eles, azis = get_viewdirs(num_frames, elevation=ele, azimuth_start=azi)
                cos_angles = np.sum(viewdirs[:, None, :] * normals.reshape(normals.shape[0], -1, 3), axis=-1).reshape(num_frames, normals.shape[1], normals.shape[2], 1)

                ## For Debug
                # print("data_id", obj_id)
                # cos_angles_vis = np.concatenate([cos_angles]*3, axis=-1)
                # cos_angles_vis[cos_angles[...,0]<0] = [1,0,0]
                # cos_angles_viss.append(cos_angles_vis)
                # for normal, cos_angle, viewdir, ele, azi in zip(normals, cos_angles, viewdirs, eles, azis):
                #     print(ele, azi)
                #     print(normal[cos_angle[...,0]<0].mean(0), viewdir, normal[cos_angle[...,0]<0].mean(0).dot(viewdir), flush=True)
                cos_angles = torch.from_numpy(cos_angles).clamp(0, 1)
                cos_angles[cos_angles==0] = -1

            srcs += src
            tgts += tgt
            uvs += uv
            inv_uvs += inv_uv
            coses.append(cos_angles)
        # cos_angles_viss = (np.hstack(np.concatenate(cos_angles_viss, axis=0).tolist())*255).astype(np.uint8)
        # imageio.imwrite("cos_angle.png", cos_angles_viss)

        cos_angles = torch.cat(coses, dim=0)
        uv = torch.from_numpy(np.stack(uvs))#.permute(0, 3, 1, 2)
        inv_uv = torch.from_numpy(np.stack(inv_uvs))
        inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1) == 2).float()
        inv_uv_mask_solid = erode_torch_batch(inv_uv_mask, 5).float()
        inv_uv_mask_boundary = inv_uv_mask - inv_uv_mask_solid
        inv_uv = torch.cat([inv_uv, inv_uv_mask_solid[...,None], inv_uv_mask_boundary[...,None]], dim=-1)

        # if self.use_uv is not None:
        #     uv_albedo = tar_content['uv_albedo'][:,:,:3] # imageio.imread(f'{self.root}/{obj_id}/{obj_id}_albedo.png')[:,:,:3]
        #     uv_normal = tar_content['uv_normal'][:,:,:3] # imageio.imread(f'{self.root}/{obj_id}/UV_normal.png')[:,:,:3]
        #     uv_normal_mask = (np.sum(uv_normal == 255, axis=-1, keepdims=True) != 3).astype(np.uint8)
        #     uv_normal = uv_normal * uv_normal_mask

        #     src = [cv2.resize(uv_normal, (self.size, self.size), cv2.INTER_CUBIC)] + srcs
        #     tgt = [cv2.resize(uv_albedo, (self.size, self.size), cv2.INTER_CUBIC)] + tgts
    
        transform1 = T.Compose([
            T.Resize(size=256),
            T.ToTensor(),
        ])
        # print(Image.fromarray(src[0]).size)
        src = torch.stack([transform1(Image.fromarray(img)) for img in srcs], dim=0).permute(0, 2, 3, 1)
        tgt = torch.stack([transform1(Image.fromarray(img)) for img in tgts], dim=0).permute(0, 2, 3, 1)

        tgt = (tgt * 2) - 1
        tgt_mask = torch.sum(src == 0, dim=-1, keepdim=True) != 3
        tgt = tgt * tgt_mask

        camera1 = get_camera(num_frames, elevation=ele1, azimuth_start=azi1)
        camera2 = get_camera(num_frames, elevation=ele2, azimuth_start=azi2)
        prompt = [prompt] * num_frames * 2
        if self.use_uv:
            num_frames = num_frames * 2
            camera = torch.cat([camera1, camera2], dim=0)
            # prompt = ['UV map, '+prompt[0]] + prompt
        # print(len(prompt), camera.shape, src.shape, tgt.shape, num_frames)
        return dict(jpg=tgt, txt=prompt, hint=src, uv=uv, inv_uv=inv_uv, cos_angles=cos_angles, 
                     camera=camera, num_frames=num_frames, data_id=obj_id, atlas_id='0')

    def grid2batch(self, grid, size=256):
        # grids = [cv2.resize(x, (size, size), cv2.INTER_NEAREST) 
        #         for patch in np.split(grid, 2, axis=0)
        #         for x in np.split(patch, 2, axis=1)]
        grids = [x for patch in np.split(grid, 2, axis=0)
                for x in np.split(patch, 2, axis=1)]
        return grids

if __name__ == "__main__":
    import nvdiffrast.torch as dr
    num_frames = 8
    dataset = UVDataset(root=f"s3://chengwei/data/gso_v2/", use_uv=True)
    
    uv_temperature = 10
    uv_maps, view_uv_maps = [], []
    # for data in dataset:
    data = dataset[0]
    x, hint, inv_uv, uv, cos_angle = data['jpg'].to("cuda"), data['hint'].to("cuda"), data['inv_uv'].to("cuda"), data['uv'].to("cuda"), data['cos_angles'].to("cuda")
    img = imageio.imread('./result-gso3/Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure.png')
    img = [img[:, i*256:(i+1)*256] for i in range(8)]
    x = torch.from_numpy(np.stack(img).astype(np.float32)).to("cuda") / 255.
    inv_uv, inv_uv_mask_valid, inv_uv_mask_boundary = torch.split(inv_uv, [2,1,1], dim=-1)
    # texture_x0, view_x0 = torch.split(x, [1, num_frames], dim=0)
    view_x0 = x

    partial_albedos = dr.texture(view_x0.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
    inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
    partial_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask).contiguous(), filter_mode='linear')
    # self.print_tensor("partial_weights", partial_weights); 
    
    alpha = (torch.sum(inv_uv_mask > 0, dim=0, keepdim=True) > 0).float()
    alpha2 = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()

    uv_temperature = 10
    origin_texture = torch.sum(partial_albedos * torch.softmax(partial_weights.double() * uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float32) 
    origin_texture = origin_texture * alpha2
    # print(origin_texture.shape)
    out = (origin_texture[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    imageio.imwrite("textured.png", out)


    prev_views = dr.texture(origin_texture.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
    uv_mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
    prev_views = prev_views * uv_mask
    
    if True:
        partial_albedos = torch.cat([torch.cat([alpha]*3, dim=-1), torch.cat([alpha2]*3, dim=-1), torch.cat([alpha - alpha2]*3, dim=-1), partial_albedos], dim=0)
        partial_albedos = (np.clip((partial_albedos.cpu().numpy()+1)/2, 0, 1)*255)
        partial_albedos = np.hstack(partial_albedos.tolist()).astype(np.uint8)
        imageio.imwrite('partial_albedos.png', partial_albedos)
        new_textured = (np.clip((prev_views.cpu().numpy()+1)/2, 0, 1)*255)
        new_textured = np.hstack(new_textured.tolist()).astype(np.uint8)
        imageio.imwrite('new_textured.png', new_textured)

        inv_uv = (np.clip(inv_uv.cpu().numpy(), 0, 1)*255)
        inv_uv = np.hstack(inv_uv.tolist()).astype(np.uint8)
        inv_uv = np.concatenate([inv_uv, np.zeros_like(inv_uv[...,:1])], axis=-1)
        imageio.imwrite('inv_uv.png', inv_uv)
        uv = (np.clip(uv.cpu().numpy(), 0, 1)*255)
        uv = np.hstack(uv.tolist()).astype(np.uint8)
        uv = np.concatenate([uv, np.zeros_like(uv[...,:1])], axis=-1)
        imageio.imwrite('uv.png', uv)

if __name__ == "__main2__":
    # obj_id = 'e7a5b206416c4c4a9ab7c347cdb47857'
    # ele = -30
    # azi = 0
    # file_dict = {
    #     f'normal_256_{ele:03d}_{azi:03d}.png': 'src',
    #     f'albedo_256_{ele:03d}_{azi:03d}.png': 'tgt',
    #     f'inv_uv_256_{ele:03d}_{azi:03d}.exr': 'inv_uv',
    #     f'uv_256_{ele:03d}_{azi:03d}.exr': 'uv',
    #     f'{obj_id}_albedo.png': 'uv_albedo',
    #     f'UV_normal.png': 'uv_normal',
    # }
    # file_dict = extract_oss_file(f"s3://chengwei/data/objs_xatlas_v2/{obj_id}.tar", file_dict)
    # imageio.imwrite('uv.png', (file_dict[f'uv']*255).astype(np.uint8))
    from torchvision.transforms import Resize
    def prepare_view_visualization(ref_tensor):
        import distinctipy
        colors = distinctipy.get_colors(ref_tensor.shape[0])
        color_tensor = []
        for i, color in enumerate(colors):
            c = torch.zeros_like(ref_tensor[i])
            for j, channel in enumerate(color):
                c[...,j] = channel
            color_tensor.append(c)
        return colors, torch.stack(color_tensor, dim=0)
    import nvdiffrast.torch as dr

    # dr.RasterizeCudaContext()
    num_frames = 8
    dataset = UVDataset(root=f"s3://chengwei/data/gso_v2/", use_uv=True)
    
    uv_temperature = 10
    uv_maps, view_uv_maps = [], []
    # for data in dataset:
    data = dataset[0]
    x, hint, inv_uv, uv, cos_angle = data['jpg'].to("cuda"), data['hint'].to("cuda"), data['inv_uv'].to("cuda"), data['uv'].to("cuda"), data['cos_angles'].to("cuda")
    # print(x.shape, inv_uv.shape, uv.shape, cos_angle.shape)
    # for i in range(10):
    inv_uv, inv_uv_mask_valid, inv_uv_mask_boundary = torch.split(inv_uv, [2,1,1], dim=-1)
    colors, color_tensor = prepare_view_visualization(x)
    albedo, x = torch.split(x, [1, num_frames], dim=0)
    inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
    partial_albedos = dr.texture(x.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
    partial_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask_valid).contiguous(), filter_mode='linear')
    boundary_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask_boundary).contiguous(), filter_mode='linear')
    
    alpha = (torch.sum(inv_uv_mask, dim=0, keepdim=True) > 0).float()
    alpha2 = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()
    alpha3 = (torch.sum(boundary_weights > 0, dim=0, keepdim=True) > 0).float()
    # partial_albedos, inv_uv_mask = torch.cat([albedo, partial_albedos], dim=0), torch.cat([torch.ones_like(inv_uv_mask[:1]), inv_uv_mask], dim=0)

    # partial_weights = torch.cat([torch.ones_like(inv_uv_mask[:1]), partial_weights * self.uv_temperature], dim=0)
    aggreated_albedos = torch.sum(partial_albedos * torch.softmax(partial_weights.double() * uv_temperature, dim=0), dim=0, keepdim=True) #.to(dtype=torch.float16) 
    aggreated_boundary_albedos = torch.sum(partial_albedos * torch.softmax(boundary_weights.double() * uv_temperature, dim=0), dim=0, keepdim=True) #.to(dtype=torch.float16) 
    part1, part2 = aggreated_albedos * alpha2, aggreated_boundary_albedos * (alpha - alpha2)
    aggreated_albedos = aggreated_albedos * alpha2 + aggreated_boundary_albedos * (alpha - alpha2)
    aggreated_albedos_np = np.clip(aggreated_albedos.cpu().numpy()[0], -1, 1)
    aggreated_albedos_np = cv2.inpaint(((aggreated_albedos_np+1)*255/2).astype(np.uint8), ((1-alpha).cpu().numpy()[0]*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    aggreated_albedos = torch.from_numpy(aggreated_albedos_np).to(device="cuda", dtype=torch.float16)[None,...] / 255. * 2 - 1
    uv_maps.append(torch.cat([aggreated_albedos, alpha], dim=-1))
    new_textured = dr.texture(aggreated_albedos.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
    uv_mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
    new_textured = new_textured * uv_mask
    x = torch.cat([aggreated_albedos, new_textured], dim=0)
        # print(x.shape)

    # if True:
    #     color = color_tensor
    #     c_albedo, c_x = torch.split(color, [1, num_frames], dim=0)
    #     partial_c_albedos = dr.texture(c_x.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
    #     partial_c_albedos = torch.cat([c_albedo, partial_c_albedos], dim=0)
    #     # fake a uv color
    #     partial_weights = torch.cat([torch.ones_like(inv_uv_mask[:1]), partial_weights], dim=0)
    #     aggreated_c_albedos = torch.sum(partial_c_albedos * torch.softmax(partial_weights.double(), dim=0), dim=0, keepdim=True).to(dtype=torch.float16)
    #     view_uv_maps.append(aggreated_c_albedos.permute(0,3,1,2)) 
    
    # if True:
    #     partial_albedos = torch.cat([albedo, aggreated_albedos, part1, part2, torch.cat([alpha]*3, dim=-1), torch.cat([alpha2]*3, dim=-1), torch.cat([alpha - alpha2]*3, dim=-1), partial_albedos], dim=0)
    #     partial_albedos = (np.clip((partial_albedos.cpu().numpy()+1)/2, 0, 1)*255)
    #     partial_albedos = np.hstack(partial_albedos.tolist()).astype(np.uint8)
    #     imageio.imwrite('partial_albedos.png', partial_albedos)
    #     new_textured = (np.clip((new_textured.cpu().numpy()+1)/2, 0, 1)*255)
    #     new_textured = np.hstack(new_textured.tolist()).astype(np.uint8)
    #     imageio.imwrite('new_textured.png', new_textured)

    #     inv_uv = (np.clip(inv_uv.cpu().numpy(), 0, 1)*255)
    #     inv_uv = np.hstack(inv_uv.tolist()).astype(np.uint8)
    #     inv_uv = np.concatenate([inv_uv, np.zeros_like(inv_uv[...,:1])], axis=-1)
    #     imageio.imwrite('inv_uv.png', inv_uv)
    #     uv = (np.clip(uv.cpu().numpy(), 0, 1)*255)
    #     uv = np.hstack(uv.tolist()).astype(np.uint8)
    #     uv = np.concatenate([uv, np.zeros_like(uv[...,:1])], axis=-1)
    #     imageio.imwrite('uv.png', uv)

    #     # imageio.imwrite(f'008.png', (np.clip((aggreated_albedos[0].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
    #     # for j, t in enumerate(partial_albedos[:8].cpu().numpy()):
    #     #     imageio.imwrite(f'x_{j:03d}.png', (np.clip((x[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
    #     #     imageio.imwrite(f'new_x_{j:03d}.png', (np.clip((new_textured[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
    #     #     imageio.imwrite(f'{j:03d}.png', (np.clip((t+1)/2, 0, 1)*255).astype(np.uint8))
    #     #     imageio.imwrite(f'mask_{j:03d}.png', (np.clip(inv_uv_mask[j,:,:,0].cpu().numpy(), 0, 1)*255).astype(np.uint8))
    #     #     imageio.imwrite(f'inv_uv_{j:03d}.png', (np.clip(inv_uv[j].cpu().numpy(), 0, 1)*255).astype(np.uint8))
        # break
