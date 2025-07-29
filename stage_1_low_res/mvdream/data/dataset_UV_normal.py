import json
import cv2, imageio
import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from mvdream.camera_utils import get_camera

def cv2_loader(path: str) -> Image.Image:
    img = cv2.imread(path)
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img


def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def gather_prompt(prompt):
    out = ''
    for key in ['category', 'description', 'texture', 'material']:
        out += (prompt[key] + ', ') if prompt[key] is not None else ''
    return out[:-2]

def get_viewdirs(num_frames, elevation, azimuth_start, radius=1, azimuth_span=360):
    angle_gap = azimuth_span / num_frames
    viewdirs = []
    for azimuth in np.arange(azimuth_start, azimuth_span+azimuth_start, angle_gap):
        elevation = np.radians(elevation)
        azimuth = np.radians(azimuth)
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = - radius * np.sin(elevation)
        z = radius * np.cos(elevation) * np.cos(azimuth)
        viewdirs.append([x, y, z])
    viewdirs = np.array(viewdirs)
    viewdirs = viewdirs / (np.linalg.norm(viewdirs, axis=-1, keepdims=True)+1e-8)
    return viewdirs

class UVDataset(Dataset):
    def __init__(self, root, use_uv, imsize=512):
        self.root = root
        self.data = np.loadtxt(f'{root}/objs_xatlas3.txt', dtype=np.str_)
        self.atlas_num = 5
        self.prompt = json.load(open(f'{root}/llm_annot.json'))
        self.elevs = [0, 15, 30]
        self.azis = [0, 15, 30, 45]
        self.elevs_num = len(self.elevs)
        self.azis_num = len(self.azis)
        self.use_uv = use_uv
        self.size = imsize


    def __len__(self):
        return len(self.data) * 12

    def __getitem__(self, idx):

        obj_id = idx // (self.elevs_num * self.azis_num)
        ele_id = (idx % (self.elevs_num * self.azis_num)) // self.azis_num
        azi_id = (idx % (self.elevs_num * self.azis_num)) % self.azis_num
        ele = self.elevs[ele_id]
        azi = self.azis[azi_id]
        obj_id = self.data[obj_id]
        num_frames = 4
        
        prompt = gather_prompt(self.prompt[obj_id])

        src = imageio.imread(f'{self.root}/{obj_id}/normal_{ele:03d}_{azi:03d}.png')[:,:,:3]
        src_mask = (np.sum(src == 255, axis=-1, keepdims=True) != 3).astype(np.uint8)
        src = src * src_mask
        tgt = imageio.imread(f'{self.root}/{obj_id}/albedo_{ele:03d}_{azi:03d}.png')[:,:,:3]

        if self.use_uv == 'inv_uv_warp':
            uv = imageio.imread(f'{self.root}/{obj_id}/inv_uv_{ele:03d}_{azi:03d}.exr')[:,:,:2]
        else:
            uv = imageio.imread(f'{self.root}/{obj_id}/uv_{ele:03d}_{azi:03d}.exr')[:,:,:2]
        src, tgt, uv, src_mask = self.grid2batch(src), self.grid2batch(tgt), self.grid2batch(uv), self.grid2batch(src_mask)
        uv = torch.from_numpy(np.stack(uv))#.permute(0, 3, 1, 2)

        cos_angles = None
        if self.use_uv is not None:
            uv_albedo = imageio.imread(f'{self.root}/{obj_id}/{obj_id}_albedo.png')[:,:,:3]
            uv_normal = imageio.imread(f'{self.root}/{obj_id}/UV_normal.png')[:,:,:3]
            uv_normal_mask = (np.sum(uv_normal == 255, axis=-1, keepdims=True) != 3).astype(np.uint8)
            uv_normal = uv_normal * uv_normal_mask

            normals = (np.stack(src).astype(np.float32) / 255. * 2 - 1) * np.stack(src_mask).astype(np.float32)[...,None]
            normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
            viewdirs = get_viewdirs(num_frames, elevation=ele, azimuth_start=azi)
            cos_angles = np.sum(viewdirs[:, None, :] * normals.reshape(normals.shape[0], -1, 3), axis=-1).reshape(num_frames, normals.shape[1], normals.shape[2], 1)
            cos_angles = torch.from_numpy(cos_angles).clamp(0, 1)
            cos_angles[cos_angles==0] = -1
            # for i, cos_angle in enumerate(cos_angles.numpy()):
            #     imageio.imwrite(f'dataloader_{i:03d}.png', (cos_angle[:,:,0]*255).astype(np.uint8))
            src = [cv2.resize(uv_normal, (self.size, self.size), cv2.INTER_CUBIC)] + src
            tgt = [cv2.resize(uv_albedo, (self.size, self.size), cv2.INTER_CUBIC)] + tgt
        transform1 = T.Compose([
            T.Resize(size=256),
            T.ToTensor(),
        ])
        # print(Image.fromarray(src[0]).size)
        src = torch.stack([transform1(Image.fromarray(img)) for img in src], dim=0).permute(0, 2, 3, 1)
        tgt = torch.stack([transform1(Image.fromarray(img)) for img in tgt], dim=0).permute(0, 2, 3, 1)

        tgt = (tgt * 2) - 1
        tgt_mask = torch.sum(src == 0, dim=-1, keepdim=True) != 3
        tgt[1:] = tgt[1:] * tgt_mask[1:]

        
        camera = get_camera(num_frames, elevation=ele, azimuth_start=azi)
        prompt = [prompt] * num_frames
        if self.use_uv:
            num_frames += 1
            camera = torch.cat([torch.ones_like(camera[:1]), camera], dim=0)
            prompt = ['UV map, '+prompt[0]] + prompt
        
        return dict(jpg=tgt, txt=prompt, hint=src, uv=uv, cos_angles=cos_angles, camera=camera, num_frames=num_frames, data_id=obj_id, atlas_id='0')

    def grid2batch(self, grid):
        grids = [cv2.resize(x, (self.size, self.size), cv2.INTER_CUBIC) 
                for patch in np.split(grid, 2, axis=0)
                for x in np.split(patch, 2, axis=1)]
        
        return grids


