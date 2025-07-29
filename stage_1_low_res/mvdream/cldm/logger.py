import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from utils.renderer import Renderer, Mesh
import nvdiffrast.torch as dr
from tqdm import tqdm, trange
import imageio

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, save_video=True):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_images_kwargs['N'] = self.max_images
        self.log_first_step = log_first_step
        self.render_size = 424
        self.renderer = Renderer(H=self.render_size, W=self.render_size)
        self.glcx = dr.RasterizeCudaContext()
        self.save_video = save_video

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, batch):
        root = os.path.join(save_dir, "image_log", split)
        os.makedirs(root, exist_ok=True)
        meshroot = '/mnt/share/cq8/wadecheng/dataset/objs_xatlas3'
        img_list = []
        view_annot = images['view_annot'].clone()
        print(view_annot.shape, images['reconstruction'].shape)
        del images['view_annot']
        for k in sorted(images.keys()):
            # print(images[k].shape)
            grid = torchvision.utils.make_grid(images[k], nrow=1)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            if grid.shape[-1] > 3 and grid.shape[-1] % 3 == 0:
                grid = np.hstack(np.split(grid, grid.shape[-1] // 3, axis=-1))
            grid = cv2.putText(grid.copy(), k, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            img_list.append(grid)
        out = np.hstack(img_list)

        if not self.save_video:
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format("img", global_step, current_epoch, batch_idx)
            imageio.imwrite(f'{root}/{filename}', out)
        else:
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.mp4".format("img", global_step, current_epoch, batch_idx)
        
            renderings = []
            B = len(images['reconstruction'])
            key = [k for k in images.keys() if 'samples_cfg_scale_' in k][0]
            # for i in range(B):
            i = 0
            B = 1
            obj_id = batch['data_id'][i]
            # atlas_id = batch['atlas_id'][i]
            mesh = Mesh.load(f'{meshroot}/{obj_id}/{obj_id}.obj')
            for albedo in [images['reconstruction'][i], images[key][i], view_annot[i]]:
                mesh.albedo = (albedo.permute(1,2,0).to(mesh.device).float().contiguous() + 1) / 2
                for azi in np.arange(0, 360, 3, dtype=np.int32):
                    self.renderer.cam.from_angle(-30, azi)
                    self.renderer.need_update = True
                    render_buffer = self.renderer.step(mesh, self.glcx)
                    renderings.append(render_buffer)
            renderings = torch.stack(renderings, dim=0).reshape(B, 3, 120, self.render_size, self.render_size, 3).permute(2, 1, 0, 5, 3, 4)
            video = []
            for frame in renderings:
                img_list = [out]
                # for i, k in enumerate(['reconstruction', key]):
                # print(frame.shape)
                grid = torchvision.utils.make_grid(frame.squeeze(), nrow=1, padding=5)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                # grid = cv2.putText(grid.copy(), k, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                img_list.append(grid)
                grid = np.hstack(img_list)
                video.append(grid)
            imageio.mimwrite(f'{root}/{filename}', video, fps=30, quality=8, macro_block_size=1)


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx, batch)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
