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
import distinctipy
from copy import deepcopy

def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: tuple = (0, 0, 255),
    bg_color_rgb: tuple =  None,
    outline_color_rgb: tuple = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, save_video=True, oss=True):
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
        self.render_size = 344
        self.renderer = Renderer(H=self.render_size, W=self.render_size)
        self.glcx = dr.RasterizeCudaContext()
        self.save_video = save_video
        self.meshroot = "s3://chengwei/data/objs_xatlas"

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, batch):
        root = os.path.join(save_dir, "image_log", split)
        os.makedirs(root, exist_ok=True)
        img_list = []
        view_annot = images['view_annot'].clone()
        
        del images['view_annot']
        for k in sorted(images.keys()):
            grid = torchvision.utils.make_grid(images[k], nrow=len(images[k]))
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            if grid.shape[-1] > 3 and grid.shape[-1] % 3 == 0:
                grid = np.hstack(np.split(grid, grid.shape[-1] // 3, axis=-1))
            grid = cv2.putText(grid.copy(), k, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            img_list.append(grid)
        out = np.vstack(img_list)

        if not self.save_video:
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format("img", global_step, current_epoch, batch_idx)
            imageio.imwrite(f'{root}/{filename}', out)
        else:
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.mp4".format("img", global_step, current_epoch, batch_idx)
            legend, mask = self.get_legend_mask()
            renderings = []
            B = len(images['reconstruction'])
            key = [k for k in images.keys() if 'samples_cfg_scale_' in k][0]
            # for i in range(B):
            i = 0
            B = 1
            obj_id = batch['data_id'][i]
            mesh = Mesh.load(f'{self.meshroot}/{obj_id}/{obj_id}.obj')
            # imageio.imwrite('test.png', (self.uv_maps[i][:3].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            for albedo in [images['reconstruction'][i], self.uv_maps[i][:3], view_annot[i]]:
                mesh.albedo = (albedo.permute(1,2,0).to(mesh.device).float().contiguous() + 1) / 2
                for azi in np.arange(0, 360, 3, dtype=np.int32):
                    self.renderer.cam.from_angle(-30, azi)
                    self.renderer.need_update = True
                    render_buffer = self.renderer.step(mesh, self.glcx)
                    render_buffer = render_buffer * (1-mask) + legend * mask
                    renderings.append(render_buffer)
            renderings = torch.stack(renderings, dim=0).reshape(B, 3, 120, self.render_size, self.render_size, 3).permute(2, 1, 0, 5, 3, 4)
            video = []
            
            for frame in renderings:
                img_list = [out]
                grid = torchvision.utils.make_grid(frame.squeeze(), nrow=1, padding=2)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                img_list.append(grid)
                grid = np.hstack(img_list)
                video.append(grid)
            imageio.mimwrite(f'{root}/{filename}', video, fps=30, quality=8, macro_block_size=1)
        # exit(0)

    def get_legend_mask(self):
        # print(colors)
        # colors = distinctipy.get_colors(5)
        height = int(self.render_size * 0.2 / 5)
        image = np.ones([self.render_size, self.render_size, 3], dtype=np.float32)
        for i, color in enumerate(self.colors):
            image = add_text_to_image(
                image,
                "UV" if i==0 else f'view{i:d}',
                font_scale=0.5,
                font_color_rgb=color,
                # outline_color_rgb=(0, 0, 0),
                top_left_xy=(10, height*i+10),
                line_spacing=1.5,
                font_face=cv2.FONT_HERSHEY_TRIPLEX,
            )
        # imageio.imwrite("test.png", (image*255).astype(np.uint8))
        mask = np.sum(image == 1, axis=-1, keepdims=True) != 3
        return image, mask

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
            self.colors = deepcopy(images["colors"])
            self.uv_maps = deepcopy(images["uv_maps"])
            del images["colors"]
            del images["uv_maps"]

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
