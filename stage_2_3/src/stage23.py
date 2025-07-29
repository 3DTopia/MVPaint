import os
from typing import Any, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.schedulers import DDPMScheduler, KarrasDiffusionSchedulers
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from .renderer.project import UVProjection as UVP
from .syncmvd.attention import replace_attention_processors
from .syncmvd.step import step_tex
from .utils import *
from torchvision import transforms
import kaolin as kal
from scipy.ndimage import binary_dilation
from .renderer.voronoi import voronoi_solve
from torchvision.transforms import Compose, Resize
from .stage2 import SpatialAware3DInpainting
from .stage3 import UVRefinement

import pdb
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

@torch.no_grad()
def stable_upscale(upscaler, low_res_texture, prompt, negative_prompt, guidance_scale=None, num_inference_steps=None,
                   generator=None, out_res=2048):
    result_tex_rgb = low_res_texture.cpu()  # 移动到 CPU
    result_tex_rgb = (result_tex_rgb * 255).byte()  # 转换为 0-255 的范围并转为 byte 类型

    # 转换为 numpy 数组并转换为 [1024, 1024, 3] 的形状
    result_tex_rgb_np = result_tex_rgb.numpy()  # permute 用于调整维度顺序

    # 使用 PIL 创建图像
    low_res_texture = Image.fromarray(result_tex_rgb_np)
    input_size = int(out_res / 4)
    low_res_texture = low_res_texture.resize((input_size, input_size))

    # low_res_texture = low_res_texture.resize(out_res/4, out_res/4)
    image = upscaler(prompt,
                     image=low_res_texture,
                     negative_prompt=negative_prompt,
                     ).images[0]
    return image

@torch.no_grad()
def tile_upscale(upscaler, low_res_texture, prompt, negative_prompt, out_res=2048, position_map=None):

    result_tex_rgb = low_res_texture.cpu()  # 移动到 CPU
    result_tex_rgb = (result_tex_rgb * 255).byte()  # 转换为 0-255 的范围并转为 byte 类型

    result_tex_rgb_np = result_tex_rgb.numpy()  # permute 用于调整维度顺序

    # 使用 PIL 创建图像
    image = Image.fromarray(result_tex_rgb_np)

    image = image.resize((out_res, out_res))

    img_list = [image]

    if position_map is not None:
        position_map = position_map.squeeze().cpu()
        position_map = (position_map * 255).byte()
        position_map_np = position_map.numpy()
        position_image = Image.fromarray(position_map_np)
        position_image = position_image.resize((out_res, out_res))
        img_list.append(position_image)


    res_image = upscaler(prompt,
                     negative_prompt=negative_prompt,
                     image=image,
                     control_image=img_list,
                     height=out_res,
                     width=out_res,
                     num_images_per_prompt=1).images

    return res_image[0]

'''

	MultiView-Diffusion Stable-Diffusion Pipeline
	Modified from a Diffusers StableDiffusionControlNetPipeline
	Just mimic the pipeline structure but did not follow any API convention

'''


class StableSyncMVDPipeline(StableDiffusionXLControlNetPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            text_encoder_2: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            unet: UNet2DConditionModel,
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
            scheduler: KarrasDiffusionSchedulers,
            force_zeros_for_empty_prompt: bool = True,
            add_watermarker: Optional[bool] = None,
            feature_extractor: CLIPImageProcessor = None,
            image_encoder: CLIPVisionModelWithProjection = None,
    ):
        super().__init__(
            vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet,
            controlnet, scheduler, force_zeros_for_empty_prompt, add_watermarker,
            feature_extractor, image_encoder
        )

        self.scheduler = DDPMScheduler.from_config(self.scheduler.config)
        self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
        self.enable_model_cpu_offload()
        self.enable_vae_slicing()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def initialize_pipeline(
            self,
            mesh_path=None,
            mesh_transform=None,
            mesh_autouv=None,
            camera_azims=None,
            camera_centers=None,
            top_cameras=True,
            ref_views=[],
            latent_size=None,
            render_rgb_size=None,
            texture_size=None,
            texture_rgb_size=None,
            max_batch_size=24,
            logging_config=None,
    ):
        # Make output dir
        output_dir = logging_config["output_dir"]

        self.result_dir = f"{output_dir}"
        self.intermediate_dir = f"{output_dir}/intermediate"

        dirs = [output_dir, self.result_dir, self.intermediate_dir]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)

        # Define the cameras for rendering
        self.camera_poses = []
        self.attention_mask = []
        self.centers = camera_centers

        cam_count = len(camera_azims)
        front_view_diff = 360
        back_view_diff = 360
        front_view_idx = 0
        back_view_idx = 0
        for i, azim in enumerate(camera_azims):
            if azim < 0:
                azim += 360
            # self.camera_poses.append((0, azim))
            up_down = -30 if i < 4 else 30
            self.camera_poses.append((up_down, azim))
            self.attention_mask.append([(cam_count + i - 1) % cam_count, i, (i + 1) % cam_count])
            if abs(azim) < front_view_diff:
                front_view_idx = i
                front_view_diff = abs(azim)
            if abs(azim - 180) < back_view_diff:
                back_view_idx = i
                back_view_diff = abs(azim - 180)
        # Add two additional cameras for painting the top surfaces
        if top_cameras:
            self.camera_poses.append((30, 0))
            self.camera_poses.append((30, 180))

            self.attention_mask.append([front_view_idx, cam_count])
            self.attention_mask.append([back_view_idx, cam_count + 1])

        # Reference view for attention (all views attend the the views in this list)
        # A forward view will be used if not specified
        if len(ref_views) == 0:
            ref_views = [front_view_idx]


        # Set up pytorch3D for projection between screen space and UV space
        # uvp is for latent and uvp_rgb for rgb color
        self.uvp = UVP(texture_size=texture_size, render_size=latent_size, sampling_mode="nearest", channels=4,
                       device=self._execution_device)
        if mesh_path.lower().endswith(".obj"):
            self.uvp.load_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
        elif mesh_path.lower().endswith(".glb"):
            self.uvp.load_glb_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0,
                                                 scale=((0.8, 0.8, 0.8),))

        self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size, sampling_mode="nearest",
                           channels=3, device=self._execution_device)
        self.uvp_rgb.mesh = self.uvp.mesh.clone()
        self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0,
                                                     scale=((0.8, 0.8, 0.8),))
        _, _, _, cos_maps, _, _ = self.uvp_rgb.render_geometry()
        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)

        # Save some VRAM
        del _, cos_maps
        self.upcast_vae()
        self.vae.config.force_upcast = True
        print("Done Initialization")

    '''
		Modified from a StableDiffusion ControlNet pipeline
		Multi ControlNet not supported yet
	'''

    @torch.no_grad()
    def __call__(
            self,
            image_path: str = None,
            height: Optional[int] = None,
            render_rgb_size=1024,
            texture_size=1536,
            texture_rgb_size=1024,
            max_batch_size=6,
            mesh_path: str = None,
            mesh_transform: dict = None,
            mesh_autouv=False,
            camera_azims=None,
            camera_centers=None,
            top_cameras=True,
            upscaler=None,
            upscaler_type="sd",
            logging_config=None,
    ):

        # Setup pipeline settings
        self.initialize_pipeline(
            mesh_path=mesh_path,
            mesh_transform=mesh_transform,
            mesh_autouv=mesh_autouv,
            camera_azims=camera_azims,
            camera_centers=camera_centers,
            top_cameras=top_cameras,
            ref_views=[3],
            latent_size=height // 8,
            render_rgb_size=render_rgb_size,
            texture_size=texture_size,
            texture_rgb_size=texture_rgb_size,
            max_batch_size=max_batch_size,
            logging_config=logging_config
        )

        mesh_name = mesh_path.split("/")[-1].split(".")[0]

        self.uvp.to(self._execution_device)
        self.uvp_rgb.to(self._execution_device)
        mesh = self.uvp_rgb.mesh
        s3i = SpatialAware3DInpainting(mesh, device, mesh_name=mesh_name)
        result_tex_rgb, position_map = s3i(image_path)
        self.uvp.save_mesh(f"{self.result_dir}/after_inpainting/textured.obj", result_tex_rgb)
        uvr = UVRefinement(mesh, upscaler_type, upscaler, device, mesh_name=mesh_name)
        result_tex_rgb = uvr(result_tex_rgb, position_map)

        self.uvp.save_mesh(f"{self.result_dir}/final/textured.obj", result_tex_rgb)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return


