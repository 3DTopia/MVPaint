import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from IPython.display import display
import numpy as np
import math
import random
import torch
from torch import functional as F
from torch import nn
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode
from transformers import (
	CLIPImageProcessor,
	CLIPTextModel,
	CLIPTextModelWithProjection,
	CLIPTokenizer,
	CLIPVisionModelWithProjection,
)

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers import DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.attention_processor import (
	AttnProcessor2_0,
	XFormersAttnProcessor,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import (
	BaseOutput, 
	# randn_tensor, 
	numpy_to_pil,
	pt_to_pil,
	# make_image_grid,
	is_accelerate_available,
	is_accelerate_version,
	# is_compiled_module,
	logging,
	# randn_tensor,
	replace_example_docstring
	)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor

from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.models.attention_processor import Attention, AttentionProcessor

from .renderer.project import UVProjection as UVP
from .renderer.voronoi import voronoi_solve
import torchvision.transforms as transforms

from .syncmvd.attention import SamplewiseAttnProcessor2_0, replace_attention_processors
from .syncmvd.prompt import *
from .syncmvd.step import step_tex
from .utils import *
import kaolin as kal
from ip_adapter.utils import is_torch2_available, get_generator
if is_torch2_available():
	from ip_adapter.attention_processor import (
		AttnProcessor2_0 as AttnProcessor,
	)
	from ip_adapter.attention_processor import (
		CNAttnProcessor2_0 as CNAttnProcessor,
	)
	from ip_adapter.attention_processor import (
		IPAttnProcessor2_0 as IPAttnProcessor,
	)
else:
	from ip_adapter.attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from scipy.ndimage import label, binary_dilation, binary_erosion

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")

# Background colors
color_constants = {"black": [-1, -1, -1], "white": [1, 1, 1], "maroon": [0, -1, -1],
			"red": [1, -1, -1], "olive": [0, 0, -1], "yellow": [1, 1, -1],
			"green": [-1, 0, -1], "lime": [-1 ,1, -1], "teal": [-1, 0, 0],
			"aqua": [-1, 1, 1], "navy": [-1, -1, 0], "blue": [-1, -1, 1],
			"purple": [0, -1 , 0], "fuchsia": [1, -1, 1]}
color_names = list(color_constants.keys())


@torch.no_grad()
def stable_upscale(upscaler,low_res_texture, prompt, negative_prompt, guidance_scale=None, num_inference_steps=None,
				generator=None, out_res=2048):

	result_tex_rgb = low_res_texture.cpu()  # 移动到 CPU
	result_tex_rgb = (result_tex_rgb * 255).byte()  # 转换为 0-255 的范围并转为 byte 类型

	# 转换为 numpy 数组并转换为 [1024, 1024, 3] 的形状
	result_tex_rgb_np = result_tex_rgb.numpy()  # permute 用于调整维度顺序

	# 使用 PIL 创建图像
	low_res_texture = Image.fromarray(result_tex_rgb_np)
	low_res_texture = low_res_texture.resize((512, 512))

	# low_res_texture = low_res_texture.resize(out_res/4, out_res/4)
	image = upscaler(prompt,
					image=low_res_texture,
					negative_prompt=negative_prompt,
					).images[0]
	return image


@torch.no_grad()
def crop(whole_image_path, sub_image_wh, out_wh):
	low_res_images = Image.open(whole_image_path).convert("RGB")
	low_res_images_list = []

	for i in range(0, low_res_images.width, sub_image_wh):
		low_res_sub_image = low_res_images.crop((i, 0, i + sub_image_wh, sub_image_wh))
		# Append the sub-images to the respective lists
		low_res_images_list.append(low_res_sub_image)

	images = []
	for i in range(len(low_res_images_list)):
		images.append(low_res_images_list[i].resize(size=(out_wh, out_wh), resample=Image.Resampling.BICUBIC))
	return images

# Used to generate depth or normal conditioning images
@torch.no_grad()
def get_conditioning_images(uvp, output_size, render_size=512, blur_filter=5, cond_type="normal"):
	verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(image_size=render_size)
	masks = normals[...,3][:,None,...]
	masks = Resize((output_size//8,)*2, antialias=True)(masks)
	normals_transforms = Compose([
		Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True), 
		GaussianBlur(blur_filter, blur_filter//3+1)]
	)

	if cond_type == "normal":
		view_normals = uvp.decode_view_normal(normals).permute(0,3,1,2) *2 - 1
		conditional_images = normals_transforms(view_normals)
	# Some problem here, depth controlnet don't work when depth is normalized
	# But it do generate using the unnormalized form as below
	elif cond_type == "depth":
		view_depths = uvp.decode_normalized_depth(depths).permute(0,3,1,2)
		conditional_images = normals_transforms(view_depths)
	
	return conditional_images, masks


# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t):
	composited_images = []
	for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
		if t > 0:
			alphas_cumprod = scheduler.alphas_cumprod[t]
			noise = torch.normal(0, 1, background.shape, device=background.device)
			background = (1-alphas_cumprod) * noise + alphas_cumprod * background
		composited = foreground * mask + background * (1-mask)
		composited_images.append(composited)
	composited_tensor = torch.stack(composited_images)
	return composited_tensor


# Split into micro-batches to use less memory in each unet prediction
# But need more investigation on reducing memory usage
# Assume it has no possitive effect and use a large "max_batch_size" to skip splitting
def split_groups(attention_mask, max_batch_size, ref_view=[]):
	group_sets = []
	group = set()
	ref_group = set()
	idx = 0
	while idx < len(attention_mask):
		new_group = group | set([idx])
		new_ref_group = (ref_group | set(attention_mask[idx] + ref_view)) - new_group 
		if len(new_group) + len(new_ref_group) <= max_batch_size:
			group = new_group
			ref_group = new_ref_group
			idx += 1
		else:
			assert len(group) != 0, "Cannot fit into a group"
			group_sets.append((group, ref_group))
			group = set()
			ref_group = set()
	if len(group)>0:
		group_sets.append((group, ref_group))

	group_metas = []
	for group, ref_group in group_sets:
		in_mask = sorted(list(group | ref_group))
		out_mask = []
		group_attention_masks = []
		for idx in in_mask:
			if idx in group:
				out_mask.append(in_mask.index(idx))
			group_attention_masks.append([in_mask.index(idxx) for idxx in attention_mask[idx] if idxx in in_mask])
		ref_attention_mask = [in_mask.index(idx) for idx in ref_view]
		group_metas.append([in_mask, out_mask, group_attention_masks, ref_attention_mask])

	return group_metas

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

		self.result_dir = f"{output_dir}/results"
		self.intermediate_dir = f"{output_dir}/intermediate"

		dirs = [output_dir, self.result_dir, self.intermediate_dir]
		for dir_ in dirs:
			if not os.path.isdir(dir_):
				os.mkdir(dir_)


		# Define the cameras for rendering
		self.camera_poses = []
		self.attention_mask=[]
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
		# for i, azim in enumerate(camera_azims):
		# 	if azim < 0:
		# 		azim += 360
		# 	self.camera_poses.append((0, azim))
		# 	self.attention_mask.append([(cam_count+i-1)%cam_count, i, (i+1)%cam_count])
		# 	if abs(azim) < front_view_diff:
		# 		front_view_idx = i
		# 		front_view_diff = abs(azim)
		# 	if abs(azim - 180) < back_view_diff:
		# 		back_view_idx = i
		# 		back_view_diff = abs(azim - 180)

		# Add two additional cameras for painting the top surfaces
		if top_cameras:
			self.camera_poses.append((30, 0))
			self.camera_poses.append((30, 180))

			self.attention_mask.append([front_view_idx, cam_count])
			self.attention_mask.append([back_view_idx, cam_count+1])

		# Reference view for attention (all views attend the the views in this list)
		# A forward view will be used if not specified
		if len(ref_views) == 0:
			ref_views = [front_view_idx]

		# Calculate in-group attention mask
		self.group_metas = split_groups(self.attention_mask, max_batch_size, ref_views)

		# Set up pytorch3D for projection between screen space and UV space
		# uvp is for latent and uvp_rgb for rgb color
		self.uvp = UVP(texture_size=texture_size, render_size=latent_size, sampling_mode="nearest", channels=4, device=self._execution_device)
		if mesh_path.lower().endswith(".obj"):
			self.uvp.load_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
		elif mesh_path.lower().endswith(".glb"):
			self.uvp.load_glb_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
		else:
			assert False, "The mesh file format is not supported. Use .obj or .glb."
		self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0, scale=((0.8, 0.8, 0.8),))


		self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size, sampling_mode="nearest", channels=3, device=self._execution_device)
		self.uvp_rgb.mesh = self.uvp.mesh.clone()
		self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0, scale=((0.8, 0.8, 0.8),))
		_,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
		self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)

		# Save some VRAM
		del _, cos_maps
		self.uvp.to("cpu")
		self.uvp_rgb.to("cpu")
		self.upcast_vae()
		self.vae.config.force_upcast = True


		color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(-1,3,1,1).to(dtype=torch.float32, device=self._execution_device)
		color_images = torch.ones(
			(1,1,latent_size*8, latent_size*8), 
			device=self._execution_device, 
			dtype=torch.float32
		) * color_images
		color_images = ((0.5*color_images)+0.5)
		color_latents = encode_latents(self.vae, color_images).to(dtype=self.text_encoder.dtype)

		self.color_latents = {color[0]:color[1] for color in zip(color_names, [latent for latent in color_latents])}		

		print("Done Initialization")




	'''
		Modified from a StableDiffusion ControlNet pipeline
		Multi ControlNet not supported yet
	'''
	@torch.no_grad()
	def __call__(
		self,
		prompt: str = None,
		prompt_2: Optional[Union[str, List[str]]] = None,
		image_path: str = None,
		# image: PipelineImageInput = None,
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: int = 50,
		timesteps: List[int] = None,
		sigmas: List[float] = None,
		denoising_end: Optional[float] = None,
		guidance_scale: float = 5.0,
		negative_prompt: str = None,
		negative_prompt_2: Optional[Union[str, List[str]]] = None,
		num_images_per_prompt: Optional[int] = 1,
		eta: float = 0.0,
		generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
		latents: Optional[torch.Tensor] = None,
		prompt_embeds: Optional[torch.Tensor] = None,
		negative_prompt_embeds: Optional[torch.Tensor] = None,
		pooled_prompt_embeds: Optional[torch.Tensor] = None,
		negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
		# ip_adapter_image: Optional[PipelineImageInput] = None,
		# ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
		output_type: Optional[str] = "pil",
		return_dict: bool = False,
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
		callback_steps: int = 1,
		max_batch_size=6,
		
		controlnet_guess_mode: bool = False,
		controlnet_conditioning_scale: Union[float, List[float]] = 0.7,
		controlnet_conditioning_end_scale: Union[float, List[float]] = 0.9,
		control_guidance_start: Union[float, List[float]] = 0.0,
		control_guidance_end: Union[float, List[float]] = 0.99,
		original_size: Tuple[int, int] = None,
		crops_coords_top_left: Tuple[int, int] = (0, 0),
		target_size: Tuple[int, int] = None,
		negative_original_size: Optional[Tuple[int, int]] = None,
		negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
		negative_target_size: Optional[Tuple[int, int]] = None,
		clip_skip: Optional[int] = None,
		guidance_rescale: float = 0.0,
		extra_strength: List[float] = [1, 1],

		mesh_path: str = None,
		mesh_transform: dict = None,
		mesh_autouv = False,
		camera_azims=None,
		camera_centers=None,
		top_cameras=True,
		texture_size = 1536,
		render_rgb_size=1024,
		texture_rgb_size = 1024,
		multiview_diffusion_end=0.8,
		exp_start=0.0,
		exp_end=6.0,
		shuffle_background_change=0.4,
		shuffle_background_end=0.99, #0.4
		upscaler=None,

		use_directional_prompt=True,
		
		ref_attention_end=0.2,

		logging_config=None,
		cond_type="depth",
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
				latent_size=height//8,
				render_rgb_size=render_rgb_size,
				texture_size=texture_size,
				texture_rgb_size=texture_rgb_size,

				max_batch_size=max_batch_size,

				logging_config=logging_config
			)


		
		num_timesteps = self.scheduler.config.num_train_timesteps
		initial_controlnet_conditioning_scale = controlnet_conditioning_scale
		log_interval = logging_config.get("log_interval", 10)
		view_fast_preview = logging_config.get("view_fast_preview", True)
		tex_fast_preview = logging_config.get("tex_fast_preview", True)

		# controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
		controlnet = self.controlnet

		# align format for control guidance
		if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
			control_guidance_start = len(control_guidance_end) * [control_guidance_start]
		elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
			control_guidance_end = len(control_guidance_start) * [control_guidance_end]
		elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
			mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
			control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
				control_guidance_end
			]


		# 0. Default height and width to unet
		height = height or self.unet.config.sample_size * self.vae_scale_factor
		width = width or self.unet.config.sample_size * self.vae_scale_factor

		for attn_processor in self.unet.attn_processors.values():
			if isinstance(attn_processor, IPAttnProcessor):
				attn_processor.scale = 1

		# 1. Check inputs. Raise error if not correct
		# print(controlnet_conditioning_scale)
		# self.check_inputs(
		# 	prompt,
		# 	torch.zeros((1,3,height,width), device=self._execution_device),
		# 	callback_steps,
		# 	negative_prompt,
		# 	None,
		# 	None,
		# 	float(controlnet_conditioning_scale),
		# 	control_guidance_start,
		# 	control_guidance_end,
		# )


		# 2. Define call parameters
		if prompt is not None and isinstance(prompt, list):
			assert len(prompt) == 1 and len(negative_prompt) == 1, "Only implemented for 1 (negative) prompt"  
		assert num_images_per_prompt == 1, "Only implemented for 1 image per-prompt"
		batch_size = len(self.uvp.cameras)

		device = self._execution_device
		# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
		# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
		# corresponds to doing no classifier free guidance.
		do_classifier_free_guidance = guidance_scale > 1.0

		if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
			controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

		global_pool_conditions = (
			controlnet.config.global_pool_conditions
			if isinstance(controlnet, ControlNetModel)
			else controlnet.nets[0].config.global_pool_conditions
		)
		guess_mode = controlnet_guess_mode or global_pool_conditions

		# 3. Encode input prompt
		prompt, negative_prompt = prepare_directional_prompt(prompt, negative_prompt)

		text_encoder_lora_scale = (
			cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
		)
		(
			prompt_embeds,
			negative_prompt_embeds,
			pooled_prompt_embeds,
			negative_pooled_prompt_embeds,
		) = self.encode_prompt(
			prompt,
			prompt_2,
			device,
			num_images_per_prompt,
			do_classifier_free_guidance,
			negative_prompt,
			negative_prompt_2,
			prompt_embeds=prompt_embeds,
			negative_prompt_embeds=negative_prompt_embeds,
			pooled_prompt_embeds=pooled_prompt_embeds,
			negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
			lora_scale=text_encoder_lora_scale,
			clip_skip=clip_skip,
		)

		# 4. Prepare image

		# negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
		prompt_embed_dict = dict(zip(direction_names, [emb for emb in prompt_embeds]))
		negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in negative_prompt_embeds]))

		# (4. Prepare image) This pipeline use internal conditional images from Pytorch3D
		self.uvp.to(self._execution_device)
		conditioning_images, masks = get_conditioning_images(self.uvp, height, cond_type=cond_type)
		conditioning_images = conditioning_images.type(prompt_embeds.dtype)
		
		low_res_image = crop(image_path, 256, 1024)
		low_res_images = torch.stack([torch.from_numpy(np.array(a).astype(np.float32)/255.) for a in low_res_image], dim=0).permute(0,3,1,2).to('cuda').type(prompt_embeds.dtype)
		cond = (conditioning_images/2+0.5).permute(0,2,3,1).cpu().numpy()
		cond = np.concatenate([img for img in cond], axis=1)
		numpy_to_pil(cond)[0].save(f"{self.intermediate_dir}/cond.jpg")

		# 5. Prepare timesteps
		# self.scheduler.set_timesteps(num_inference_steps, device=device)
		# timesteps = self.scheduler.timesteps
		timesteps, num_inference_steps = retrieve_timesteps(
			self.scheduler, num_inference_steps, device, timesteps, sigmas
		)
		self._num_timesteps = len(timesteps)

		# 6. Prepare latent variables
		num_channels_latents = self.unet.config.in_channels
		latents = self.prepare_latents(
			batch_size * num_images_per_prompt,
			num_channels_latents,
			height,
			width,
			prompt_embeds.dtype,
			device,
			generator,
			latents,
		)

		latent_tex = self.uvp.set_noise_texture()
		noise_views = self.uvp.render_textured_views()
		foregrounds = [view[:-1] for view in noise_views]
		masks = [view[-1:] for view in noise_views]
		composited_tensor = composite_rendered_view(self.scheduler, latents, foregrounds, masks, timesteps[0]+1)
		latents = composited_tensor.type(latents.dtype)
		self.uvp.to("cpu")

		

		# 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

		# 7.1 Create tensor stating which controlnets to keep
		controlnet_keep = []

		for i in range(len(timesteps)):
			keeps = [
				1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
				for s, e in zip(control_guidance_start, control_guidance_end)
			]
			controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

		# 7.2 Prepare added time ids & embeddings
		# if isinstance(image, list):
		# 	original_size = original_size or image[0].shape[-2:]
		# else:
		# 	original_size = original_size or image.shape[-2:]
		original_size = (height, width)
		target_size = target_size or (height, width)

		add_text_embeds = pooled_prompt_embeds
		if self.text_encoder_2 is None:
			text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
		else:
			text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

		add_time_ids = self._get_add_time_ids(
			original_size,
			crops_coords_top_left,
			target_size,
			dtype=prompt_embeds.dtype,
			text_encoder_projection_dim=text_encoder_projection_dim,
		)

		if negative_original_size is not None and negative_target_size is not None:
			negative_add_time_ids = self._get_add_time_ids(
				negative_original_size,
				negative_crops_coords_top_left,
				negative_target_size,
				dtype=prompt_embeds.dtype,
				text_encoder_projection_dim=text_encoder_projection_dim,
			)
		else:
			negative_add_time_ids = add_time_ids

		if do_classifier_free_guidance:
			prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
			add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
			add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)



		prompt_embeds = prompt_embeds.to(device)
		add_text_embeds = add_text_embeds.to(device)
		add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

		# 8. Denoising loop
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
		intermediate_results = []
		background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
		dbres_sizes_list = []
		mbres_size_list = []
		# 8.1 Apply denoising_end
		if (
			denoising_end is not None
			and isinstance(denoising_end, float)
			and denoising_end > 0
			and denoising_end < 1
		):
			discrete_timestep_cutoff = int(
				round(
					self.scheduler.config.num_train_timesteps
					- (denoising_end * self.scheduler.config.num_train_timesteps)
				)
			)
			num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
			timesteps = timesteps[:num_inference_steps]

		is_unet_compiled = is_compiled_module(self.unet)
		is_controlnet_compiled = is_compiled_module(self.controlnet)
		is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):

				# mix prompt embeds according to azim angle
				positive_prompt_embeds = [azim_prompt(prompt_embed_dict, pose) for pose in self.camera_poses]
				positive_prompt_embeds = torch.stack(positive_prompt_embeds, axis=0)

				negative_prompt_embeds = [azim_neg_prompt(negative_prompt_embed_dict, pose) for pose in self.camera_poses]
				negative_prompt_embeds = torch.stack(negative_prompt_embeds, axis=0)

				positive_added_cond_kwargs = {
					"text_embeds":  torch.stack([azim_index(add_text_embeds.chunk(2)[1], pose) for pose in self.camera_poses], dim=0),
					"time_ids": add_time_ids.chunk(2)[1],
				}
				negative_added_cond_kwargs = {
					"text_embeds":  torch.stack([azim_neg_index(add_text_embeds.chunk(2)[0], pose) for pose in self.camera_poses], dim=0),
					"time_ids": add_time_ids.chunk(2)[0],
				}

				# expand the latents if we are doing classifier free guidance
				latent_model_input = self.scheduler.scale_model_input(latents, t)

				'''
					Use groups to manage prompt and results
					Make sure negative and positive prompt does not perform attention together
				'''
				prompt_embeds_groups = {"positive": positive_prompt_embeds}
				result_groups = {}
				if do_classifier_free_guidance:
					prompt_embeds_groups["negative"] = negative_prompt_embeds

				for prompt_tag, prompt_embeds in prompt_embeds_groups.items():
					if prompt_tag == "positive" or not guess_mode:
						# controlnet(s) inference
						control_model_input = latent_model_input
						controlnet_prompt_embeds = prompt_embeds
						# controlnet_added_cond_kwargs = positive_added_cond_kwargs if prompt_tag == "positive" else negative_added_cond_kwargs

						if isinstance(controlnet_keep[i], list):
							cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
						else:
							controlnet_cond_scale = controlnet_conditioning_scale
							if isinstance(controlnet_cond_scale, list):
								controlnet_cond_scale = controlnet_cond_scale[0]
							cond_scale = controlnet_cond_scale * controlnet_keep[i]

						# Split into micro-batches according to group meta info
						# Ignore this feature for now
						down_block_res_samples_list = []
						mid_block_res_sample_list = []

						active_added_cond_kwargs = positive_added_cond_kwargs if prompt_tag == "positive" else negative_added_cond_kwargs
						model_input_batches = [torch.index_select(control_model_input, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
						prompt_embeds_batches = [torch.index_select(controlnet_prompt_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
						conditioning_images_batches = [torch.index_select(conditioning_images, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
						conditioning_images_batches2 = [torch.index_select(low_res_images, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
						controlnet_added_cond_kwargs = [{
							"text_embeds": torch.index_select(active_added_cond_kwargs["text_embeds"], dim=0, index=torch.tensor(meta[0], device=self._execution_device)),
							"time_ids": torch.index_select(active_added_cond_kwargs["time_ids"], dim=0, index=torch.tensor(meta[0], device=self._execution_device)),
						}  for meta in self.group_metas]

						for model_input_batch ,prompt_embeds_batch, conditioning_images_batch, conditioning_images_batch2, controlnet_added_cond_kwargs_batch \
							in zip (model_input_batches, prompt_embeds_batches, conditioning_images_batches, conditioning_images_batches2, controlnet_added_cond_kwargs):
							down_block_res_samples, mid_block_res_sample = self.controlnet(
								model_input_batch,
								t,
								encoder_hidden_states=prompt_embeds_batch,
								controlnet_cond=[conditioning_images_batch, conditioning_images_batch2],
								conditioning_scale=[c*s for c, s in zip(cond_scale, extra_strength)],
								guess_mode=guess_mode,
								added_cond_kwargs=controlnet_added_cond_kwargs_batch,
								return_dict=False,
							)
							down_block_res_samples_list.append(down_block_res_samples)
							mid_block_res_sample_list.append(mid_block_res_sample)

						''' For the ith element of down_block_res_samples, concat the ith element of all mini-batch result '''
						model_input_batches = prompt_embeds_batches = conditioning_images_batches = None

						if guess_mode:
							for dbres in down_block_res_samples_list:
								dbres_sizes = []
								for res in dbres:
									dbres_sizes.append(res.shape)
								dbres_sizes_list.append(dbres_sizes)

							for mbres in mid_block_res_sample_list:
								mbres_size_list.append(mbres.shape)

					else:
						# Infered ControlNet only for the conditional batch.
						# To apply the output of ControlNet to both the unconditional and conditional batches,
						# add 0 to the unconditional batch to keep it unchanged.
						# We copy the tensor shapes from a conditional batch
						down_block_res_samples_list = []
						mid_block_res_sample_list = []
						for dbres_sizes in dbres_sizes_list:
							down_block_res_samples_list.append([torch.zeros(shape, device=self._execution_device, dtype=latents.dtype) for shape in dbres_sizes])
						for mbres in mbres_size_list:
							mid_block_res_sample_list.append(torch.zeros(mbres, device=self._execution_device, dtype=latents.dtype))
						dbres_sizes_list = []
						mbres_size_list = []


					'''
					
						predict the noise residual, split into mini-batches
						Downblock res samples has n samples, we split each sample into m batches
						and re group them into m lists of n mini batch samples.
					
					'''
					noise_pred_list = []
					model_input_batches = [torch.index_select(latent_model_input, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
					prompt_embeds_batches = [torch.index_select(prompt_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]

					active_added_cond_kwargs = positive_added_cond_kwargs if prompt_tag == "positive" else negative_added_cond_kwargs
					input_added_cond_kwargs = [{
						"text_embeds": torch.index_select(active_added_cond_kwargs["text_embeds"], dim=0, index=torch.tensor(meta[0], device=self._execution_device)),
						"time_ids": torch.index_select(active_added_cond_kwargs["time_ids"], dim=0, index=torch.tensor(meta[0], device=self._execution_device)),
					}  for meta in self.group_metas]

					for model_input_batch, prompt_embeds_batch, down_block_res_samples_batch, mid_block_res_sample_batch, input_added_cond_batch, meta \
						in zip(model_input_batches, prompt_embeds_batches, down_block_res_samples_list, mid_block_res_sample_list, input_added_cond_kwargs, self.group_metas):
						if t > num_timesteps * (1- ref_attention_end):
							replace_attention_processors(self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3], ref_weight=1)
						else:
							replace_attention_processors(self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3], ref_weight=0)

						noise_pred = self.unet(
							model_input_batch,
							t,
							encoder_hidden_states=prompt_embeds_batch,
							cross_attention_kwargs=cross_attention_kwargs,
							down_block_additional_residuals=down_block_res_samples_batch,
							mid_block_additional_residual=mid_block_res_sample_batch,
							added_cond_kwargs=input_added_cond_batch,
							return_dict=False,
						)[0]
						noise_pred_list.append(noise_pred)

					noise_pred_list = [torch.index_select(noise_pred, dim=0, index=torch.tensor(meta[1], device=self._execution_device)) for noise_pred, meta in zip(noise_pred_list, self.group_metas)]
					noise_pred = torch.cat(noise_pred_list, dim=0)
					down_block_res_samples_list = None
					mid_block_res_sample_list = None
					noise_pred_list = None
					model_input_batches = prompt_embeds_batches = down_block_res_samples_batches = mid_block_res_sample_batches = None

					result_groups[prompt_tag] = noise_pred

				positive_noise_pred = result_groups["positive"]

				# perform guidance
				if do_classifier_free_guidance:
					noise_pred = result_groups["negative"] + guidance_scale * (positive_noise_pred - result_groups["negative"])


				if do_classifier_free_guidance and guidance_rescale > 0.0:
					# Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
					noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

				self.uvp.to(self._execution_device)
				# compute the previous noisy sample x_t -> x_t-1
				# Multi-View step or individual step
				current_exp = ((exp_end-exp_start) * i / num_inference_steps) + exp_start
				if t > (1-multiview_diffusion_end)*num_timesteps:
					step_results = step_tex(
						scheduler=self.scheduler, 
						uvp=self.uvp, 
						model_output=noise_pred, 
						timestep=t, 
						sample=latents, 
						texture=latent_tex,
						return_dict=True, 
						main_views=[], 
						exp= current_exp,
						**extra_step_kwargs
					)

					pred_original_sample = step_results["pred_original_sample"]
					latents = step_results["prev_sample"]
					latent_tex = step_results["prev_tex"]

					# Composit latent foreground with random color background
					background_latents = [self.color_latents[color] for color in background_colors]
					composited_tensor = composite_rendered_view(self.scheduler, background_latents, latents, masks, t)
					latents = composited_tensor.type(latents.dtype)

					intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))
				else:
					step_results = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)

					pred_original_sample = step_results["pred_original_sample"]
					latents = step_results["prev_sample"]
					latent_tex = None

					intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))

				del noise_pred, result_groups
					


				# Update pipeline settings after one step:
				# 1. Annealing ControlNet scale
				if (1-t/num_timesteps) < control_guidance_start[0]:
					controlnet_conditioning_scale = initial_controlnet_conditioning_scale
				elif (1-t/num_timesteps) > control_guidance_end[0]:
					controlnet_conditioning_scale = controlnet_conditioning_end_scale
				else:
					alpha = ((1-t/num_timesteps) - control_guidance_start[0]) / (control_guidance_end[0] - control_guidance_start[0])
					controlnet_conditioning_scale = alpha * initial_controlnet_conditioning_scale + (1-alpha) * controlnet_conditioning_end_scale

				if isinstance(controlnet, MultiControlNetModel):
					if isinstance(controlnet_conditioning_scale, torch.Tensor):
						controlnet_conditioning_scale = controlnet_conditioning_scale.item()
					controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

				# 2. Shuffle background colors; only black and white used after certain timestep
				if (1-t/num_timesteps) < shuffle_background_change:
					background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
				elif (1-t/num_timesteps) < shuffle_background_end:
					background_colors = [random.choice(["black","white"]) for i in range(len(self.camera_poses))]
				else:
					background_colors = background_colors



				# Logging at "log_interval" intervals and last step
				# Choose to uses color approximation or vae decoding
				if i % log_interval == log_interval-1 or t == 1:
					if view_fast_preview:
						decoded_results = []
						for latent_images in intermediate_results[-1]:
							images = latent_preview(latent_images.to(self._execution_device))
							images = np.concatenate([img for img in images], axis=1)
							decoded_results.append(images)
						result_image = np.concatenate(decoded_results, axis=0)
						numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")
					else:
						decoded_results = []
						for latent_images in intermediate_results[-1]:
							images = decode_latents(self.vae, latent_images.to(self._execution_device))

							images = np.concatenate([img for img in images], axis=1)

							decoded_results.append(images)
						result_image = np.concatenate(decoded_results, axis=0)
						numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")

					if not t < (1-multiview_diffusion_end)*num_timesteps:
						if tex_fast_preview:
							tex = latent_tex.clone()
							texture_color = latent_preview(tex[None, ...])
							numpy_to_pil(texture_color)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.jpg")
						else:
							self.uvp_rgb.to(self._execution_device)
							result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, pred_original_sample)
							numpy_to_pil(result_tex_rgb_output)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")
							self.uvp_rgb.to("cpu")

				self.uvp.to("cpu")

				# call the callback, if provided
				if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
					progress_bar.update()
					if callback is not None and i % callback_steps == 0:
						callback(i, t, latents)

				# Signal the program to skip or end
				import select
				import sys
				if select.select([sys.stdin],[],[],0)[0]:
					userInput = sys.stdin.readline().strip()
					if userInput == "skip":
						return None
					elif userInput == "end":
						exit(0)
		image = self.latent2img(latents)
		
		self.uvp.to(self._execution_device)
		self.uvp_rgb.to(self._execution_device)
		result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, latents)
		self.uvp_rgb.set_texture_map(result_tex_rgb)

		position_map = self.UV_pos_render()

		result_tex_rgb = self.update_texture(position_map)  # 点云补全颜色
		result_tex_rgb = torch.from_numpy(result_tex_rgb).to(device)

		bkgd_mask = position_map.sum(-1) == 0

		# 进行膨胀操作

		fore_mask = (1 - bkgd_mask.int()).squeeze()
		bigger_region = binary_dilation(fore_mask.cpu().numpy(), iterations=8)

		result_tex_rgb = voronoi_solve(result_tex_rgb, position_map.squeeze()[..., 0])
		bigger_region = torch.from_numpy(bigger_region).unsqueeze(-1).to(device)
		result_tex_rgb = torch.where(bigger_region>0,result_tex_rgb,1)

		# bkgd_mask = bkgd_mask.squeeze().unsqueeze(-1).expand(-1, -1, 3)
		# result_tex_rgb[bkgd_mask == 1] = 1
		
		upscaled_texture = stable_upscale(upscaler, result_tex_rgb,
											'bright, high quality, sharp, best quality',
											"moiré pattern, black spot, speckles, blur, low quality, noisy image, over-exposed, shadow",
															)
		# result_tex_rgb = transform(upscaled_texture).permute(1, 2, 0)
		# self.uvp.save_mesh(f"{self.result_dir}/upscale.obj", result_tex_rgb)
		transform = Compose([
			transforms.ToTensor()  # 将图像转换为 tensor 并归一化到 [0, 1] 范围
		])
		result_tex_rgb = transform(upscaled_texture).permute(1, 2, 0)  # (2048,2048,3)
		self.uvp_rgb.set_texture_map(result_tex_rgb.permute(2,0,1))
		position_map = self.UV_pos_render()

		result_tex_rgb = self.fix_seams(position_map)
		result_tex_rgb = torch.from_numpy(result_tex_rgb).to(device)
		result_tex_rgb = voronoi_solve(result_tex_rgb, position_map.squeeze()[..., 0])
		self.uvp.save_mesh(f"{self.result_dir}/textured.obj", result_tex_rgb)

		self.uvp_rgb.set_texture_map(result_tex_rgb.permute(2,0,1))
		textured_views = self.uvp_rgb.render_textured_views()
		textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1, ...]
		textured_views_rgb = textured_views_rgb.permute(1, 2, 0).cpu().numpy()[None, ...]
		
		images_load_rgb = np.hstack([np.array(img.resize((render_rgb_size, render_rgb_size))) for img in low_res_image]).astype(np.float32) / 255.
		# images_sr_rgb = np.hstack([np.array(img.resize((render_rgb_size, render_rgb_size))) for img in image]).astype(np.float32) / 255.
		textured_views_rgb = np.vstack([images_load_rgb, textured_views_rgb[0]])[None,...]
		
		v = numpy_to_pil(textured_views_rgb)[0]
		v.save(f"{self.result_dir}/textured_views_rgb.png")
		display(v)

		# Offload last model to CPU
		if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
			self.final_offload_hook.offload()

		self.uvp.to("cpu")
		self.uvp_rgb.to("cpu")

		return result_tex_rgb, textured_views, v

	@torch.no_grad()
	def latent2img(self, latents, output_type="pil"):
		if not output_type == "latent":
			# make sure the VAE is in float32 mode, as it overflows in float16
			# needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

			if self.vae.config.force_upcast:
				self.upcast_vae()
				latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

			# unscale/denormalize the latents
			# denormalize with the mean and std if available and not None
			has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
			has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
			if has_latents_mean and has_latents_std:
				latents_mean = (
					torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
				)
				latents_std = (
					torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
				)
				latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
			else:
				latents = latents / self.vae.config.scaling_factor

			image = self.vae.decode(latents, return_dict=False)[0]

			# # cast back to fp16 if needed
			# if needs_upcasting:
			# 	self.vae.to(dtype=torch.float16)
		else:
			image = latents

		if not output_type == "latent":
			# apply watermark if available
			if self.watermark is not None:
				image = self.watermark.apply_watermark(image)

			image = self.image_processor.postprocess(image, output_type=output_type)

		return image

	@torch.no_grad()
	def UV_pos_render(self):
		"""
		:param verts: (V, 3)
		:param faces: (F, 3)
		:param uv_face_attr: shape (1, F, 3, 2), range [0, 1]
		:param theta:
		:param phi:
		:param radius:
		:param view_target:
		:param texture_dims:
		:return:
		"""

		mesh = self.uvp_rgb.mesh

		verts = mesh.verts_packed()
		faces = mesh.faces_packed()
		verts_uv = mesh.textures.verts_uvs_padded()[0]  # 获取打包后的 UV 坐标 (V, 2)
		faces_uv = mesh.textures.faces_uvs_padded()[0]
		# normals = mesh.normals_packed()

		# 使用 faces 作为索引，从 verts_uv 中获取每个面的 UV 坐标
		uv_face_attr = torch.index_select(verts_uv, 0, faces_uv.view(-1))  # 选择对应顶点的 UV 坐标
		uv_face_attr = uv_face_attr.view(faces.shape[0], faces_uv.shape[1], 2).unsqueeze(0)
		texture_map = mesh.textures.maps_padded()[0]
		texture_dim = texture_map.shape[0]

		x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
		mesh_out_of_range = False
		if x.min() < -1 or x.max() > 1 or y.min() < -1 or y.max() > 1 or z.min() < -1 or z.max() > 1:
			mesh_out_of_range = True
		face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
		face_vertices_z = torch.zeros_like(face_vertices_world[:, :, :, -1], device=verts.device)
		uv_position, face_idx = kal.render.mesh.rasterize(texture_dim, texture_dim, face_vertices_z,
														uv_face_attr * 2 - 1,
														face_features=face_vertices_world, )
		uv_position = torch.clamp(uv_position, -1, 1)

		uv_position = uv_position / 2 + 0.5
		uv_position[face_idx == -1] = 0
		return uv_position

	@torch.no_grad()
	def update_texture(self, position_map):
		texture = self.uvp_rgb.mesh.textures.maps_padded()[0]
		points = position_map.reshape(-1, 3).cpu().numpy()
		texture_map_np = texture.cpu().numpy()
		h, w = texture_map_np.shape[:2]
		texture = texture_map_np.reshape(-1, 3)

		colored_points = np.concatenate([points, texture], 1)

		mask = points[:, 0] != 0
		to_be_update = colored_points[mask]
		np.save('./to_be_update.npy', to_be_update)
		np.save('./update_texture_mask.npy', mask)
		updated_colored_points = update_colored_points(to_be_update)
		colored_points[mask] = updated_colored_points
		colors = colored_points[:, 3:]
		colors = colors.reshape(h, w, 3)
		return colors

	@torch.no_grad()
	def fix_seams(self, position_map):
		texture = self.uvp_rgb.mesh.textures.maps_padded()[0]
		points = position_map.reshape(-1, 3).cpu().numpy()
		texture_map_np = texture.cpu().numpy()
		h, w = texture_map_np.shape[:2]
		texture = texture_map_np.reshape(-1, 3)

		colored_points = np.concatenate([points, texture], 1)
		mask = points[:, 0] != 0  # (2048*2048,)

		colored_points = smooth_seams(mask, colored_points)
		colors = colored_points[:, 3:]
		colors = colors.reshape(h, w, 3)
		return colors

