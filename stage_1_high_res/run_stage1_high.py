import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusionUpscalePipeline
from diffusers import DDPMScheduler, UniPCMultistepScheduler, DDIMScheduler
from src.pipeline_sdxl import StableSyncMVDPipeline
from src.configs import *
from shutil import copy
import json
from tqdm import tqdm, trange 
import trimesh, shutil
import numpy as np


def gather_prompt(prompt):
    out = 'Bright, highres, Best quality, photographic'
    for key in ['category', 'description', 'texture']:
        out += (prompt[key] + ', ') if prompt[key] is not None else ''
    return out

if __name__ == "__main__":

	print("=== parsing args", flush=True)
	opt = parse_config()
	
	print("==== loading model", flush=True)
	print("== depth_controlnet", flush=True)
	depth_controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",
															torch_dtype=torch.float16)
	print("== tile_controlnet", flush=True)
	tile_controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-tile-sdxl-1.0",
															torch_dtype=torch.float16)
	print("== pipe", flush=True)
	pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
		"stabilityai/stable-diffusion-xl-base-1.0", 
		controlnet=[depth_controlnet, tile_controlnet],
		torch_dtype=torch.float16, variant="fp16", use_safetensors=True
	)
	pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

	print("== upscaler", flush=True)
	upscaler = StableDiffusionUpscalePipeline.from_pretrained(
		"stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
	)
	upscaler.scheduler = DDIMScheduler.from_config(upscaler.scheduler.config)
	# upscaler.enable_model_cpu_offload()
	upscaler.enable_xformers_memory_efficient_attention()
	upscaler.to('cuda')

	print("===== init StableSyncMVDPipeline", flush=True)
	syncmvd = StableSyncMVDPipeline(**pipe.components)
	
	opt.negative_prompt = 'strong light, Bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses'
	
	prompt_file = opt.prompt_file
	with open(prompt_file, 'r') as fin:
		prompt_dict = json.load(fin)

	mesh_folder = opt.mesh_folder
	low_res_folder = opt.lowres_folder
	out_root = opt.output
	os.makedirs(out_root, exist_ok=True)
	
	mesh_filenames = sorted([x for x in os.listdir(mesh_folder) if os.path.isdir(os.path.join(mesh_folder, x))])
	mesh_len = len(mesh_filenames)
	print("=== mesh_len", mesh_len, flush=True)
	for i in trange(mesh_len):
		# if i != 1:
		# 	continue 

		mesh_id = i
		mesh_name = mesh_filenames[i] 
		print("=== mesh_name", mesh_name, flush=True)

		output_dir = f'{out_root}/{mesh_name}'
		os.makedirs(output_dir, exist_ok=True)

		mesh_path = f'{mesh_folder}/{mesh_name}/{mesh_name}.obj'
		intermediate_low_res_path = f'{low_res_folder}/{mesh_name}.png'

		mesh = trimesh.load(mesh_path)
		logging_config = {
			"output_dir":output_dir, 
			"log_interval":opt.log_interval,
			"view_fast_preview": opt.view_fast_preview,
			"tex_fast_preview": opt.tex_fast_preview,
		}
		opt.prompt = gather_prompt(prompt_dict[mesh_name])

		result_tex_rgb, textured_views, v = syncmvd(
			image_path=intermediate_low_res_path,
			prompt=opt.prompt,
			height=opt.latent_view_size*8,
			width=opt.latent_view_size*8,
			num_inference_steps=opt.steps,
			guidance_scale=opt.guidance_scale,
			negative_prompt=opt.negative_prompt,
			
			generator=torch.manual_seed(opt.seed),
			max_batch_size=48,
			controlnet_guess_mode=opt.guess_mode,
			controlnet_conditioning_scale = opt.conditioning_scale,
			controlnet_conditioning_end_scale= opt.conditioning_scale_end,
			control_guidance_start= opt.control_guidance_start,
			control_guidance_end = opt.control_guidance_end,
			guidance_rescale = opt.guidance_rescale,
			extra_strength=opt.extra_strength,
			use_directional_prompt=True,

			mesh_path=mesh_path,
			mesh_transform={"scale":opt.mesh_scale},

			camera_azims=opt.camera_azims,
			top_cameras=not opt.no_top_cameras,
			texture_size=opt.latent_tex_size,
			render_rgb_size=opt.rgb_view_size,
			texture_rgb_size=opt.rgb_tex_size,
			multiview_diffusion_end=opt.mvd_end,
			ref_attention_end=opt.ref_attention_end,
			shuffle_background_change=opt.shuffle_bg_change,
			shuffle_background_end=opt.shuffle_bg_end,

			logging_config=logging_config,
			cond_type=opt.cond_type,
			upscaler=upscaler,	
		)
		shutil.rmtree(f'{output_dir}/intermediate')

	print("=== done", flush=True)