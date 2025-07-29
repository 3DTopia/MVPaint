import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusionUpscalePipeline, \
    StableDiffusionControlNetImg2ImgPipeline
from diffusers import DDPMScheduler, UniPCMultistepScheduler, DDIMScheduler

from src.configs import *
from shutil import copy
import json
from tqdm import tqdm, trange
import trimesh, shutil
import pdb
import tarfile
from megfile import smart_open
import numpy as np
from glob import glob

# from src.pipeline_sdxl import StableSyncMVDPipeline
from src.stage23 import StableSyncMVDPipeline

import argparse

import sys
import logging
logging.getLogger("diffusers").setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

if __name__ == "__main__":

    opt = parse_config()
    # opt.negative_prompt = 'strong light, Bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses, oil style painting'
    middle_res_folder = opt.stage1_folder
    mesh_folder = opt.mesh_folder
    out_root = opt.output
    os.makedirs(out_root, exist_ok=True)

    print("====== loading model", flush=True)
    print("=== depth_controlnet", flush=True)
    depth_controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16,
    )
    print("=== tile_controlnet", flush=True)
    tile_controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-tile-sdxl-1.0",
        torch_dtype=torch.float16
    )
    print("=== pipe", flush=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[depth_controlnet, tile_controlnet],
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    print("=== upscaler_type", flush=True)
    upscaler_type = "sd"  # sd or tile
    upscaler = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    upscaler.scheduler = DDIMScheduler.from_config(upscaler.scheduler.config)
    upscaler.enable_xformers_memory_efficient_attention()
    upscaler.to('cuda')

    syncmvd = StableSyncMVDPipeline(**pipe.components)

    mesh_filenames = sorted([x for x in os.listdir(mesh_folder) if os.path.isdir(os.path.join(mesh_folder, x))])
    mesh_len = len(mesh_filenames)
    print("=== mesh_len", mesh_len, flush=True)
    for i in trange(mesh_len):
        # if i != 1:
        #     continue 

        mesh_name = mesh_filenames[i]
        print("=== mesh_name", mesh_name, flush=True)

        out_dir = os.path.join(out_root, mesh_name)
        os.makedirs(out_dir, exist_ok=True)

        out_inpaint_folder = os.path.join(out_dir, "after_inpainting")
        out_final_folder = os.path.join(out_dir, "final")
        os.makedirs(out_inpaint_folder, exist_ok=True)
        os.makedirs(out_final_folder, exist_ok=True)
        
        mesh_path = os.path.join(mesh_folder, mesh_name, f'{mesh_name}.obj')
        middle_result_path = os.path.join(middle_res_folder, mesh_name, f'results/textured.png')

        logging_config = {
            "output_dir": out_dir,
            "log_interval": opt.log_interval,
            "view_fast_preview": opt.view_fast_preview,
            "tex_fast_preview": opt.tex_fast_preview,
        }
        syncmvd(
            image_path=middle_result_path,
            height=opt.latent_view_size * 8,
            render_rgb_size=opt.rgb_view_size,
            texture_size=opt.latent_tex_size,
            texture_rgb_size=opt.rgb_tex_size,
            max_batch_size=6,
            mesh_path=mesh_path,
            mesh_transform={"scale": opt.mesh_scale},
            mesh_autouv=False,
            camera_azims=opt.camera_azims,
            top_cameras=not opt.no_top_cameras,
            upscaler=upscaler,
            upscaler_type=upscaler_type,
            logging_config=logging_config,
        )

        shutil.rmtree(f'{out_dir}/intermediate')

    print("=== done", flush=True)
