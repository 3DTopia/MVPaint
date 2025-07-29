
import sys
import os
import argparse
import datetime
import glob
import time
import json
from omegaconf import OmegaConf
from collections import OrderedDict
from mvdream.ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
import torch
import numpy as np
from PIL import Image
import nvdiffrast.torch as dr
glcx = dr.RasterizeCudaContext()
import random, os
from tqdm import tqdm

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(90)

def t2i(model, batch, sampler, outdir, step=50, scale=7.5, batch_size=9, dtype=torch.float32, device="cuda"):
    negative_promt = "strong light, Bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses"

    with torch.autocast(device_type="cuda", dtype=torch.float32):
        N = batch_size
        z, c, uv, inv_uv, cos_angles = model.get_input(batch, model.first_stage_key, bs=N)
        c_cat, c = c["control"][0][:N].to(c["context"][0][:N].dtype), c["context"][0][:N]
        camera = batch['camera'].to(device).reshape(-1, batch['camera'].shape[-1])[:N].to(c.dtype)
        N = min(z.shape[0], N)
        text = np.array([[x for x in patch] for patch in batch['txt']]).T.reshape(-1).tolist()[:N]
        data_id = batch["data_id"][:1] # * N
        uv = uv[:1]
        inv_uv = inv_uv[:1]
        cos_angles = cos_angles[:1]
        cos_angles = cos_angles.to(device)

        uc_cross = model.get_learned_conditioning([negative_promt] * N)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_full = {"control": [uc_cat], "context": [uc_cross],  "camera": camera}
        _, _, h, w = c_cat.shape
        shape = (4, h // 8, w // 8)

        samples_cfg, _, uv_maps, view_annot, colors = sampler.sample(step, N, shape,
                                        {"control": [c_cat], "context": [c], "camera": camera}, ddim=True,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_full,
                                        uv=uv,
                                        inv_uv=inv_uv,
                                        cos_angles=cos_angles,
                                        x_T=None,
                                        verbose=False,
                                        )
        x_sample = model.decode_first_stage(samples_cfg).detach()
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
        x_sample = np.hstack(list(x_sample.astype(np.uint8)))
        Image.fromarray(x_sample).save(f'{outdir}/{data_id[0]}.png')


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--using_time_prefix",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--auto_resume",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

if __name__ == '__main__':
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    print("=== loading parser", flush=True)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    print("=== loading dataset", flush=True)
    # 1. data
    dataset = instantiate_from_config(config.data)
    data = DataLoader(dataset, num_workers=config.dataloader.num_workers,
                            batch_size=1, shuffle=False)
    
    print("=== loading model", flush=True)
    model = instantiate_from_config(config.model)
    if hasattr(config, "resume_model_from_checkpoint") and config.resume_model_from_checkpoint:
        print(f"resume_model_from_checkpoint {config.resume_model_from_checkpoint}")
        if os.path.isdir(config.resume_model_from_checkpoint):
            config.resume_model_from_checkpoint += "/checkpoint/mp_rank_00_model_states.pt"
        
        print("=== loading checkpoint", flush=True)
        ckpt = torch.load(config.resume_model_from_checkpoint, map_location="cpu")
        # bask_ckpt = torch.load(config.base_model, map_location="cpu")
        # for k in ckpt['state_dict'].keys():
        #     if k.startswith('control_model.'):
        #         bask_ckpt[k] = ckpt['state_dict'][k]
        if "module" in ckpt:
            ckpt["state_dict"] = OrderedDict()
            for k in list(ckpt["module"].keys()):
                ckpt["state_dict"][k[len("module."):]] = ckpt["module"].pop(k)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        del config.resume_model_from_checkpoint
    model = model.to('cuda')
    sampler = DDIMSampler(model)

    print("=== processing", flush=True)
    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)
    data_len = len(data)
    for i, batch in enumerate(tqdm(data, desc='eval')):
        # if i != 1:
        #     continue 

        t2i(model, batch, sampler, outdir)
        # break

    print("=== done", flush=True)