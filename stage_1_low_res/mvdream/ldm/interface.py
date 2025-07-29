from typing import List
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from .util import exists, default, instantiate_from_config
from .modules.distributions.distributions import DiagonalGaussianDistribution
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange, repeat
import nvdiffrast.torch as dr
import imageio
import distinctipy
import cv2
import pdb
import torchvision.transforms as T
import random


@torch.no_grad()
def latent_preview(x):
    # adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
    v1_4_latent_rgb_factors = torch.tensor([
        #   R        G        B
        [0.298, 0.207, 0.208],  # L1
        [0.187, 0.286, 0.173],  # L2
        [-0.158, 0.189, 0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ], dtype=x.dtype, device=x.device)
    # image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
    print(x.shape, v1_4_latent_rgb_factors.shape)
    image = x @ v1_4_latent_rgb_factors
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()
    return image

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)

class LatentDiffusionInterface(pl.LightningModule):
    ''' a simple interface class for LDM inference ''' 
    def __init__(self, 
                unet_config, 
                cond_stage_config, 
                first_stage_config, 
                parameterization = "eps",
                scale_factor = 0.18215,
                beta_schedule="linear",
                timesteps=1000,
                linear_start=0.00085,
                linear_end=0.0120,
                cosine_s=8e-3,
                first_stage_key="image",
                cond_stage_key="image",
                log_every_t=100,
                given_betas=None,
                use_uv=None,
                uv_temperature=1,
                sync_direction="both",
                *args, **kwargs):
        super().__init__()
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        unet = instantiate_from_config(unet_config)
        self.model = DiffusionWrapper(unet)
        self.cond_stage_model = instantiate_from_config(cond_stage_config).to(torch.float16)
        self.first_stage_model = instantiate_from_config(first_stage_config).to(torch.float16)
        self.use_uv = use_uv
        self.uv_temperature = uv_temperature
        self.sync_direction = sync_direction

        self.parameterization = parameterization
        self.scale_factor = scale_factor
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                            linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                    cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.v_posterior = 0
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        color_constants = {"black": [-1, -1, -1], "white": [1, 1, 1], "maroon": [0, -1, -1],
                    "red": [1, -1, -1], "olive": [0, 0, -1], "yellow": [1, 1, -1],
                    "green": [-1, 0, -1], "lime": [-1 ,1, -1], "teal": [-1, 0, 0],
                    "aqua": [-1, 1, 1], "navy": [-1, -1, 0], "blue": [-1, -1, 1],
                    "purple": [0, -1 , 0], "fuchsia": [1, -1, 1]}
        self.color_names = list(color_constants.keys())
        color_images = torch.FloatTensor([color_constants[name] for name in self.color_names]).reshape(-1,3,1,1)
        color_images = torch.ones(
            (1,1,256, 256),
        ) * color_images
        self.color_images = ((0.5*color_images)+0.5)
        self.background_colors = [random.choice(list(color_constants.keys())) for i in range(8)]
        self.color_latents = None


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def apply_model(self, x_noisy, t, cond, **kwargs):
        assert isinstance(cond, dict)
        return self.model(x_noisy, t, **cond, **kwargs)

    def get_learned_conditioning(self, prompts: List[str]):
        return self.cond_stage_model(prompts)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)
    
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def get_loss(self, pred, target, mean=True):
        if mean:
            loss = torch.nn.functional.mse_loss(target, pred)
        else:
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')

        return loss
    
    @torch.no_grad()
    def prepare_view_visualization(self, ref_tensor):
        colors = distinctipy.get_colors(ref_tensor.shape[0])
        color_tensor = []
        for i, color in enumerate(colors):
            c = torch.zeros_like(ref_tensor[i])
            for j, channel in enumerate(color):
                c[...,j] = channel
            color_tensor.append(c)
        return colors, torch.stack(color_tensor, dim=0)
    
    def save_tensor(self, tensor):
        tensor = np.hstack(list(tensor.detach().cpu().numpy()))
        # print(tensor.shape)
        return (np.clip((tensor+1)/2, 0, 1)*255).astype(np.uint8)
    
    def print_tensor(self, name, tensor):
        print(name, tensor.shape, tensor.min(), tensor.max())

    @torch.no_grad()
    def composite_rendered_view(self, backgrounds, foregrounds, masks, alphas_cumprod):
        composited_images = []
        for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
            mask, foreground = mask[None,...], foreground[None,...]
            noise = torch.normal(0, 1, background.shape, device=foreground.device)
            background = (1-alphas_cumprod) * noise + alphas_cumprod * background
            composited = foreground * mask + background * (1-mask)
            composited_images.append(composited)
        composited_tensor = torch.stack(composited_images)
        return composited_tensor
        
    @torch.no_grad()
    def noise_synchornize_v5(self, xts, pred_x0s, x_prevs, a_ts, a_prevs, sigma_ts, sqrt_one_minus_ats, temperature, uvs, inv_uvs, cos_angles, ts, step, latent_tex=None, view_color_debug=False, debug=False):
    # def noise_synchornize_v4(self, xts, pred_x0s, x_prevs, a_prevs, sigma_ts, temperature, uvs, inv_uvs, cos_angles, ts, latent_tex=None, view_color_debug=False, debug=False):
        batch_size, num_frames = uvs.shape[0], uvs.shape[1]
        # decoded_xts = self.decode_first_stage(xts)
        # decoded_pred_x0s = self.decode_first_stage(pred_x0s)
        # decoded_prev_xts = self.decode_first_stage(x_prevs)
        if self.color_latents is None:
            color_posterior = self.encode_first_stage(torch.ones_like(self.color_images.to(dtype=xts.dtype, device=xts.device))*2)
            color_latents = self.get_first_stage_encoding(color_posterior).detach()
            self.color_latents = {color[0]:color[1] for color in zip(self.color_names, [latent for latent in color_latents])}
            self.background_latents = [self.color_latents[color] for color in self.background_colors]
        if view_color_debug:
            colors, color_tensor = self.prepare_view_visualization(decoded_pred_x0s.permute(0,2,3,1))
            view_uv_maps = []

        uv_maps = []
        synced_latents = []
        # torchvision.transfroms
        for i, (xt, x0, x_prev, a_t, a_prev, sigma_t, sqrt_one_minus_at, t, inv_uv, uv, cos_angle) in enumerate(zip(xts.chunk(batch_size), pred_x0s.chunk(batch_size), x_prevs.chunk(batch_size), \
                                                                                    a_ts.chunk(batch_size), a_prevs.chunk(batch_size), sigma_ts.chunk(batch_size), sqrt_one_minus_ats.chunk(batch_size),\
                                                                                    ts.chunk(batch_size), inv_uvs, uvs, cos_angles)):
            inv_uv, inv_uv_mask_valid, inv_uv_mask_boundary = torch.split(inv_uv, [2,1,1], dim=-1)
            a_prev_texture = a_prev[:1]
            sigma_t_texture = sigma_t[:1]
            t = t[:1]
            sqrt_one_minus_at= sqrt_one_minus_at[:1]
            a_t = a_t[:1]

            alpha_prod_t = a_t
            alpha_prod_t_prev = a_prev_texture

            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t

            partial_albedos = dr.texture(x0.permute(0,2,3,1).float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            partial_albedos_xt = dr.texture(xt.permute(0,2,3,1).float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
            partial_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask).contiguous(), filter_mode='linear')
            
            alpha = (torch.sum(inv_uv_mask > 0, dim=0, keepdim=True) > 0).float()
            alpha2 = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()

            origin_texture = torch.sum(partial_albedos * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float32) 
            current_texture = torch.sum(partial_albedos_xt * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float32)
            origin_texture = origin_texture * alpha2
            print("origin_texture", origin_texture.shape, origin_texture.min(), origin_texture.max(), x0.shape, x0.min(), x0.max())
            current_texture = current_texture * alpha2

            # current_texture = latent_tex
            if latent_tex is not None:
                current_texture = latent_tex
            # # pred_texture_latent = (current_texture - sqrt_one_minus_at * origin_texture) / a_t.sqrt()
            pred_texture_latent = origin_texture
            # pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            e_texture_t = (current_texture - pred_texture_latent * a_t.sqrt()) / sqrt_one_minus_at
            dir_texture = (1. - a_prev_texture - sigma_t_texture**2).sqrt() * e_texture_t
            noise = sigma_t_texture * torch.rand_like(current_texture) * temperature
            prev_texture_latent = a_prev_texture.sqrt() * pred_texture_latent + dir_texture + noise

            # pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
            # current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
            # prev_texture_latent = pred_original_sample_coeff * origin_texture + current_sample_coeff * latent_tex
            print("prev_texture_latent", prev_texture_latent.shape, prev_texture_latent.min(), prev_texture_latent.max())
            print("origin_texture", origin_texture.shape, origin_texture.min(), origin_texture.max())
            # print(pred_original_sample_coeff, current_sample_coeff)

            uv = uv.permute(0,3,1,2)
            resize = T.Resize((32,32))
            uv = resize(uv).permute(0,2,3,1)
            prev_views = dr.texture(prev_texture_latent.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
            uv_mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
            uv_mask = uv_mask.permute(0,3,1,2)
            # if step is not None:
            #     start = 10
            #     end = 45
            #     if step < start:
            #         prev_alpha = 1; x_prev_alpha = 0
            #     elif step > end:
            #         prev_alpha = 0; x_prev_alpha = 1
            #     else:
            #         length = end - start
            #         prev_alpha = (1 - (step-start)/length)
            #         x_prev_alpha = (step-start)/length
            #     # print(prev_alpha, x_prev_alpha)
            #     prev_views = prev_views.permute(0,3,1,2) * uv_mask * prev_alpha + x_prev * uv_mask * x_prev_alpha + x_prev* (1 - uv_mask)
            # else:
            # print(prev_views.shape, uv_mask.shape, x_prev.shape)
            prev_views = prev_views.permute(0,3,1,2) * uv_mask + x_prev * (1 - uv_mask)

            # prev_views = self.composite_rendered_view(self.background_latents, prev_views.permute(0,3,1,2), uv_mask, alpha_prod_t).squeeze()
            # prev_views = composited_tensor.type(latents.dtype)
            # prev_views = view_x_prev
            # imageio.imwrite('prev_views.png', self.save_tensor(prev_views))

            # new_posterior = self.encode_first_stage(prev_views.permute(0,3,1,2))
            # prev_latent = self.get_first_stage_encoding(new_posterior).detach()
            # synced_latents.append(prev_views)
            synced_latents.append(prev_views)

            if step % 10 == 0:
                origin_texture_np = latent_preview(origin_texture)
                prev_texture_latent_np = latent_preview(prev_texture_latent)
                prev_views_np = latent_preview(prev_views.permute(0,2,3,1))
                prev_views_np = np.hstack(list(prev_views_np))
                x0_np = latent_preview(x0.permute(0,2,3,1))
                x0_np = np.hstack(list(x0_np))
                prev_views_np = np.vstack([prev_views_np, x0_np])
                imageio.imwrite(f'origin_texture_{step:03d}.png', (origin_texture_np[0]*255).astype(np.uint8))
                imageio.imwrite(f'prev_texture_{step:03d}.png', (prev_texture_latent_np[0]*255).astype(np.uint8))
                imageio.imwrite(f'prev_views_np_{step:03d}.png', (prev_views_np*255).astype(np.uint8))
            # pdb.set_trace()

            if debug:
                imageio.imwrite(f'008.png', (np.clip((aggreated_albedos[0].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                for j, t in enumerate(partial_albedos[:8].cpu().numpy()):
                    imageio.imwrite(f'x_{j:03d}.png', (np.clip((x[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'new_x_{j:03d}.png', (np.clip((new_textured[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'{j:03d}.png', (np.clip((t+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'mask_{j:03d}.png', (np.clip(inv_uv_mask[j,:,:,0].cpu().numpy(), 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'uv_{j:03d}.png', (np.clip(inv_uv[j].cpu().numpy(), 0, 1)*255).astype(np.uint8))

                exit(0)

            synced_latents = torch.cat(synced_latents, dim=0)
            if not view_color_debug:
                return synced_latents, prev_texture_latent
            else:
                uv_maps = torch.cat(uv_maps, dim=0)
                view_uv_maps = torch.cat(view_uv_maps, dim=0)
                return synced_latents, uv_maps, view_uv_maps, colors  
            
    @torch.no_grad()
    def noise_synchornize_v4(self, xts, pred_x0s, x_prevs, a_prevs, sigma_ts, temperature, uvs, inv_uvs, cos_angles, ts, latent_tex=None, view_color_debug=False, debug=False):
        batch_size, num_frames = uvs.shape[0], uvs.shape[1]
        decoded_xts = self.decode_first_stage(xts)
        decoded_pred_x0s = self.decode_first_stage(pred_x0s)
        decoded_x_prevs = self.decode_first_stage(x_prevs)
        if view_color_debug:
            colors, color_tensor = self.prepare_view_visualization(decoded_pred_x0s.permute(0,2,3,1))
            view_uv_maps = []

        uv_maps = []
        synced_latents = []
        # torchvision.transfroms
        for i, (xt, x0, x_prev, a_prev, sigma_t, t, inv_uv, uv, cos_angle) in enumerate(zip(decoded_xts.chunk(batch_size), decoded_pred_x0s.chunk(batch_size), decoded_x_prevs.chunk(batch_size), \
                                                                                a_prevs.chunk(batch_size), sigma_ts.chunk(batch_size), ts.chunk(batch_size), inv_uvs, uvs, cos_angles)):
            inv_uv, inv_uv_mask_valid, inv_uv_mask_boundary = torch.split(inv_uv, [2,1,1], dim=-1)
            # texture_x0, view_x0 = torch.split(x0.permute(0,2,3,1), [1, num_frames], dim=0)
            # texture_xt, view_xt = torch.split(xt.permute(0,2,3,1), [1, num_frames], dim=0)
            # _, view_x_prev = torch.split(x_prev.permute(0,2,3,1), [1, num_frames], dim=0)
            view_x0 = x0.permute(0,2,3,1)
            view_xt = xt.permute(0,2,3,1)
            view_x_prev = x_prev.permute(0,2,3,1)
            a_prev_texture = a_prev[:1]
            sigma_t_texture = sigma_t[:1]
            t = t[:1]

            partial_albedos = dr.texture(view_x0.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            # imageio.imwrite('partial_albedos.png', self.save_tensor(partial_albedos))
            partial_albedos_xt = dr.texture(view_xt.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
            partial_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask).contiguous(), filter_mode='linear')
            
            alpha = (torch.sum(inv_uv_mask > 0, dim=0, keepdim=True) > 0).float()
            alpha2 = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()

            origin_texture = torch.sum(partial_albedos * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float32) 
            current_texture = torch.sum(partial_albedos_xt * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float32)
            origin_texture = origin_texture * alpha2
            current_texture = current_texture * alpha2
            # origin_texture_np = np.clip(origin_texture.cpu().numpy()[0], -1, 1)
            # origin_texture_np = cv2.inpaint(((origin_texture_np+1)*255/2).astype(np.uint8), ((1-alpha).cpu().numpy()[0]*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            # origin_texture = torch.from_numpy(origin_texture_np).to(device="cuda", dtype=torch.float32)[None,...] / 255. * 2 - 1

            # current_texture_np = np.clip(current_texture.cpu().numpy()[0], -1, 1)
            # current_texture_np = cv2.inpaint(((current_texture_np+1)*255/2).astype(np.uint8), ((1-alpha).cpu().numpy()[0]*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            # current_texture = torch.from_numpy(current_texture_np).to(device="cuda", dtype=torch.float32)[None,...] / 255. * 2 - 1
            # self.print_tensor("current_texture", origin_texture); self.print_tensor("current_texture", current_texture); 
            textures = torch.cat([origin_texture, current_texture], dim=0).to(dtype=torch.float32)
            
            texture_posterior = self.encode_first_stage(textures.permute(0,3,1,2))
            texture_latents = self.get_first_stage_encoding(texture_posterior).detach()
            origin_texture_latent, current_texture_latent = torch.split(texture_latents, [1, 1], dim=0)
            
            if latent_tex is not None:
                current_texture_latent = latent_tex

            pred_texture_latent = self.predict_start_from_z_and_v(current_texture_latent, t, origin_texture_latent)
            dir_texture = (1. - a_prev_texture - sigma_t_texture**2).sqrt() * origin_texture_latent
            noise = sigma_t_texture * torch.rand_like(current_texture_latent) * temperature
            prev_texture_latent = a_prev_texture.sqrt() * pred_texture_latent + dir_texture + noise

            prev_texture = self.decode_first_stage(prev_texture_latent).permute(0,2,3,1)

            prev_views = dr.texture(prev_texture.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
            uv_mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
            prev_views = prev_views.clamp(-1,1) * uv_mask + view_x_prev.clamp(-1,1) * (1 - uv_mask)
            # prev_views = view_x_prev
            # imageio.imwrite('prev_views.png', self.save_tensor(prev_views))

            new_posterior = self.encode_first_stage(prev_views.permute(0,3,1,2))
            prev_latent = self.get_first_stage_encoding(new_posterior).detach()
            synced_latents.append(prev_latent)
            # pdb.set_trace()

            if debug:
                imageio.imwrite(f'008.png', (np.clip((aggreated_albedos[0].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                for j, t in enumerate(partial_albedos[:8].cpu().numpy()):
                    imageio.imwrite(f'x_{j:03d}.png', (np.clip((x[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'new_x_{j:03d}.png', (np.clip((new_textured[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'{j:03d}.png', (np.clip((t+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'mask_{j:03d}.png', (np.clip(inv_uv_mask[j,:,:,0].cpu().numpy(), 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'uv_{j:03d}.png', (np.clip(inv_uv[j].cpu().numpy(), 0, 1)*255).astype(np.uint8))

                exit(0)

            synced_latents = torch.cat(synced_latents, dim=0)
            if not view_color_debug:
                return synced_latents, prev_texture_latent
            else:
                uv_maps = torch.cat(uv_maps, dim=0)
                view_uv_maps = torch.cat(view_uv_maps, dim=0)
                return synced_latents, uv_maps, view_uv_maps, colors     

    @torch.no_grad()
    def noise_synchornize_v3(self, xts, pred_x0s, prev_xs, a_prevs, sigma_ts, temperature, uvs, inv_uvs, cos_angles, latent_tex=None, view_color_debug=False, debug=False):
        batch_size, num_frames = uvs.shape[0], uvs.shape[1]
        decoded_xts = self.decode_first_stage(xts)
        decoded_pred_x0s = self.decode_first_stage(pred_x0s)
        decoded_prev_xs = self.decode_first_stage(prev_xs)
        if view_color_debug:
            colors, color_tensor = self.prepare_view_visualization(decoded_pred_x0s.permute(0,2,3,1))
            view_uv_maps = []

        uv_maps = []
        synced_latents = []
        # torchvision.transfroms
        for i, (xt, x0, prev_x, a_prev, sigma_t, inv_uv, uv, cos_angle) in enumerate(zip(decoded_xts.chunk(batch_size), decoded_pred_x0s.chunk(batch_size), decoded_prev_xs.chunk(batch_size), \
                                                                                a_prevs.chunk(batch_size), sigma_ts.chunk(batch_size), inv_uvs, uvs, cos_angles)):
            # self.print_tensor("xt", xt); self.print_tensor("x0", x0); self.print_tensor("prev_x", prev_x); self.print_tensor("prev_x", prev_x); self.print_tensor("sigma_t", sigma_t); self.print_tensor("inv_uv", inv_uv); 
            # self.print_tensor("uv", uv); self.print_tensor("cos_angle", cos_angle); 
            inv_uv, inv_uv_mask_valid, inv_uv_mask_boundary = torch.split(inv_uv, [2,1,1], dim=-1)
            texture_x0, view_x0 = torch.split(x0.permute(0,2,3,1), [1, num_frames], dim=0)
            # imageio.imwrite('view_x0.png', self.save_tensor(view_x0))
            texture_xt, view_xt = torch.split(xt.permute(0,2,3,1), [1, num_frames], dim=0)
            texture_prev_xt, view_prev_x = torch.split(prev_x.permute(0,2,3,1), [1, num_frames], dim=0)

            a_prev_texture, a_prev_views = torch.split(a_prev, [1, num_frames], dim=0)
            sigma_t_texture, a_prev_views = torch.split(sigma_t, [1, num_frames], dim=0)

            partial_albedos = dr.texture(view_x0.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            # imageio.imwrite('partial_albedos.png', self.save_tensor(partial_albedos))
            partial_albedos_xt = dr.texture(view_xt.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            # self.print_tensor("partial_albedos", partial_albedos); self.print_tensor("partial_albedos_xt", partial_albedos_xt); 
            inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
            partial_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask).contiguous(), filter_mode='linear')
            # self.print_tensor("partial_weights", partial_weights); 
            
            alpha = (torch.sum(inv_uv_mask > 0, dim=0, keepdim=True) > 0).float()
            alpha2 = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()

            origin_texture = torch.sum(partial_albedos * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float32) 
            current_texture = torch.sum(partial_albedos_xt * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float32)
            origin_texture = origin_texture * alpha2
            current_texture = current_texture * alpha2
            # origin_texture_np = np.clip(origin_texture.cpu().numpy()[0], -1, 1)
            # origin_texture_np = cv2.inpaint(((origin_texture_np+1)*255/2).astype(np.uint8), ((1-alpha).cpu().numpy()[0]*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            # origin_texture = torch.from_numpy(origin_texture_np).to(device="cuda", dtype=torch.float32)[None,...] / 255. * 2 - 1

            # current_texture_np = np.clip(current_texture.cpu().numpy()[0], -1, 1)
            # current_texture_np = cv2.inpaint(((current_texture_np+1)*255/2).astype(np.uint8), ((1-alpha).cpu().numpy()[0]*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            # current_texture = torch.from_numpy(current_texture_np).to(device="cuda", dtype=torch.float32)[None,...] / 255. * 2 - 1
            # self.print_tensor("current_texture", origin_texture); self.print_tensor("current_texture", current_texture); 
            textures = torch.cat([origin_texture, current_texture], dim=0).to(dtype=torch.float16)
            
            texture_posterior = self.encode_first_stage(textures.permute(0,3,1,2))
            texture_latents = self.get_first_stage_encoding(texture_posterior).detach()
            # self.print_tensor("texture_latents", texture_latents); 
            origin_texture_latent, current_texture_latent = torch.split(texture_latents, [1, 1], dim=0)
            
            if latent_tex is not None:
                current_texture_latent = latent_tex

            dir_texture = (1. - a_prev_texture - sigma_t_texture**2).sqrt() * current_texture_latent
            # self.print_tensor("current_texture_latent", current_texture_latent); self.print_tensor("sigma_t_texture", sigma_t_texture); self.print_tensor("a_prev_texture", a_prev_texture);
            noise = sigma_t_texture * torch.rand_like(current_texture_latent) * temperature
            # print(sigma_t_texture, temperature, (1. - a_prev_texture - sigma_t_texture**2).sqrt())
            prev_texture_latent = a_prev_texture.sqrt() * origin_texture_latent + dir_texture + noise
            # self.print_tensor("dir_texture", dir_texture); self.print_tensor("noise", noise); self.print_tensor("prev_texture_latent", prev_texture_latent);

            prev_texture = self.decode_first_stage(prev_texture_latent).permute(0,2,3,1)

            prev_views = dr.texture(prev_texture.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
            uv_mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
            prev_views = prev_views * uv_mask + view_prev_x * (1 - uv_mask)
            # prev_views = prev_views * uv_mask
            prev_views = torch.cat([texture_prev_xt, prev_views], dim=0)
            # self.print_tensor("prev_texture", prev_texture); self.print_tensor("uv_mask", uv_mask); self.print_tensor("prev_views", prev_views);
            # new_x = torch.cat([aggreated_albedos, new_textured], dim=0)

            # print(prev_views.shape)
            new_posterior = self.encode_first_stage(prev_views.permute(0,3,1,2))
            prev_latent = self.get_first_stage_encoding(new_posterior).detach()
            synced_latents.append(prev_latent)
            # pdb.set_trace()

            if debug:
                imageio.imwrite(f'008.png', (np.clip((aggreated_albedos[0].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                for j, t in enumerate(partial_albedos[:8].cpu().numpy()):
                    imageio.imwrite(f'x_{j:03d}.png', (np.clip((x[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'new_x_{j:03d}.png', (np.clip((new_textured[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'{j:03d}.png', (np.clip((t+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'mask_{j:03d}.png', (np.clip(inv_uv_mask[j,:,:,0].cpu().numpy(), 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'uv_{j:03d}.png', (np.clip(inv_uv[j].cpu().numpy(), 0, 1)*255).astype(np.uint8))

                exit(0)

            synced_latents = torch.cat(synced_latents, dim=0)
            if not view_color_debug:
                return synced_latents, prev_texture_latent
            else:
                uv_maps = torch.cat(uv_maps, dim=0)
                view_uv_maps = torch.cat(view_uv_maps, dim=0)
                return synced_latents, uv_maps, view_uv_maps, colors      
            
    @torch.no_grad()
    def noise_synchornize_v2(self, xts, pred_x0s, a_prevs, dir_xts, temperature, uvs, inv_uvs, cos_angles, view_color_debug=False, debug=False):
        batch_size, num_frames = uvs.shape[0], uvs.shape[1]
        decoded_xts = self.decode_first_stage(xts)
        decoded_pred_xts = self.decode_first_stage(pred_x0s)
        if view_color_debug:
            colors, color_tensor = self.prepare_view_visualization(decoded_pred_xts.permute(0,2,3,1))
            view_uv_maps = []

        uv_maps = []
        synced_latents = []
        resize = T.Resize((256, 256))
        for i, (xt, x0, a_prev, dir_xt, inv_uv, uv, cos_angle) in enumerate(zip(decoded_xts.chunk(batch_size), decoded_pred_xts.chunk(batch_size), a_prevs.chunk(batch_size), \
                                                                dir_xts.chunk(batch_size), inv_uvs, uvs, cos_angles)):
            inv_uv, inv_uv_mask_valid, inv_uv_mask_boundary = torch.split(inv_uv, [2,1,1], dim=-1)
            albedo_x0, x0 = torch.split(x0.permute(0,2,3,1), [1, num_frames], dim=0)
            _, xt = torch.split(xt.permute(0,2,3,1), [1, num_frames], dim=0)

            a_prev_albedo = a_prev[:1]
            dir_xt_albedo = dir_xt[:1]

            partial_albedos = dr.texture(x0.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            partial_albedos_xt = dr.texture(xt.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            partial_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask_valid).contiguous(), filter_mode='linear')
            boundary_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask_boundary).contiguous(), filter_mode='linear')
            inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
            alpha = (torch.sum(inv_uv_mask > 0, dim=0, keepdim=True) > 0).float()
            alpha2 = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()

            aggreated_albedos = torch.sum(partial_albedos * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float16) 
            aggreated_albedos_prev = torch.sum(partial_albedos_xt * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float16) 
            aggreated_boundary_albedos = torch.sum(partial_albedos * torch.softmax(boundary_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float16) 
            aggreated_albedos = aggreated_albedos * alpha2 + aggreated_boundary_albedos * (alpha - alpha2)
            aggreated_albedos_np = np.clip(aggreated_albedos.cpu().numpy()[0], -1, 1)
            aggreated_albedos_np = cv2.inpaint(((aggreated_albedos_np+1)*255/2).astype(np.uint8), ((1-alpha).cpu().numpy()[0]*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            aggreated_albedos = torch.from_numpy(aggreated_albedos_np).to(device="cuda", dtype=torch.float16)[None,...] / 255. * 2 - 1

            # noise_albedo = dir_xt_albedo * torch.randn_like(aggreated_albedos) * temperature
            # # print(a_prev_albedo.shape, aggreated_albedos.shape, dir_xt_albedo.shape, noise_albedo.shape, aggreated_albedos_prev.shape)
            # # print(a_prev_albedo.shape, dir_xt_albedo.shape, aggreated_albedos_prev.shape)
            # dir_xt_albedo = (1. - a_prev_albedo - dir_xt_albedo**2).sqrt() * aggreated_albedos_prev
            
            # # print(a_prev_albedo.shape, aggreated_albedos.shape, dir_xt_albedo.shape, noise_albedo.shape)
            # prev_albedo_latent = a_prev_albedo.sqrt() * aggreated_albedos + dir_xt_albedo + noise_albedo

            aggreated_posterior = self.encode_first_stage(aggreated_albedos.permute(0,3,1,2))
            albedo_x0 = self.get_first_stage_encoding(aggreated_posterior).detach()
            synced_latents.append(albedo_x0)

            # prev_albedo_image = self.decode_first_stage(prev_albedo_latent).permute(0,2,3,1)

            uv_maps.append(torch.cat([aggreated_albedos, alpha], dim=-1).permute(0,3,1,2))
            new_textured = dr.texture(aggreated_albedos.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
            uv_mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
            new_textured = new_textured * uv_mask
            # new_x = torch.cat([aggreated_albedos, new_textured], dim=0)

            new_posterior = self.encode_first_stage(new_textured.permute(0,3,1,2))
            new_latent = self.get_first_stage_encoding(new_posterior).detach()
            print(albedo_x0.shape, new_latent.shape)
            synced_latents.append(new_latent)

            if view_color_debug:
                color = color_tensor.chunk(batch_size)[i]
                c_albedo, c_x = torch.split(color, [1, num_frames], dim=0)
                partial_c_albedos = dr.texture(c_x.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
                partial_c_albedos = torch.cat([c_albedo, partial_c_albedos], dim=0)
                # fake a uv color
                partial_weights = torch.cat([torch.ones_like(inv_uv_mask[:1]), partial_weights], dim=0)
                aggreated_c_albedos = torch.sum(partial_c_albedos * torch.softmax(partial_weights.double(), dim=0), dim=0, keepdim=True).to(dtype=torch.float16)
                view_uv_maps.append(aggreated_c_albedos.permute(0,3,1,2)) 
            
            if debug:
                imageio.imwrite(f'008.png', (np.clip((aggreated_albedos[0].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                for j, t in enumerate(partial_albedos[:8].cpu().numpy()):
                    imageio.imwrite(f'x_{j:03d}.png', (np.clip((x[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'new_x_{j:03d}.png', (np.clip((new_textured[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'{j:03d}.png', (np.clip((t+1)/2, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'mask_{j:03d}.png', (np.clip(inv_uv_mask[j,:,:,0].cpu().numpy(), 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f'uv_{j:03d}.png', (np.clip(inv_uv[j].cpu().numpy(), 0, 1)*255).astype(np.uint8))

                exit(0)

            if not view_color_debug:
                return torch.cat(synced_latents, dim=0)
            else:
                uv_maps = torch.cat(uv_maps, dim=0)
                view_uv_maps = torch.cat(view_uv_maps, dim=0)
                return torch.cat(synced_latents, dim=0), uv_maps, view_uv_maps, colors           
            

    @torch.no_grad()
    def noise_synchornize(self, x_noisy, uvs, inv_uvs, cos_angles, sync_direction="both", view_color_debug=False, debug=False):
        batch_size, num_frames = uvs.shape[0], uvs.shape[1]
        decoded_x_nosiy = self.decode_first_stage(x_noisy).clip(-1, 1)
        if view_color_debug:
            colors, color_tensor = self.prepare_view_visualization(decoded_x_nosiy.permute(0,2,3,1))
            view_uv_maps = []

        uv_maps = []
        if sync_direction == 'uv':
            for i, (x, uv) in enumerate(zip(decoded_x_nosiy.chunk(batch_size), uvs)):
                albedo, x = torch.split(x.permute(0,2,3,1), [1, num_frames], dim=0)
                textured = dr.texture(albedo.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
                mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
                # print(uv.min(), uv.max())
                # textured = (textured * mask) + (1 - mask) * x
                textured = textured * mask
                # print(textured.shape, textured.min(), textured.max())
                # imageio.imwrite(f'textured.png', ((albedo[0]+1).detach().cpu().numpy()*255/2).astype(np.uint8))
                # for i, t in enumerate(textured):
                #     imageio.imwrite(f'textured_{i:03d}.png', ((t+1).detach().cpu().numpy()*255/2).astype(np.uint8))
                # for i, t in enumerate(mask):
                #     imageio.imwrite(f'mask_{i:03d}.png', (t[...,0].detach().cpu().numpy()*255).astype(np.uint8))
                # for i, t in enumerate(uv):
                #     t = (t.detach().cpu().numpy()*255).astype(np.uint8)
                #     t = np.concatenate([t, np.zeros_like(t[...,:1])], axis=-1)
                #     imageio.imwrite(f'uv_{i:03d}.png', t)
                textured_posterior = self.encode_first_stage(textured.permute(0,3,1,2))
                textured_x = self.get_first_stage_encoding(textured_posterior).detach()
                x_noisy[i*(num_frames+1)+1:(i+1)*(num_frames+1)] = textured_x
                uv_maps.append(albedo)
            if not view_color_debug:
                return x_noisy
            else:
                uv_maps = torch.cat(uv_maps, dim=0)
                return x_noisy, uv_maps, None, None
        elif sync_direction == 'inv_uv':
            # print(decoded_x_nosiy.shape, uvs.shape, cos_angles.shape)
            for i, (x, inv_uv, cos_angle) in enumerate(zip(decoded_x_nosiy.chunk(batch_size), inv_uvs, cos_angles)):
                albedo, x = torch.split(x.permute(0,2,3,1), [1, num_frames], dim=0)
                partial_albedos = dr.texture(x.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
                # print(cos_angle.device, inv_uv.device, partial_albedos.device)
                partial_weights = dr.texture(cos_angle.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
                mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
                alpha = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()
                partial_albedos, mask = torch.cat([albedo, partial_albedos], dim=0), torch.cat([torch.ones_like(mask[:1]), mask], dim=0)

                partial_weights = torch.cat([torch.ones_like(mask[:1]), partial_weights * self.uv_temperature], dim=0)
                aggreated_albedos = torch.sum(partial_albedos * torch.softmax(partial_weights.double(), dim=0), dim=0, keepdim=True).to(dtype=torch.float16)
                # aggreated_albedos_np = aggreated_albedos.cpu().numpy()
                # aggreated_albedos_np = cv2.inpaint(((aggreated_albedos_np+1)*255/2).astype(np.uint8), ((1-alpha).cpu().numpy()[0]*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
                # aggreated_albedos = torch.from_numpy(aggreated_albedos_np).to(partial_albedos)[None,...] / 255. * 2 - 1
                albedo_posterior = self.encode_first_stage(aggreated_albedos.permute(0,3,1,2))
                aggreated_albedo = self.get_first_stage_encoding(albedo_posterior).detach()
                x_noisy[i*(num_frames+1):i*(num_frames+1)+1] = aggreated_albedo
                uv_maps.append(torch.cat([aggreated_albedos, alpha], dim=-1).permute(0,3,1,2))

                if view_color_debug:
                    color = color_tensor.chunk(batch_size)[i]
                    c_albedo, c_x = torch.split(color, [1, num_frames], dim=0)
                    partial_c_albedos = dr.texture(c_x.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
                    partial_c_albedos = torch.cat([c_albedo, partial_c_albedos], dim=0)
                    aggreated_c_albedos = torch.sum(partial_c_albedos * torch.softmax(partial_weights.double(), dim=0), dim=0, keepdim=True).to(dtype=torch.float16)
                    view_uv_maps.append(aggreated_c_albedos.permute(0,3,1,2))
                
                if debug:
                    imageio.imwrite(f'005.png', (np.clip((aggreated_albedos[0].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    for j, t in enumerate(partial_albedos[:4].cpu().numpy()):
                        imageio.imwrite(f'x_{j:03d}.png', (np.clip((x[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                        imageio.imwrite(f'{j:03d}.png', (np.clip((t+1)/2, 0, 1)*255).astype(np.uint8))
                        imageio.imwrite(f'mask_{j:03d}.png', (np.clip(mask[j+1,:,:,0].cpu().numpy(), 0, 1)*255).astype(np.uint8))
                        imageio.imwrite(f'uv_{j:03d}.png', (np.clip(inv_uv[j].cpu().numpy(), 0, 1)*255).astype(np.uint8))

            if not view_color_debug:
                return x_noisy
            else:
                uv_maps = torch.cat(uv_maps, dim=0)
                view_uv_maps = torch.cat(view_uv_maps, dim=0)
                return x_noisy, uv_maps, view_uv_maps, colors
        # sync both direction
        else:
            for i, (x, inv_uv, uv, cos_angle) in enumerate(zip(decoded_x_nosiy.chunk(batch_size), inv_uvs, uvs, cos_angles)):
                inv_uv, inv_uv_mask_valid, inv_uv_mask_boundary = torch.split(inv_uv, [2,1,1], dim=-1)
                x = x.permute(0,2,3,1)
                partial_albedos = dr.texture(x.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
                partial_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask_valid).contiguous(), filter_mode='linear')
                boundary_weights = dr.texture(cos_angle.float().contiguous(), (inv_uv.float()*inv_uv_mask_boundary).contiguous(), filter_mode='linear')
                inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
                alpha = (torch.sum(inv_uv_mask > 0, dim=0, keepdim=True) > 0).float()
                alpha2 = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()
                # partial_albedos, inv_uv_mask = torch.cat([albedo, partial_albedos], dim=0), torch.cat([torch.ones_like(inv_uv_mask[:1]), inv_uv_mask], dim=0)

                # partial_weights = torch.cat([torch.ones_like(inv_uv_mask[:1]), partial_weights * self.uv_temperature], dim=0)
                aggreated_albedos = torch.sum(partial_albedos * torch.softmax(partial_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float16) 
                aggreated_boundary_albedos = torch.sum(partial_albedos * torch.softmax(boundary_weights.double() * self.uv_temperature, dim=0), dim=0, keepdim=True).to(dtype=torch.float16) 
                aggreated_albedos = aggreated_albedos * alpha2 + aggreated_boundary_albedos * (alpha - alpha2)
                # aggreated_albedos_np = np.clip(aggreated_albedos.cpu().numpy()[0], -1, 1)
                # aggreated_albedos_np = cv2.inpaint(((aggreated_albedos_np+1)*255/2).astype(np.uint8), ((1-alpha).cpu().numpy()[0]*255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
                # aggreated_albedos = torch.from_numpy(aggreated_albedos_np).to(device="cuda", dtype=torch.float16)[None,...] / 255. * 2 - 1
                uv_maps.append(torch.cat([aggreated_albedos, alpha], dim=-1).permute(0,3,1,2))
                new_textured = dr.texture(aggreated_albedos.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
                uv_mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
                new_textured = new_textured * uv_mask + x * (1 - uv_mask)
                new_textured = new_textured.clip(-1, 1)
                # new_x = torch.cat([aggreated_albedos, new_textured], dim=0)

                new_posterior = self.encode_first_stage(new_textured.permute(0,3,1,2))
                new_x = self.get_first_stage_encoding(new_posterior).detach()
                
                x_noisy[i*num_frames:(i+1)*num_frames] = new_x
            # for i, (x, inv_uv, uv, cos_angle) in enumerate(zip(decoded_x_nosiy.chunk(batch_size), inv_uvs, uvs, cos_angles)):
            #     albedo, x = torch.split(x.permute(0,2,3,1), [1, num_frames], dim=0)
            #     partial_albedos = dr.texture(x.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            #     partial_weights = dr.texture(cos_angle.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
            #     inv_uv_mask = (torch.sum((inv_uv > 0) & (inv_uv < 1), dim=-1, keepdim=True) == 2).float()
            #     alpha = (torch.sum(partial_weights > 0, dim=0, keepdim=True) > 0).float()
            #     # partial_albedos, inv_uv_mask = torch.cat([albedo, partial_albedos], dim=0), torch.cat([torch.ones_like(inv_uv_mask[:1]), inv_uv_mask], dim=0)

            #     # partial_weights = torch.cat([torch.ones_like(inv_uv_mask[:1]), partial_weights * self.uv_temperature], dim=0)
            #     partial_weights *= self.uv_temperature
            #     aggreated_albedos = torch.sum(partial_albedos * torch.softmax(partial_weights.double(), dim=0), dim=0, keepdim=True).to(dtype=torch.float16) 
            #     uv_maps.append(torch.cat([aggreated_albedos, alpha], dim=-1).permute(0,3,1,2))
            #     new_textured = dr.texture(aggreated_albedos.float().contiguous(), uv.float().contiguous(), filter_mode='linear')
            #     uv_mask = (torch.sum((uv > 0) & (uv < 1), dim=-1, keepdim=True) == 2).float()
            #     new_textured = new_textured * uv_mask
            #     new_x = torch.cat([aggreated_albedos, new_textured], dim=0)

            #     new_posterior = self.encode_first_stage(new_x.permute(0,3,1,2))
            #     new_x = self.get_first_stage_encoding(new_posterior).detach()
            #     x_noisy[i*(num_frames+1):(i+1)*(num_frames+1)] = new_x
                
                if False:
                    partial_albedos_ = torch.cat([albedo, aggreated_albedos, partial_albedos], dim=0)
                    partial_albedos_ = (np.clip((partial_albedos_.cpu().numpy()+1)/2, 0, 1)*255)
                    partial_albedos_ = np.hstack(partial_albedos_.tolist()).astype(np.uint8)
                    imageio.imwrite('partial_albedos2.png', partial_albedos_)
                    new_textured = (np.clip((new_textured.cpu().numpy()+1)/2, 0, 1)*255)
                    new_textured = np.hstack(new_textured.tolist()).astype(np.uint8)
                    imageio.imwrite('new_textured2.png', new_textured)

                    inv_uv_ = (np.clip(inv_uv.cpu().numpy(), 0, 1)*255)
                    inv_uv_ = np.hstack(inv_uv_.tolist()).astype(np.uint8)
                    inv_uv_ = np.concatenate([inv_uv_, np.zeros_like(inv_uv_[...,:1])], axis=-1)
                    imageio.imwrite('inv_uv2.png', inv_uv_)
                    uv_ = (np.clip(uv.cpu().numpy(), 0, 1)*255)
                    uv_ = np.hstack(uv_.tolist()).astype(np.uint8)
                    uv_ = np.concatenate([uv_, np.zeros_like(uv_[...,:1])], axis=-1)
                    imageio.imwrite('uv2.png', uv_)

                if view_color_debug:
                    color = color_tensor.chunk(batch_size)[i]
                    c_albedo, c_x = torch.split(color, [1, num_frames], dim=0)
                    partial_c_albedos = dr.texture(c_x.float().contiguous(), inv_uv.float().contiguous(), filter_mode='linear')
                    partial_c_albedos = torch.cat([c_albedo, partial_c_albedos], dim=0)
                    # fake a uv color
                    partial_weights = torch.cat([torch.ones_like(inv_uv_mask[:1]), partial_weights], dim=0)
                    aggreated_c_albedos = torch.sum(partial_c_albedos * torch.softmax(partial_weights.double(), dim=0), dim=0, keepdim=True).to(dtype=torch.float16)
                    view_uv_maps.append(aggreated_c_albedos.permute(0,3,1,2)) 
                
                if debug:
                    imageio.imwrite(f'008.png', (np.clip((aggreated_albedos[0].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                    for j, t in enumerate(partial_albedos[:8].cpu().numpy()):
                        imageio.imwrite(f'x_{j:03d}.png', (np.clip((x[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                        imageio.imwrite(f'new_x_{j:03d}.png', (np.clip((new_textured[j].cpu().numpy()+1)/2, 0, 1)*255).astype(np.uint8))
                        imageio.imwrite(f'{j:03d}.png', (np.clip((t+1)/2, 0, 1)*255).astype(np.uint8))
                        imageio.imwrite(f'mask_{j:03d}.png', (np.clip(inv_uv_mask[j,:,:,0].cpu().numpy(), 0, 1)*255).astype(np.uint8))
                        imageio.imwrite(f'uv_{j:03d}.png', (np.clip(inv_uv[j].cpu().numpy(), 0, 1)*255).astype(np.uint8))

                    exit(0)

            if not view_color_debug:
                return x_noisy
            else:
                uv_maps = torch.cat(uv_maps, dim=0)
                view_uv_maps = torch.cat(view_uv_maps, dim=0)
                return x_noisy, uv_maps, view_uv_maps, colors           
            
    def p_losses(self, x_start, cond, t, uv=None, inv_uv=None, cos_angles=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # print(x_start.shape, noise.shape, t, len(t), uv.shape, cos_angles.shape, flush=True)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # if self.use_uv is not None and uv is not None and inv_uv is not None:
        #     x_noisy = self.noise_synchornize(x_noisy, uv, inv_uv, cos_angles, sync_direction=self.sync_direction)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss = self.get_loss(model_output, target.to(model_output.dtype), mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss.mean()})

        loss = loss.mean()

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, c, uv, inv_uv, cos_angles, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0]//(uv.shape[1]+1),), device=self.device).long()
        t = torch.repeat_interleave(t, uv.shape[1]+1)
        # t = torch.zeros_like(t)
        return self.p_losses(x, c, t, uv, inv_uv, cos_angles, *args, **kwargs)
    
    def shared_step(self, batch):
        x, c, uv, inv_uv, cos_angles = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, uv, inv_uv, cos_angles)
        return loss

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        # with self.ema_scope():
        #     _, loss_dict_ema = self.shared_step(batch)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def get_input_(self, batch, k):
        # print("=========", flush=True)
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.size(), flush=True)
        #     else:
        #         print(k, type(v), flush=True)
        # exit(0)
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        elif len(x.shape) == 5:
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).to(torch.float16)
        return x
    
    @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None):
        x = self.get_input_(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if cond_key is None:
            cond_key = self.cond_stage_key
        if cond_key != self.first_stage_key:
            if cond_key in ['caption', 'coordinates_bbox', "txt"]:
                xc = batch[cond_key]
                xc = np.array([[x for x in patch] for patch in xc]).T.reshape(-1).tolist()
            elif cond_key in ['class_label', 'cls']:
                xc = batch
            else:
                xc = self.get_input_(batch, cond_key).to(self.device)
        else:
            xc = x

        
        if isinstance(xc, dict) or isinstance(xc, list):
            c = self.get_learned_conditioning(xc)
        else:
            c = self.get_learned_conditioning(xc.to(self.device))

        uv = batch['uv']
        inv_uv = batch['inv_uv']
        cos_angles = batch['cos_angles']
    
        if bs is not None:
            c = c[:bs]
            uv = uv[:bs]
            inv_uv = inv_uv[:bs]

        out = [z, c, uv, inv_uv, cos_angles]
        return out
