import einops
import torch
import torch as th
import torch.nn as nn

from mvdream.ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from mvdream.ldm.modules.attention import SpatialTransformer, SpatialTransformer3D
from mvdream.ldm.modules.diffusionmodules.openaimodel import MultiViewUNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from mvdream.ldm.interface import LatentDiffusionInterface
from mvdream.ldm.util import log_txt_as_img, exists, instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
import nvdiffrast.torch as dr
import logging
import numpy as np
logging.getLogger('nvdiffrast').setLevel(logging.ERROR)


class ControlledUnetModel(MultiViewUNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, y=None, camera=None, **kwargs):
        # print("ControlledUnetModel forward")
        assert x.shape[0] % self.num_frames == 0, "[UNet] input batch size must be dividable by num_frames!"
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            # Add camera embeddings
            if camera is not None:
                assert camera.shape[0] == emb.shape[0]
                emb = emb + self.camera_embed(camera)

            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context, num_frames=self.num_frames)
                hs.append(h)
            h = self.middle_block(h, emb, context, num_frames=self.num_frames)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context, num_frames=self.num_frames)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            camera_dim=None,
            num_frames=1,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.num_frames = num_frames

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if camera_dim is not None:
            time_embed_dim = model_channels * 4
            self.camera_embed = nn.Sequential(
                linear(camera_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer3D(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer3D(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, camera=None, **kwargs):
        # print("ControlNet forward", x.device)
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        # Add camera embeddings
        if camera is not None:
            assert camera.shape[0] == emb.shape[0]
            emb = emb + self.camera_embed(camera)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context, num_frames=self.num_frames)
                h += guided_hint
                guided_hint = None
            else:
                # print(h.shape, emb.shape, context.shape)
                h = module(h, emb, context, num_frames=self.num_frames)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context, num_frames=self.num_frames)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusionInterface):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # print("ControlLDM get_input")
        x, c, uv, inv_uv, cos_angles = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        if len(control.shape) == 5:
            control = control.reshape(control.shape[0]*control.shape[1], control.shape[2], control.shape[3], control.shape[4])
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        camera = batch['camera'].to(self.device).reshape(-1, batch['camera'].shape[-1])
        uv = uv.to(self.device)
        inv_uv = inv_uv.to(self.device)
        return x, dict(context=[c], control=[control], camera=camera), uv, inv_uv, cos_angles

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        # print("ControlLDM apply_model")
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        context = torch.cat(cond['context'], 1)
        camera = cond['camera']
        # print("ControlLDM apply_model", context.shape, camera.shape)

        #forward(self, x, timesteps=None, context=None, y=None, camera=None, num_frames=1, **kwargs)
        if cond['control'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=context, control=None, only_mid_control=self.only_mid_control)
        else:
            # print("controling", x_noisy.device)
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['control'], 1), timesteps=t, context=context, camera=camera)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            # print("controlded", [c.shape for c in control])
            eps = diffusion_model(x=x_noisy, timesteps=t, context=context, control=control, only_mid_control=self.only_mid_control, camera=camera)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            log = dict()
            z, c, uv, inv_uv, cos_angles = self.get_input(batch, self.first_stage_key, bs=N)
            c_cat, c = c["control"][0][:N].to(c["context"][0][:N].dtype), c["context"][0][:N]
            camera = batch['camera'].to(self.device).reshape(-1, batch['camera'].shape[-1])[:N].to(c.dtype)
            # print("log_images", c_cat.dtype, c.dtype, camera.dtype)
            N = min(z.shape[0], N)
            n_row = min(z.shape[0], n_row)
            # recon = dr.texture(z.permute(0,2,3,1).contiguous(), batch['uv'].permute(0,2,3,1).contiguous(), filter_mode='linear')
            # log["reconstruction"] = self.decode_first_stage(recon.permute(0,3,1,2))
            log["reconstruction"] = self.decode_first_stage(z)
            log["control"] = c_cat * 2.0 - 1.0
            text = np.array([[x for x in patch] for patch in batch[self.cond_stage_key]]).T.reshape(-1).tolist()[:N]
            data_id = batch["data_id"][:1] * N
            log["conditioning"] = log_txt_as_img((256, 256), text, data_id, size=12)
            uv = uv[:1]
            inv_uv = inv_uv[:1]

            if plot_diffusion_rows:
                # get diffusion row
                diffusion_row = list()
                z_start = z[:n_row]
                for t in range(self.num_timesteps):
                    if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                        t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                        t = t.to(self.device).long()
                        noise = torch.randn_like(z_start)
                        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                        diffusion_row.append(self.decode_first_stage(z_noisy))

                diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
                diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
                diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
                diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
                log["diffusion_row"] = diffusion_grid

            if sample:
                # get denoise row
                samples, z_denoise_row = self.sample_log(cond={"control": [c_cat], "context": [c], "camera": camera},
                                                        batch_size=N, ddim=use_ddim,
                                                        ddim_steps=ddim_steps, eta=ddim_eta)
                x_samples = self.decode_first_stage(samples)
                log["samples"] = x_samples
                if plot_denoise_rows:
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                    log["denoise_row"] = denoise_grid

            if unconditional_guidance_scale > 1.0:
                uc_cross = self.get_unconditional_conditioning(N)
                uc_cat = c_cat  # torch.zeros_like(c_cat)
                uc_full = {"control": [uc_cat], "context": [uc_cross],  "camera": camera}
                samples_cfg, _, uv_maps, view_annot, colors = self.sample_log(cond={"control": [c_cat], "context": [c], "camera": camera},
                                                batch_size=N, ddim=use_ddim,
                                                ddim_steps=ddim_steps, eta=ddim_eta,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc_full,
                                                uv=uv,
                                                inv_uv=inv_uv,
                                                cos_angles=cos_angles,
                                                )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
                log["view_annot"] = view_annot
                log["colors"] = colors
                log["uv_maps"] = uv_maps

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["control"][0].shape
        shape = (4, h // 8, w // 8)
        samples, intermediates, uv_maps, view_annot, colors = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates, uv_maps, view_annot, colors

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
