# import argparse
# import json
# import os

# from PIL import Image
# from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode

import numpy as np
import torch
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler, DDPMScheduler
from diffusers.models.attention_processor import Attention, AttentionProcessor


def replace_attention_processors(module, processor, attention_mask=None, ref_attention_mask=None, ref_weight=0):
	attn_processors = module.attn_processors
	for k, v in attn_processors.items():
		if "attn1" in k:
			attn_processors[k] = processor(custom_attention_mask=attention_mask, ref_attention_mask=ref_attention_mask, ref_weight=ref_weight)
	module.set_attn_processor(attn_processors)


class SamplewiseAttnProcessor2_0:
	r"""
	Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
	"""

	def __init__(self, custom_attention_mask=None, ref_attention_mask=None, ref_weight=0):
		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
		self.ref_weight = ref_weight
		self.custom_attention_mask = custom_attention_mask
		self.ref_attention_mask = ref_attention_mask

	def __call__(
		self,
		attn: Attention,
		hidden_states,
		encoder_hidden_states=None,
		attention_mask=None,
		temb=None,
	):

		residual = hidden_states

		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)

		input_ndim = hidden_states.ndim


		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

		batch_size, sequence_length, channels = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)

		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			# scaled_dot_product_attention expects attention_mask shape to be
			# (batch, heads, source_length, target_length)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = torch.clone(hidden_states)
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


		'''
			reshape encoder hidden state to a single batch
		'''
		encoder_hidden_states_f = encoder_hidden_states.reshape(1, -1, channels)



		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads

		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		'''
			each time select 1 sample from q and compute with concated kv
			concat result hidden states afterwards
		'''
		hidden_state_list = []

		for b_idx in range(batch_size):
			
			query_b = query[b_idx:b_idx+1]

			if self.ref_weight > 0 or True:
				key_ref = key.clone()
				value_ref = value.clone()

				keys = [key_ref[view_idx] for view_idx in self.ref_attention_mask]
				values = [value_ref[view_idx] for view_idx in self.ref_attention_mask]

				key_ref = torch.stack(keys)
				key_ref = key_ref.view(key_ref.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

				value_ref = torch.stack(values)
				value_ref = value_ref.view(value_ref.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

			key_a = key.clone()
			value_a = value.clone()

			# key_a = key_a[max(0,b_idx-1):min(b_idx+1,batch_size)+1]

			keys = [key_a[view_idx] for view_idx in self.custom_attention_mask[b_idx]]
			values = [value_a[view_idx] for view_idx in self.custom_attention_mask[b_idx]]

			# keys = (key_a[b_idx-1], key_a[b_idx], key_a[(b_idx+1)%batch_size])
			# values = (value_a[b_idx-1], value_a[b_idx], value_a[(b_idx+1)%batch_size])
			
			# if b_idx not in [0, batch_size-1, batch_size//2]:
			# 	keys = keys + (key_a[min(batch_size-2, 2*(batch_size//2) - b_idx)],)
			# 	values = values + (value_a[min(batch_size-2, 2*(batch_size//2) - b_idx)],)
			key_a = torch.stack(keys)
			key_a = key_a.view(key_a.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

			# value_a = value_a[max(0,b_idx-1):min(b_idx+1,batch_size)+1]
			value_a = torch.stack(values)
			value_a = value_a.view(value_a.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

			hidden_state_a = F.scaled_dot_product_attention(
				query_b, key_a, value_a, attn_mask=None, dropout_p=0.0, is_causal=False
			)

			if self.ref_weight > 0 or True:
				hidden_state_ref = F.scaled_dot_product_attention(
					query_b, key_ref, value_ref, attn_mask=None, dropout_p=0.0, is_causal=False
				)

				hidden_state = (hidden_state_a + self.ref_weight * hidden_state_ref) / (1+self.ref_weight)
			else:
				hidden_state = hidden_state_a

			# the output of sdp = (batch, num_heads, seq_len, head_dim)
			# TODO: add support for attn.scale when we move to Torch 2.1
			
			hidden_state_list.append(hidden_state)

		hidden_states = torch.cat(hidden_state_list)


		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		hidden_states = hidden_states.to(query.dtype)

		# linear proj
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)

		if input_ndim == 4:
			hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

		if attn.residual_connection:
			hidden_states = hidden_states + residual

		hidden_states = hidden_states / attn.rescale_output_factor

		return hidden_states
