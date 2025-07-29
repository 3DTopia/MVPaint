import torch
# from diffusers.utils import randn_tensor
from diffusers.utils.torch_utils import randn_tensor

'''

	Customized Step Function
	step on texture
	texture

'''
@torch.no_grad()
def step_tex(
		scheduler,
		uvp,
		model_output: torch.FloatTensor,
		timestep: int,
		sample: torch.FloatTensor,
		texture: None,
		generator=None,
		return_dict: bool = True,
		guidance_scale = 1,
		main_views = [],
		hires_original_views = True,
		exp=None,
		cos_weighted=True
):
	t = timestep

	prev_t = scheduler.previous_timestep(t)

	if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
		model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
	else:
		predicted_variance = None

	# 1. compute alphas, betas
	alpha_prod_t = scheduler.alphas_cumprod[t]
	alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
	beta_prod_t = 1 - alpha_prod_t
	beta_prod_t_prev = 1 - alpha_prod_t_prev
	current_alpha_t = alpha_prod_t / alpha_prod_t_prev
	current_beta_t = 1 - current_alpha_t

	# 2. compute predicted original sample from predicted noise also called
	# "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
	if scheduler.config.prediction_type == "epsilon":
		pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
	elif scheduler.config.prediction_type == "sample":
		pred_original_sample = model_output
	elif scheduler.config.prediction_type == "v_prediction":
		pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
	else:
		raise ValueError(
			f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
			" `v_prediction`  for the DDPMScheduler."
		)
	# 3. Clip or threshold "predicted x_0"
	if scheduler.config.thresholding:
		pred_original_sample = scheduler._threshold_sample(pred_original_sample)
	elif scheduler.config.clip_sample:
		pred_original_sample = pred_original_sample.clamp(
			-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
		)

	# 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
	# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
	pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
	current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

	'''
		Add multidiffusion here
	'''

	if texture is None:
		sample_views = [view for view in sample]
		sample_views, texture, _ = uvp.bake_texture(views=sample_views, main_views=main_views, exp=exp)
		sample_views = torch.stack(sample_views, axis=0)[:,:-1,...]


	original_views = [view for view in pred_original_sample]
	original_views, original_tex, visibility_weights = uvp.bake_texture(views=original_views, main_views=main_views, exp=exp)
	uvp.set_texture_map(original_tex)
	original_views = uvp.render_textured_views()
	original_views = torch.stack(original_views, axis=0)[:,:-1,...]

	# 5. Compute predicted previous sample µ_t
	# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
	# pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
	prev_tex = pred_original_sample_coeff * original_tex + current_sample_coeff * texture

	# 6. Add noise
	variance = 0

	if predicted_variance is not None:
		variance_views = [view for view in predicted_variance]
		variance_views, variance_tex, visibility_weights = uvp.bake_texture(views=variance_views, main_views=main_views, cos_weighted=cos_weighted, exp=exp)
		variance_views = torch.stack(variance_views, axis=0)[:,:-1,...]
	else:
		variance_tex = None

	if t > 0:
		device = texture.device
		variance_noise = randn_tensor(
			texture.shape, generator=generator, device=device, dtype=texture.dtype
		)
		if scheduler.variance_type == "fixed_small_log":
			variance = scheduler._get_variance(t, predicted_variance=variance_tex) * variance_noise
		elif scheduler.variance_type == "learned_range":
			variance = scheduler._get_variance(t, predicted_variance=variance_tex)
			variance = torch.exp(0.5 * variance) * variance_noise
		else:
			variance = (scheduler._get_variance(t, predicted_variance=variance_tex) ** 0.5) * variance_noise
	prev_tex = prev_tex + variance

	uvp.set_texture_map(prev_tex)
	prev_views = uvp.render_textured_views()
	pred_prev_sample = torch.clone(sample)
	for i, view in enumerate(prev_views):
		pred_prev_sample[i] = view[:-1]
	masks = [view[-1:] for view in prev_views]

	return {"prev_sample": pred_prev_sample, "pred_original_sample":pred_original_sample, "prev_tex": prev_tex}

	if not return_dict:
		return pred_prev_sample, pred_original_sample
	pass

