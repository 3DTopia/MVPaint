import torch

direction_names = ["", "front", "side", "back", "top", "bottom"]

# Append directions to the prompts
def prepare_directional_prompt(prompt, negative_prompt):
	directional_prompt = [prompt + f", {v} view." for v in direction_names]
	negative_prompt = [negative_prompt + f", {v} view." for v in direction_names ]
	return directional_prompt, negative_prompt


# Choose which prompt to use depending on the view azim angle
@torch.no_grad()
def azim_prompt(embeddings, pose):

	elev, azim = pose
	if elev > 30:
		pos_z = embeddings["top"]
	elif elev < -30:
		pos_z = embeddings["bottom"]
	else:
		if azim > 180:
			azim -= 360
		if azim >= -30 and azim <= 30:
			pos_z = embeddings["front"]
		elif azim <=-120 or azim >= 120:
			pos_z = embeddings["back"]
		else:
			pos_z = embeddings["side"]
	return pos_z

@torch.no_grad()
def azim_index(embeddings, pose):

	elev, azim = pose
	if elev > 30:
		pos_z = embeddings[direction_names.index("top")]
	elif elev < -30:
		pos_z = embeddings[direction_names.index("bottom")]
	else:
		if azim > 180:
			azim -= 360
		if azim >= -30 and azim <= 30:
			pos_z = embeddings[direction_names.index("front")]
		elif azim <=-120 or azim >= 120:
			pos_z = embeddings[direction_names.index("back")]
		else:
			pos_z = embeddings[direction_names.index("side")]
	return pos_z


# Choose an opposite prompt for negative prompt
@torch.no_grad()
def azim_neg_prompt(embeddings, pose):
	elev, azim = pose
	if azim > 180:
		azim -= 360
	if azim > -30 and azim < 30:
		pos_z = embeddings[""]
	elif azim <=-120 or azim >= 120:
		pos_z = embeddings["front"]
	else:
		pos_z = embeddings["back"]
	return pos_z

@torch.no_grad()
def azim_neg_index(embeddings, pose):
	elev, azim = pose
	if azim > 180:
		azim -= 360
	if azim > -30 and azim < 30:
		pos_z = embeddings[direction_names.index("")]
	elif azim <=-120 or azim >= 120:
		pos_z = embeddings[direction_names.index("front")]
	else:
		pos_z = embeddings[direction_names.index("back")]
	return pos_z


# We can also linearly blend the prompt
# Currently not in use
@torch.no_grad()
def azim_prompt_mix(embeddings, pose):
	elev, azim = pose
	if elev >= 30:
		pos_z = embeddings["top"]
	elif elev <= -30:
		pos_z = embeddings["bottom"]
	else:
		# print(azim)
		if azim > 180:
			azim -= 360
		if azim >= -90 and azim < 90:
			if azim >= 0:
				r = 1 - azim / 90
			else:
				r = 1 + azim / 90
			start_z = embeddings['front']
			end_z = embeddings['side']
			pos_z = r * start_z + (1 - r) * end_z
		else:
			if azim >= 0:
				r = 1 - (azim - 90) / 90
			else:
				r = 1 + (azim + 90) / 90
			start_z = embeddings['side']
			end_z = embeddings['back']
			pos_z = r * start_z + (1 - r) * end_z
	return pos_z