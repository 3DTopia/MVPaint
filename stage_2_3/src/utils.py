from PIL import Image
import numpy as np
import math
import random
import torch
from torchvision.transforms import Resize, InterpolationMode
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
from scipy.ndimage import label
import cv2
import open3d as o3d


def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    r"""Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.reshape(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output


def get_normals(pcd: np.array):
    pcd = n2o(pcd)
    pcd.estimate_normals()
    return np.asarray(pcd.normals)


def n2o(__points, __colors=None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(__points[:, :3])
    if __points.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(__points[:, 3:])
    if __colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(__colors)
    return pcd


'''
    Encoding and decoding functions similar to diffusers library implementation
'''
@torch.no_grad()
def encode_latents(vae, imgs):
    imgs = (imgs-0.5)*2
    latents = vae.encode(imgs).latent_dist.sample()
    latents = vae.config.scaling_factor * latents
    return latents


@torch.no_grad()
def decode_latents(vae, latents):

    latents = 1 / vae.config.scaling_factor * latents

    image = vae.decode(latents, return_dict=False)[0]
    torch.cuda.current_stream().synchronize()

    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.permute(0, 2, 3, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()
    
    return image


# A fast decoding method based on linear projection of latents to rgb
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
    image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()
    return image


# Decode each view and bake them into a rgb texture
def get_rgb_texture(vae, uvp_rgb, latents, smooth_colorize=False):
    latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
    result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
    result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
    textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False, smooth_colorize=smooth_colorize)
    result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
    return result_tex_rgb, result_tex_rgb_output


def construct_sparse_L(knn_indices, distance_score, m, n):
    """
    knn_indices: a list of arrays where each array contains the k-nearest neighbor indices for one unseen point.
    m: number of unseen points.
    n: total number of points.
    """
    row_indices = []
    col_indices = []

    for i, neighbors in enumerate(knn_indices):
        row_indices.extend([i] * len(neighbors))  # Add the same row index for each neighbor
        col_indices.extend(neighbors)  # Add the column indices of the neighbors

    # Convert to PyTorch tensor
    row_indices = torch.tensor(row_indices, dtype=torch.long)
    col_indices = torch.tensor(col_indices, dtype=torch.long)
    ones_data = torch.ones(len(row_indices))

    # Construct the sparse tensor in COO format
    L = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=distance_score.reshape(-1),
        size=(m, n),
        dtype=torch.float
    )

    return L

