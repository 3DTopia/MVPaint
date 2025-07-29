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
def get_rgb_texture(vae, uvp_rgb, latents):
    latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
    result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
    result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
    textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
    result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
    return result_tex_rgb, result_tex_rgb_output


def update_colored_points(colored_points):
    points = colored_points[:, :3]
    colors = colored_points[:, 3:]
    color_mask = colors.sum(axis=1) != 0

    if color_mask.sum() != len(points):  # 还有未上色点
        colors, invalid_index = knn_color_completion(points, colors, color_mask)

    return np.concatenate([points, colors], 1)




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


def knn_color_completion(points, colors, color_mask, n_neighbors=60):
    """
    points: (N, 3) 点云坐标
    colors: (N, 3) 点云颜色
    color_mask: (N,) bool 数组，表示哪些点已经有颜色
    n_neighbors: int, 用于KNN的邻居数量
    """


    normals = get_normals(points)
    normals = torch.tensor(normals, dtype=torch.float32)
    normals = torch.nn.functional.normalize(normals, p=2, dim=1)

    points = torch.tensor(points, dtype=torch.float32)
    colors = torch.tensor(colors, dtype=torch.float32)
    color_mask = torch.tensor(color_mask, dtype=torch.bool)
    invalid_index = torch.where(color_mask == False)[0]

    invalid_index_ori = deepcopy(invalid_index)
    # 将已有颜色的点和无色的点分开
    unknown_points = points[~color_mask]

    unknown_normals = normals[~color_mask]

    # 使用 NearestNeighbors 找到无色点的 k 近邻
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')

    knn.fit(points)
    distances, indices = knn.kneighbors(unknown_points)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    edge_neighbors_normals = index_select(normals, torch.LongTensor(indices), 0)
    cos = torch.einsum('ec,enc->en', unknown_normals, edge_neighbors_normals)
    cos = cos / 2 + 0.5
    cos = torch.where(cos < 0.5, 1e-8, cos)
    cos = torch.where(cos > 0.9, 10, cos)

    distances = torch.from_numpy(distances)
    distance_score = torch.nn.functional.normalize((1 / distances), p=2, dim=1)

    weight = cos * distance_score

    colored_count = torch.ones_like(colors[:, 0])  # [V]
    colored_count[invalid_index] = 0
    L_invalid = construct_sparse_L(indices, weight, m=invalid_index.shape[0], n=colors.shape[0])
    # 根据 k 近邻插值颜色
    total_colored = colored_count.sum()
    coloring_round = 0
    stage = "uncolored"
    from tqdm import tqdm
    pbar = tqdm(miniters=100)
    while stage == "uncolored" or coloring_round > 0:
        new_color = torch.matmul(L_invalid, colors * colored_count[:, None])  # [IV, 3] 邻接点贡献和
        new_count = torch.matmul(L_invalid, colored_count)[:, None]  # [IV, 1] 有几个邻接点是有贡献的

        # 找到仍未被更新的那些点：
        # colorless_mask = torch.where(new_count == 0)[0]
        # L_invalid = torch.index_select(L_invalid, 0, colorless_mask)

        colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, colors[invalid_index])

        colored_count[invalid_index] = (new_count[:, 0] > 0).float()

        # invalid_index = torch.index_select(invalid_index, 0, colorless_mask)

        new_total_colored = colored_count.sum()
        color_num = new_total_colored - total_colored
        if color_num > 0:
            total_colored = new_total_colored
            coloring_round += 1
        else:
            stage = "colored"
            coloring_round -= 1
        pbar.update(1)
        if coloring_round > 10000:
            print("coloring_round > 10000, break")
            break

    return colors.numpy(), invalid_index_ori.numpy()


def knn_seam_smooth(valid_points, non_edge_points_num, non_edge_colors, n_neighbors=30):
    """
    points: (N, 3) 点云坐标
    colors: (N, 3) 点云颜色
    color_mask: (N,) bool 数组，表示哪些点已经有颜色
    n_neighbors: int, 用于KNN的邻居数量
    """

    normals = get_normals(valid_points)
    normals = torch.tensor(normals, dtype=torch.float32)
    normals = torch.nn.functional.normalize(normals, p=2, dim=1)

    non_edge_normals = normals[:non_edge_points_num]
    edge_normals = normals[non_edge_points_num:]

    non_edge_points = valid_points[:non_edge_points_num]
    edge_points = valid_points[non_edge_points_num:]
    colors = torch.tensor(non_edge_colors, dtype=torch.float32)
    edge_points = torch.tensor(edge_points, dtype=torch.float32)
    non_edge_points = torch.tensor(non_edge_points, dtype=torch.float32)

    # 使用 NearestNeighbors 找到接缝点的 k 近邻
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(non_edge_points)
    distances, indices = knn.kneighbors(edge_points)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    edge_neighbors_normals = index_select(non_edge_normals, torch.LongTensor(indices), 0)
    cos = torch.einsum('ec,enc->en', edge_normals, edge_neighbors_normals)
    cos = cos / 2 + 0.5
    cos = torch.where(cos < 0.5, 1e-8, cos)
    cos = torch.where(cos > 0.9, 10, cos)

    distances = torch.from_numpy(distances)
    distance_score = torch.nn.functional.normalize((1 / distances), p=2, dim=1)

    weight = cos * distance_score

    colored_count = torch.ones_like(colors[:, 0])  # [V]
    # colored_count[invalid_index] = 0
    L_invalid = construct_sparse_L(indices, weight, m=edge_points.shape[0], n=colors.shape[0])
    new_color = torch.matmul(L_invalid, colors)  # [seam, 3] 邻接点贡献和
    new_count = torch.matmul(L_invalid, colored_count)[:, None]  # [IV, 1] 有几个邻接点是有贡献的
    new_color = new_color / new_count

    return new_color.numpy()


def smooth_seams(mask, color_pcd, seam_width=3):
    np.save('./mask.npy', mask)
    np.save('./color_pcd.npy', color_pcd)

    valid_pcd = color_pcd[mask]

    mask = mask.astype(np.uint8).reshape(2048, 2048)
    # show_img(mask)
    # 找到连片区域
    labeled_mask, num_features = label(mask)

    # 初始化边缘结果
    edges = np.zeros_like(mask, dtype=np.uint8)

    # 对每个连片区域提取边缘
    for region in range(1, num_features + 1):
        # 创建当前连片的二值图像
        region_mask = (labeled_mask == region).astype(np.uint8)

        # 使用findContours提取边缘
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在edges图像上绘制边缘
        cv2.drawContours(edges, contours, -1, 255, seam_width)  # 255为白色，1为边缘的厚度
    edges = edges / 255

    valid_edges = edges * mask
    valid_edges_mask = (valid_edges > 0).reshape(-1)  # (N,)

    valid_non_edges = (1 - edges) * mask
    valid_non_edges_mask = (valid_non_edges > 0).reshape(-1)  # (N,)

    edge_points = color_pcd[valid_edges_mask][:, :3]

    non_edge_points = color_pcd[valid_non_edges_mask][:, :3]
    non_edge_colors = color_pcd[valid_non_edges_mask][:, 3:]

    valid_points = np.concatenate([non_edge_points, edge_points], 0)
    pcd = n2o(valid_pcd)
    pcd.estimate_normals()
    valid_normals = np.asarray(pcd.normals)
    non_edge_points_num = non_edge_points.shape[0]

    new_colors = knn_seam_smooth(valid_points, non_edge_points_num, non_edge_colors)
    edge_color_pcd = np.concatenate([edge_points, new_colors], -1)

    color_pcd[valid_edges_mask] = edge_color_pcd
    return color_pcd
