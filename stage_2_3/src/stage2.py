import os
from PIL import Image
import numpy as np
import torch
from .renderer.project import UVProjection as UVP
from .utils import *
from torchvision import transforms
import kaolin as kal
from scipy.ndimage import binary_dilation
from .renderer.voronoi import voronoi_solve
from pytorch3d.renderer import TexturesUV

class SpatialAware3DInpainting:
    def __init__(self, mesh, device, mesh_name=''):
        self.mesh = mesh
        self.device = device
        self.mesh_name = mesh_name

    def set_texture_map(self, texture):
        new_map = texture.permute(1, 2, 0)
        new_map = new_map.to(self.device)
        new_tex = TexturesUV(
            [new_map],
            self.mesh.textures.faces_uvs_padded(),
            self.mesh.textures.verts_uvs_padded(),
            sampling_mode="nearest"
        )
        self.mesh.textures = new_tex

    @torch.no_grad()
    def UV_pos_render(self, texture_dim):
        verts = self.mesh.verts_packed()
        faces = self.mesh.faces_packed()
        verts_uv = self.mesh.textures.verts_uvs_padded()[0]
        faces_uv = self.mesh.textures.faces_uvs_padded()[0]
        uv_face_attr = torch.index_select(verts_uv, 0, faces_uv.view(-1)).view(faces.shape[0], faces_uv.shape[1],
                                                                               2).unsqueeze(0)
        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_z = torch.zeros_like(face_vertices_world[:, :, :, -1], device=verts.device)
        uv_position, face_idx = kal.render.mesh.rasterize(texture_dim, texture_dim, face_vertices_z,
                                                          uv_face_attr * 2 - 1, face_features=face_vertices_world)
        uv_position = torch.clamp(uv_position, -1, 1) / 2 + 0.5
        uv_position[face_idx == -1] = 0
        return uv_position

    @torch.no_grad()
    def update_colored_points(self, colored_points):
        points = colored_points[:, :3]
        colors = colored_points[:, 3:]
        color_mask = colors.sum(axis=1) != 0

        if color_mask.sum() != len(points):
            colors, invalid_index = self.knn_color_completion(points, colors, color_mask)

        return np.concatenate([points, colors], 1)


    def knn_color_completion(self, points, colors, color_mask, n_neighbors=60):
        """
        Completes missing colors for points using k-Nearest Neighbors (kNN).

        Args:
            points (np.ndarray): Array of shape (N, 3), where N is the number of points, representing point cloud coordinates.
            colors (np.ndarray): Array of shape (N, 3), representing the RGB colors of the points.
            color_mask (np.ndarray): Boolean array of shape (N,), indicating which points have valid colors.
            n_neighbors (int): Number of neighbors to consider for kNN. Default is 60.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Updated colors of shape (N, 3) with missing values completed.
                - Indices of points that had missing colors initially (before completion).
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
            colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, colors[invalid_index])
            colored_count[invalid_index] = (new_count[:, 0] > 0).float()
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

    @torch.no_grad()
    def update_texture(self, position_map):
        """
        Updates the texture map by interpolating missing colors using the position map.

        Args:
            position_map (torch.Tensor): Tensor of shape (H, W, 3), representing the 3D positions of points in the UV space.

        Returns:
            np.ndarray: Updated texture map of shape (H, W, 3) with completed colors.
        """
        texture = self.mesh.textures.maps_padded()[0]
        points = position_map.reshape(-1, 3).cpu().numpy()
        texture_map_np = texture.cpu().numpy()
        h, w = texture_map_np.shape[:2]
        texture = texture_map_np.reshape(-1, 3)

        colored_points = np.concatenate([points, texture], 1)

        mask = points[:, 0] != 0
        to_be_update = colored_points[mask]
        updated_colored_points = self.update_colored_points(to_be_update)
        colored_points[mask] = updated_colored_points
        colors = colored_points[:, 3:]
        colors = colors.reshape(h, w, 3)
        return colors

    def __call__(self, texture_path):
        """
        Processes a given texture and updates it with spatial-aware inpainting.

        Args:
            texture_path (str): Path to the input texture image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Resultant texture map as a tensor of shape (H, W, 3) after inpainting.
                - Position map of shape (H, W, 3), representing the 3D positions in UV space.
        """
        totensor = transforms.ToTensor()
        result_tex_rgb = totensor(Image.open(texture_path))

        self.set_texture_map(result_tex_rgb)
        position_map = self.UV_pos_render(result_tex_rgb.shape[1])

        result_tex_rgb = self.update_texture(position_map)
        result_tex_rgb = torch.from_numpy(result_tex_rgb).to(self.device)

        bkgd_mask = position_map.sum(-1) == 0
        fore_mask = (1 - bkgd_mask.int()).squeeze()
        bigger_region = binary_dilation(fore_mask.cpu().numpy(), iterations=8)

        result_tex_rgb = voronoi_solve(result_tex_rgb, position_map.squeeze()[..., 0])
        bigger_region = torch.from_numpy(bigger_region).unsqueeze(-1).to(self.device)
        result_tex_rgb = torch.where(bigger_region > 0, result_tex_rgb, 1)

        return result_tex_rgb, position_map


