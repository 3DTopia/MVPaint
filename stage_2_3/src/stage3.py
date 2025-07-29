import os
from typing import Any, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import torch
from .utils import *
from torchvision import transforms
import kaolin as kal
from .renderer.voronoi import voronoi_solve
from torchvision.transforms import Compose, Resize
from pytorch3d.renderer import TexturesUV

class UVRefinement:
    def __init__(self, mesh, upscaler_type, upscaler, device, mesh_name=''):
        self.mesh = mesh
        self.upscaler_type = upscaler_type
        self.upscaler = upscaler
        self.device = device
        self.mesh_name = mesh_name

    def __call__(self,
                 result_tex_rgb,
                 position_map,
                 positive_prompt="bright, high quality, sharp, best quality",
                 negative_prompt="moiré pattern, black spot, speckles, blur, low quality, noisy image, over-exposed, shadow, oil painting style",
                 ):
        """
        Executes the UV texture refinement process, including upscaling, seam fixing,
        and Voronoi interpolation.

        Args:
        - result_tex_rgb (Tensor): The initial texture image.
        - position_map (Tensor): The UV position map indicating texture coordinates.
        - positive_prompt (str): Desired positive characteristics for the texture.
        - negative_prompt (str): Undesired negative characteristics to avoid.

        Returns:
        - Tensor: The refined texture image.
        """
        if self.upscaler_type == "sd":
            upscaled_texture = self.stable_upscale(
                self.upscaler, result_tex_rgb,
                positive_prompt,
                negative_prompt,
            )
        else:
            upscaled_texture = self.tile_upscale(
                self.upscaler, result_tex_rgb,
                positive_prompt,
                negative_prompt,
                position_map=position_map
            )

        transform = Compose([transforms.ToTensor()])
        result_tex_rgb = transform(upscaled_texture).permute(1, 2, 0)
        self.set_texture_map(result_tex_rgb.permute(2, 0, 1))
        position_map = self.UV_pos_render(result_tex_rgb.shape[1])

        result_tex_rgb = self.fix_seams(position_map)
        result_tex_rgb = torch.from_numpy(result_tex_rgb).to(self.device)
        result_tex_rgb = voronoi_solve(result_tex_rgb, position_map.squeeze()[..., 0])

        return result_tex_rgb

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
    def fix_seams(self, position_map):

        """
        Fixes seams in the UV texture by interpolating missing texture regions.

        Args:
        - position_map (Tensor): UV position map used to determine texture coordinates.

        Returns:
        - numpy.ndarray: The seam-fixed texture image.
        """

        texture = self.mesh.textures.maps_padded()[0]
        points = position_map.reshape(-1, 3).cpu().numpy()
        texture_map_np = texture.cpu().numpy()
        h, w = texture_map_np.shape[:2]
        texture = texture_map_np.reshape(-1, 3)

        colored_points = np.concatenate([points, texture], 1)
        mask = points[:, 0] != 0  # (2048*2048,)

        colored_points = self.smooth_seams(mask, colored_points)
        colors = colored_points[:, 3:]
        colors = colors.reshape(h, w, 3)
        return colors

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
    def stable_upscale(self, upscaler, low_res_texture, prompt, negative_prompt, out_res=2048):
        result_tex_rgb = low_res_texture.cpu()  # 移动到 CPU
        result_tex_rgb = (result_tex_rgb * 255).byte()
        result_tex_rgb_np = result_tex_rgb.numpy()
        low_res_texture = Image.fromarray(result_tex_rgb_np)
        input_size = int(out_res / 4)
        low_res_texture = low_res_texture.resize((input_size, input_size))
        image = upscaler(prompt,
                         image=low_res_texture,
                         negative_prompt=negative_prompt,
                         ).images[0]
        return image

    @torch.no_grad()
    def tile_upscale(self, upscaler, low_res_texture, prompt, negative_prompt, out_res=2048, position_map=None):
        result_tex_rgb = low_res_texture.cpu()
        result_tex_rgb = (result_tex_rgb * 255).byte()
        result_tex_rgb_np = result_tex_rgb.numpy()
        image = Image.fromarray(result_tex_rgb_np)
        image = image.resize((out_res, out_res))
        img_list = [image]
        if position_map is not None:
            position_map = position_map.squeeze().cpu()
            position_map = (position_map * 255).byte()
            position_map_np = position_map.numpy()
            position_image = Image.fromarray(position_map_np)
            position_image = position_image.resize((out_res, out_res))
            img_list.append(position_image)
        res_image = upscaler(prompt,
                         negative_prompt=negative_prompt,
                         image=image,
                         control_image=img_list,
                         height=out_res,
                         width=out_res,
                         num_images_per_prompt=1).images
        return res_image[0]

    def smooth_seams(self, mask, color_pcd, seam_width=3):
        """
        Smooths texture seams using KNN-based interpolation and normal similarity.

        Args:
        - mask (numpy.ndarray): Binary mask indicating valid points.
        - color_pcd (numpy.ndarray): Point cloud with colors (shape: N x 6).
        - seam_width (int, optional): Width of the seam to smooth. Defaults to 3.

        Returns:
        - numpy.ndarray: The point cloud with smoothed colors.
        """
        valid_pcd = color_pcd[mask]
        mask = mask.astype(np.uint8).reshape(2048, 2048)
        labeled_mask, num_features = label(mask)
        edges = np.zeros_like(mask, dtype=np.uint8)
        for region in range(1, num_features + 1):
            region_mask = (labeled_mask == region).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edges, contours, -1, 255, seam_width)
        edges = edges / 255
        valid_edges = edges * mask
        valid_edges_mask = (valid_edges > 0).reshape(-1)
        valid_non_edges = (1 - edges) * mask
        valid_non_edges_mask = (valid_non_edges > 0).reshape(-1)
        edge_points = color_pcd[valid_edges_mask][:, :3]
        non_edge_points = color_pcd[valid_non_edges_mask][:, :3]
        non_edge_colors = color_pcd[valid_non_edges_mask][:, 3:]
        valid_points = np.concatenate([non_edge_points, edge_points], 0)
        pcd = n2o(valid_pcd)
        pcd.estimate_normals()
        non_edge_points_num = non_edge_points.shape[0]
        new_colors = self.knn_seam_smooth(valid_points, non_edge_points_num, non_edge_colors)
        edge_color_pcd = np.concatenate([edge_points, new_colors], -1)
        color_pcd[valid_edges_mask] = edge_color_pcd
        return color_pcd

    def knn_seam_smooth(self, valid_points, non_edge_points_num, non_edge_colors, n_neighbors=30):
        """
        Performs KNN-based seam smoothing using distance and normal similarity.

        Args:
        - valid_points (numpy.ndarray): Point cloud coordinates (N x 3).
        - non_edge_points_num (int): Number of non-edge points in the point cloud.
        - non_edge_colors (numpy.ndarray): Colors of non-edge points (N x 3).
        - n_neighbors (int, optional): Number of nearest neighbors for KNN. Defaults to 30.

        Returns:
        - numpy.ndarray: Smoothed colors for edge points.
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

        colored_count = torch.ones_like(colors[:, 0])
        L_invalid = construct_sparse_L(indices, weight, m=edge_points.shape[0], n=colors.shape[0])
        new_color = torch.matmul(L_invalid, colors)
        new_count = torch.matmul(L_invalid, colored_count)[:, None]
        new_color = new_color / new_count

        return new_color.numpy()