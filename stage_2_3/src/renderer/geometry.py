import torch
import pytorch3d
import torch.nn.functional as F

from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras, 
	AmbientLights,
	PointLights, 
	DirectionalLights, 
	Materials, 
	RasterizationSettings, 
	MeshRenderer, 
	MeshRasterizer,  
	SoftPhongShader,
	SoftSilhouetteShader,
	HardPhongShader,
	TexturesVertex,
	TexturesUV,
	Materials,

)
from pytorch3d.renderer.blending import BlendParams,hard_rgb_blend
from pytorch3d.renderer.utils import convert_to_tensors_and_broadcast, TensorProperties
from pytorch3d.renderer.mesh.shader import ShaderBase


def get_cos_angle(
	points, normals, camera_position
):
	'''
		calculate cosine similarity between view->surface and surface normal.
	'''

	if points.shape != normals.shape:
		msg = "Expected points and normals to have the same shape: got %r, %r"
		raise ValueError(msg % (points.shape, normals.shape))

	# Ensure all inputs have same batch dimension as points
	matched_tensors = convert_to_tensors_and_broadcast(
		points, camera_position, device=points.device
	)
	_, camera_position = matched_tensors

	# Reshape direction and color so they have all the arbitrary intermediate
	# dimensions as points. Assume first dim = batch dim and last dim = 3.
	points_dims = points.shape[1:-1]
	expand_dims = (-1,) + (1,) * len(points_dims)

	if camera_position.shape != normals.shape:
		camera_position = camera_position.view(expand_dims + (3,))

	normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)

	# Calculate the cosine value.
	view_direction = camera_position - points
	view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
	cos_angle = torch.sum(view_direction * normals, dim=-1, keepdim=True)
	cos_angle = cos_angle.clamp(0, 1)

	# Cosine of the angle between the reflected light ray and the viewer
	return cos_angle


def _geometry_shading_with_pixels(
	meshes, fragments, lights, cameras, materials, texels
):
	"""
	Render pixel space vertex position, normal(world), depth, and cos angle

	Args:
		meshes: Batch of meshes
		fragments: Fragments named tuple with the outputs of rasterization
		lights: Lights class containing a batch of lights
		cameras: Cameras class containing a batch of cameras
		materials: Materials class containing a batch of material properties
		texels: texture per pixel of shape (N, H, W, K, 3)

	Returns:
		colors: (N, H, W, K, 3)
		pixel_coords: (N, H, W, K, 3), camera coordinates of each intersection.
	"""
	verts = meshes.verts_packed()  # (V, 3)
	faces = meshes.faces_packed()  # (F, 3)
	vertex_normals = meshes.verts_normals_packed()  # (V, 3)
	faces_verts = verts[faces]
	faces_normals = vertex_normals[faces]
	pixel_coords_in_camera = interpolate_face_attributes(
		fragments.pix_to_face, fragments.bary_coords, faces_verts
	)
	pixel_normals = interpolate_face_attributes(
		fragments.pix_to_face, fragments.bary_coords, faces_normals
	)

	cos_angles = get_cos_angle(pixel_coords_in_camera, pixel_normals, cameras.get_camera_center())

	return pixel_coords_in_camera, pixel_normals, fragments.zbuf[...,None], cos_angles 


class HardGeometryShader(ShaderBase):
	"""
	renders common geometric informations.
	
	
	"""

	def forward(self, fragments, meshes, **kwargs):
		cameras = super()._get_cameras(**kwargs)
		texels = self.texel_from_uv(fragments, meshes)

		lights = kwargs.get("lights", self.lights)
		materials = kwargs.get("materials", self.materials)
		blend_params = kwargs.get("blend_params", self.blend_params)
		verts, normals, depths, cos_angles = _geometry_shading_with_pixels(
			meshes=meshes,
			fragments=fragments,
			texels=texels,
			lights=lights,
			cameras=cameras,
			materials=materials,
		)
		verts = hard_rgb_blend(verts, fragments, blend_params)
		normals = hard_rgb_blend(normals, fragments, blend_params)
		depths = hard_rgb_blend(depths, fragments, blend_params)
		cos_angles = hard_rgb_blend(cos_angles, fragments, blend_params)
		texels = hard_rgb_blend(texels, fragments, blend_params)
		return verts, normals, depths, cos_angles, texels, fragments

	def texel_from_uv(self, fragments, meshes):
		texture_tmp = meshes.textures
		maps_tmp = texture_tmp.maps_padded()
		uv_color = [ [[1,0],[1,1]],[[0,0],[0,1]] ]
		uv_color = torch.FloatTensor(uv_color).to(maps_tmp[0].device).type(maps_tmp[0].dtype)
		uv_texture = TexturesUV([uv_color.clone() for t in maps_tmp], texture_tmp.faces_uvs_padded(), texture_tmp.verts_uvs_padded(), sampling_mode="bilinear")
		meshes.textures = uv_texture
		texels = meshes.sample_textures(fragments)
		meshes.textures = texture_tmp
		texels  = torch.cat((texels, texels[...,-1:]*0), dim=-1)
		return texels
