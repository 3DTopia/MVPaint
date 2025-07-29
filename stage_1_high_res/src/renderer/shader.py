from typing import Optional

import torch
import pytorch3d


from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.structures import Meshes
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

from pytorch3d.renderer.lighting import AmbientLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.mesh.shading import _apply_lighting, flat_shading
from pytorch3d.renderer.mesh.rasterizer import Fragments


'''
	Customized the original pytorch3d hard flat shader to support N channel flat shading
'''
class HardNChannelFlatShader(ShaderBase):
	"""
	Per face lighting - the lighting model is applied using the average face
	position and the face normal. The blending function hard assigns
	the color of the closest face for each pixel.

	To use the default values, simply initialize the shader with the desired
	device e.g.

	.. code-block::

		shader = HardFlatShader(device=torch.device("cuda:0"))
	"""

	def __init__(
		self,
		device = "cpu",
		cameras: Optional[TensorProperties] = None,
		lights: Optional[TensorProperties] = None,
		materials: Optional[Materials] = None,
		blend_params: Optional[BlendParams] = None,
		channels: int = 3,
	):
		self.channels = channels
		ones = ((1.0,)*channels,)
		zeros = ((0.0,)*channels,)
		
		if not isinstance(lights, AmbientLights) or not lights.ambient_color.shape[-1] == channels:
			lights = AmbientLights(
				ambient_color=ones,
				device=device,
			)

		if not materials or not materials.ambient_color.shape[-1] == channels:
			materials = Materials(
				device=device,
				diffuse_color=zeros,
				ambient_color=ones,
				specular_color=zeros,
				shininess=0.0,
			)

		blend_params_new = BlendParams(background_color=(1.0,)*channels)
		if not isinstance(blend_params, BlendParams):
			blend_params = blend_params_new
		else:
			background_color_ = blend_params.background_color
			if isinstance(background_color_, Sequence[float]) and not len(background_color_) == channels:
				blend_params = blend_params_new
			if isinstance(background_color_, torch.Tensor) and not background_color_.shape[-1] == channels:
				blend_params = blend_params_new

		super().__init__(
			device,
			cameras,
			lights,
			materials,
			blend_params,
		)
		

	def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
		cameras = super()._get_cameras(**kwargs)
		texels = meshes.sample_textures(fragments)
		lights = kwargs.get("lights", self.lights)
		materials = kwargs.get("materials", self.materials)
		blend_params = kwargs.get("blend_params", self.blend_params)
		colors = flat_shading(
			meshes=meshes,
			fragments=fragments,
			texels=texels,
			lights=lights,
			cameras=cameras,
			materials=materials,
		)
		images = hard_rgb_blend(colors, fragments, blend_params)
		return images