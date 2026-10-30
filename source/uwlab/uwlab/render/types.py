# SPDX-License-Identifier: BSD-3-Clause
"""Instance-owned render settings and persistent buffers; no global Newton patches."""

from dataclasses import dataclass
import math
from numbers import Integral

import warp as wp
from newton._src.geometry import Gaussian
from newton._src.sensors.warp_raytrace.types import MeshData, RenderConfig as NewtonRenderConfig, TextureData

ENV_SAMPLE_WIDTH = 128
ENV_SAMPLE_HEIGHT = 64
ENV_SAMPLE_BINS = ENV_SAMPLE_WIDTH * ENV_SAMPLE_HEIGHT
IBL_ROUGHNESS_LEVELS = 6


@dataclass(unsafe_hash=True)
class RenderConfig(NewtonRenderConfig):
    ambient_sky: tuple[float, float, float] = (0.4, 0.4, 0.45)
    ambient_ground: tuple[float, float, float] = (0.1, 0.1, 0.12)
    ambient_intensity: float = 0.5
    spot_inner: float = 0.95
    spot_outer: float = 0.85
    distance_attenuation: float = 0.02
    shadow_floor: float = 0.3
    shadow_samples: int = 1
    samples_per_pixel: int = 1
    per_world_intrinsics: bool = True
    per_world_lights: bool = True
    enable_pbr: bool = False
    enable_reflections: bool = False
    enable_dome_light: bool = False
    enable_precomputed_ibl: bool = False
    dome_samples: int = 1
    aperture_radius: float = 0.0
    focus_distance: float = 1.0
    shutter_fraction: float = 0.0
    sample_seed: int = 17

    def validate(self):
        if any(
            not isinstance(n, Integral) or not 1 <= n <= 64
            for n in (self.samples_per_pixel, self.shadow_samples, self.dome_samples)
        ):
            raise ValueError("sample counts must be in [1, 64]")
        if not -1.0 <= self.spot_outer < self.spot_inner <= 1.0:
            raise ValueError("spot cones require -1 <= outer < inner <= 1")
        if not 0.0 <= self.shadow_floor <= 1.0 or not 0.0 <= self.shutter_fraction <= 1.0:
            raise ValueError("shadow_floor and shutter_fraction must be in [0, 1]")
        if self.focus_distance <= 0 or self.aperture_radius < 0 or self.distance_attenuation < 0:
            raise ValueError("invalid lens or attenuation parameters")
        if self.enable_reflections and not self.enable_pbr:
            raise ValueError("reflections require enable_pbr=True")
        if self.enable_precomputed_ibl and self.enable_dome_light:
            raise ValueError("precomputed IBL and sampled dome lighting are mutually exclusive")
        if self.enable_precomputed_ibl and not self.enable_pbr:
            raise ValueError("precomputed IBL requires enable_pbr=True")
        if (self.aperture_radius > 0 or self.shutter_fraction > 0) and self.samples_per_pixel < 2:
            raise ValueError("depth of field and motion blur require multiple samples")
        if len(self.ambient_sky) != 3 or len(self.ambient_ground) != 3:
            raise ValueError("ambient colors require three channels")
        for value in (
            self.ambient_intensity,
            self.aperture_radius,
            self.focus_distance,
            self.distance_attenuation,
            self.max_distance,
            *self.ambient_sky,
            *self.ambient_ground,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("ambient settings must be finite and nonnegative")


@wp.struct
class Material:
    metallic: float
    roughness: float
    specular: float
    emissive: wp.vec3f
    roughness_map: int
    normal_map: int
    emissive_map: int
    normal_strength: float


@wp.struct
class SceneData:
    world_count: int
    light_count: int
    bvh_shapes_size: int
    bvh_shapes_id: wp.uint64
    bvh_shapes_group_roots: wp.array[wp.int32]
    shape_enabled: wp.array[wp.uint32]
    shape_types: wp.array[wp.int32]
    shape_sizes: wp.array[wp.vec3f]
    shape_colors: wp.array[wp.vec3f]
    shape_transforms: wp.array[wp.transformf]
    shape_source_ptr: wp.array[wp.uint64]
    shape_texture_ids: wp.array[wp.int32]
    shape_mesh_data_ids: wp.array[wp.int32]
    materials: wp.array[Material]
    bvh_particles_size: int
    bvh_particles_id: wp.uint64
    bvh_particles_group_roots: wp.array[wp.int32]
    particles_position: wp.array[wp.vec3f]
    particles_radius: wp.array[wp.float32]
    topology_particle_mask: wp.array[wp.bool]
    triangle_mesh_id: wp.uint64
    triangle_mesh_group_roots: wp.array[wp.int32]
    mesh_data: wp.array[MeshData]
    gaussians_data: wp.array[Gaussian.Data]
    texture_data: wp.array[TextureData]
    light_active: wp.array[wp.bool]
    light_type: wp.array[wp.int32]
    light_cast_shadow: wp.array[wp.bool]
    light_positions: wp.array[wp.vec3f]
    light_orientations: wp.array[wp.vec3f]
    light_colors: wp.array[wp.vec3f]
    light_intensities: wp.array[wp.float32]
    light_radii: wp.array[wp.float32]
    environment_data: wp.array[TextureData]
    environment_ids: wp.array[wp.int32]
    environment_rotation: wp.array[wp.float32]
    environment_intensity: wp.array[wp.float32]
    environment_cdf: wp.array[wp.float32]
    environment_probability: wp.array[wp.float32]
    ibl_diffuse_data: wp.array[TextureData]
    ibl_specular_data: wp.array[TextureData]
    ibl_brdf_data: wp.array[TextureData]


@wp.struct
class FrameData:
    camera_count: int
    width: int
    height: int
    rays: wp.array4d[wp.vec3f]
    intrinsics: wp.array[wp.mat33f]
    transforms: wp.array2d[wp.transformf]
    previous_transforms: wp.array2d[wp.transformf]


@wp.struct
class Outputs:
    color: wp.array[wp.uint32]
    depth: wp.array[wp.float32]
    forward_depth: wp.array[wp.float32]
    shape_index: wp.array[wp.uint32]
    normal: wp.array[wp.vec3f]
    albedo: wp.array[wp.uint32]
    hdr: wp.array[wp.vec3f]
