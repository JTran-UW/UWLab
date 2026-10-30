# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in Newton renderer patches; all dependencies are shipped through source/."""

from isaaclab.utils import configclass
from isaaclab_newton.renderers import NewtonWarpRendererCfg


@configclass
class UWLabNewtonWarpRendererCfg(NewtonWarpRendererCfg):
    class_type: str = "uwlab.utils.newton_warp_renderer:UWLabNewtonWarpRenderer"
    enable_camera_response: bool = False
    surface_maps: dict = {}
    texture_paths: tuple[str, ...] = ()
    load_texture_banks: bool = False
    enable_patches: bool = False
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
    tone_mapper: str = "none"
    exposure_range: tuple[float, float] = (0.0, 0.0)
    environment_paths: tuple[str, ...] = ()
    material_overrides: dict = {}
