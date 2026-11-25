# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in Newton renderer patches; all dependencies are shipped through source/."""

from isaaclab.utils import configclass
from isaaclab_newton.renderers import NewtonWarpRendererCfg


@configclass
class UWLabNewtonWarpRendererCfg(NewtonWarpRendererCfg):
    class_type: str = "uwlab.utils.newton_warp_renderer:UWLabNewtonWarpRenderer"
    backend: str = "warp"
    """Shape traversal: ``"warp"`` (Warp BVH) or ``"optix"`` (OptiX, see ``uwlab/render/optix.py``; needs
    ``warp-optix`` and has no CUDA-graph capture). ``UWLAB_NEWTON_RENDER_BACKEND`` overrides it."""
    optix_extent: float | None = None
    """Half-width [m] of the region about each env's centre that OptiX traces; shapes that leave it are hidden.
    ``None``: twice the reach of the shapes when the acceleration structure is built."""
    optix_rebuild_interval: int = 0
    """Rebuild the OptiX instance structure every N refits instead of refitting it (0: never)."""
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
    path_samples: int = 0
    """Secondary rays per pixel sample and lobe for one-bounce dome lighting (0: off)."""
    path_bounce: float = 0.3
    path_sky_clamp: float = 0.0
    path_sun: bool = False
    """Re-add the sky energy above ``path_sky_clamp`` as up to two shadow-casting suns per map."""
    path_outputs: bool = False
    path_aa: int = 1
    """Primary samples per pixel that anti-alias edges under path lighting."""
    path_filter_step: int = 6
    """Pixel spacing of the second, dilated path-lighting filter pass (0: tent filter only)."""
    tone_mapper: str = "none"
    tone_blur: float = 0.0
    """Sigma [px] of a 3x3 Gaussian applied to the HDR image before tone mapping (0: off)."""
    exposure_range: tuple[float, float] = (0.0, 0.0)
    environment_paths: tuple[str, ...] = ()
    material_overrides: dict = {}
