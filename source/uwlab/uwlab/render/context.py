# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) 2026 The UW Lab Project Developers
# SPDX-License-Identifier: Apache-2.0
"""Extended launch path for Newton 1.5.2; geometry/BVH ownership stays upstream."""

from dataclasses import astuple, replace
from pathlib import Path
import logging
import math

import numpy as np
import warp as wp
from newton._src.sensors.warp_raytrace.render_context import RenderContext as NewtonRenderContext
from newton._src.sensors.warp_raytrace.types import TextureData, RenderOrder
from .types import ENV_SAMPLE_HEIGHT, ENV_SAMPLE_WIDTH, SUN_SLOTS, RenderConfig, SceneData, FrameData, Outputs, Material
from .kernel import create_kernel
from .ibl import precompute_ibl, precompute_brdf_lut


_IBL_CACHE = {}
_OPTIX_MAX_LAUNCH = 1 << 30
logger = logging.getLogger(__name__)


def _float_texture(pixels, device):
    rgba = np.concatenate((pixels, np.ones((*pixels.shape[:2], 1), dtype=np.float32)), axis=-1)
    record = TextureData()
    record.texture = wp.Texture2D(
        np.ascontiguousarray(rgba),
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.WRAP,
        normalized_coords=True,
        dtype=wp.float32,
        num_channels=4,
        device=device,
    )
    record.repeat = wp.vec2f(1.0)
    return record




def extract_suns(pixels: np.ndarray, clamp: float, cluster_deg: float = 10.0):
    """Radiance above ``clamp`` (what path lighting clips) as up to ``SUN_SLOTS`` directional lights in the map frame
    (z up, u = 0.5 along +x): directions (x, y, z, angular radius [rad]) and colours (radiance x solid angle). The
    brightest excess texel seeds each light, which takes the excess within ``cluster_deg``; the rest goes to the nearest."""
    direction = np.zeros((SUN_SLOTS, 4), np.float32)
    color = np.zeros((SUN_SLOTS, 3), np.float32)
    if clamp <= 0.0:
        return direction, color
    height, width = pixels.shape[:2]
    excess = np.maximum(np.asarray(pixels, np.float64) - clamp, 0.0)
    theta = np.pi * (np.arange(height) + 0.5) / height
    phi = 2.0 * np.pi * ((np.arange(width) + 0.5) / width - 0.5)
    omega = (2.0 * np.pi / width) * (np.pi / height) * np.sin(theta)[:, None]
    dirs = np.stack(np.broadcast_arrays(np.sin(theta)[:, None] * np.cos(phi)[None], np.sin(theta)[:, None] * np.sin(phi)[None],
                                        np.cos(theta)[:, None]), -1)
    weight = (excess @ np.array([0.2126, 0.7152, 0.0722])) * omega
    remaining = weight > 0.0
    for slot in range(SUN_SLOTS):
        if not remaining.any():
            break
        peak = dirs[np.unravel_index(np.argmax(np.where(remaining, weight, 0.0)), weight.shape)]
        member = remaining & ((dirs @ peak) > np.cos(np.radians(cluster_deg)))
        core = np.where(member & ((dirs @ peak) > np.cos(np.radians(4.0))), weight, 0.0)
        mean = (dirs * core[..., None]).sum((0, 1))
        mean /= max(np.linalg.norm(mean), 1.0e-12)
        spread = (np.arccos(np.clip(dirs @ mean, -1.0, 1.0)) * core).sum() / max(core.sum(), 1.0e-12)
        direction[slot] = (*mean, np.clip(spread, 0.005, 0.02))
        color[slot] = (excess * (omega * member)[..., None]).sum((0, 1))
        remaining &= ~member
    if remaining.any():  # keep the total energy: the rest goes to the light it is closest to
        nearest = np.argmax(np.stack([np.where(color[k].sum() > 0, dirs @ direction[k, :3], -2.0) for k in range(SUN_SLOTS)]), 0)
        for slot in range(SUN_SLOTS):
            color[slot] += (excess * (omega * (remaining & (nearest == slot)))[..., None]).sum((0, 1))
    return direction, color

class RenderContext(NewtonRenderContext):
    Config = RenderConfig

    def __init__(
        self,
        world_count=1,
        device=None,
        config=None,
        light_slots=3,
        backend="warp",
        optix_extent=None,
        optix_rebuild_interval=0,
    ):
        """``backend="optix"`` traces shapes with OptiX (see ``optix.py``) instead of the Warp BVH.
        ``optix_extent`` is the half-width [m] of the region about each world's centre that its shapes
        may occupy (default: twice the build-time reach); ``optix_rebuild_interval`` rebuilds the
        instance structure every N refits (0: refit only)."""
        super().__init__(world_count=world_count, device=device)
        if backend not in ("warp", "optix"):
            raise ValueError(f"backend must be 'warp' or 'optix', got {backend!r}")
        self.extension_config = config or RenderConfig()
        self.extension_config.validate()
        self.light_slots = light_slots
        self.backend = backend
        self.optix_extent = optix_extent
        self.optix_rebuild_interval = int(optix_rebuild_interval)
        self.camera_intrinsics = None
        self.previous_camera_transforms = None
        self._environment_textures = []
        self.frame_seed = 0
        self.path_samples = 0
        self.path_aa = 0
        self.path_images = None

    @property
    def light_count(self):
        return self.light_slots

    def init_from_model(self, model, load_textures=True):
        super().init_from_model(model, load_textures)
        count = self.world_count * self.light_slots if self.extension_config.per_world_lights else self.light_slots
        active = np.zeros(count, dtype=bool)
        active[:: self.light_slots] = True
        types = np.tile(np.array([1] + [2] * (self.light_slots - 1), dtype=np.int32), count // self.light_slots)
        self.lights_active = wp.array(active, dtype=wp.bool, device=self.device)
        self.lights_type = wp.array(types, dtype=wp.int32, device=self.device)
        self.lights_cast_shadow = wp.full(count, value=True, dtype=wp.bool, device=self.device)
        self.lights_position = wp.zeros(count, dtype=wp.vec3f, device=self.device)
        self.lights_orientation = wp.array(
            np.tile([-0.57735026, 0.57735026, -0.57735026], (count, 1)), dtype=wp.vec3f, device=self.device
        )
        self.lights_color = wp.array(np.ones((count, 3), dtype=np.float32), dtype=wp.vec3f, device=self.device)
        self.lights_intensity = wp.ones(count, dtype=wp.float32, device=self.device)
        self.lights_radius = wp.zeros(count, dtype=wp.float32, device=self.device)
        self._materials = []
        for source in model.shape_source:
            material = Material()
            metallic = getattr(source, "metallic", None)
            roughness = getattr(source, "roughness", None)
            material.metallic = 0.0 if metallic is None else float(metallic)
            material.roughness = 0.5 if roughness is None else max(0.05, float(roughness))
            material.specular = 0.5
            material.roughness_map = material.normal_map = material.emissive_map = -1
            material.normal_strength = 1.0
            material.uv_scale = 0.0
            self._materials.append(material)
        self.materials = wp.array(self._materials, dtype=Material, device=self.device)
        self.environment_data = wp.array([], dtype=TextureData, device=self.device)
        self.environment_ids = wp.full(self.world_count, value=-1, dtype=wp.int32, device=self.device)
        self.environment_rotation = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        # map frame -> world rotation of each world's environment map (identity: z-up latlong)
        self.environment_orientation = wp.array([wp.quat_identity(dtype=wp.float32)] * self.world_count, dtype=wp.quatf, device=self.device)
        self.environment_intensity = wp.ones(self.world_count, dtype=wp.float32, device=self.device)
        self.environment_ambient = wp.zeros(self.world_count, dtype=wp.vec3f, device=self.device)
        self.environment_mean = wp.zeros(1, dtype=wp.vec3f, device=self.device)
        self.environment_sun_direction = wp.zeros(SUN_SLOTS, dtype=wp.vec4f, device=self.device)
        self.environment_sun_color = wp.zeros(SUN_SLOTS, dtype=wp.vec3f, device=self.device)
        self.environment_cdf = wp.empty(0, dtype=wp.float32, device=self.device)
        self.environment_probability = wp.empty(0, dtype=wp.float32, device=self.device)
        self.ibl_diffuse_data = wp.array([], dtype=TextureData, device=self.device)
        self.ibl_specular_data = wp.array([], dtype=TextureData, device=self.device)
        self.ibl_brdf_data = wp.array([], dtype=TextureData, device=self.device)
        self._empty_intrinsics = wp.empty(0, dtype=wp.mat33f, device=self.device)

    def set_materials(self, shape_ids, **values):
        """Set linear scalar/emissive values or texture-bank indices without reallocating buffers."""
        allowed = {
            "metallic",
            "roughness",
            "specular",
            "emissive",
            "roughness_map",
            "normal_map",
            "emissive_map",
            "normal_strength",
            "uv_scale",
        }
        if values.keys() - allowed:
            raise ValueError(f"unknown material fields: {values.keys() - allowed}")
        for key, value in values.items():
            if key in {"metallic", "roughness", "specular"} and not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be in [0, 1]")
            if key.endswith("_map") and not -1 <= value < self.texture_data.shape[0]:
                raise ValueError(f"invalid texture id {value}")
        for i in shape_ids:
            if not 0 <= int(i) < len(self._materials):
                raise ValueError(f"invalid shape id {i}")
            for key, value in values.items():
                setattr(self._materials[int(i)], key, wp.vec3f(*value) if key == "emissive" else value)
        self.materials.assign(self._materials)

    def add_texture(self, path):
        """Append an albedo/data map before graph capture and return its material-map id."""
        if self.kernel_cache:
            raise RuntimeError("load material maps before the first render")
        from newton.utils import load_texture, normalize_texture

        pixels = normalize_texture(load_texture(path), require_channels=True)
        if pixels is None:
            raise ValueError(f"cannot load texture {path}")
        record = TextureData()
        record.texture = wp.Texture2D(
            pixels,
            filter_mode=wp.TextureFilterMode.LINEAR,
            address_mode=wp.TextureAddressMode.WRAP,
            normalized_coords=True,
            dtype=wp.uint8,
            num_channels=4,
            device=self.device,
        )
        record.repeat = wp.vec2f(1.0)
        data = self._RenderContext__texture_data
        data.append(record)
        self.texture_data = wp.array(data, dtype=TextureData, device=self.device)
        return len(data) - 1

    def add_textures(self, images):
        """Append sRGB ``(H, W, 4)`` uint8 arrays as albedo maps before graph capture; returns their ids."""
        if self.kernel_cache:
            raise RuntimeError("load material maps before the first render")
        data = self._RenderContext__texture_data
        first = len(data)
        for pixels in images:
            record = TextureData()
            record.texture = wp.Texture2D(
                np.ascontiguousarray(pixels),
                filter_mode=wp.TextureFilterMode.LINEAR,
                address_mode=wp.TextureAddressMode.WRAP,
                normalized_coords=True,
                dtype=wp.uint8,
                num_channels=4,
                device=self.device,
            )
            record.repeat = wp.vec2f(1.0)
            data.append(record)
        self.texture_data = wp.array(data, dtype=TextureData, device=self.device)
        return list(range(first, len(data)))

    def load_environment_maps(self, paths, precompute=True):
        """Load linear .npy HDR arrays, or imageio-supported HDR/EXR and sRGB images, before graph capture.
        Entries may also be linear ``(H, W, 3/4)`` float arrays. ``precompute=False`` skips the precomputed-IBL
        maps (path lighting samples the environment itself)."""
        if self.kernel_cache:
            raise RuntimeError("load environment maps before the first render")
        from newton.utils import load_texture

        records = []
        means = []
        suns = []
        probabilities = []
        cdfs = []
        diffuse_records = []
        specular_records = []
        for path in paths:
            given = path if isinstance(path, np.ndarray) else None
            path = Path("array") if given is not None else Path(path)
            if given is not None:
                pixels = given
            elif path.suffix.lower() == ".npy":
                pixels = np.load(path, allow_pickle=False)
            elif path.suffix.lower() in (".exr", ".hdr"):
                import imageio.v3 as iio

                pixels = iio.imread(path)
            else:
                pixels = load_texture(path)
                if pixels is None:
                    raise ValueError(f"cannot load environment map {path}")
            pixels = np.asarray(pixels)
            if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
                raise ValueError(f"environment map must be HxWx3/4: {path}")
            if pixels.dtype == np.uint8:
                pixels = pixels.astype(np.float32) / 255.0
                pixels = np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)
            pixels = np.asarray(pixels, dtype=np.float32)
            if pixels.shape[2] == 3:
                pixels = np.concatenate((pixels, np.ones((*pixels.shape[:2], 1), dtype=np.float32)), axis=2)
            if not np.isfinite(pixels).all() or (pixels < 0).any():
                raise ValueError(f"environment map must be finite and nonnegative: {path}")
            height, width = pixels.shape[:2]
            suns.append(extract_suns(pixels[..., :3], self.extension_config.path_sky_clamp))
            luminance = np.dot(pixels[..., :3], np.array([0.2126, 0.7152, 0.0722]))
            solid_angle = np.sin(np.pi * (np.arange(height) + 0.5) / height)
            clamped = np.minimum(pixels[..., :3], 10.0 * max(float(np.median(luminance)), 1.0e-6))
            means.append((clamped * solid_angle[:, None, None]).sum((0, 1)) / (solid_angle.sum() * width))
            row = np.arange(height) * ENV_SAMPLE_HEIGHT // height
            col = np.arange(width) * ENV_SAMPLE_WIDTH // width
            weights = np.zeros((ENV_SAMPLE_HEIGHT, ENV_SAMPLE_WIDTH), dtype=np.float64)
            np.add.at(weights, (row[:, None], col[None, :]), luminance * solid_angle[:, None])
            total = weights.sum()
            weights += max(total * 1.0e-6 / weights.size, 1.0e-12)
            probability = (weights / weights.sum()).reshape(-1)
            cdf = np.cumsum(probability)
            cdf[-1] = 1.0
            probabilities.append(probability.astype(np.float32))
            cdfs.append(cdf.astype(np.float32))
            record = TextureData()
            record.texture = wp.Texture2D(
                np.ascontiguousarray(pixels),
                filter_mode=wp.TextureFilterMode.LINEAR,
                address_mode=wp.TextureAddressMode.WRAP,
                normalized_coords=True,
                dtype=wp.float32,
                num_channels=4,
                device=self.device,
            )
            record.repeat = wp.vec2f(1.0)
            records.append(record)
            if not precompute:
                continue
            if given is not None:
                diffuse, specular = precompute_ibl(pixels)
            else:
                stat = path.stat()
                cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
                if cache_key not in _IBL_CACHE:
                    _IBL_CACHE[cache_key] = precompute_ibl(pixels)
                diffuse, specular = _IBL_CACHE[cache_key]
            diffuse_records.append(_float_texture(diffuse, self.device))
            specular_records.extend(_float_texture(level, self.device) for level in specular)
        self._environment_textures = records
        self._ibl_textures = (diffuse_records, specular_records)
        self.environment_data = wp.array(records, dtype=TextureData, device=self.device)
        self.environment_mean = wp.array(
            np.asarray(means or [[0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3f, device=self.device
        )
        self.ibl_diffuse_data = wp.array(diffuse_records, dtype=TextureData, device=self.device)
        self.ibl_specular_data = wp.array(specular_records, dtype=TextureData, device=self.device)
        if records and precompute:
            brdf = precompute_brdf_lut()
            self._ibl_brdf_texture = _float_texture(brdf[..., :3], self.device)
            self.ibl_brdf_data = wp.array([self._ibl_brdf_texture], dtype=TextureData, device=self.device)
        else:
            self.ibl_brdf_data = wp.array([], dtype=TextureData, device=self.device)
        if suns:
            self.environment_sun_direction = wp.array(np.concatenate([s[0] for s in suns]), dtype=wp.vec4f, device=self.device)
            self.environment_sun_color = wp.array(np.concatenate([s[1] for s in suns]), dtype=wp.vec3f, device=self.device)
        self.environment_cdf = (
            wp.array(np.concatenate(cdfs), dtype=wp.float32, device=self.device)
            if cdfs
            else wp.empty(0, dtype=wp.float32, device=self.device)
        )
        self.environment_probability = (
            wp.array(np.concatenate(probabilities), dtype=wp.float32, device=self.device)
            if probabilities
            else wp.empty(0, dtype=wp.float32, device=self.device)
        )
        self.environment_ids.fill_(0 if records else -1)

    def optix_shapes(self, model, state, refit=True):
        """The model's OptiX acceleration, built on first use and shared by every context of the model."""
        from . import optix as optix_backend

        if self.state.has_particles or self.triangle_mesh is not None:
            raise optix_backend.UnsupportedError("the OptiX backend does not support particles or deformable surfaces")
        shapes = getattr(model, "_uwlab_optix_shapes", None)
        if shapes is None:
            shapes = optix_backend.build_shapes(model, state, self.shape_render_type, extent=self.optix_extent)
            model._uwlab_optix_shapes = shapes
            logger.info(
                "OptiX shapes: %d world + %d global instances, %d planes, %d geometries, extent %.3f m, grid %d",
                shapes.instance_shapes.shape[0],
                shapes.global_instance_shapes.shape[0],
                len(shapes.planes) + int((shapes.world_planes.numpy() >= 0).sum()),
                len(shapes.geometry),
                shapes.extent,
                shapes.grid,
            )
        elif refit:
            interval = self.optix_rebuild_interval
            optix_backend.refit_shapes(
                model, state, shapes, rebuild=interval > 0 and (shapes.refit_count + 1) % interval == 0
            )
            if shapes.refit_count % 4096 == 0 and int(shapes.out_of_cell.numpy()[0]):
                shapes.out_of_cell.zero_()
                logger.warning(
                    "OptiX hid shapes that left their world's %.2f m cell or were not finite; raise optix_extent"
                    " if they should stay visible",
                    shapes.extent,
                )
        return shapes

    def render(
        self,
        model,
        state,
        *,
        camera_transforms,
        camera_rays,
        color_image=None,
        hdr_color_image=None,
        depth_image=None,
        forward_depth_image=None,
        shape_index_image=None,
        normal_image=None,
        albedo_image=None,
        light_image=None,
        gloss_image=None,
        tint_image=None,
        cover_image=None,
        clear_data=None,
        config=None,
        kernel_block_dim=64,
    ):
        config = config or self.extension_config
        config.validate()
        if config.path_outputs and light_image is None and self.path_images is not None:
            # the camera's own buffers (the sensor API has no slot for them); guides only when not requested
            light_image, gloss_image, tint_image = (self.path_images[k] for k in ("light", "gloss", "tint"))
            cover_image = self.path_images["cover"]
            normal_image = normal_image if normal_image is not None else self.path_images["normal"]
            depth_image = depth_image if depth_image is not None else self.path_images["depth"]
            self.path_images["guides"] = (normal_image, depth_image)
        if config.enable_dome_light and self.environment_data.shape[0] == 0:
            raise ValueError("sampled dome lighting requires a loaded environment map")
        if config.enable_precomputed_ibl and self.ibl_diffuse_data.shape[0] == 0:
            raise ValueError("precomputed IBL requires a loaded environment map")
        if config.per_world_lights != self.extension_config.per_world_lights:
            raise ValueError("per_world_lights cannot change after context initialization")
        clear_data = clear_data or self.DEFAULT_CLEAR_DATA
        width, height = camera_rays.shape[2], camera_rays.shape[1]
        cameras = camera_transforms.shape[0]
        if camera_transforms.shape != (cameras, self.world_count) or camera_rays.shape != (cameras, height, width, 2):
            raise ValueError("inconsistent camera buffer shapes")
        flags = dict(
            render_color=color_image,
            render_depth=depth_image,
            render_forward_depth=forward_depth_image,
            render_shape_index=shape_index_image,
            render_normal=normal_image,
            render_albedo=albedo_image,
            render_hdr_color=hdr_color_image,
        )
        for name, data in flags.items():
            if data is not None and data.shape != (self.world_count, cameras, height, width):
                raise ValueError(f"{name} has incorrect output dimensions")
            setattr(self.state, name, data is not None)
        if not any(x is not None for x in flags.values()):
            raise ValueError("at least one render output is required")
        optix = self.optix_shapes(model, state) if self.backend == "optix" else None
        if optix is None and model.shape_count > 0 and model.bvh_shape_enabled is None:
            raise RuntimeError("missing shape BVH")
        scene = SceneData()
        scene.world_count, scene.light_count = self.world_count, self.light_slots
        scene.path_samples = max(int(self.path_samples or config.path_samples), 1)
        scene.path_aa = max(int(self.path_aa or config.path_aa), 1)
        scene.bvh_shapes_size = model.bvh_shape_count_enabled
        scene.bvh_shapes_id = model.bvh_shapes.id if model.bvh_shapes is not None else 0
        scene.bvh_shapes_group_roots = model.bvh_shapes_group_roots
        for target, value in {
            "shape_enabled": model.bvh_shape_enabled,
            "shape_types": self.shape_render_type,
            "shape_sizes": model.shape_scale,
            "shape_colors": self.shape_colors,
            "shape_transforms": model.bvh_shape_world_transforms,
            "shape_source_ptr": self.shape_source_ptr,
            "shape_texture_ids": self.shape_texture_ids,
            "shape_mesh_data_ids": self.shape_mesh_data_ids,
            "mesh_data": self.mesh_data,
            "gaussians_data": self.gaussians_data,
            "texture_data": self.texture_data,
            "materials": self.materials,
            "light_active": self.lights_active,
            "light_type": self.lights_type,
            "light_cast_shadow": self.lights_cast_shadow,
            "light_positions": self.lights_position,
            "light_orientations": self.lights_orientation,
            "light_colors": self.lights_color,
            "light_intensities": self.lights_intensity,
            "light_radii": self.lights_radius,
            "environment_data": self.environment_data,
            "environment_ids": self.environment_ids,
            "environment_rotation": self.environment_rotation,
            "environment_orientation": self.environment_orientation,
            "environment_intensity": self.environment_intensity,
            "environment_cdf": self.environment_cdf,
            "environment_probability": self.environment_probability,
            "environment_ambient": self.environment_ambient,
            "environment_mean": self.environment_mean,
            "environment_sun_direction": self.environment_sun_direction,
            "environment_sun_color": self.environment_sun_color,
            "ibl_diffuse_data": self.ibl_diffuse_data,
            "ibl_specular_data": self.ibl_specular_data,
            "ibl_brdf_data": self.ibl_brdf_data,
        }.items():
            setattr(scene, target, value)
        if optix is not None:
            scene.bvh_shapes_size, scene.bvh_shapes_id = 0, 0
            scene.shape_transforms = optix.shape_world_transforms
            scene.optix_id, scene.optix_global_id = optix.id, optix.global_id
            scene.optix_world_planes = optix.world_planes_flat
            scene.optix_world_centers = optix.world_centers
        has_particles = config.enable_particles and self.state.has_particles and state.particle_q is not None
        if has_particles:
            scene.bvh_particles_size = state.particle_q.shape[0]
            scene.bvh_particles_id = model.bvh_particles.id if model.bvh_particles is not None else 0
            scene.bvh_particles_group_roots = model.bvh_particles_group_roots
            scene.particles_position, scene.particles_radius = state.particle_q, model.particle_radius
            scene.topology_particle_mask = self._RenderContext__topology_particle_mask
        scene.triangle_mesh_id = self.triangle_mesh.id if self.triangle_mesh is not None else 0
        scene.triangle_mesh_group_roots = self.triangle_mesh_group_roots
        frame = FrameData()
        frame.camera_count, frame.width, frame.height = cameras, width, height
        frame.rays, frame.transforms = camera_rays, camera_transforms
        frame.intrinsics = self.camera_intrinsics if self.camera_intrinsics is not None else self._empty_intrinsics
        if frame.intrinsics.shape[0] not in (0, cameras * self.world_count):
            raise ValueError("intrinsics must contain camera_count * world_count matrices")
        if config.samples_per_pixel > 1 and frame.intrinsics.shape[0] == 0:
            raise ValueError("MSAA requires camera intrinsics")
        frame.previous_transforms = (
            self.previous_camera_transforms if self.previous_camera_transforms is not None else camera_transforms
        )
        frame.seed = int(self.frame_seed)
        out = Outputs()
        for name, data in (
            ("color", color_image),
            ("depth", depth_image),
            ("forward_depth", forward_depth_image),
            ("normal", normal_image),
            ("albedo", albedo_image),
            ("shape_index", shape_index_image),
            ("hdr", hdr_color_image),
            ("light", light_image),
            ("gloss", gloss_image),
            ("tint", tint_image),
            ("cover", cover_image),
        ):
            if data is not None:
                setattr(out, name, data.flatten())
        if config.path_outputs and any(x is None for x in (light_image, gloss_image, tint_image, cover_image)):
            raise ValueError("path_outputs requires light_image, gloss_image, tint_image and cover_image")
        key = (astuple(config), astuple(self.state), astuple(clear_data))
        pixels_per_view = width * height
        if config.render_order == RenderOrder.TILED:
            pixels_per_view = (
                math.ceil(width / config.tile_width)
                * math.ceil(height / config.tile_height)
                * config.tile_width
                * config.tile_height
            )
        dim = self.world_count * cameras * pixels_per_view
        if optix is not None:
            from . import optix as optix_backend

            if dim > _OPTIX_MAX_LAUNCH:
                raise ValueError(f"OptiX launches are limited to {_OPTIX_MAX_LAUNCH} rays, got {dim}")
            # Pipelines are compiled against the shapes they trace, so they are cached with them.
            pipeline = optix.pipelines.get(key)
            if pipeline is None:
                kernel = create_kernel(replace(config), replace(self.state), replace(clear_data), optix)
                pipeline = optix.pipelines[key] = optix_backend.create_pipeline(optix, kernel)
            self.kernel_cache[("optix", key)] = pipeline
            optix_backend.wo.launch(pipeline, dim=dim, inputs=[scene, frame, out])
            return
        kernel = self.kernel_cache.get(key)
        if kernel is None:
            kernel = create_kernel(config, self.state, clear_data)
            self.kernel_cache[key] = kernel
        wp.launch(
            kernel,
            dim=dim,
            inputs=[scene, frame, out],
            device=self.device,
            block_dim=kernel_block_dim,
        )
