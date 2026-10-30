# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Warp camera intrinsics and scene-wide lighting for OmniReset."""

from __future__ import annotations

from pathlib import Path
import dataclasses
import re

import newton
import torch
import warp as wp

from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer, RenderData


@wp.kernel
def _rays_from_intrinsics(
    intrinsics: wp.array(dtype=wp.mat33f),
    rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    y, x = wp.tid()
    k = intrinsics[0]
    dx = (float(x) + 0.5 - k[0, 2]) / k[0, 0]
    dy = (k[1, 2] - float(y) - 0.5) / k[1, 1]
    rays[0, y, x, 0] = wp.vec3f(0.0)
    rays[0, y, x, 1] = wp.normalize(wp.vec3f(dx, dy, -1.0))


class _IntrinsicsRenderData(RenderData):
    def update(self, positions, orientations, intrinsics):
        if self.camera_rays is None:
            self.camera_rays = wp.empty(
                (1, self.height, self.width, 2), dtype=wp.vec3f, device=self.newton_sensor.model.device
            )
        # The upstream update owns the pose buffers and their graph-captured pointers.
        super().update(positions, orientations, intrinsics)
        wp.launch(
            _rays_from_intrinsics,
            dim=(self.height, self.width),
            inputs=[intrinsics.warp, self.camera_rays],
            device=self.newton_sensor.model.device,
        )


class _PatchedRenderData(_IntrinsicsRenderData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._intrinsics = None
        self._previous_transforms = None

    def update(self, positions, orientations, intrinsics):
        if self._previous_transforms is not None:
            wp.copy(self._previous_transforms, self.camera_transforms)
        super().update(positions, orientations, intrinsics)
        if self._previous_transforms is None:
            self._previous_transforms = wp.clone(self.camera_transforms)
        if self._intrinsics is None:
            self._intrinsics = wp.empty_like(intrinsics.warp)
        wp.copy(self._intrinsics, intrinsics.warp)

    def set_outputs(self, output_data):
        super().set_outputs(output_data)
        if self._renderer_cfg.tone_mapper != "none":
            if self.ppisp_pipeline is not None:
                raise ValueError("select either PPISP or the Newton tone mapper")
            if self.outputs.color_image is None:
                raise ValueError("tone mapping requires RGB/RGBA output")
            if self.outputs.hdr_color_image is None:
                self.outputs.hdr_color_image = wp.empty(
                    (self.newton_sensor.model.world_count, 1, self.height, self.width),
                    dtype=wp.vec3f,
                    device=self.newton_sensor.model.device,
                )

    def _convert_plane_depth(self):
        from uwlab.render.postprocess import planar_depth

        for name, dest in self._depth_dests.items():
            if name in self._PLANE_DEPTH_KINDS:
                wp.launch(
                    planar_depth,
                    dim=dest.shape,
                    inputs=[
                        self.outputs.depth_image,
                        self._intrinsics,
                        dest,
                        int(self._renderer_cfg.per_world_intrinsics),
                    ],
                    device=dest.device,
                )


class UWLabNewtonWarpRenderer(NewtonWarpRenderer):
    """Newton renderer with full shared-camera K and mutable global light slots.

    Newton 1.5.2 indexes rays by camera, not by world. The first world's K is therefore
    shared by all worlds; per-world intrinsics require a Newton raytrace kernel change.
    """

    def initialize(self) -> None:
        super().initialize()
        sensor = self.newton_sensor
        # Newton exposes no public render-context accessor in 1.5.2. Keep the
        # version-specific access here rather than spreading it through event terms.
        self._context = sensor._SensorTiledCamera__render_context
        self._patches = bool(getattr(self.cfg, "enable_patches", False))
        if self._patches:
            self._initialize_patches()
            self.texture_banks = {}
            if self.cfg.enable_textures and self.cfg.load_texture_banks:
                self._load_texture_banks()
            self._apply_material_overrides()
            self.matched_surface_ids = {
                name: {kind: self._context.add_texture(path) for kind, path in maps.items()}
                for name, maps in self.cfg.surface_maps.items()
            }
            self.matched_texture_ids = [self._context.add_texture(path) for path in self.cfg.texture_paths]
            return
        device = self._context.device
        self._context.lights_active = wp.array([True, False, False], dtype=wp.bool, device=device)
        self._context.lights_type = wp.array([1, 2, 2], dtype=wp.int32, device=device)
        self._context.lights_cast_shadow = wp.array([self.cfg.enable_shadows] * 3, dtype=wp.bool, device=device)
        self._context.lights_position = wp.array([wp.vec3f(0.0)] * 3, dtype=wp.vec3f, device=device)
        self._context.lights_orientation = wp.array(
            [wp.vec3f(-0.57735026, 0.57735026, -0.57735026), wp.vec3f(0.0), wp.vec3f(0.0)],
            dtype=wp.vec3f,
            device=device,
        )
        self.texture_banks = {}
        if self.cfg.enable_textures and self.cfg.load_texture_banks:
            self._load_texture_banks()

    def _initialize_patches(self):
        from newton._src.sensors.warp_raytrace.types import RenderConfig as NewtonRenderConfig
        from newton._src.sensors.warp_raytrace.utils import Utils
        from uwlab.render.context import RenderContext
        from uwlab.render.types import RenderConfig
        from uwlab.render.postprocess import create_tone_mapper, create_camera_mapper

        if newton.__version__.split("+")[0] != "1.5.2":
            raise RuntimeError("UWLab renderer patches require Newton 1.5.2; revalidate before upgrading")
        sensor = self.newton_sensor
        base = sensor.default_render_config
        base_fields = {f.name for f in dataclasses.fields(NewtonRenderConfig)}
        params = {name: getattr(base, name) for name in base_fields}
        params.update(
            {f.name: getattr(self.cfg, f.name) for f in dataclasses.fields(RenderConfig) if f.name not in base_fields}
        )
        config = RenderConfig(**params)
        config.validate()
        context = RenderContext(
            world_count=self._newton_model.world_count, device=self._newton_model.device, config=config
        )
        context.init_from_model(self._newton_model, load_textures=self.cfg.enable_textures)
        if self.cfg.environment_paths:
            context.load_environment_maps(self.cfg.environment_paths)
        sensor._SensorTiledCamera__render_context = context
        sensor._SensorTiledCamera__default_render_config = config
        sensor._SensorTiledCamera__utils = Utils(context, config)
        self._context = context
        self._exposure = wp.zeros(context.world_count, dtype=wp.float32, device=context.device)
        self._tone_mapper = None if self.cfg.tone_mapper == "none" else create_tone_mapper(self.cfg.tone_mapper)
        self._white_balance = wp.full(context.world_count, wp.vec3f(1.0), dtype=wp.vec3f, device=context.device)
        self._camera_response = wp.full(context.world_count, wp.vec4f(1.0, 0.0, 0.0, 17.0), dtype=wp.vec4f, device=context.device)
        self._response_frame = 0
        if self.cfg.enable_camera_response:
            assert self.cfg.tone_mapper != "none", 'Camera response requires tone mapping'
            self._tone_mapper = create_camera_mapper(self.cfg.tone_mapper)
        lo, hi = self.cfg.exposure_range
        if lo > hi or (self._tone_mapper is None and (lo != 0.0 or hi != 0.0)):
            raise ValueError("exposure_range requires an ordered range and an enabled tone mapper")

    def _load_texture_banks(self) -> None:
        """Add the bundled CC0 albedo maps once, before any render graph is captured."""
        from newton._src.sensors.warp_raytrace.types import TextureData

        root = Path(__file__).with_name("newton_textures")
        groups = {
            "table": ("wood_floor", "fine_grained_wood"),
            "insertive_object": ("metal_plate_02", "polystyrene"),
            "receptive_object": ("metal_plate_02", "polystyrene"),
            "curtain_back": ("polystyrene",),
            "curtain_left": ("polystyrene",),
            "curtain_right": ("polystyrene",),
        }
        context = self._context
        data = list(context._RenderContext__texture_data)
        ids = {}
        for names in groups.values():
            for name in names:
                if name in ids:
                    continue
                pixels = newton.utils.normalize_texture(
                    newton.utils.load_texture(root / f"{name}.jpg"), require_channels=True
                )
                if pixels is None:
                    raise FileNotFoundError(root / f"{name}.jpg")
                item = TextureData()
                item.texture = wp.Texture2D(
                    pixels,
                    filter_mode=wp.TextureFilterMode.LINEAR,
                    address_mode=wp.TextureAddressMode.WRAP,
                    normalized_coords=True,
                    dtype=wp.uint8,
                    num_channels=4,
                    device=context.device,
                )
                item.repeat = wp.vec2f(1.0, 1.0)
                ids[name] = len(data)
                data.append(item)
        context.texture_data = wp.array(data, dtype=TextureData, device=context.device)
        context._RenderContext__texture_data = data
        self.texture_banks = {group: tuple(ids[name] for name in names) for group, names in groups.items()}

    def _apply_material_overrides(self):
        texture_ids = {}
        for pattern, spec in self.cfg.material_overrides.items():
            shapes = [i for i, label in enumerate(self._newton_model.shape_label) if re.search(pattern, label or "")]
            if not shapes:
                raise ValueError(f"material override matched no shapes: {pattern}")
            values = dict(spec)
            for name in ("normal", "roughness", "emissive"):
                path = values.pop(f"{name}_texture", None)
                if path is not None:
                    if not self.cfg.enable_textures:
                        raise ValueError("material maps require enable_textures=True")
                    if path not in texture_ids:
                        texture_ids[path] = self._context.add_texture(path)
                    values[f"{name}_map"] = texture_ids[path]
            self._context.set_materials(shapes, **values)

    def create_render_data(self, spec):
        # Preserve upstream segmentation and output setup, then use the same sensor
        # and mapper with our ray generation. No renderer or Newton kernel is copied.
        data = super().create_render_data(spec)
        data_type = _PatchedRenderData if self._patches else _IntrinsicsRenderData
        return data_type(data.newton_sensor, spec, seg_mapper=data._seg_mapper, renderer_cfg=data._renderer_cfg)

    def set_lights(
        self,
        positions,
        orientations,
        active,
        *,
        env_ids=None,
        colors=None,
        intensities=None,
        types=None,
        cast_shadow=None,
        radii=None,
    ) -> None:
        """Update three fixed slots in place so captured render graphs retain their pointers."""
        if self._patches:
            per_world = self.newton_sensor.default_render_config.per_world_lights
            n = self._context.world_count if per_world else 1
            ids = slice(None) if env_ids is None or not per_world else env_ids
            values = {
                "lights_position": positions,
                "lights_orientation": orientations,
                "lights_active": active,
                "lights_color": colors,
                "lights_intensity": intensities,
                "lights_type": types,
                "lights_cast_shadow": cast_shadow,
                "lights_radius": radii,
            }
            for name, value in values.items():
                if value is not None:
                    dest = wp.to_torch(getattr(self._context, name))
                    dest = dest.reshape((n, 3, 3) if dest.ndim == 2 else (n, 3))
                    dest[ids] = torch.as_tensor(value, device=dest.device, dtype=dest.dtype)
            return
        if positions.shape != (3, 3) or orientations.shape != (3, 3) or active.shape != (3,):
            raise ValueError("expected positions/orientations (3, 3) and active (3,)")
        wp.to_torch(self._context.lights_position).copy_(positions.to(dtype=torch.float32))
        wp.to_torch(self._context.lights_orientation).copy_(orientations.to(dtype=torch.float32))
        wp.to_torch(self._context.lights_active).copy_(active.to(dtype=torch.bool))

    def set_exposure(self, env_ids, stops):
        if not self._patches:
            raise RuntimeError("per-world exposure requires renderer patches")
        wp.to_torch(self._exposure)[env_ids] = stops

    def _launch_render(self, render_data):
        if self._patches:
            self._context.camera_intrinsics = render_data._intrinsics
            self._context.previous_camera_transforms = render_data._previous_transforms
        super()._launch_render(render_data)

    def render(self, render_data):
        super().render(render_data)
        if self._patches and self._tone_mapper is not None:
            inputs = [render_data.outputs.hdr_color_image, render_data.outputs.color_image, self._exposure]
            if self.cfg.enable_camera_response:
                inputs += [self._white_balance, self._camera_response, self._response_frame]
                self._response_frame += 1
            wp.launch(
                self._tone_mapper,
                dim=render_data.outputs.color_image.shape,
                inputs=inputs,
                device=self._context.device,
            )
