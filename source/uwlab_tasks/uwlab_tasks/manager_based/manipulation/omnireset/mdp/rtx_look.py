# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The RTX RGB look and randomization on the UWLab Newton renderer (path lighting).

Mirrors the RTX scene of ``RGBEventCfg`` plus a global dome: a uniform white dome and a sky environment map, and per
surface an OmniPBR material that is either a projected texture times a tint or a solid colour, with random roughness,
metallic and specular level.
"""

from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase

from uwlab.render.types import Material

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

DEFAULT_SKY = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Materials/Textures/"
    "Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
)
"""The sky dome texture of the RGB scenes (``sky_light``)."""

RTX_SURFACES = {
    "wrist_mount": ("Robot/robotiq_base_link/visuals/D415_to_Robotiq_Mount",),
    "left_inner_finger": ("Robot/left_inner_finger/visuals/mesh_1",),
    "right_inner_finger": ("Robot/right_inner_finger/visuals/mesh_1",),
    "insertive_object": ("InsertiveObject/visuals",),
    "receptive_object": ("ReceptiveObject/visuals",),
    "table": ("Table/visuals/vention_mat",),
    "curtain_left": ("CurtainLeft/geometry",),
    "curtain_back": ("CurtainBack/geometry",),
    "curtain_right": ("CurtainRight/geometry",),
}
"""Surfaces the RTX randomization gives a material each, as shape-label prefixes under an env root."""

RTX_SURFACE_RANGES = {
    "insertive_object": ((0.0, 1.0), (0.0, 1.0)),
    "receptive_object": ((0.0, 1.0), (0.0, 1.0)),
    "table": ((0.3, 0.9), (0.0, 0.3)),
    "curtain_left": ((0.0, 1.0), (0.0, 1.0)),
    "curtain_back": ((0.0, 1.0), (0.0, 1.0)),
    "curtain_right": ((0.0, 1.0), (0.0, 1.0)),
}
"""(roughness, metallic) ranges of the RTX task per surface; the others use ``roughness_range`` / ``metallic_range``."""


@wp.kernel
def _write_materials(
    ids: wp.array(dtype=wp.int32),
    roughness: wp.array(dtype=wp.float32),
    metallic: wp.array(dtype=wp.float32),
    specular: wp.array(dtype=wp.float32),
    uv_scale: wp.array(dtype=wp.float32),
    materials: wp.array(dtype=Material),
):
    i = wp.tid()
    item = materials[ids[i]]
    item.roughness = roughness[i]
    item.metallic = metallic[i]
    item.specular = specular[i]
    item.uv_scale = uv_scale[i]
    item.roughness_map = -1
    item.normal_map = -1
    materials[ids[i]] = item


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1.0e-8).pow(1.0 / 2.4) - 0.055)


class RandomizeRtxLook(ManagerTermBase):
    """Dome lighting and, with ``randomize``, the RTX appearance randomization, for the worlds of ``env_ids``.

    Lighting: a uniform dome of radiance ``dome`` plus a sky map scaled by ``sky_intensity_range`` (both in units of
    1000 RTX intensity) and rotated by ``sky_rotation_range`` [turns]. Without ``randomize`` the sky is the scene's
    default one at the lower intensity and rotation; with it, a map of ``hdri_config_path`` per world. ``exposure``
    [stops] is the tone mapper's; 0.3 gave the closest image to the RTX GlobalDome7k reference.

    Materials (``randomize`` only), one draw per surface of ``surfaces`` and world, as
    ``randomize_visual_appearance_multiple_meshes``: with ``texture_prob`` a texture of ``texture_config_path``
    projected at ``texture_scale_range`` repeats per metre and multiplied by a tint from ``diffuse_tint_range``,
    else a solid colour from ``color_range``; roughness, metallic and specular level from their ranges. As on RTX,
    a solid colour is multiplied by the tint of the surface's last textured draw.

    RTX has one sky for all envs and redraws every env's materials on each call; here both are per world.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        from uwlab_assets import resolve_cloud_path

        from uwlab.utils.newton_warp_renderer import UWLabNewtonWarpRenderer

        from .utils import load_asset_paths_from_config

        params = cfg.params
        self.randomize = bool(params.get("randomize", False))
        self.renderers = list(
            {
                id(s._renderer): s._renderer
                for s in env.scene.sensors.values()
                if isinstance(getattr(s, "_renderer", None), UWLabNewtonWarpRenderer)
            }.values()
        )
        assert self.renderers, "the RTX look requires UWLabNewtonWarpRenderer cameras"
        skies = [resolve_cloud_path(params.get("sky", DEFAULT_SKY))]
        textures = []
        if self.randomize:
            skies += load_asset_paths_from_config(params["hdri_config_path"], cache_subdir="hdris")
            if params.get("randomize_appearance", True):
                textures = load_asset_paths_from_config(params["texture_config_path"], cache_subdir="textures")
        self.sky_count = len(skies)
        self.texture_ids = {}
        for renderer in self.renderers:
            renderer.load_environment_bank(skies)
            if textures:
                self.texture_ids[id(renderer)] = renderer.load_texture_bank(textures)
        self.shapes = None
        self.drawn = None
        self.tints = None

    def _bind(self, env: ManagerBasedEnv, surfaces: dict, tint_range) -> None:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        labels = list(NewtonManager.get_model().shape_label)
        roots = [f"/World/envs/env_{i}/" for i in range(env.num_envs)]
        per_env = [[[] for _ in surfaces] for _ in roots]
        index = re.compile(r"^/World/envs/env_(\d+)/(.*)$")
        for shape, label in enumerate(labels):
            m = index.match(label or "")
            if m is None or int(m.group(1)) >= env.num_envs:
                continue
            for k, prefixes in enumerate(surfaces.values()):
                if any(m.group(2) == p or m.group(2).startswith(p + "/") for p in prefixes):
                    per_env[int(m.group(1))][k].append(shape)
        self.shapes = []
        for k, name in enumerate(surfaces):
            ids = [per_env[i][k] for i in range(env.num_envs)]
            assert ids[0] and all(len(x) == len(ids[0]) for x in ids), f"surface {name}: no / uneven shapes"
            self.shapes.append(torch.tensor(ids, dtype=torch.long, device=env.device))
        # as after an earlier textured draw: RTX's steady state, where every surface carries a random tint
        lo, hi = (torch.as_tensor(b, dtype=torch.float32, device=env.device) for b in tint_range)
        self.tints = lo + (hi - lo) * torch.rand(env.num_envs, len(surfaces), 3, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        randomize: bool = False,
        randomize_appearance: bool = True,
        sky_random_orientation: bool = False,
        dome: float = 7.0,
        exposure: float = 0.45,
        sky_intensity_range: tuple[float, float] = (1.0, 4.0),
        sky_rotation_range: tuple[float, float] = (0.0, 1.0),
        sky: str = DEFAULT_SKY,
        hdri_config_path: str | None = None,
        texture_config_path: str | None = None,
        surfaces: dict = RTX_SURFACES,
        texture_prob: float = 0.5,
        color_range: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        diffuse_tint_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        texture_scale_range: tuple[float, float] = (0.7, 5.0),
        roughness_range: tuple[float, float] = (0.2, 1.0),
        metallic_range: tuple[float, float] = (0.0, 0.8),
        specular_range: tuple[float, float] = (0.0, 1.0),
        surface_ranges: dict = RTX_SURFACE_RANGES,
        redraw_on_reset: bool = True,
        redraw: bool = False,
    ) -> None:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        ids = torch.arange(env.num_envs, device=env.device)
        ids = ids if env_ids is None else ids[env_ids] if isinstance(env_ids, slice) else env_ids.long().to(env.device)
        if self.drawn is None:
            self.drawn = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if not (redraw_on_reset or redraw):  # as RTX: one draw at the start, then only the interval's
            ids = ids[~self.drawn[ids]]
        self.drawn[ids] = True
        n = ids.numel()
        if n == 0:
            return
        device = env.device

        def uniform(bounds, *shape):
            lo, hi = (torch.as_tensor(b, dtype=torch.float32, device=device) for b in bounds)
            return lo + (hi - lo) * torch.rand(*shape, device=device)

        if randomize:
            maps = torch.randint(1, self.sky_count, (n,), device=device, dtype=torch.int32)
            intensity, rotation = uniform(sky_intensity_range, n), uniform(sky_rotation_range, n)
            # OmniReset's randomize_hdri: a uniformly random 3D orientation of the sky dome (not only its yaw)
            orientation = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=1) if sky_random_orientation else None
        else:
            maps = torch.zeros(n, device=device, dtype=torch.int32)
            intensity = torch.full((n,), float(sky_intensity_range[0]), device=device)
            rotation = torch.full((n,), float(sky_rotation_range[0]), device=device)
        for renderer in self.renderers:
            renderer.set_environment(
                ids, ambient=torch.full((n, 3), float(dome), device=device), map_ids=maps, intensity=intensity,
                rotation=rotation, orientation=orientation if randomize else None,
            )
            renderer.set_exposure(ids, exposure)
            renderer.set_lights(None, None, [[False] * 3] * n, env_ids=ids)  # the dome is the only light
        if not (randomize and randomize_appearance):  # sky only, or nothing: the surfaces keep their materials
            return
        if self.shapes is None:
            self._bind(env, surfaces, diffuse_tint_range)
        model = NewtonManager.get_model()
        colors = wp.to_torch(model.shape_color)
        for k, (name, shapes) in enumerate(zip(surfaces, self.shapes)):
            count = shapes.shape[1]
            rough, metal = surface_ranges.get(name, (roughness_range, metallic_range))
            textured = torch.rand(n, device=device) < texture_prob
            tint = torch.where(textured[:, None], uniform(diffuse_tint_range, n, 3), self.tints[ids, k])
            self.tints[ids, k] = tint
            albedo = torch.where(textured[:, None], tint, uniform(color_range, n, 3) * tint)
            scale = uniform(texture_scale_range, n)
            pick = torch.rand(n, device=device)
            flat = shapes[ids].reshape(-1)
            colors[flat] = _linear_to_srgb(albedo).repeat_interleave(count, dim=0)
            repeat = lambda v: v.repeat_interleave(count).contiguous()
            material = [repeat(uniform(r, n)) for r in (rough, metal, specular_range)]
            material.append(repeat(torch.where(textured, scale, 0.0)))
            for renderer in self.renderers:
                context = renderer._context
                bank = self.texture_ids[id(renderer)]
                # mip level by projected scale: one more halving per doubling of the repeats per metre
                level = torch.log2(scale / texture_scale_range[0]).floor().clamp(0, bank.shape[1] - 1).long()
                texture = bank[(pick * bank.shape[0]).long().clamp_max(bank.shape[0] - 1), level]
                wp.to_torch(context.shape_texture_ids)[flat] = repeat(torch.where(textured, texture, -1)).to(torch.int32)
                wp.launch(
                    _write_materials,
                    dim=flat.numel(),
                    inputs=[wp.from_torch(flat.to(torch.int32).contiguous()), *(wp.from_torch(v) for v in material), context.materials],
                    device=model.device,
                )
