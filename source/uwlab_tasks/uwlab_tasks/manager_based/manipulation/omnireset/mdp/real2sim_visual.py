"""Reset-local appearance variation for the matched Newton scene."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase
from uwlab.render.types import Material
from .newton_visual import _ShapeColorWriter, _linear_to_srgb

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@wp.kernel
def _materials(ids: wp.array(dtype=wp.int32), roughness: wp.array(dtype=wp.float32),
               metallic: wp.array(dtype=wp.float32), normal_map: int, roughness_map: int, materials: wp.array(dtype=Material)):
    i = wp.tid()
    item = materials[ids[i]]
    item.roughness = roughness[i]
    item.metallic = metallic[i]
    item.normal_map = normal_map
    item.roughness_map = roughness_map
    item.normal_strength = 0.3
    materials[ids[i]] = item


class RandomizeMatchedAppearance(ManagerTermBase):
    """Update only resetting worlds; renderer arrays keep their graph-captured pointers."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.bindings = None
        self.renderers = []

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None,
                 groups: dict, color_delta: float = 0.035, material_delta: float = 0.08,
                 texture_probability: float = 0.5, light_position_delta: float = 0.12,
                 light_strength_range: tuple[float, float] = (0.8, 1.2),
                 exposure_range: tuple[float, float] = (-0.2, 0.2),
                 white_balance_delta: float = 0.04, gamma_range: tuple[float, float] = (.95, 1.05),
                 read_noise_range: tuple[float, float] = (0.0, .003),
                 shot_noise_range: tuple[float, float] = (0.0, .006),
                 ibl_strength_range: tuple[float, float] = (.65, 1.0)) -> None:
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from uwlab.utils.newton_warp_renderer import UWLabNewtonWarpRenderer

        ids = torch.arange(env.num_envs, device=env.device)
        ids = ids if env_ids is None else ids[env_ids] if isinstance(env_ids, slice) else env_ids.long()
        if ids.numel() == 0:
            return
        model = NewtonManager.get_model()
        if self.bindings is None:
            self.renderers = list({id(s._renderer): s._renderer for s in env.scene.sensors.values()
                                   if isinstance(getattr(s, '_renderer', None), UWLabNewtonWarpRenderer)}.values())
            assert self.renderers, 'Matched appearance requires initialized UWLabNewtonWarpRenderer cameras'
            self.bindings = []
            colors = wp.to_torch(model.shape_color)
            for name, spec in groups.items():
                root = getattr(env.scene.cfg, spec['asset']).prim_path.replace('{ENV_REGEX_NS}', env.scene.env_regex_ns)
                writer = _ShapeColorWriter(model, root, env.num_envs, None, spec.get('filter'), False)
                assert writer.count, f'No Newton shapes for matched appearance group {name}'
                base = colors[writer._shapes].clone()
                if 'color' in spec:
                    base[:] = _linear_to_srgb(torch.tensor(spec['color'], device=env.device))
                self.bindings.append((writer, spec, base))
            self.texture_ids = {id(r): r.matched_texture_ids for r in self.renderers}
            assert all(len(bank) == 3 for bank in self.texture_ids.values()), 'Preload the three matched textures'
        n = len(ids)
        for writer, spec, base in self.bindings:
            shapes = writer._shapes[ids].reshape(-1)
            # Base is sRGB from Newton; apply bounded additive jitter in linear space.
            b = base[ids]
            linear = torch.where(b <= .04045, b / 12.92, ((b + .055) / 1.055).pow(2.4))
            jitter = (torch.rand(n, 1, 3, device=env.device) * 2 - 1) * spec.get('color_delta', color_delta)
            wp.to_torch(model.shape_color)[shapes] = _linear_to_srgb(linear + jitter).reshape(-1, 3)
            rough = (spec.get('roughness', .65) + (torch.rand(n, device=env.device)*2-1)*material_delta).clamp(.05, 1)
            metal = (spec.get('metallic', 0) + (torch.rand(n, device=env.device)*2-1)*material_delta*.5).clamp(0, 1)
            rough = rough.repeat_interleave(writer.count).contiguous()
            metal = metal.repeat_interleave(writer.count).contiguous()
            shape_ids = shapes.to(torch.int32).contiguous()
            choice = torch.where(torch.rand(n, device=env.device) < texture_probability,
                                 torch.full((n,), spec.get('texture', -1), device=env.device), -1).long()
            for renderer in self.renderers:
                context = renderer._context
                family = spec.get('surface', ('smooth', 'cloth', 'printed')[spec['texture']] if 'texture' in spec else 'metal')
                maps = renderer.matched_surface_ids.get(family, {})
                wp.launch(_materials, dim=len(shapes), inputs=[wp.from_torch(shape_ids), wp.from_torch(rough),
                          wp.from_torch(metal), maps.get('normal', -1), maps.get('roughness', -1), context.materials], device=model.device)
                bank = torch.tensor(self.texture_ids[id(renderer)], device=env.device)
                tex = torch.where(choice >= 0, bank[choice.clamp_min(0)], -1)
                wp.to_torch(context.shape_texture_ids)[shapes] = tex.repeat_interleave(writer.count).to(torch.int32)
        positions = torch.zeros(n, 3, 3, device=env.device)
        positions[:, 0] = torch.tensor([.7, -.4, 1.0], device=env.device)
        positions[:, 0] += (torch.rand(n, 3, device=env.device)*2-1)*light_position_delta
        positions += env.scene.env_origins[ids, None, :]
        orientations = torch.tensor([[-.35, .45, -.82], [.6, -.2, -.4], [0, 0, -1]], device=env.device).expand(n, -1, -1).clone()
        orientations[:, 1] += (torch.rand(n, 3, device=env.device)*2-1)*.1
        strengths = torch.tensor([2.5, .8, 0], device=env.device).expand(n, -1).clone()
        lo, hi = light_strength_range
        strengths *= lo + (hi-lo)*torch.rand(n, 3, device=env.device)
        colors = torch.tensor([[1, .91, .95], [.82, .9, 1], [1, 1, 1]], device=env.device).expand(n, -1, -1).clone()
        colors = (colors + (torch.rand_like(colors)*2-1)*.04).clamp(0, 1)
        for renderer in self.renderers:
            renderer.set_lights(positions, orientations, [[True, True, False]]*n, env_ids=ids,
                                colors=colors, intensities=strengths, types=[[2, 1, 1]]*n,
                                cast_shadow=[[True, False, False]]*n, radii=[[.025, 0, 0]]*n)

            # Sample camera and IBL parameters only for worlds participating in this reset.
            def sample(bounds: tuple[float, float]) -> torch.Tensor:
                return bounds[0] + (bounds[1]-bounds[0])*torch.rand(n, device=env.device)
            renderer.set_exposure(ids, sample(exposure_range))
            wp.to_torch(renderer._white_balance)[ids] = 1 + (torch.rand(n, 3, device=env.device)*2-1)*white_balance_delta
            response = torch.stack((sample(gamma_range), sample(read_noise_range), sample(shot_noise_range),
                                    torch.randint(1, 100000, (n,), device=env.device).float()), dim=-1)
            wp.to_torch(renderer._camera_response)[ids] = response
            context = renderer._context
            wp.to_torch(context.environment_ids)[ids] = torch.randint(
                len(renderer.cfg.environment_paths), (n,), device=env.device, dtype=torch.int32)
            wp.to_torch(context.environment_intensity)[ids] = sample(ibl_strength_range)
            wp.to_torch(context.environment_rotation)[ids] = (torch.rand(n, device=env.device)*2-1)*.035
