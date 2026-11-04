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


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    k = (torch.stack((5 + h * 6, 3 + h * 6, 1 + h * 6), -1)) % 6
    return v[..., None] - v[..., None] * s[..., None] * (torch.minimum(k, 4 - k).clamp(0, 1))


def _kelvin_to_rgb(kelvin: torch.Tensor, luminance: float = 0.9) -> torch.Tensor:
    """Blackbody colour (Tanner Helland fit, 1000-40000 K) as a linear light tint with luminance ``luminance``."""
    t = kelvin / 100.0
    r = torch.where(t <= 66, torch.full_like(t, 255.0), 329.698727446 * (t - 60).clamp_min(1e-3).pow(-0.1332047592))
    g = torch.where(t <= 66, 99.4708025861 * t.log() - 161.1195681661,
                    288.1221695283 * (t - 60).clamp_min(1e-3).pow(-0.0755148492))
    b = torch.where(t >= 66, torch.full_like(t, 255.0),
                    torch.where(t <= 19, torch.zeros_like(t), 138.5177312231 * (t - 10).clamp_min(1e-3).log() - 305.0447927307))
    srgb = (torch.stack((r, g, b), -1) / 255.0).clamp(0, 1)
    linear = torch.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055).pow(2.4))
    return linear * luminance / (linear @ torch.tensor([.2126, .7152, .0722], device=linear.device))[..., None]


def _rotate_within_cone(d: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Uniformly sample directions within ``max_deg`` of the unit vectors ``d`` (n, 3)."""
    n = d.shape[0]
    cos_max = torch.cos(torch.deg2rad(torch.tensor(max_deg, device=d.device)))
    cos_t = 1 - torch.rand(n, device=d.device) * (1 - cos_max)
    sin_t = (1 - cos_t * cos_t).clamp_min(0).sqrt()
    phi = torch.rand(n, device=d.device) * 2 * torch.pi
    helper = torch.where(d[:, 2:3].abs() < .9, torch.tensor([0., 0., 1.], device=d.device), torch.tensor([1., 0., 0.], device=d.device))
    u = torch.nn.functional.normalize(torch.cross(d, helper, dim=-1), dim=-1)
    w = torch.cross(d, u, dim=-1)
    return cos_t[:, None] * d + sin_t[:, None] * (torch.cos(phi)[:, None] * u + torch.sin(phi)[:, None] * w)


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
                 ibl_strength_range: tuple[float, float] = (.65, 1.0),
                 random_colors: dict | None = None,
                 key_light_half_extent: tuple[float, float, float] | None = None,
                 fill_light_cone_deg: float | None = None,
                 light_temperature_range: tuple[float, float] | None = None, light_tint_delta: float = .04,
                 key_strength_range: tuple[float, float] | None = None,
                 fill_strength_range: tuple[float, float] | None = None,
                 camera_exposure_ranges: dict | None = None,
                 white_balance_rb_range: tuple[float, float] | None = None,
                 camera_white_balance_rb_ranges: dict | None = None,
                 key_light_radius_range: tuple[float, float] | None = None) -> None:
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
            self.cameras = {name: sensor for name, sensor in env.scene.sensors.items()
                            if isinstance(getattr(sensor, '_renderer', None), UWLabNewtonWarpRenderer)}
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
                self.bindings.append((name, writer, spec, base))
            self.texture_ids = {id(r): r.matched_texture_ids for r in self.renderers}
            assert all(len(bank) == 3 for bank in self.texture_ids.values()), 'Preload the three matched textures'
        n = len(ids)
        random_colors = random_colors or {}
        u = lambda lo_hi: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(n, device=env.device)
        linked = {}  # one colour draw per link name (e.g. the three curtains)
        for name, writer, spec, base in self.bindings:
            shapes = writer._shapes[ids].reshape(-1)
            # Base is sRGB from Newton; apply bounded additive jitter in linear space.
            b = base[ids]
            linear = torch.where(b <= .04045, b / 12.92, ((b + .055) / 1.055).pow(2.4))
            jitter = (torch.rand(n, 1, 3, device=env.device) * 2 - 1) * spec.get('color_delta', color_delta)
            linear = linear + jitter
            rc = random_colors.get(name)
            if rc is not None:
                key = rc.get('link', name)
                if key not in linked:
                    rgb = _hsv_to_rgb(torch.rand(n, device=env.device), u(rc['saturation']), u(rc['value']))
                    rgb = torch.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055).pow(2.4))
                    linked[key] = (rgb, torch.rand(n, device=env.device) < rc['probability'],
                                   torch.rand(n, device=env.device) < texture_probability, torch.randint(0, 3, (n,), device=env.device))
                rgb, pick, _, _ = linked[key]
                linear = torch.where(pick[:, None, None], rgb[:, None, :].expand_as(linear), linear)
            wp.to_torch(model.shape_color)[shapes] = _linear_to_srgb(linear).reshape(-1, 3)
            rough = (spec.get('roughness', .65) + (torch.rand(n, device=env.device)*2-1)*material_delta).clamp(.05, 1)
            metal = (spec.get('metallic', 0) + (torch.rand(n, device=env.device)*2-1)*material_delta*.5).clamp(0, 1)
            rough = rough.repeat_interleave(writer.count).contiguous()
            metal = metal.repeat_interleave(writer.count).contiguous()
            shape_ids = shapes.to(torch.int32).contiguous()
            textured = torch.rand(n, device=env.device) < texture_probability
            any_texture = rc is not None and rc.get('any_texture', False)
            if any_texture:
                _, _, textured, family_draw = linked[rc.get('link', name)]
                choice = torch.where(textured, family_draw, -1).long()
            else:
                choice = torch.where(textured, torch.full((n,), spec.get('texture', -1), device=env.device), -1).long()
            default_family = spec.get('surface', ('smooth', 'cloth', 'printed')[spec['texture']] if 'texture' in spec else 'metal')
            env_family = [default_family] if not any_texture else [default_family, 'smooth', 'cloth', 'printed']
            for renderer in self.renderers:
                context = renderer._context
                for k, family in enumerate(env_family):
                    sel = torch.ones(n, dtype=torch.bool, device=env.device) if len(env_family) == 1 else \
                        (choice < 0) if k == 0 else (choice == k - 1)
                    if not bool(sel.any()):
                        continue
                    sel_s = sel.repeat_interleave(writer.count)
                    maps = renderer.matched_surface_ids.get(family, {})
                    wp.launch(_materials, dim=int(sel_s.sum()), inputs=[wp.from_torch(shape_ids[sel_s].contiguous()),
                              wp.from_torch(rough[sel_s].contiguous()), wp.from_torch(metal[sel_s].contiguous()),
                              maps.get('normal', -1), maps.get('roughness', -1), context.materials], device=model.device)
                bank = torch.tensor(self.texture_ids[id(renderer)], device=env.device)
                tex = torch.where(choice >= 0, bank[choice.clamp_min(0)], -1)
                wp.to_torch(context.shape_texture_ids)[shapes] = tex.repeat_interleave(writer.count).to(torch.int32)
        positions = torch.zeros(n, 3, 3, device=env.device)
        positions[:, 0] = torch.tensor([.7, -.4, 1.0], device=env.device)
        if key_light_half_extent is None:
            positions[:, 0] += (torch.rand(n, 3, device=env.device)*2-1)*light_position_delta
        else:
            positions[:, 0] += (torch.rand(n, 3, device=env.device)*2-1)*torch.tensor(key_light_half_extent, device=env.device)
        positions += env.scene.env_origins[ids, None, :]
        orientations = torch.tensor([[-.35, .45, -.82], [.6, -.2, -.4], [0, 0, -1]], device=env.device).expand(n, -1, -1).clone()
        if fill_light_cone_deg is None:
            orientations[:, 1] += (torch.rand(n, 3, device=env.device)*2-1)*.1
        else:
            orientations[:, 1] = _rotate_within_cone(torch.nn.functional.normalize(orientations[:, 1], dim=-1), fill_light_cone_deg)
        strengths = torch.tensor([2.5, .8, 0], device=env.device).expand(n, -1).clone()
        lo, hi = light_strength_range
        scale = lo + (hi-lo)*torch.rand(n, 3, device=env.device)
        for slot, rng in ((0, key_strength_range), (1, fill_strength_range)):
            if rng is not None:
                scale[:, slot] = rng[0] + (rng[1] - rng[0]) * torch.rand(n, device=env.device)
        strengths *= scale
        colors = torch.tensor([[1, .91, .95], [.82, .9, 1], [1, 1, 1]], device=env.device).expand(n, -1, -1).clone()
        if light_temperature_range is not None:
            lo_k, hi_k = light_temperature_range
            colors[:, :2] = _kelvin_to_rgb(lo_k + (hi_k - lo_k) * torch.rand(n, 2, device=env.device))
        colors = colors + (torch.rand_like(colors)*2-1)*light_tint_delta
        colors = colors.clamp(0, 1) if light_temperature_range is None else colors.clamp_min(0)
        radii = torch.zeros(n, 3, device=env.device)
        radii[:, 0] = .025 if key_light_radius_range is None else \
            key_light_radius_range[0] + (key_light_radius_range[1] - key_light_radius_range[0]) * torch.rand(n, device=env.device)
        for renderer in self.renderers:
            renderer.set_lights(positions, orientations, [[True, True, False]]*n, env_ids=ids,
                                colors=colors, intensities=strengths, types=[[2, 1, 1]]*n,
                                cast_shadow=[[True, False, False]]*n, radii=radii)

            # Sample camera and IBL parameters only for worlds participating in this reset.
            def sample(bounds: tuple[float, float]) -> torch.Tensor:
                return bounds[0] + (bounds[1]-bounds[0])*torch.rand(n, device=env.device)
            def white_balance(cam_name: str | None = None) -> torch.Tensor:
                wb = 1 + (torch.rand(n, 3, device=env.device)*2-1)*white_balance_delta
                rb_range = (camera_white_balance_rb_ranges or {}).get(cam_name, white_balance_rb_range)
                if rb_range is not None:
                    lo_rb, hi_rb = (torch.tensor(v, device=env.device).log() for v in rb_range)
                    ratio = (lo_rb + (hi_rb - lo_rb) * torch.rand(n, device=env.device)).exp()
                    wb[:, 0] *= ratio.sqrt()
                    wb[:, 2] /= ratio.sqrt()
                return wb
            if camera_exposure_ranges is None and white_balance_rb_range is None and camera_white_balance_rb_ranges is None:
                renderer.set_exposure(ids, sample(exposure_range))
                wp.to_torch(renderer._white_balance)[ids] = white_balance()
            else:  # each camera its own exposure / white balance (per-camera arrays on its render data)
                for name, sensor in self.cameras.items():
                    if sensor._renderer is not renderer:
                        continue
                    rd = sensor._render_data
                    if not hasattr(rd, 'uw_exposure'):
                        rd.uw_exposure = wp.clone(renderer._exposure)
                        rd.uw_white_balance = wp.clone(renderer._white_balance)
                    wp.to_torch(rd.uw_exposure)[ids] = sample((camera_exposure_ranges or {}).get(name, exposure_range))
                    wp.to_torch(rd.uw_white_balance)[ids] = white_balance(name)
            response = torch.stack((sample(gamma_range), sample(read_noise_range), sample(shot_noise_range),
                                    torch.randint(1, 100000, (n,), device=env.device).float()), dim=-1)
            wp.to_torch(renderer._camera_response)[ids] = response
            context = renderer._context
            wp.to_torch(context.environment_ids)[ids] = torch.randint(
                len(renderer.cfg.environment_paths), (n,), device=env.device, dtype=torch.int32)
            wp.to_torch(context.environment_intensity)[ids] = sample(ibl_strength_range)
            wp.to_torch(context.environment_rotation)[ids] = (torch.rand(n, device=env.device)*2-1)*.035
