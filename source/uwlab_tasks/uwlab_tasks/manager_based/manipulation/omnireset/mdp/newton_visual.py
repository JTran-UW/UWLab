# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_CONTRAST_TRIES = 20
_ENV_RE = re.compile(r"/env_(\d+)(?:/|$)")


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    # same transfer as isaaclab_newton.renderers.visual_material._linear_to_srgb
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.pow(1.0 / 2.4) - 0.055)


class _ShapeColorWriter:
    """Writes ``model.shape_color`` rows of the shapes under one scene asset, addressed by ``shape_label``.

    Not ``NewtonManager.create_visual_shape_color_writer``: on this scene its per-world shape ids land on other
    assets' shapes (receptive/pad colours showed up on the hole and the UR5 plate), and it cannot reach the static
    curtains (world-attached, body -1). Shape labels are per-env prim paths, so they address every case.
    """

    def __init__(self, model, prim_path: str, num_envs: int, body_names, shape_filter, per_body: bool):
        import warp as wp

        root = re.compile(prim_path + "(?:/.*)?")
        filt = re.compile(shape_filter) if shape_filter else None
        rows_of = {n: i for i, n in enumerate(body_names)} if body_names is not None else None
        shape_body = model.shape_body.numpy()
        body_label = list(model.body_label)
        per_env: list[list[tuple[str, int, int]]] = [[] for _ in range(num_envs)]
        for idx, label in enumerate(model.shape_label):
            if not label or not root.fullmatch(label) or (filt is not None and not filt.search(label)):
                continue
            m = _ENV_RE.search(label)
            if m is None or int(m.group(1)) >= num_envs:
                continue
            row = 0
            if rows_of is not None:
                b = int(shape_body[idx])
                body = body_label[b].rsplit("/", 1)[-1] if b >= 0 else None
                if body not in rows_of:
                    continue
                row = rows_of[body] if per_body else 0
            per_env[int(m.group(1))].append((label.split(m.group(0), 1)[-1], idx, row))
        self.count = len(per_env[0])
        assert all(len(s) == self.count for s in per_env), f"uneven shape count per env under {prim_path}"
        for s in per_env:
            s.sort()
        self.device = torch.device(wp.device_to_torch(model.device))
        self._shapes = torch.tensor([[i for _, i, _ in s] for s in per_env], dtype=torch.long, device=self.device)
        self._rows = torch.tensor([r for _, _, r in per_env[0]], dtype=torch.long, device=self.device)
        self.labels = [label for label, _, _ in per_env[0]]
        self.model = None
        self.rebind(model)

    def rebind(self, model) -> None:
        import warp as wp

        self.model = model
        self._shape_color = wp.to_torch(model.shape_color)

    def __call__(self, colors: torch.Tensor, env_ids: torch.Tensor) -> None:
        """``colors``: (len(env_ids), num_rows, 3), linear RGB."""
        if self.count == 0 or env_ids.numel() == 0:
            return
        shapes = self._shapes[env_ids.long()].reshape(-1)
        self._shape_color[shapes] = _linear_to_srgb(colors[:, self._rows]).reshape(-1, 3)


class RandomizeNewtonSceneColors(ManagerTermBase):
    """Per-env flat colour DR written straight into the Newton model (``model.shape_color``) on reset.

    The Newton-Warp renderer reads USD colours once at model build, so the replicator material/texture/HDRI terms do
    nothing there. ``groups``: name -> dict(asset_cfg, color_range=((lo,lo,lo),(hi,hi,hi)), per_body=False,
    shape_filter=None); ``asset_cfg.body_names`` (regex) restricts to those bodies, ``shape_filter`` (regex searched in
    the shape prim path) to matching shapes. The table colour is rejection-sampled to stay >= ``min_table_contrast``
    (RGB L2) away from both the insertive and the receptive colour.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from isaaclab_newton.physics.newton_manager import NewtonManager

        self._manager = NewtonManager
        model = NewtonManager.get_model()
        self._groups: dict[str, dict] = {}
        for name, spec in cfg.params["groups"].items():
            asset_cfg = spec["asset_cfg"]
            if not isinstance(asset_cfg, SceneEntityCfg):
                asset_cfg = SceneEntityCfg(**asset_cfg)
            asset = env.scene[asset_cfg.name]
            lo, hi = (torch.tensor(v, dtype=torch.float32, device=env.device) for v in spec["color_range"])
            per_body = bool(spec.get("per_body", False))
            body_names = None
            # find_bodies, not asset_cfg.resolve: a resolved cfg rejects its own regex body_names on re-resolve
            if asset_cfg.body_names is not None:
                body_names = tuple(asset.find_bodies(asset_cfg.body_names, preserve_order=True)[1])
            elif per_body:
                body_names = tuple(asset.body_names)
            # static AssetBaseCfg prims live in scene extras (no .cfg); read the path off the scene cfg
            prim_path = getattr(env.scene.cfg, asset_cfg.name).prim_path
            prim_path = prim_path.replace("{ENV_REGEX_NS}", env.scene.env_regex_ns)
            writer = _ShapeColorWriter(model, prim_path, env.num_envs, body_names, spec.get("shape_filter"), per_body)
            assert writer.count > 0, f"group {name}: no Newton shapes found for asset {asset_cfg.name}"
            num_rows = len(body_names) if per_body else 1
            print(f"[newton colours] {name}: {writer.count} shapes, {num_rows} colour(s)/env", flush=True)
            self._groups[name] = dict(writer=writer, lo=lo, hi=hi, per_body=per_body, num_rows=num_rows)
        self._order = [n for n in ("insertive_object", "receptive_object") if n in self._groups]
        self._order += [n for n in self._groups if n not in self._order]
        self.last_colors: dict[str, torch.Tensor] = {
            n: torch.zeros((env.num_envs, g["num_rows"], 3) if g["per_body"] else (env.num_envs, 3), device=env.device)
            for n, g in self._groups.items()
        }

    def _sample(self, g: dict, *shape: int) -> torch.Tensor:
        return g["lo"] + (g["hi"] - g["lo"]) * torch.rand((*shape, 3), device=self._env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        groups: dict,
        min_table_contrast: float = 0.35,
    ) -> None:
        del groups
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = env_ids.to(env.device)
        n = env_ids.numel()
        if n == 0:
            return
        model = self._manager.get_model()
        sampled: dict[str, torch.Tensor] = {}
        for name in self._order:
            g = self._groups[name]
            refs = [sampled[r] for r in ("insertive_object", "receptive_object") if r in sampled]
            if name == "table" and not g["per_body"] and refs:
                cand = self._sample(g, n, _CONTRAST_TRIES)
                dist = torch.full(cand.shape[:2], float("inf"), device=env.device)
                for ref in refs:
                    ref = ref if ref.dim() == 2 else ref.mean(1)
                    dist = torch.minimum(dist, (cand - ref.unsqueeze(1)).norm(dim=-1))
                ok = dist >= min_table_contrast
                pick = torch.where(ok.any(1), ok.float().argmax(1), dist.argmax(1))
                colors = cand[torch.arange(n, device=env.device), pick]
            else:
                colors = self._sample(g, n, g["num_rows"]) if g["per_body"] else self._sample(g, n)
            sampled[name] = colors
            self.last_colors[name][env_ids] = colors

            writer = g["writer"]
            if writer.model is not model:
                writer.rebind(model)
            rows = colors if g["per_body"] else colors.unsqueeze(1)
            writer(rows.to(writer.device), env_ids.to(writer.device))
