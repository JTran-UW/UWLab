# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl model with a PointNet set encoder for point-cloud observation groups.

Point-cloud groups have shape (B, N, C); each is encoded by the PointNet set encoder from the UWLab-patrick-private
``uwlab_rl.networks.point_net.PointNet`` (shared per-point MLP with LayerNorm, max-pool over points, LayerNorm), which
is permutation invariant. 1D groups are normalized as in ``MLPModel`` and concatenated with the pooled features; the
MLP head and output distribution are rsl_rl's. Use as an actor/critic ``class_name``:
``uwlab_rl.rsl_rl.point_net_model:PointNetModel``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchvision.ops import MLP as TorchvisionMLP

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState


class PointNetEncoder(nn.Module):
    """PointNet set encoder: per-point MLP (LayerNorm) -> max over points -> LayerNorm. Input (B, N, C), output (B, D)."""

    def __init__(self, point_dim: int, hidden_dims: list[int] | tuple[int, ...]):
        super().__init__()
        self.set_encoder = TorchvisionMLP(in_channels=point_dim, hidden_channels=list(hidden_dims), norm_layer=nn.LayerNorm)
        self.point_norm = nn.LayerNorm(hidden_dims[-1])
        self.output_dim = int(hidden_dims[-1])

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.point_norm(self.set_encoder(points).amax(dim=1))


class PointNetModel(MLPModel):
    """``MLPModel`` whose 3D (B, N, C) observation groups are encoded by a :class:`PointNetEncoder`."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        point_encoder_cfg: dict | None = None,
    ) -> None:
        enc_cfg = dict(point_encoder_cfg or {})
        self.obs_groups_pc = [g for g in obs_groups[obs_set] if obs[g].ndim == 3]
        if not self.obs_groups_pc:
            raise ValueError(f"No point-cloud (B, N, C) groups in '{obs_set}'; use MLPModel instead.")
        encoders = {g: PointNetEncoder(int(obs[g].shape[-1]), enc_cfg.get("hidden_dims", (64, 128, 256))) for g in self.obs_groups_pc}
        self.pc_latent_dim = sum(e.output_dim for e in encoders.values())
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)
        self.point_encoders = nn.ModuleDict(encoders)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent_pc = torch.cat([self.point_encoders[g](obs[g]) for g in self.obs_groups_pc], dim=-1)
        if not self.obs_groups:
            return latent_pc
        return torch.cat([super().get_latent(obs), latent_pc], dim=-1)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("PointNetModel has no JIT export yet.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        raise NotImplementedError("PointNetModel has no ONNX export yet.")

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        groups_1d = [g for g in obs_groups[obs_set] if obs[g].ndim == 2]
        for g in obs_groups[obs_set]:
            if obs[g].ndim not in (2, 3):
                raise ValueError(f"Unsupported observation shape for '{g}': {tuple(obs[g].shape)}")
        return groups_1d, sum(int(obs[g].shape[-1]) for g in groups_1d)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.pc_latent_dim
