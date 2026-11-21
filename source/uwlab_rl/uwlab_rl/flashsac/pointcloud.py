# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PointNet + state encoder for :class:`uwlab_rl.flashsac.FlashSACRGB` (``encoder_type="pointnet"``).

Same per-point MLP / max-pool / LayerNorm set encoder as ``uwlab_rl.rsl_rl.point_net_model.PointNetEncoder``, with
unit-norm linears (FlashSAC weight normalization) and the RGB encoder's unit-linear projections of the pooled feature
and the state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .rgb import UnitLinear


class PointCloudStateEncoder(nn.Module):
    """points (B, N, C) -> per-point [unit linear + LayerNorm + ReLU]* -> unit linear -> max over N -> LayerNorm ->
    unit linear to ``embed_dim``; state -> unit linear to ``state_embed_dim``; output = concat."""

    def __init__(
        self,
        points_shape: tuple[int, int],
        state_dim: int,
        embed_dim: int,
        state_embed_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        cin = int(points_shape[-1])
        for h in hidden_dims[:-1]:
            layers += [UnitLinear(cin, h), nn.LayerNorm(h), nn.ReLU()]
            cin = h
        layers.append(UnitLinear(cin, hidden_dims[-1]))
        self.point_mlp = nn.Sequential(*layers)
        self.pool_norm = nn.LayerNorm(hidden_dims[-1])
        self.points_fc = UnitLinear(hidden_dims[-1], embed_dim, bias=False)
        self.state_fc = UnitLinear(state_dim, state_embed_dim, bias=False)
        self.out_dim = embed_dim + state_embed_dim
        self.normalize_parameters()

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

    def forward(self, points: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_norm(self.point_mlp(points).amax(dim=1))
        return torch.cat([self.points_fc(pooled), self.state_fc(state)], dim=-1)
