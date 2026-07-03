# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical PointNet architecture — single source of truth.

Both the offline PyTorch-Lightning BC trainer and the online rsl_rl DAgger
student build their networks from these classes, so iterating on the
architecture here changes both at once and keeps BC-init -> DAgger state_dicts
compatible.

Import sites:

* ``scripts/imitation_learning/point_cloud/point_net.py`` and
  ``residual_point_net.py`` are thin shims re-exporting these names (so
  ``from point_net import PointNet`` / ``from residual_point_net import
  ResidualPointNet`` keep working for the BC pipeline).
* ``uwlab_rl.rsl_rl.student_teacher_pointcloud.StudentTeacherPointCloud`` wraps
  one of these as its trainable ``student`` module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP


class PointNet(nn.Module):
    def __init__(self, encoder_hidden_dims, action_hidden_dims, proprio_dim, action_dim, predict_std, point_dim=3):
        super().__init__()
        self.predict_std = predict_std
        self.point_dim = point_dim  # 3 = xyz; 4 = xyz + per-point segmentation label
        self.pooled_dim = encoder_hidden_dims[-1]  # width of the pooled global feature (aux-probe input)
        self.set_encoder = MLP(
            in_channels=point_dim,
            hidden_channels=encoder_hidden_dims,
            norm_layer=nn.LayerNorm,
        )
        self.point_norm = torch.nn.LayerNorm(encoder_hidden_dims[-1])
        self.proprio_encoder = torch.nn.Sequential(torch.nn.Linear(proprio_dim, encoder_hidden_dims[-1]), torch.nn.LayerNorm(encoder_hidden_dims[-1]))
        out_dim = action_dim * 2 if predict_std else action_dim
        self.action_head = MLP(
            in_channels=encoder_hidden_dims[-1]*2,
            hidden_channels=[*action_hidden_dims, out_dim],
        )

    def forward(self, x, proprio, return_pooled=False):  # x: (B, N, point_dim)
        assert x.ndim == 3 and x.shape[-1] == self.point_dim
        assert proprio.ndim == 2
        feats = self.set_encoder(x)
        pooled = self.point_norm(feats.amax(dim=1))      # (B, pooled_dim) global PC feature
        proprio_feats = self.proprio_encoder(proprio)
        feats = torch.cat([pooled, proprio_feats], dim=-1)
        out = self.action_head(feats)
        if self.predict_std:
            out = out.chunk(2, dim=-1)                    # (mean, log_std)
        # return_pooled exposes the pooled feature for an auxiliary probe (e.g. object-pose head).
        return (out, pooled) if return_pooled else out

    def calculate_loss(self, points, proprio, targets):
        if self.predict_std:
            mean, log_std = self.forward(points, proprio)
            var = torch.exp(2 * log_std)      # std = exp(log_std) > 0, so var > 0
            return F.gaussian_nll_loss(mean, targets, var)
        return F.mse_loss(self.forward(points, proprio), targets)


class ResidualMLP(nn.Module):
    """Drop-in replacement for ``torchvision.ops.MLP`` with per-layer residual connections.

    Each hidden layer is ``relu(norm(Linear(x)))``, with an identity skip added *only* when
    ``d_in == d_out`` (no projection shortcut -- width-changing layers stay plain, so no extra
    parameters are spent where a residual can't apply cleanly). Operates on the last dim, so it
    works on both per-point ``(B, N, C)`` and pooled ``(B, C)`` tensors, exactly like the MLPs it
    replaces in :class:`PointNet`.

    The final entry of ``hidden_channels`` is treated as the output projection: a plain ``Linear``
    with no norm / activation / residual (matching how ``torchvision.ops.MLP`` ends and how the
    PointNet action head emits raw action logits).
    """

    def __init__(self, in_channels, hidden_channels, norm_layer=nn.LayerNorm, activation=nn.ReLU):
        super().__init__()
        dims = [in_channels, *hidden_channels]
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.has_skip = []  # identity residual only where input/output widths match
        self.n_hidden = len(hidden_channels) - 1  # all but the output projection get norm + activation
        for i in range(len(hidden_channels)):
            d_in, d_out = dims[i], dims[i + 1]
            self.linears.append(nn.Linear(d_in, d_out))
            if i < self.n_hidden:
                self.norms.append(norm_layer(d_out))
                self.has_skip.append(d_in == d_out)
        self.act = activation()

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            if i < self.n_hidden:
                h = self.act(self.norms[i](linear(x)))
                x = h + x if self.has_skip[i] else h
            else:
                x = linear(x)  # output projection: raw, no residual
        return x


class ResidualPointNet(PointNet):
    """:class:`PointNet` with residual connections in the set encoder and action head.

    Identical to :class:`PointNet` (same forward, pooling, proprio fusion, loss) -- only the two deep
    MLPs are swapped for :class:`ResidualMLP`. Motivation: as the encoder/head grow deep, residual
    connections ease optimization. ``point_norm`` and ``proprio_encoder`` are shallow and left as-is.
    """

    def __init__(self, encoder_hidden_dims, action_hidden_dims, proprio_dim, action_dim, predict_std, point_dim=3):
        super().__init__(encoder_hidden_dims, action_hidden_dims, proprio_dim, action_dim, predict_std, point_dim)
        self.set_encoder = ResidualMLP(in_channels=point_dim, hidden_channels=encoder_hidden_dims)
        out_dim = action_dim * 2 if predict_std else action_dim
        self.action_head = ResidualMLP(in_channels=encoder_hidden_dims[-1] * 2, hidden_channels=[*action_hidden_dims, out_dim])
