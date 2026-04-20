#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a state-expert training checkpoint (``model_state_dict`` from rsl_rl PPO) into a
JIT-scripted teacher suitable for :class:`StudentTeacherVision`.

Two modes:

* Default (``--no-std``): ``forward(obs) -> mean``. Rebuilds the actor MLP and
  the empirical obs normalizer from the state dict; the output JIT's
  ``forward(obs) -> mean`` matches the exported-policy contract used by our
  MSE-on-mean DAgger.

* ``--std``: ``forward(obs) -> (mean, std)``. Additionally reconstructs the
  gSDE std head (``variance = features² @ exp(log_std)²`` per DEXTRAH-style
  inverse-variance-weighted distillation). ``features`` is the last hidden
  layer before the final Linear, ``log_std`` is the ``(hidden_dim, num_actions)``
  gSDE matrix from the training ckpt.

Framework-free (no sim, no isaaclab import).

Usage::

    # mean-only (legacy)
    python scripts_v2/tools/convert_state_expert_to_jit.py \
        --input  peg_state_rl_expert_seed42.pt \
        --output teachers/peg_teacher_mean.pt

    # mean + std (for weighted-L2 DAgger)
    python scripts_v2/tools/convert_state_expert_to_jit.py \
        --input  peg_state_rl_expert_seed42.pt \
        --output teachers/peg_teacher_full.pt \
        --std
"""

from __future__ import annotations

import argparse
import re

import torch
import torch.nn as nn


def build_mlp(state_dict: dict, prefix: str, activation: type[nn.Module]) -> nn.Sequential:
    """Reconstruct a plain Linear+activation Sequential from an ``rsl_rl.networks.MLP`` state dict.

    rsl_rl's MLP stores layers as indices ``0, 2, 4, ...`` (Linear) with activation modules at
    the odd indices. We scan for weights, build ``Linear(in, out)`` in order, and interleave
    the requested activation between them. No activation after the final layer.
    """
    idx_re = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    indices = sorted(int(m.group(1)) for k in state_dict if (m := idx_re.match(k)))
    assert indices, f"no Linear weights found under prefix '{prefix}'"

    layers: list[nn.Module] = []
    for i, idx in enumerate(indices):
        w = state_dict[f"{prefix}.{idx}.weight"]
        b = state_dict[f"{prefix}.{idx}.bias"]
        out_dim, in_dim = w.shape
        lin = nn.Linear(in_dim, out_dim)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(indices) - 1:
            layers.append(activation())

    return nn.Sequential(*layers)


class MeanOnlyTeacher(nn.Module):
    """``forward(obs) -> mean``: normalize then apply the actor MLP. No gSDE std."""

    def __init__(self, in_dim: int, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-2):
        super().__init__()
        assert mean.shape == (1, in_dim) and std.shape == (1, in_dim)
        self.register_buffer("_mean", mean.clone())
        self.register_buffer("_std", std.clone())
        self.eps = eps
        self.actor: nn.Sequential = nn.Identity()  # filled in by caller

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self._mean) / (self._std + self.eps)
        return self.actor(x)


class MeanStdTeacher(nn.Module):
    """``forward(obs) -> (mean, std)`` with gSDE state-dependent std.

    Splits the actor into ``actor_features`` (all layers except the last
    Linear) and ``actor_final`` (the last Linear). Variance is
    ``features² @ exp(log_std)²`` per UWLab's gSDE exporter
    (``uwlab_rl/rsl_rl/exporter.py:114-145``); std adds an epsilon for
    numerical stability.
    """

    def __init__(
        self,
        in_dim: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        log_std: torch.Tensor,
        norm_eps: float = 1e-2,
        var_eps: float = 1e-6,
    ):
        super().__init__()
        assert mean.shape == (1, in_dim) and std.shape == (1, in_dim)
        assert log_std.ndim == 2, f"expected log_std (hidden_dim, num_actions); got {log_std.shape}"
        self.register_buffer("_mean", mean.clone())
        self.register_buffer("_std", std.clone())
        # gSDE parameter: same mapping used at training time (see exporter.py).
        self.register_buffer("_log_std", log_std.clone())
        self.norm_eps = norm_eps
        self.var_eps = var_eps
        self.actor_features: nn.Sequential = nn.Identity()  # filled in by caller
        self.actor_final: nn.Linear = nn.Linear(1, 1)  # placeholder; replaced by caller

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (obs - self._mean) / (self._std + self.norm_eps)
        features = self.actor_features(x)
        mean = self.actor_final(features)
        variance = torch.mm(features**2, torch.exp(self._log_std) ** 2)
        std = torch.sqrt(variance + self.var_eps)
        return mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to training checkpoint (.pt)")
    p.add_argument("--output", required=True, help="Path to write JIT module to")
    p.add_argument(
        "--activation",
        default="elu",
        choices=["elu", "relu", "tanh", "gelu"],
        help="Actor activation (must match training config; Base_PPORunnerCfg defaults to elu)",
    )
    p.add_argument(
        "--std",
        action="store_true",
        help="Export (mean, std) using gSDE formula instead of mean-only.",
    )
    args = p.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    assert isinstance(sd, dict), f"unexpected checkpoint format: {type(ckpt)}"

    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[args.activation]

    # Obs normalizer: mean/var/std are registered as (1, in_dim) buffers.
    norm_mean = sd["actor_obs_normalizer._mean"]
    norm_std = sd["actor_obs_normalizer._std"]
    in_dim = norm_mean.shape[-1]

    if not args.std:
        actor = build_mlp(sd, "actor", act_cls)
        teacher = MeanOnlyTeacher(in_dim=in_dim, mean=norm_mean, std=norm_std)
        teacher.actor = actor
        teacher.eval()
        first_lin = next(m for m in actor.modules() if isinstance(m, nn.Linear))
        assert first_lin.in_features == in_dim, (
            f"normalizer in_dim={in_dim} disagrees with actor in_features={first_lin.in_features}"
        )
        with torch.no_grad():
            out = teacher(torch.randn(1, in_dim))
        print(f"Mean-only teacher in_dim={in_dim}, out_dim={tuple(out.shape)}")
    else:
        # Build the full actor, then split into features (all but last) + final Linear.
        full_actor = build_mlp(sd, "actor", act_cls)
        linear_indices = [i for i, m in enumerate(full_actor) if isinstance(m, nn.Linear)]
        assert len(linear_indices) >= 2, f"expected ≥2 Linear layers in actor; got {len(linear_indices)}"
        last_linear_idx = linear_indices[-1]
        actor_features = nn.Sequential(*list(full_actor.children())[:last_linear_idx])
        actor_final = full_actor[last_linear_idx]
        assert isinstance(actor_final, nn.Linear)

        log_std = sd["log_std"]
        hidden_dim, num_actions = log_std.shape
        assert actor_final.in_features == hidden_dim, (
            f"log_std hidden_dim={hidden_dim} but actor final Linear in_features={actor_final.in_features}"
        )
        assert actor_final.out_features == num_actions, (
            f"log_std num_actions={num_actions} but actor final Linear out_features={actor_final.out_features}"
        )

        teacher = MeanStdTeacher(in_dim=in_dim, mean=norm_mean, std=norm_std, log_std=log_std)
        teacher.actor_features = actor_features
        teacher.actor_final = actor_final
        teacher.eval()
        with torch.no_grad():
            m, s = teacher(torch.randn(1, in_dim))
        print(
            f"Mean+Std teacher in_dim={in_dim}, mean_shape={tuple(m.shape)}, std_shape={tuple(s.shape)}, "
            f"log_std_shape={tuple(log_std.shape)}"
        )
        # Sanity: std must be positive and finite.
        assert torch.all(s > 0) and torch.isfinite(s).all(), f"invalid std sample: {s}"

    scripted = torch.jit.script(teacher)
    scripted.save(args.output)
    print(f"Saved JIT teacher to: {args.output}")


if __name__ == "__main__":
    main()
