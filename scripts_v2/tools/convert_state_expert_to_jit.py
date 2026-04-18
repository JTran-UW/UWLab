#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a state-expert training checkpoint (``model_state_dict`` from rsl_rl PPO) into a
JIT-scripted mean-only teacher suitable for :class:`StudentTeacherVision`.

Framework-free (no sim, no isaaclab import). Rebuilds the actor MLP and the
empirical obs normalizer from the state dict alone; the output JIT's
``forward(obs) -> mean`` matches the exported-policy contract used by our DAgger
loop (normalizer baked in, mean only — gSDE std is dropped).

Usage::

    python scripts_v2/tools/convert_state_expert_to_jit.py \
        --input  checkpoints/peg_teacher/peg_state_rl_expert_finetuned_seed42.pt \
        --output checkpoints/peg_teacher/peg_teacher_mean.pt
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
    args = p.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    assert isinstance(sd, dict), f"unexpected checkpoint format: {type(ckpt)}"

    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[args.activation]
    actor = build_mlp(sd, "actor", act_cls)

    # Obs normalizer: mean/var/std are registered as (1, in_dim) buffers.
    norm_mean = sd["actor_obs_normalizer._mean"]
    norm_std = sd["actor_obs_normalizer._std"]
    in_dim = norm_mean.shape[-1]

    teacher = MeanOnlyTeacher(in_dim=in_dim, mean=norm_mean, std=norm_std)
    teacher.actor = actor
    teacher.eval()

    # Sanity check: in_dim should match actor's first Linear.
    first_lin = next(m for m in actor.modules() if isinstance(m, nn.Linear))
    assert first_lin.in_features == in_dim, (
        f"normalizer in_dim={in_dim} disagrees with actor in_features={first_lin.in_features}"
    )
    # Sanity forward: one random sample runs to the end.
    with torch.no_grad():
        out = teacher(torch.randn(1, in_dim))
    print(f"Teacher in_dim={in_dim}, out_dim={tuple(out.shape)}, activation={args.activation}")

    scripted = torch.jit.script(teacher)
    scripted.save(args.output)
    print(f"Saved JIT teacher to: {args.output}")


if __name__ == "__main__":
    main()
