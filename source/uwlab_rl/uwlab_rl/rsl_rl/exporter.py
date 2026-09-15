# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JIT export of an rsl-rl >= 5.0 actor with its output distribution.

rsl-rl's own ``OnPolicyRunner.export_policy_to_jit`` exports the mean action only. Demo collection
(``scripts_v2/tools/collect_demos.py``) samples from the expert, so the exported module also needs
``compute_distribution(obs) -> (mean, std)``. The distribution's standard deviation is evaluated once at
export time through the distribution's own code path, so gSDE (state-dependent std), log-std and scalar-std
Gaussians all export exactly.
"""

from __future__ import annotations

import copy
import os
import torch
from torch import nn


def export_policy_as_jit(actor: nn.Module, path: str, filename: str = "policy.pt") -> None:
    """Export an rsl-rl ``MLPModel`` actor into a TorchScript file with ``forward`` and ``compute_distribution``.

    Args:
        actor: The actor model (``rsl_rl.models.MLPModel``), uncompiled.
        path: The directory to save into.
        filename: The file name. Defaults to "policy.pt".
    """
    os.makedirs(path, exist_ok=True)
    exporter = _TorchPolicyExporter(actor).to("cpu")  # device-neutral artifact; consumers move it as needed
    torch.jit.script(exporter).save(os.path.join(path, filename))


class _TorchPolicyExporter(nn.Module):
    """TorchScript-able snapshot of an ``MLPModel`` actor: normalizer, MLP and output distribution."""

    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.normalizer = copy.deepcopy(actor.obs_normalizer)
        # ``MLP`` is an ``nn.Sequential`` subclass whose ``__init__`` takes positional arguments,
        # so it cannot be sliced; rebuild the split around the last layer explicitly.
        layers = [copy.deepcopy(layer) for layer in actor.mlp]
        self.actor_features = nn.Sequential(*layers[:-1])
        self.actor_final = layers[-1]
        self.epsilon = 1e-6

        dist = actor.distribution
        with torch.no_grad():
            if dist is None:
                self.state_dependent = False
                self.register_buffer("std", torch.ones(1))
                self.register_buffer("std_matrix", torch.ones(1, 1))
            elif hasattr(dist, "_get_std") and getattr(dist, "requires_latent_sde", False):
                # gSDE: marginal std = sqrt(phi(s)^2 @ std_matrix^2), std_matrix is (latent_dim, num_actions)
                # after the distribution's own clamp / full_std / expln handling.
                self.state_dependent = True
                self.register_buffer("std", torch.ones(1))
                self.register_buffer("std_matrix", dist._get_std().detach().clone())
            else:
                # Gaussian: evaluate the clamped std once from the distribution's parameterization.
                self.state_dependent = False
                if getattr(dist, "std_type", "scalar") == "log":
                    std = torch.exp(dist.log_std_param.clamp(dist.log_std_range[0], dist.log_std_range[1]))
                else:
                    std = dist.std_param.clamp(dist.std_range[0], dist.std_range[1])
                self.register_buffer("std", std.detach().clone())
                self.register_buffer("std_matrix", torch.ones(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_final(self.actor_features(self.normalizer(x)))

    @torch.jit.export
    def compute_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.actor_features(self.normalizer(x))
        mean = self.actor_final(features)
        if self.state_dependent:
            variance = torch.mm(features**2, self.std_matrix**2)
            std = torch.sqrt(variance + self.epsilon)
        else:
            std = self.std.expand_as(mean)
        return mean, std
