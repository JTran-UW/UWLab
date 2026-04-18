# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation algorithm with two teacher-injection modes.

Two mutually exclusive mixing strategies on top of
``rsl_rl.algorithms.Distillation``:

1. β-annealed per-step coin flip (default): each env each step, teacher acts
   with probability ``beta = max(0, 1 - num_updates / beta_anneal_iters)``.
   Mixes teacher into rollouts early, anneals to pure student.

2. Fixed per-env pool split: caller provides ``student_mask`` of shape
   ``(num_envs,)``; student-pool envs always run student actions, teacher-pool
   envs always run teacher actions. Enables clean per-pool success logging.

Loss is always MSE-on-mean over every transition (both pools contribute data).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation


class DistillationDAgger(Distillation):
    """DAgger with either β-annealed action mixing or a fixed per-env pool split."""

    def __init__(
        self,
        *args,
        beta_anneal_iters: int = 0,
        student_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta_anneal_iters = int(beta_anneal_iters)
        if student_mask is not None:
            student_mask = student_mask.to(self.device).bool()
        self.student_mask = student_mask

    @property
    def beta(self) -> float:
        if self.beta_anneal_iters <= 0:
            return 0.0
        return max(0.0, 1.0 - self.num_updates / self.beta_anneal_iters)

    def act(self, obs: TensorDict) -> torch.Tensor:
        student_action = self.policy.act(obs).detach()
        teacher_action = self.policy.evaluate(obs).detach()

        if self.student_mask is not None:
            mask = self.student_mask.unsqueeze(-1)
            action = torch.where(mask, student_action, teacher_action)
        else:
            beta = self.beta
            if beta > 0.0:
                num_envs = student_action.shape[0]
                use_teacher = torch.rand(num_envs, device=student_action.device) < beta
                action = torch.where(use_teacher.unsqueeze(-1), teacher_action, student_action)
            else:
                action = student_action

        self.transition.actions = action
        self.transition.privileged_actions = teacher_action
        self.transition.observations = obs
        return action

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self.student_mask is None:
            loss_dict["beta"] = self.beta
        else:
            loss_dict["student_fraction"] = float(self.student_mask.float().mean().item())
        return loss_dict
