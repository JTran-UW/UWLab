# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation algorithm with β-annealed teacher action injection.

Standard ``rsl_rl.algorithms.Distillation`` runs pure β=0 DAgger: student drives
every env every step, teacher relabels student-visited states. For vision
students trained from scratch that wastes rollouts before the student can reach
states the teacher cares about. This subclass mixes teacher actions into the
rollout with probability ``beta``, linearly annealed 1 → 0 over
``beta_anneal_iters`` update steps. Loss remains MSE on mean.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation


class DistillationDAgger(Distillation):
    """DAgger with linear β annealing on teacher action injection."""

    def __init__(self, *args, beta_anneal_iters: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 0 => pure β=0 from the start (behaves like vanilla Distillation).
        self.beta_anneal_iters = int(beta_anneal_iters)

    @property
    def beta(self) -> float:
        if self.beta_anneal_iters <= 0:
            return 0.0
        return max(0.0, 1.0 - self.num_updates / self.beta_anneal_iters)

    def act(self, obs: TensorDict) -> torch.Tensor:
        student_action = self.policy.act(obs).detach()
        teacher_action = self.policy.evaluate(obs).detach()

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
        loss_dict["beta"] = self.beta
        return loss_dict
