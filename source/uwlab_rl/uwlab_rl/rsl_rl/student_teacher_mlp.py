# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MLP student + JIT teacher for rsl_rl ``DistillationRunner`` DAgger.

Sanity-check sibling of ``StudentTeacherVision``: the student reads the same
1D obs groups as the teacher (typically the full 215d state). No CNN, no image
path. Intended to isolate whether the distillation loop / teacher / action
plumbing is correct independent of any vision representation issue.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import MLP, EmpiricalNormalization


class StudentTeacherMLP(StudentTeacher):
    """MLP student + JIT teacher (state→state DAgger)."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_jit_path: str,
        student_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        student_obs_normalization: bool = True,
        **kwargs,
    ) -> None:
        # Bypass StudentTeacher.__init__ (it builds an MLP teacher from config;
        # we want a frozen JIT teacher instead).
        nn.Module.__init__(self)
        Normal.set_default_validate_args(False)

        self.obs_groups = obs_groups
        self.num_actions = num_actions

        student_obs = torch.cat([obs[g] for g in obs_groups["policy"]], dim=-1)
        assert student_obs.ndim == 2, f"policy groups must be 1D; got shape {student_obs.shape}"
        num_student_obs = student_obs.shape[-1]

        self.student = MLP(num_student_obs, num_actions, list(student_hidden_dims), activation)
        print(f"StudentTeacherMLP student (obs={num_student_obs}, act={num_actions}): {self.student}")

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = nn.Identity()

        # JIT teacher — normalizer baked in, forward returns action mean.
        self.teacher = torch.jit.load(teacher_jit_path)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_obs_normalization = False
        self.teacher_obs_normalizer = nn.Identity()
        self.loaded_teacher = True

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"noise_std_type must be 'scalar' or 'log'; got {self.noise_std_type}")

        self.distribution = None

    def _encode_student(self, obs: TensorDict) -> torch.Tensor:
        x = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        return self.student_obs_normalizer(x)

    def _encode_teacher(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["teacher"]], dim=-1)

    def _update_distribution(self, x: torch.Tensor) -> None:
        mean = self.student(x)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        x = self._encode_student(obs)
        self._update_distribution(x)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        x = self._encode_student(obs)
        return self.student(x)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self._encode_teacher(obs)
        with torch.no_grad():
            return self.teacher(teacher_obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_student(obs)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_teacher(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            x = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
            self.student_obs_normalizer.update(x)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.teacher.eval()
        return self

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("teacher.")}
        nn.Module.load_state_dict(self, filtered, strict=False)
        self.loaded_teacher = True
        self.teacher.eval()
        return True
