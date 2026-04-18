# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DistillationRunner with a fixed per-env student/teacher pool split.

Assigns the first ``round(num_envs * student_fraction)`` envs to the student
pool and the rest to the teacher pool; the assignment is fixed for the entire
training run. The mask is:

* passed to :class:`DistillationDAgger` via ``student_mask`` so action routing
  honors the split, and
* stashed on ``env.unwrapped.pool_mask`` so the reset event
  (:class:`MultiResetManager`) can log ``Metrics/success_student_only`` and
  ``Metrics/success_teacher_only`` alongside the usual per-task success rates.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.runners import DistillationRunner
from rsl_rl.utils import resolve_obs_groups


class DistillationRunnerSplit(DistillationRunner):
    """DistillationRunner that fixes a per-env student/teacher pool split."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Strip split-only keys before the parent sees the cfg.
        self.student_fraction = float(train_cfg.pop("student_fraction", 0.5))
        if not 0.0 <= self.student_fraction <= 1.0:
            raise ValueError(f"student_fraction must be in [0, 1]; got {self.student_fraction}")

        # DistillationRunner's __init__ builds the algorithm. We need the mask ready
        # before _construct_algorithm is called so DistillationDAgger sees it. The
        # mask depends on env.num_envs which is available right away.
        num_envs = env.num_envs
        num_student = round(num_envs * self.student_fraction)
        mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        mask[:num_student] = True
        self.student_mask = mask
        self.num_student = num_student
        self.num_teacher = num_envs - num_student

        # Expose mask to the env for the reset event's per-pool success logging.
        # Event-side ops are on env.device, runner-side ops on `device`; push a
        # per-device copy to avoid cross-device indexing failures.
        env.unwrapped.pool_mask = mask.to(env.unwrapped.device)

        # Inject student_mask into the algorithm cfg so _construct_algorithm
        # passes it to DistillationDAgger.__init__.
        train_cfg["algorithm"] = dict(train_cfg["algorithm"])
        train_cfg["algorithm"]["student_mask"] = mask

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        print(
            f"[DistillationRunnerSplit] pool split: {num_student} student / "
            f"{self.num_teacher} teacher (fraction={self.student_fraction})"
        )
