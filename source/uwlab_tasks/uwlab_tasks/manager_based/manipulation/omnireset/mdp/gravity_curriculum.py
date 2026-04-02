# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Gravity curriculum for OmniReset: ramps gravity from 0 to full based on success rate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GravityScheduler(ManagerTermBase):
    """Adaptive gravity scheduler based on task success rate.

    Tracks per-environment difficulty. Promotes on success, demotes on failure.
    Exposes `difficulty_frac` (0→1) used by `gravity_interpolate_fn` to ramp gravity.
    Uses the success signal from ProgressContext (via string eval).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.current_difficulties = torch.zeros(env.num_envs, device=env.device)
        self.difficulty_frac = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        success_str: str = "env.reward_manager.get_term_cfg('progress_context').func.success",
        max_difficulty: int = 10,
    ):
        if env_ids is None or len(env_ids) == 0:
            return self.difficulty_frac

        success = eval(success_str)  # (num_envs,) bool tensor
        success_mask = success[env_ids]

        self.current_difficulties[env_ids] = torch.where(
            success_mask,
            self.current_difficulties[env_ids] + 1,
            self.current_difficulties[env_ids] - 1,
        ).clamp(min=0, max=max_difficulty)

        self.difficulty_frac = self.current_difficulties.mean().item() / max(max_difficulty, 1)
        return self.difficulty_frac


def gravity_interpolate_fn(env, env_id, data, initial_gravity, final_gravity, scheduler_term_str):
    """Interpolate gravity based on GravityScheduler difficulty fraction.

    Args:
        initial_gravity: tuple of (mean, std) for zero-g, e.g. ((0,0,0), (0,0,0))
        final_gravity: tuple of (mean, std) for full gravity, e.g. ((0,0,-9.81), (0,0,-9.81))
        scheduler_term_str: name of the GravityScheduler curriculum term
    """
    from isaaclab.envs.mdp import modify_term_cfg

    scheduler: GravityScheduler = getattr(env.curriculum_manager.cfg, scheduler_term_str).func
    frac = scheduler.difficulty_frac

    if frac < 0.05:
        return modify_term_cfg.NO_CHANGE

    # Linear interpolation between initial and final gravity
    result = []
    for ig, fg in zip(initial_gravity, final_gravity):
        interpolated = tuple(ig_i + frac * (fg_i - ig_i) for ig_i, fg_i in zip(ig, fg))
        result.append(interpolated)
    return tuple(result)
