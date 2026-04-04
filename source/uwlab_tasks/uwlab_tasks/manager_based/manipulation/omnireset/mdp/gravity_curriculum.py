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


class GravityCurriculum(ManagerTermBase):
    """Curriculum that ramps gravity from zero-g to full gravity based on success rate.

    Tracks a global difficulty fraction (0→1). On each reset batch, increments
    difficulty for successful envs and decrements for failed ones. Sets the physics
    scene gravity to ``(0, 0, -9.81 * difficulty_frac)`` each time it's called.

    Returns the current difficulty fraction for wandb logging.
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
        full_gravity: float = -9.81,
    ) -> dict[str, float]:
        if env_ids is not None and len(env_ids) > 0:
            success = eval(success_str)  # (num_envs,) bool tensor
            success_mask = success[env_ids]

            self.current_difficulties[env_ids] = torch.where(
                success_mask,
                self.current_difficulties[env_ids] + 1,
                self.current_difficulties[env_ids] - 1,
            ).clamp(min=0, max=max_difficulty)

            self.difficulty_frac = self.current_difficulties.mean().item() / max(max_difficulty, 1)

        # Set gravity based on current difficulty
        import carb
        from isaaclab.sim import SimulationContext

        gravity = carb.Float3(0.0, 0.0, full_gravity * self.difficulty_frac)
        physics_sim_view = SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(gravity)

        return {"gravity_frac": self.difficulty_frac}
