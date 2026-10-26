# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Algorithm-independent env guards shared by the rsl_rl (PPO) and FlashSAC training scripts."""

from __future__ import annotations

import torch


def uses_newton_physics(env) -> bool:
    """Whether the env's resolved physics config is a Newton one (e.g. ``presets=newton_mjwarp``)."""
    physics = getattr(env.unwrapped.cfg.sim, "physics", None)
    return any(c.__name__ == "NewtonCfg" for c in type(physics).__mro__)


def guard_non_finite(wrapped) -> None:
    """Zero non-finite rewards/obs (rare Newton blow-ups) instead of letting rsl_rl's check_nan kill the run.

    ``wrapped`` is an rsl_rl-style vec env whose ``step`` returns ``(obs TensorDict, rewards, dones, extras)``; the env's
    blow-up termination resets the affected envs.
    """
    step = wrapped.step
    count = [0]

    def guarded_step(actions):
        obs, rewards, dones, extras = step(actions)
        bad = ~torch.isfinite(rewards)
        for key in obs.keys():
            if obs[key].is_floating_point():
                bad_obs = ~torch.isfinite(obs[key]).flatten(1).all(dim=1)
                if bad_obs.any():
                    bad |= bad_obs
                    obs[key] = torch.nan_to_num(obs[key], nan=0.0, posinf=0.0, neginf=0.0)
        if bad.any():
            count[0] += int(bad.sum())
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"[env_guard] zeroed non-finite reward/obs in {int(bad.sum())} envs (total {count[0]})", flush=True)
        return obs, rewards, dones, extras

    wrapped.step = guarded_step


def apply_newton_env_guards(wrapped) -> None:
    """Install the Newton-only guards; PhysX envs are left untouched."""
    if uses_newton_physics(wrapped):
        guard_non_finite(wrapped)
        print("[env_guard] Newton physics: non-finite reward/obs guard on", flush=True)
