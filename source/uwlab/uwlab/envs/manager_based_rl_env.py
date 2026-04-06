# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""UWLab ManagerBasedRLEnv with terminal observation caching for correct value bootstrapping."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv


class UWLabManagerBasedRLEnv(ManagerBasedRLEnv):
    """Extends ManagerBasedRLEnv to cache terminal observations before auto-reset.

    This enables correct value bootstrapping in PPO: truncated episodes use
    V(s_terminal) while true terminations use V=0. Without this, the observations
    returned after reset are post-reset states, making it impossible to compute
    V(s_terminal).
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        # Cache terminal observations before any reset logic runs
        if len(env_ids) > 0:
            terminal_obs = self.observation_manager.compute()
            self.extras["terminal_observations"] = {
                key: value[env_ids].clone() for key, value in terminal_obs.items()
            }
            self.extras["terminal_env_ids"] = env_ids

        super()._reset_idx(env_ids)
