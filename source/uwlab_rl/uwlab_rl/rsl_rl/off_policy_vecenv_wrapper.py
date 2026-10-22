# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL vec-env wrapper that adds ``extras["time_outs_obs"]`` (pre-reset observations) for off-policy bootstrapping."""

from __future__ import annotations

import torch
from functools import partial
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class RslRlOffPolicyVecEnvWrapper(RslRlVecEnvWrapper):
    """Isaac Lab resets done envs inside ``step()``, so their returned observation is already post-reset. This caches
    the terminal observation in ``_reset_idx`` and republishes it as ``extras["time_outs_obs"]`` so SAC bootstraps
    truncations from the true final state."""

    def __init__(self, env, clip_actions: float | None = None):
        super().__init__(env, clip_actions)
        self._final_obs: dict[str, torch.Tensor] | None = None
        base = self.unwrapped
        base._reset_idx = partial(self._reset_idx_with_final_obs, base, base._reset_idx)

    def _reset_idx_with_final_obs(self, base, orig_reset_idx, env_ids, *args, **kwargs):
        obs = {k: v for k, v in base.observation_manager.compute().items() if isinstance(v, torch.Tensor)}
        if self._final_obs is None:
            self._final_obs = {k: torch.zeros_like(v) for k, v in obs.items()}
        ids = slice(None) if env_ids is None else env_ids
        for k, v in obs.items():
            self._final_obs[k][ids] = v[ids]
        return orig_reset_idx(env_ids, *args, **kwargs)

    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = super().step(actions)
        if "time_outs" in extras and self._final_obs is not None:
            extras["time_outs_obs"] = TensorDict(
                {k: v.clone() for k, v in self._final_obs.items()}, batch_size=[self.num_envs]
            )
        return obs, rew, dones, extras
