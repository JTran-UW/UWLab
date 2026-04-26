# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""UWLab RslRlVecEnvWrapper that adds the terminated flag to extras."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class UWLabRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Extends RslRlVecEnvWrapper to pass the terminated flag in extras.

    This allows PPO to distinguish true terminations (V=0) from truncations
    (V=V(s_terminal)) for correct value bootstrapping.
    """

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # pass terminated flag for correct value bootstrapping
        extras["terminated"] = terminated
        # return the step information
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras
