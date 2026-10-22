# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManiSkill-style vector view of the PushT ManiSkill-parity RGB task for the FlashSAC-RGB port (same API as
``UWLabVisionVecEnv``): ``{"rgb": uint8 (N, 1, 64, 64, 1), "state": float (N, 1, 21)}``, fixed horizon, all envs reset
together, pre-reset observation in ``infos["final_observation"]``. Set ``env_cfg.episode_length_s`` past the horizon
(``extend_episode_length``) so the env's own time-out never fires first.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from uwlab_tasks.manager_based.manipulation.push_t.mdp import maniskill


def extend_episode_length(env_cfg, horizon: int):
    env_cfg.episode_length_s = 10 * horizon * env_cfg.sim.dt * env_cfg.decimation


class _NoSGS:
    sgs = None


class PushTVisionVecEnv:
    def __init__(self, env, horizon: int, action_scale: float = 1.0, state_group: str = "state"):
        self.env = env.unwrapped
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.horizon = horizon
        self.action_scale = action_scale
        self.action_dim = self.env.action_manager.total_action_dim
        self.state_group = state_group
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)
        self.image_keys = ("gray",)
        self._reset_term = _NoSGS()
        self.blank_images = False
        print(f"[push_t_vision_env] num_rerenders_on_reset={self.env.cfg.num_rerenders_on_reset} horizon={horizon} episode_length={int(self.env.max_episode_length)}", flush=True)
        obs, _ = self.reset()
        self.single_observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(0, 255, tuple(obs["rgb"].shape[1:]), np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, tuple(obs["state"].shape[1:]), np.float32),
        })

    def _convert(self, obs_dict) -> dict:
        rgb = obs_dict["image"].permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).unsqueeze(1)
        if self.blank_images:
            rgb.zero_()
        state = torch.nan_to_num(obs_dict[self.state_group].float(), nan=0.0, posinf=0.0, neginf=0.0)
        return {"rgb": rgb, "state": state.unsqueeze(1)}

    def success(self) -> torch.Tensor:
        return maniskill.intersection_success(self.env)

    def reset(self):
        self.obs_dict, _ = self.env.reset()
        self.t = 0
        self.ep_return = torch.zeros(self.num_envs, device=self.device)
        self.ep_success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._convert(self.obs_dict), {}

    def reset_for_eval(self):
        return self.reset()

    def sgs_stats(self) -> dict[str, float]:
        return {}

    def step(self, actions: torch.Tensor):
        self.obs_dict, rewards, _, _, _ = self.env.step(actions * self.action_scale)
        rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.t += 1
        self.ep_return += rewards
        success = self.success()
        self.ep_success_once |= success
        obs = self._convert(self.obs_dict)
        none = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.t < self.horizon:
            return obs, rewards, none, none, {}
        infos = {
            "final_observation": obs,
            "_final_info": torch.ones_like(none),
            "final_info": {
                "episode": {
                    "return": self.ep_return.clone(),
                    "success_once": self.ep_success_once.float(),
                    "success_at_end": success.float(),
                    "final_overlap": maniskill.intersection_area(self.env).float(),
                },
                "task_id": torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
            },
        }
        obs, _ = self.reset()
        return obs, rewards, none, torch.ones_like(none), infos
