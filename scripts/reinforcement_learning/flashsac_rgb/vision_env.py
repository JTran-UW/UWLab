# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManiSkill-style vector view of the OmniReset vision task for the FlashSAC-RGB port.

Observations are ``{"rgb": uint8 (N, 1, H, W, n_cams), "state": float (N, 1, D)}`` (frame stack 1, cameras on the
channel axis, as ManiSkill's FlattenRGBDObservationWrapper + GrayscaleObservationWrapper + FrameStack produce).
The wrapper owns a fixed horizon: every env is reset together after ``horizon`` steps and the pre-reset observation
is returned in ``infos["final_observation"]``, which is what the episodic ImageReplayBuffer needs.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

IMAGE_KEYS = ("front", "wrist")  # default 2-camera task; the wrapper reads the actual list from the camera group


class UWLabVisionVecEnv:
    def __init__(self, env, horizon: int, action_scale: float, state_group: str = "vision_state"):
        self.env = env.unwrapped
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.horizon = horizon
        action_dim = self.env.action_manager.total_action_dim
        # the policy acts in [-1, 1]; env actions are action_scale times that
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device) if isinstance(action_scale, (list, tuple)) else action_scale
        self.action_dim = action_dim
        self.state_group = state_group
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32)
        self._cmd = self.env.command_manager.get_term("task_command")
        self._ctx = self.env.reward_manager.get_term_cfg("progress_context").func
        self._reset_term = self.env.event_manager.get_term_cfg("reset_from_reset_states").func
        self.image_keys = None
        self.push_outcomes = True
        obs, _ = self.reset()
        self.single_observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(0, 255, tuple(obs["rgb"].shape[1:]), np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, tuple(obs["state"].shape[1:]), np.float32),
        })

    def _convert(self, obs_dict) -> dict:
        if self.image_keys is None:
            self.image_keys = tuple(obs_dict["camera"].keys())
        rgb = torch.cat([obs_dict["camera"][k] for k in self.image_keys], dim=-1)
        rgb = (rgb * 255.0).round_().clamp_(0, 255).to(torch.uint8).unsqueeze(1)
        groups = ["vision_state", "policy"] if self.state_group in ("asym", "privstore") else [self.state_group]
        parts = []
        for g in groups:
            part = torch.nan_to_num(obs_dict[g].float(), nan=0.0, posinf=0.0, neginf=0.0)
            # both state groups start with prev_actions (env units): report them in policy units
            part[:, : self.action_dim] /= self.action_scale
            parts.append(part)
        # "asym": non-privileged state first (actor), privileged appended (critic only)
        state = torch.cat(parts, dim=-1)
        return {"rgb": rgb, "state": state.unsqueeze(1)}

    def success(self) -> torch.Tensor:
        return self._cmd.orientation_aligned & self._cmd.position_aligned

    def reset(self):
        self.obs_dict, _ = self.env.reset()
        self.t = 0
        self.ep_return = torch.zeros(self.num_envs, device=self.device)
        self.ep_success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ep_ctx_success_once = torch.zeros_like(self.ep_success_once)
        self.ep_lenient_success_once = torch.zeros_like(self.ep_success_once)
        self.ep_task_id = self._reset_term.task_id.clone()
        self.ep_state_id = self._reset_term.state_id.clone()
        self.ep_abnormal = torch.zeros_like(self.ep_success_once)
        return self._convert(self.obs_dict), {}

    def _restart(self, ids: torch.Tensor):
        """Close the episodes of envs the task terminated (and already reset) and start tracking their new ones."""
        if self.push_outcomes:
            self._reset_term.sgs_update(self.ep_state_id[ids], self.ep_success_once[ids])
        self.ep_abnormal[ids] = True
        self.ep_return[ids] = 0.0
        self.ep_success_once[ids] = False
        self.ep_ctx_success_once[ids] = False
        self.ep_lenient_success_once[ids] = False
        self.ep_task_id[ids] = self._reset_term.task_id[ids]
        self.ep_state_id[ids] = self._reset_term.state_id[ids]

    def reset_for_eval(self):
        """Restart all envs from the uniform (probs) reset distribution; this episode's outcomes stay out of SGS."""
        self._reset_term.sgs_bypass = True
        self.push_outcomes = False
        return self.reset()

    def sgs_stats(self) -> dict[str, float]:
        return self._reset_term.sgs_stats()

    def step(self, actions: torch.Tensor):
        self.obs_dict, rewards, terms, _, _ = self.env.step(actions * self.action_scale)
        terms = terms.bool()
        # rare Newton blow-ups make pose-based rewards non-finite; one NaN in the replay buffer poisons every update
        rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1.0, 1.0)
        self.t += 1
        self.ep_return += rewards
        success = self.success()
        self.ep_success_once |= success
        self.ep_ctx_success_once |= self._ctx.success
        # thresholds of the jtran-uw main-branch hole (2.5 mm / 0.1 rad) the symmetry expert was trained under
        self.ep_lenient_success_once |= (self._ctx.xyz_distance < 0.0025) & (self._ctx.euler_xy_distance < 0.1)
        restarted = terms.nonzero(as_tuple=True)[0]
        if len(restarted):
            self._restart(restarted)
        obs = self._convert(self.obs_dict)
        none = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        infos = {}
        if self.t < self.horizon:
            return obs, rewards, terms, none, infos
        infos["final_observation"] = obs
        infos["_final_info"] = torch.ones_like(none)
        infos["final_info"] = {
            "episode": {
                "return": self.ep_return.clone(),
                "success_once": self.ep_success_once.float(),
                "success_at_end": success.float(),
                "ctx_success_once": self.ep_ctx_success_once.float(),
                "ctx_success_at_end": self._ctx.success.float(),
                "lenient_success_once": self.ep_lenient_success_once.float(),
                "abnormal": self.ep_abnormal.float(),
            },
            "task_id": self.ep_task_id,
            "state_id": self.ep_state_id,
        }
        if self.push_outcomes:
            keep = ~terms
            self._reset_term.sgs_update(self.ep_state_id[keep], self.ep_success_once[keep])
        self._reset_term.sgs_bypass = False
        self.push_outcomes = True
        obs, _ = self.reset()
        return obs, rewards, terms, torch.ones_like(none), infos
