# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asymmetric ActorCritic for depth-image PPO + BC distillation.

Actor: depth CNN encoder + proprio MLP -> action mean (+ scalar/log std). Sees
only what a deployed robot sees.

Critic: state-only MLP on the privileged ``critic`` obs group (typically the
43d teacher state). Sees full ground-truth state for accurate value estimates.
This is the standard asymmetric AC pattern -- privileged critic, depth actor.

Designed to plug into rsl_rl's ``PPO`` algorithm. Exposes the same ``act``,
``act_inference``, ``evaluate``, ``get_actions_log_prob``, ``action_mean``,
``action_std``, ``entropy`` API as ``rsl_rl.modules.ActorCritic``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization

from .student_teacher_vision import DepthCNN, ResNet18Encoder


class ActorCriticDepth(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        vision_groups: list[str],
        embed_dim: int = 128,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        encoder_type: str = "depth_cnn",
        encoder_pretrained_path: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        Normal.set_default_validate_args(False)

        self.obs_groups = obs_groups
        self.vision_groups = list(vision_groups)
        self.num_actions = num_actions

        # ---- Actor ----
        # Proprio: concatenation of all groups in obs_groups["policy"].
        proprio = torch.cat([obs[g] for g in obs_groups["policy"]], dim=-1)
        assert proprio.ndim == 2, f"policy groups must be 1D; got {proprio.shape}"
        num_proprio = proprio.shape[-1]

        first_img = obs[self.vision_groups[0]]
        assert first_img.ndim == 4, f"vision group {self.vision_groups[0]} must be 4D; got {first_img.shape}"
        _, in_channels, H, W = first_img.shape
        for g in self.vision_groups[1:]:
            assert obs[g].shape == first_img.shape, (
                f"all vision groups must share shape; {g} has {obs[g].shape} vs {first_img.shape}"
            )

        if encoder_type == "depth_cnn":
            self.depth_encoder = DepthCNN(in_channels, H, W, embed_dim=embed_dim)
        elif encoder_type == "resnet18":
            self.depth_encoder = ResNet18Encoder(in_channels, embed_dim=embed_dim, pretrained_path=encoder_pretrained_path)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        vision_feat_dim = len(self.vision_groups) * embed_dim

        self.actor = MLP(num_proprio + vision_feat_dim, num_actions, list(actor_hidden_dims), activation)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            # Normalize proprio only; images already in [0, 1] from process_image.
            self.actor_obs_normalizer = EmpiricalNormalization(num_proprio)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # ---- Critic (asymmetric, state-only) ----
        critic_dim = 0
        for g in obs_groups["critic"]:
            assert obs[g].ndim == 2, f"critic groups must be 1D; got {obs[g].shape}"
            critic_dim += obs[g].shape[-1]
        self.critic = MLP(critic_dim, 1, list(critic_hidden_dims), activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_dim)
        else:
            self.critic_obs_normalizer = nn.Identity()

        # ---- Action noise ----
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"noise_std_type must be 'scalar' or 'log'; got {noise_std_type}")

        self.distribution: Normal | None = None

    # ---- ActorCritic API ----
    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _encode_actor(self, obs: TensorDict) -> torch.Tensor:
        proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        proprio = self.actor_obs_normalizer(proprio)
        img_feats = [self.depth_encoder(obs[g]) for g in self.vision_groups]
        return torch.cat([proprio] + img_feats, dim=-1)

    def _encode_critic(self, obs: TensorDict) -> torch.Tensor:
        obs_cat = torch.cat([obs[g] for g in self.obs_groups["critic"]], dim=-1)
        return self.critic_obs_normalizer(obs_cat)

    def _update_distribution(self, feat: torch.Tensor) -> None:
        mean = self.actor(feat)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        feat = self._encode_actor(obs)
        self._update_distribution(feat)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        feat = self._encode_actor(obs)
        return self.actor(feat)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        return self.critic(self._encode_critic(obs))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> TensorDict:
        # PPO storage stores the whole TensorDict, so just return it; the
        # selected groups are pulled in _encode_actor.
        return obs

    def get_critic_obs(self, obs: TensorDict) -> TensorDict:
        return obs

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
            self.actor_obs_normalizer.update(proprio)
        if self.critic_obs_normalization:
            critic_obs = torch.cat([obs[g] for g in self.obs_groups["critic"]], dim=-1)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
