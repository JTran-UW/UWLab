# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with Potential-Based Reward Shaping (PBRS) using a frozen expert V function.

PBRS theorem (Ng et al. 1999): adding ``γ * Φ(s') − Φ(s)`` to every reward
preserves the optimal policy but provides dense guidance toward expert-valued
states. We use the trained state-PPO expert's critic as the potential function
``Φ = V_expert``.

Compared to BC: BC pulls the student toward expert *actions* (broken when
student obs ⊂ teacher obs, e.g. depth student vs state teacher). PBRS pulls
toward expert-*valued states*, leaving the student to find its own action
mapping. For state→state both are valid; for state→depth PBRS sidesteps the
obs-mismatch issue.

Use with ``ActorCriticDepth`` (or any rsl_rl ActorCritic). The expert checkpoint
must contain ``critic.*`` and ``critic_obs_normalizer.*`` keys at the same
architecture as ``hidden_dims`` here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.networks import EmpiricalNormalization, MLP


class PPOPBRS(PPO):
    def __init__(
        self,
        policy,
        expert_critic_path: str,
        expert_obs_group: str = "policy",
        expert_obs_dim: int = 43,
        expert_hidden_dims: list[int] = (512, 256, 128, 64),
        expert_activation: str = "elu",
        init_critic_from_expert: bool = False,
        pbrs_coef: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(policy, **kwargs)
        self.expert_obs_group = expert_obs_group
        self.pbrs_coef = float(pbrs_coef)

        # Build V_expert MLP and normalizer with the same shape as the saved expert critic.
        self.v_expert = MLP(
            input_dim=int(expert_obs_dim),
            output_dim=1,
            hidden_dims=list(expert_hidden_dims),
            activation=expert_activation,
        ).to(self.device)
        self.v_expert_norm = EmpiricalNormalization(int(expert_obs_dim)).to(self.device)

        # Load expert critic + normalizer from checkpoint.
        ckpt = torch.load(expert_critic_path, map_location=self.device, weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        critic_sd = {k[len("critic."):]: v for k, v in sd.items() if k.startswith("critic.")}
        norm_sd = {k[len("critic_obs_normalizer."):]: v for k, v in sd.items() if k.startswith("critic_obs_normalizer.")}
        self.v_expert.load_state_dict(critic_sd, strict=True)
        self.v_expert_norm.load_state_dict(norm_sd, strict=True)
        for p in self.v_expert.parameters():
            p.requires_grad = False
        for p in self.v_expert_norm.parameters():
            p.requires_grad = False
        self.v_expert.eval()
        self.v_expert_norm.eval()
        # Freeze the expert normalizer: its `until` (count threshold) is what
        # blocks .update() in EmpiricalNormalization. Set well below current
        # count so future calls skip the running-stats update.
        if hasattr(self.v_expert_norm, "until") and hasattr(self.v_expert_norm, "count"):
            self.v_expert_norm.until = int(self.v_expert_norm.count)
        print(
            f"[PPOPBRS] loaded V_expert from '{expert_critic_path}': "
            f"{sum(p.numel() for p in self.v_expert.parameters())} params, normalizer count={int(self.v_expert_norm.count)}"
        )

        # Optionally seed the policy critic with the same weights.
        if init_critic_from_expert:
            policy_sd = {f"critic.{k}": v for k, v in critic_sd.items()}
            policy_sd.update({f"critic_obs_normalizer.{k}": v for k, v in norm_sd.items()})
            result = nn.Module.load_state_dict(self.policy, policy_sd, strict=False)
            unexpected = list(result.unexpected_keys) if hasattr(result, "unexpected_keys") else []
            print(f"[PPOPBRS] init_critic_from_expert: loaded {len(policy_sd)} keys; unexpected={len(unexpected)}")

        # Logging buffers
        self._last_pbrs_mean = 0.0
        self._last_pbrs_abs_mean = 0.0
        self._last_v_expert_mean = 0.0

    @torch.no_grad()
    def _v(self, obs: TensorDict) -> torch.Tensor:
        s = obs[self.expert_obs_group]
        s = self.v_expert_norm(s)
        return self.v_expert(s).squeeze(-1)  # [N]

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        # Compute PBRS shaping using s_t (cached in self.transition.observations
        # by parent's act()) and s_{t+1} (the new `obs`):
        #   r' = r + pbrs_coef * (gamma * V(s') * (1 - done) - V(s))
        # The (1 - done) factor zeros out V(s') at episode boundaries (terminal
        # states have V=0 by MDP convention; PBRS theorem requires this for
        # optimal-policy invariance).
        if (
            self.transition.observations is not None
            and isinstance(obs, TensorDict)
            and self.expert_obs_group in obs.keys()
        ):
            with torch.no_grad():
                v_prev = self._v(self.transition.observations)
                v_next = self._v(obs)
                shaping = self.pbrs_coef * (self.gamma * v_next * (1.0 - dones.float()) - v_prev)
                rewards = rewards + shaping
                self._last_pbrs_mean = float(shaping.mean().item())
                self._last_pbrs_abs_mean = float(shaping.abs().mean().item())
                self._last_v_expert_mean = float(v_prev.mean().item())
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        out = super().update()
        out["pbrs_mean"] = self._last_pbrs_mean
        out["pbrs_abs_mean"] = self._last_pbrs_abs_mean
        out["v_expert_mean"] = self._last_v_expert_mean
        return out
