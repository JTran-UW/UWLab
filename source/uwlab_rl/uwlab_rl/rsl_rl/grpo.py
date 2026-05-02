# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRPO (Group Relative Policy Optimization) for finetuning a distilled policy.

Per DeepSeek's R1 recipe and Doorman's Phase-3 finetune of a DAgger student.
Key differences vs vanilla PPO:

* **No learned value function**. Advantages are computed from per-batch
  (or per-group) baseline of trajectory returns. Sidesteps the
  hard-to-learn V(depth) regression that BCPPO got stuck on.
* **KL penalty against a frozen reference policy** (the DAgger snapshot at
  init). Prevents the policy from drifting away from the distilled init when
  the reward signal is sparse.
* **Trajectory MC returns** (not GAE), with bootstrap-by-zero on incomplete
  trajectories — incomplete-episode rows get advantage ~0 (no gradient).

Implementation: subclass rsl_rl's PPO so we reuse its rollout / minibatch
generator / clipping logic. Override:

* ``__init__``: snapshot the policy as reference, store kl_coeff and DAgger
  checkpoint path (optional weight init).
* ``compute_returns``: replace GAE with truncated MC returns, per-batch
  advantage normalization. Skip critic value-target setup.
* ``update``: identical to PPO loop but adds ``kl_coeff * kl(student || ref)``
  per minibatch. Value-loss coefficient is forced to 0 (no critic update).

Init from DAgger: the DAgger student (StudentTeacherVision) keys are
``depth_encoder.*``, ``student.*``, ``student_obs_normalizer.*``, ``std`` /
``log_std``. ActorCriticDepth keys are ``depth_encoder.*``, ``actor.*``,
``actor_obs_normalizer.*``, ``std`` / ``log_std``. Loader renames + skips
the teacher / aux / std_head bits.
"""

from __future__ import annotations

import copy
from collections import deque

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


def load_dagger_into_policy(policy: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """Load DAgger StudentTeacherVision weights into an ActorCriticDepth policy.

    Renames ``student.*`` -> ``actor.*`` and ``student_obs_normalizer.*`` ->
    ``actor_obs_normalizer.*``. Drops teacher / aux_head / std_head keys.
    Critic + critic_obs_normalizer stay at random init (DAgger has no critic).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # rsl_rl checkpoints typically wrap state under 'model_state_dict' or 'model'.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    new_sd = {}
    skipped = []
    for k, v in sd.items():
        if k.startswith("teacher.") or k.startswith("aux_head.") or k.startswith("std_head."):
            skipped.append(k)
            continue
        if k.startswith("student."):
            new_sd[k.replace("student.", "actor.", 1)] = v
        elif k.startswith("student_obs_normalizer."):
            new_sd[k.replace("student_obs_normalizer.", "actor_obs_normalizer.", 1)] = v
        else:
            new_sd[k] = v
    missing, unexpected = policy.load_state_dict(new_sd, strict=False)
    print(
        f"[GRPO] Loaded DAgger checkpoint '{ckpt_path}': mapped {len(new_sd)} keys, "
        f"skipped {len(skipped)} (teacher/aux/std_head). missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"  missing keys (will use random init): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  unexpected keys (ignored): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")


class GRPO(PPO):
    def __init__(
        self,
        policy,
        kl_coeff: float = 0.04,
        kl_target: float | None = None,
        init_from_dagger_path: str = "",
        clamp_advantage_quantile: float | None = None,
        **kwargs,
    ) -> None:
        # Force value_loss_coef=0 so critic doesn't train (we don't use V).
        kwargs.setdefault("value_loss_coef", 0.0)
        super().__init__(policy, **kwargs)
        # Optionally init policy from DAgger checkpoint (renaming keys).
        if init_from_dagger_path:
            load_dagger_into_policy(self.policy, init_from_dagger_path)
            self.policy.to(self.device)
        # Snapshot reference policy (frozen) AFTER potentially loading DAgger.
        self.reference_policy = copy.deepcopy(self.policy)
        for p in self.reference_policy.parameters():
            p.requires_grad = False
        self.reference_policy.eval()
        self.kl_coeff = float(kl_coeff)
        self.kl_target = kl_target  # if set, adapt kl_coeff toward this
        # Optional: clamp advantages to a quantile range to stabilize updates
        # when occasional huge returns dominate.
        self.clamp_advantage_quantile = clamp_advantage_quantile
        self._update_count = 0
        # Logging buffers
        self._buf_traj_return: deque[float] = deque(maxlen=2048)

    def compute_returns(self, obs) -> None:
        """Truncated MC returns. No V bootstrap. Per-batch advantage norm."""
        T = self.storage.num_transitions_per_env
        rewards = self.storage.rewards  # [T, N, 1]
        dones = self.storage.dones.float()  # [T, N, 1]
        # Backward accumulate. R_t = r_t + gamma * (1 - done_t) * R_{t+1}.
        R = torch.zeros_like(rewards[0])  # [N, 1]
        for t in reversed(range(T)):
            R = rewards[t] + self.gamma * (1.0 - dones[t]) * R
            self.storage.returns[t] = R
        # Track trajectory returns for logging (use only completed episodes).
        with torch.no_grad():
            done_mask = (dones > 0).flatten()
            if done_mask.any():
                done_returns = self.storage.returns.flatten()[done_mask]
                self._buf_traj_return.extend(done_returns.detach().cpu().tolist())
        # Per-batch advantage normalization.
        with torch.no_grad():
            A = self.storage.returns.clone()
            A_flat = A.flatten()
            A_mean = A_flat.mean()
            A_std = A_flat.std() + 1e-8
            A = (A - A_mean) / A_std
            if self.clamp_advantage_quantile is not None and 0 < self.clamp_advantage_quantile < 1:
                lo = torch.quantile(A_flat, self.clamp_advantage_quantile)
                hi = torch.quantile(A_flat, 1.0 - self.clamp_advantage_quantile)
                A = A.clamp(min=lo, max=hi)
            self.storage.advantages = A

    def _ref_distribution(self, obs_batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu_ref, sigma_ref) for KL penalty. No grad."""
        with torch.no_grad():
            self.reference_policy.act(obs_batch)
            mu_ref = self.reference_policy.action_mean.detach()
            sigma_ref = self.reference_policy.action_std.detach()
        return mu_ref, sigma_ref

    @staticmethod
    def _kl_gaussian(mu1, sigma1, mu2, sigma2) -> torch.Tensor:
        """KL(N(mu1, sigma1) || N(mu2, sigma2)), summed over action dims, mean over batch."""
        var1 = sigma1.pow(2)
        var2 = sigma2.pow(2)
        kl = 0.5 * (
            (var1 + (mu1 - mu2).pow(2)) / (var2 + 1e-8)
            + torch.log(var2 + 1e-8)
            - torch.log(var1 + 1e-8)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def update(self) -> dict[str, float]:
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl_loss = 0.0
        mean_ref_kl = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,  # unused (no V update)
            advantages_batch,
            returns_batch,        # unused (no V update)
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            # Normalize per minibatch if requested (else GRPO uses per-batch from compute_returns).
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Forward student
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Surrogate (PPO clipped)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # KL against reference policy
            mu_ref, sigma_ref = self._ref_distribution(obs_batch)
            ref_kl = self._kl_gaussian(mu_batch, sigma_batch, mu_ref, sigma_ref)

            loss = surrogate_loss - self.entropy_coef * entropy_batch.mean() + self.kl_coeff * ref_kl

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl_loss += (self.kl_coeff * ref_kl).item()
            mean_ref_kl += ref_kl.item()

        n = self.num_learning_epochs * self.num_mini_batches
        mean_surrogate_loss /= n
        mean_entropy /= n
        mean_kl_loss /= n
        mean_ref_kl /= n

        # Adaptive KL coefficient (DeepSeek-style: target a fixed kl).
        if self.kl_target is not None:
            if mean_ref_kl > 1.5 * self.kl_target:
                self.kl_coeff = self.kl_coeff * 1.5
            elif mean_ref_kl < self.kl_target / 1.5:
                self.kl_coeff = max(self.kl_coeff / 1.5, 1e-6)

        self._update_count += 1
        self.storage.clear()

        out = {
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_loss": mean_kl_loss,
            "ref_kl": mean_ref_kl,
            "kl_coeff": float(self.kl_coeff),
        }
        if len(self._buf_traj_return) > 0:
            out["mean_traj_return"] = sum(self._buf_traj_return) / len(self._buf_traj_return)
        return out
