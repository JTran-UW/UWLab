# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + behavior cloning auxiliary loss.

Standard rsl_rl PPO with one extra term added to the per-minibatch loss:

    bc_loss = MSE(student_action_mean, teacher_action_mean.detach())
    total_loss = ppo_loss + cloning_loss_coeff * bc_loss

The teacher is a JIT module loaded from disk that consumes the ``teacher`` obs
group (43d state) and returns action mean. Used to lift a depth student past
the DAgger-only ceiling (~50% on peg) by combining PPO's reward signal with a
constant pull toward the teacher's action distribution.

Usage: pair with ``ActorCriticDepth`` (depth actor + state critic) and the
existing rsl_rl ``OnPolicyRunner``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


class BCPPO(PPO):
    def __init__(
        self,
        policy,
        teacher_jit_path: str,
        teacher_obs_groups: list[str],
        cloning_loss_coeff: float = 1.0,
        cloning_loss_decay: float = 1.0,
        bc_loss_type: str = "mse",
        **kwargs,
    ) -> None:
        super().__init__(policy, **kwargs)
        self.teacher = torch.jit.load(teacher_jit_path, map_location=self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_obs_groups = list(teacher_obs_groups)
        self.cloning_loss_coeff = float(cloning_loss_coeff)
        self.cloning_loss_decay = float(cloning_loss_decay)
        self._cur_cloning_loss_coeff = self.cloning_loss_coeff
        self._update_count = 0
        # bc_loss_type: "mse" = MSE on action means (raw scale ~250 for our peg
        #   teacher, see comments in agents/rsl_rl_cfg.py).
        # "weighted_mse" = DEXTRAH-style inverse-variance-weighted L2:
        #   sqrt(sum_i (1/sigma_t,i)^2 (mu_s,i - mu_t,i)^2) per sample, then
        #   mean over batch. Down-weights action dims where teacher is uncertain.
        #   Requires the JIT teacher to return (mean, std) tuple.
        if bc_loss_type not in ("mse", "weighted_mse"):
            raise ValueError(f"bc_loss_type must be 'mse' or 'weighted_mse'; got {bc_loss_type}")
        self.bc_loss_type = bc_loss_type

    def _teacher_distribution(self, obs_batch) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build teacher input and return (mean, std). std is None if teacher
        only returns mean."""
        teacher_obs = torch.cat([obs_batch[g] for g in self.teacher_obs_groups], dim=-1)
        with torch.no_grad():
            out = self.teacher(teacher_obs)
        if isinstance(out, (tuple, list)):
            return out[0].detach(), out[1].detach()
        return out.detach(), None

    def _bc_loss(self, mu_batch: torch.Tensor, obs_batch) -> torch.Tensor:
        teacher_mean, teacher_std = self._teacher_distribution(obs_batch)
        if self.bc_loss_type == "weighted_mse":
            if teacher_std is None:
                # Fall back silently if teacher doesn't expose std.
                return (mu_batch - teacher_mean).pow(2).mean()
            weights = (1.0 / teacher_std.clamp_min(1e-6)) ** 2
            return torch.sqrt(torch.sum(weights * (mu_batch - teacher_mean).pow(2), dim=-1)).mean()
        # default: plain MSE on means
        return (mu_batch - teacher_mean).pow(2).mean()

    def update(self) -> dict[str, float]:
        # Reuse parent PPO's update loop but inject BC into per-minibatch loss.
        # Easiest way: monkey-patch the optimizer step inside the loop is invasive;
        # instead, replicate the loop here. Keeps logic local & explicit.
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_bc_loss = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Adaptive LR via KL divergence.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate.
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value.
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # BC auxiliary.
            bc_loss = self._bc_loss(mu_batch, obs_batch)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self._cur_cloning_loss_coeff * bc_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_bc_loss += bc_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_bc_loss /= num_updates

        self._update_count += 1
        self._cur_cloning_loss_coeff = self.cloning_loss_coeff * (self.cloning_loss_decay ** self._update_count)

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc_loss": mean_bc_loss,
            "bc_coeff": self._cur_cloning_loss_coeff,
        }
