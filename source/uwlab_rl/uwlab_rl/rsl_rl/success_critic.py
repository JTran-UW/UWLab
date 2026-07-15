# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Auxiliary V_success value head trained from terminal-success reward.

Predicts P(success | s, pi) via TD/GAE regression on a reward stream where
r_t = 1 on a success-termination step, 0 otherwise. Consumed by the GPS
curriculum to score candidate reset states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.networks import MLP, EmpiricalNormalization


class SuccessCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str,
        obs_normalization: bool,
        lr: float,
        device: str,
    ) -> None:
        super().__init__()
        self.device = device
        self.normalizer = EmpiricalNormalization(input_dim) if obs_normalization else nn.Identity()
        self.net = MLP(input_dim, 1, hidden_dims, activation)
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(obs)
        return self.net(x)

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(obs)
        return self.net(x).squeeze(-1).clamp(0.0, 1.0)

    def update_normalization(self, obs: torch.Tensor) -> None:
        if isinstance(self.normalizer, EmpiricalNormalization):
            self.normalizer.update(obs)

    def update(
        self,
        obs_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        old_values_batch: torch.Tensor,
        num_epochs: int,
        num_mini_batches: int,
        use_clipped_value_loss: bool,
        clip_param: float,
        max_grad_norm: float,
    ) -> dict[str, float]:
        total = obs_batch.shape[0]
        mb_size = max(1, total // num_mini_batches)

        total_loss = 0.0
        n_updates = 0
        for _ in range(num_epochs):
            indices = torch.randperm(total, device=obs_batch.device)
            for i in range(num_mini_batches):
                sl = indices[i * mb_size : (i + 1) * mb_size]
                if sl.numel() == 0:
                    continue
                pred = self.forward(obs_batch[sl])
                target = returns_batch[sl]
                if use_clipped_value_loss:
                    old = old_values_batch[sl]
                    clipped = old + (pred - old).clamp(-clip_param, clip_param)
                    loss = torch.max((pred - target).pow(2), (clipped - target).pow(2)).mean()
                else:
                    loss = (pred - target).pow(2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
                n_updates += 1

        with torch.no_grad():
            final_pred = self.forward(obs_batch).squeeze(-1)
            final_target = returns_batch.squeeze(-1)
            target_var = final_target.var(unbiased=False)
            explained_var = 1.0 - (final_target - final_pred).var(unbiased=False) / (target_var + 1e-8)

        return {
            "value_loss": total_loss / max(1, n_updates),
            "explained_var": float(explained_var.item()),
            "mean_return": float(final_target.mean().item()),
            "mean_pred": float(final_pred.mean().item()),
        }


def compute_success_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    time_outs: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with timeout-aware bootstrap.

    All buffers shape (T, N, 1). On time_out=1, treat step as non-terminal
    (bootstrap through). On done=1 without time_out (i.e. success or fail),
    treat as absolute terminal (no bootstrap).
    """
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    advantage = torch.zeros_like(rewards[0])

    # Absolute terminal = done AND NOT time_out. Bootstrap is gated by this.
    is_absolute_terminal = (dones > 0).float() * (1.0 - (time_outs > 0).float())
    next_is_not_terminal = 1.0 - is_absolute_terminal

    for step in reversed(range(T)):
        next_v = last_values if step == T - 1 else values[step + 1]
        nt = next_is_not_terminal[step]
        delta = rewards[step] + nt * gamma * next_v - values[step]
        advantage = delta + nt * gamma * lam * advantage
        returns[step] = advantage + values[step]
        advantages[step] = advantage
    return returns, advantages
