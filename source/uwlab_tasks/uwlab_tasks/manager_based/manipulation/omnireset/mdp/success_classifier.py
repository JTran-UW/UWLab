# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Online success classifier for GPS reset curriculum.

Takes a per-reset-state feature vector (arm/gripper joints + object poses + EE pose),
predicts P(success | reset). Trained critic-style: on the current rollout only,
K epochs x M minibatches, triggered from the runner after PPO's update.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SuccessClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = device

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Return sigmoid-probabilities for a batch of features. Does NOT flip train/eval mode."""
        logits = self.net(features).squeeze(-1)
        return torch.sigmoid(logits)

    def train_on_pairs(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int,
        minibatch_size: int,
    ) -> dict[str, float]:
        """K epochs x ceil(N / minibatch_size) minibatches of BCE over (feats, labels).

        Returns mean BCE loss over the final epoch.
        """
        n = feats.shape[0]
        if n == 0:
            return {"update_loss": 0.0, "samples_per_iter": 0}

        last_epoch_losses: list[float] = []
        for epoch in range(n_epochs):
            perm = torch.randperm(n, device=feats.device)
            epoch_losses = []
            for start in range(0, n, minibatch_size):
                idx = perm[start : start + minibatch_size]
                logits = self.net(feats[idx]).squeeze(-1)
                loss = self.loss_fn(logits, labels[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            last_epoch_losses = epoch_losses

        return {
            "update_loss": sum(last_epoch_losses) / max(len(last_epoch_losses), 1),
            "samples_per_iter": n,
        }
