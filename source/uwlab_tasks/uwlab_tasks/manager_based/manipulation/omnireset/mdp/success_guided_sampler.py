# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success Guided Sampling (SGS, arXiv 2610.12465) over a fixed set of reset states."""

from __future__ import annotations

import math

import torch


class SuccessGuidedSampler:
    """Per-state ring buffers of Boolean outcomes, scored with the paper's Beta-shaped kernel.

    p_i is the mean over filled slots (``empty_rate`` for unvisited states). Sampling probability is
    ``base_i * exp(log(max(w_i, eps) + eps) / max(T, 1))`` with ``w_i = (p_i+eps)^(kappa t) (1-p_i+eps)^(kappa (1-t))``.
    """

    def __init__(
        self,
        num_states: int,
        history: int = 100,
        target: float = 0.5,
        kappa: float = 1.0,
        temperature: float = 2.0,
        eps: float = 1e-4,
        empty_rate: float = 0.5,
        base_weights: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ):
        self.num_states, self.history = num_states, history
        self.target, self.kappa, self.eps, self.empty_rate = target, kappa, eps, empty_rate
        self.temperature = max(temperature, 1.0)
        self.device = torch.device(device)
        self.outcomes = torch.zeros(num_states, history, dtype=torch.bool, device=self.device)
        self.count = torch.zeros(num_states, dtype=torch.long, device=self.device)
        base = torch.ones(num_states, device=self.device) if base_weights is None else base_weights.to(self.device, torch.float32)
        self.log_base = torch.log(base / base.sum())

    def success_rate(self) -> torch.Tensor:
        n = self.count.clamp(max=self.history)
        s = self.outcomes.sum(dim=1)
        return torch.where(n > 0, s.float() / n.clamp(min=1), torch.full_like(s, self.empty_rate, dtype=torch.float32))

    def probs(self) -> torch.Tensor:
        p = self.success_rate()
        w = (p + self.eps) ** (self.kappa * self.target) * (1.0 - p + self.eps) ** (self.kappa * (1.0 - self.target))
        logits = torch.log(w.clamp(min=self.eps) + self.eps) / self.temperature + self.log_base
        return torch.softmax(logits, dim=0)

    def sample(self, n: int) -> torch.Tensor:
        return torch.multinomial(self.probs(), n, replacement=True)

    def update(self, state_ids: torch.Tensor, outcomes: torch.Tensor) -> None:
        """Append one outcome per entry; repeated ids in one call each get their own slot (oldest dropped past H)."""
        ids = state_ids.to(self.device, torch.long).flatten()
        out = outcomes.to(self.device).bool().flatten()
        if ids.numel() == 0:
            return
        order = torch.argsort(ids, stable=True)
        ids, out = ids[order], out[order]
        idx = torch.arange(ids.numel(), device=self.device)
        first = torch.ones_like(ids, dtype=torch.bool)
        first[1:] = ids[1:] != ids[:-1]
        rank = idx - torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), dim=0).values
        mult = torch.bincount(ids, minlength=self.num_states)[ids]
        keep = rank >= mult - self.history
        ids, out, rank = ids[keep], out[keep], rank[keep]
        slot = (self.count[ids] + rank) % self.history
        self.outcomes[ids, slot] = out
        self.count.index_add_(0, ids, torch.ones_like(ids))

    def stats(self, groups: torch.Tensor | None = None, num_groups: int = 0) -> dict[str, float]:
        p, q = self.success_rate(), self.probs()
        visited = self.count > 0
        out = {
            "sgs/effective_states": math.exp(-(q * torch.log(q.clamp(min=1e-30))).sum().item()),
            "sgs/visited_frac": visited.float().mean().item(),
            "sgs/filled_frac": (self.count >= self.history).float().mean().item(),
            "sgs/mean_p_visited": p[visited].mean().item() if visited.any() else float("nan"),
        }
        if groups is not None:
            for g in range(num_groups):
                m = groups == g
                out[f"sgs/mass_type{g}"] = q[m].sum().item()
                vm = m & visited
                out[f"sgs/mean_p_type{g}"] = p[vm].mean().item() if vm.any() else float("nan")
        return out

    def state_dict(self) -> dict:
        return {"outcomes": self.outcomes, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.outcomes.copy_(d["outcomes"])
        self.count.copy_(d["count"])
