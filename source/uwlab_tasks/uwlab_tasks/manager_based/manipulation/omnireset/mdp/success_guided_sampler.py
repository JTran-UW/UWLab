# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success Guided Sampling (SGS, arXiv 2610.12465) over a fixed set of reset states."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


def distributed_world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def gather_ragged(x: torch.Tensor) -> torch.Tensor:
    """All-gather a 1-D tensor whose length differs per rank; the rank-ordered concatenation, identical on every rank."""
    ws = dist.get_world_size()
    n = torch.tensor([x.numel()], dtype=torch.long, device=x.device)
    ns = [torch.zeros_like(n) for _ in range(ws)]
    dist.all_gather(ns, n)
    sizes = [int(k.item()) for k in ns]
    if max(sizes) == 0:
        return x[:0]
    pad = torch.zeros(max(sizes), dtype=x.dtype, device=x.device)
    pad[: x.numel()] = x
    outs = [torch.zeros_like(pad) for _ in range(ws)]
    dist.all_gather(outs, pad)
    return torch.cat([o[:k] for o, k in zip(outs, sizes)])


class RollingMean:
    """Mean of the last ``window`` pushed values (fewer until the window fills)."""

    def __init__(self, window: int, device: str | torch.device = "cpu"):
        self.ring = torch.zeros(max(int(window), 1), device=device)
        self.next, self.filled = 0, 0

    def push(self, values: torch.Tensor) -> None:
        values = values.flatten().to(self.ring)
        n = self.ring.numel()
        if values.numel() >= n:  # only the newest window survives
            self.ring.copy_(values[-n:].roll(self.next))
            self.filled = n
            return
        idx = (self.next + torch.arange(values.numel(), device=self.ring.device)) % n
        self.ring[idx] = values
        self.next = (self.next + values.numel()) % n
        self.filled = min(self.filled + values.numel(), n)

    def mean(self) -> float | None:
        return self.ring[: self.filled].mean().item() if self.filled else None


class OutcomeQueue:
    """Multi-GPU SGS: each rank queues (key, outcome) pairs and ``flush`` pools every rank's queue in rank order.

    ``flush`` is a collective: call it at the same step on every rank (a global-time interval event), never from a
    reset event, which only runs on steps where that rank has resets.
    """

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.parts: list[torch.Tensor] = []

    def push(self, keys: torch.Tensor, outcomes: torch.Tensor) -> None:
        self.parts.append(keys.to(self.device, torch.long).flatten() * 2 + outcomes.to(self.device).long().flatten())

    def flush(self) -> tuple[torch.Tensor, torch.Tensor]:
        local = torch.cat(self.parts) if self.parts else torch.zeros(0, dtype=torch.long, device=self.device)
        self.parts = []
        pooled = gather_ragged(local)
        return pooled // 2, (pooled % 2).bool()


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
