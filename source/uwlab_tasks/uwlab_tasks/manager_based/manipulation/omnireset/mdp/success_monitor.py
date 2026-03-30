# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .success_monitor_cfg import SuccessMonitorCfg


class SuccessMonitor:
    def __init__(self, cfg: SuccessMonitorCfg):

        # uniform success buff
        self.monitored_history_len = cfg.monitored_history_len
        self.device = cfg.device
        self.success_buf = torch.zeros((cfg.num_monitored_data, self.monitored_history_len), device=self.device)
        self.success_rate = torch.zeros((cfg.num_monitored_data), device=self.device)
        self.success_pointer = torch.zeros((cfg.num_monitored_data), device=self.device, dtype=torch.int32)
        self.success_size = torch.zeros((cfg.num_monitored_data), device=self.device, dtype=torch.int32)

    def failure_rate_sampling(self, env_ids):
        failure_rate = (1 - self.success_rate).clamp(min=1e-6)
        return torch.multinomial(failure_rate.view(-1), len(env_ids), replacement=True).to(torch.int32)

    def success_update(self, ids_all, success_mask):
        unique_indices, inv, counts = torch.unique(ids_all, return_inverse=True, return_counts=True)
        counts_clamped = counts.clamp(max=self.monitored_history_len).to(dtype=self.success_pointer.dtype)

        ptrs = self.success_pointer[unique_indices]
        values = (success_mask[torch.argsort(inv)]).to(device=self.device, dtype=self.success_buf.dtype)
        values_splits = torch.split(values, counts.tolist())
        clamped_values = torch.cat([grp[-n:] for grp, n in zip(values_splits, counts_clamped.tolist())])
        state_indices = torch.repeat_interleave(unique_indices, counts_clamped)
        buf_indices = torch.cat([
            torch.arange(start, start + n, dtype=torch.int64, device=self.device) % self.monitored_history_len
            for start, n in zip(ptrs.tolist(), counts_clamped.tolist())
        ])

        self.success_buf.index_put_((state_indices, buf_indices), clamped_values)

        self.success_pointer.index_add_(0, unique_indices, counts_clamped)
        self.success_pointer = self.success_pointer % self.monitored_history_len

        self.success_size.index_add_(0, unique_indices, counts_clamped)
        self.success_size = self.success_size.clamp(max=self.monitored_history_len)
        self.success_rate[:] = self.success_buf.sum(dim=1) / self.success_size.clamp(min=1)

    def get_success_rate(self):
        return self.success_rate.clone()

    @staticmethod
    def sample_by_target_rate_from_rates(
        rates: torch.Tensor,
        num_samples: int,
        target: float = 0.33,
        kappa: float = 2.0,
        temperature: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample indices preferring success rates near `target` using a Beta-shaped kernel.

        Weight per partition: w(p) = p^(a-1) * (1-p)^(b-1)
        where a = 1 + kappa*target, b = 1 + kappa*(1-target).

        Args:
            rates: Success rates for each partition, shape [K].
            num_samples: Number of indices to draw.
            target: Desired success rate peak in [0, 1].
            kappa: Concentration. Larger = tighter around target.
            temperature: Softmax temperature. Larger = more uniform.

        Returns:
            (choices, probs): choices shape [num_samples], probs shape [K].
        """
        t = float(max(0.0, min(1.0, target)))
        k = float(max(0.0, kappa))
        a = 1.0 + k * t
        b = 1.0 + k * (1.0 - t)

        eps = 1e-8
        w = ((rates + eps).pow(a - 1.0) * (1.0 - rates + eps).pow(b - 1.0)).clamp_min(eps)
        logits = torch.log(w + eps)
        probs = torch.softmax(logits / float(temperature), dim=0)
        choices = torch.multinomial(probs, num_samples, replacement=True).to(torch.int64)
        return choices, probs
