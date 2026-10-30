# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FlashSAC with expert demo mixing (RLPD-style): a fixed share of every mini-batch comes from an expert buffer."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_flashsac.rl_cfg import FlashSACAlgorithmCfg
from rsl_rl_flashsac.algorithms import FlashSAC


class ExpertBuffer:
    """Fixed buffer saved by ``scripts/tools/collect_flashsac_expert_buffer.py`` (``TorchUniformBuffer.save``)."""

    def __init__(self, path: str, device: str, n_step: int, gamma: float, groups: list[str], mmap: bool = True):
        # memory-map a CPU buffer: its pages are reclaimable page cache rather than process RAM
        # (mmap=False reads it into RAM once: random reads from a network filesystem otherwise stall every update)
        data = torch.load(path, map_location=device, weights_only=False, mmap=mmap and str(device) == "cpu")
        meta = data.get("metadata", {})
        for key, want in (("n_step", n_step), ("gamma", gamma), ("obs_groups", groups)):
            assert meta.get(key) == want, f"expert buffer {path}: {key} {meta.get(key)!r} != run's {want!r}"
        self.size = int(data["num_in_buffer"])
        self.observation = {k: v[: self.size] for k, v in data["observation"].items() if k in groups}
        self.next_observation = {k: v[: self.size] for k, v in data["next_observation"].items() if k in groups}
        self.action = data["action"][: self.size]
        self.reward = data["reward"][: self.size]
        self.terminated = data["terminated"][: self.size]
        self.truncated = data["truncated"][: self.size]
        self.metadata = meta
        self.device = device

    def sample(self, n: int) -> dict:
        idxs = torch.randint(0, self.size, (n,), device=self.device)
        return {
            "observation": TensorDict({k: v[idxs].float() for k, v in self.observation.items()}, batch_size=[n]),
            "action": self.action[idxs],
            "reward": self.reward[idxs],
            "terminated": self.terminated[idxs],
            "truncated": self.truncated[idxs],
            "next_observation": TensorDict(
                {k: v[idxs].float() for k, v in self.next_observation.items()}, batch_size=[n]
            ),
        }


class FlashSACDemo(FlashSAC):
    """FlashSAC whose mini-batches replace ``expert_ratio`` of the online samples with expert transitions.

    Rewards stay raw in both buffers and go through the same (online-statistics) reward normalizer.
    """

    def __init__(
        self,
        *args,
        expert_buffer_path: str = "",
        expert_ratio: float = 0.0,
        expert_buffer_device: str = "",
        expert_buffer_mmap: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert 0.0 <= expert_ratio < 1.0, f"expert_ratio must be in [0, 1), got {expert_ratio}"
        self.expert_ratio = expert_ratio
        self.expert_buffer = None
        if expert_buffer_path and expert_ratio > 0.0:
            assert self.symmetry is None, "FlashSACDemo does not support symmetry augmentation"
            rb = self.replay_buffer
            self.expert_buffer = ExpertBuffer(
                expert_buffer_path, expert_buffer_device or rb._device, self.n_steps, self.gamma, sorted(rb._store_groups),
                mmap=expert_buffer_mmap,
            )
            self.num_expert = int(round(expert_ratio * rb._sample_batch_size))
            print(
                f"[FlashSACDemo] expert buffer {expert_buffer_path}: {self.expert_buffer.size} transitions,"
                f" success {self.expert_buffer.metadata.get('success_rate')}, {self.num_expert}/"
                f"{rb._sample_batch_size} per mini-batch"
            )

    def _sample_batch(self) -> dict:
        if self.expert_buffer is None:
            return super()._sample_batch()
        online = self.replay_buffer.sample()
        n_online = self.replay_buffer._sample_batch_size - self.num_expert
        expert = self.expert_buffer.sample(self.num_expert)
        return {key: torch.cat([online[key][:n_online], expert[key].to(online[key].device)], dim=0) for key in online}


@configclass
class FlashSACDemoAlgorithmCfg(FlashSACAlgorithmCfg):
    class_name: str = "uwlab_rl.flashsac:FlashSACDemo"

    expert_buffer_path: str = ""
    """Expert buffer from ``scripts/tools/collect_flashsac_expert_buffer.py`` (empty: no demo mixing)."""

    expert_ratio: float = 0.2
    """Share of every mini-batch drawn from the expert buffer."""

    expert_buffer_device: str = ""
    """Device of the expert buffer ("" = the replay buffer's); on "cpu" each sample is moved to the GPU."""

    expert_buffer_mmap: bool = True
    """Memory-map a CPU expert buffer (low RSS) instead of reading it into RAM; use False on network filesystems."""
