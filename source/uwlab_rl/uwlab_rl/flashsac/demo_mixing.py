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

    def __init__(
        self, path: str, device: str, n_step: int, gamma: float, groups: list[str], mmap: bool = True,
        shard: tuple[int, int] | None = None,
    ):
        # memory-map a CPU buffer: its pages are reclaimable page cache rather than process RAM
        # (mmap=False reads it into RAM once: random reads from a network filesystem otherwise stall every update)
        # shard=(rank, world_size): keep every world_size-th transition from rank (loaded via CPU mmap)
        if shard is None:
            data = torch.load(path, map_location=device, weights_only=False, mmap=mmap and str(device) == "cpu")
        else:
            data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        meta = data.get("metadata", {})
        for key, want in (("n_step", n_step), ("gamma", gamma), ("obs_groups", groups)):
            assert meta.get(key) == want, f"expert buffer {path}: {key} {meta.get(key)!r} != run's {want!r}"
        n = int(data["num_in_buffer"])

        def take(v: torch.Tensor) -> torch.Tensor:
            if shard is None:
                return v[:n]
            return v[:n][shard[0] :: shard[1]].contiguous().to(device)

        self.observation = {k: take(v) for k, v in data["observation"].items() if k in groups}
        self.next_observation = {k: take(v) for k, v in data["next_observation"].items() if k in groups}
        self.action = take(data["action"])
        self.reward = take(data["reward"])
        self.terminated = take(data["terminated"])
        self.truncated = take(data["truncated"])
        self.size = self.action.shape[0]
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
        expert_ratio_final: float = -1.0,
        expert_anneal_start: int = 0,
        expert_anneal_end: int = 0,
        expert_buffer_device: str = "",
        expert_buffer_mmap: bool = True,
        expert_buffer_shard: bool = False,
        expert_policy_checkpoint: str = "",
        expert_policy_agent: str = "rsl_rl_cfg_entry_point",
        expert_obs_group: str = "policy",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # expert envs (set by expert_envs.attach_expert_envs, which needs the env)
        self.expert_policy_checkpoint, self.expert_policy_agent = expert_policy_checkpoint, expert_policy_agent
        self.expert_obs_group = expert_obs_group
        self.expert_policy = self.expert_env_ids = None
        self.expert_clip_fraction = 0.0
        assert 0.0 <= expert_ratio < 1.0, f"expert_ratio must be in [0, 1), got {expert_ratio}"
        self.expert_ratio = expert_ratio
        self.expert_ratio_final = expert_ratio if expert_ratio_final < 0 else expert_ratio_final
        assert 0.0 <= self.expert_ratio_final < 1.0, f"expert_ratio_final must be in [0, 1), got {expert_ratio_final}"
        assert expert_anneal_end >= expert_anneal_start >= 0
        self.expert_anneal_start, self.expert_anneal_end = expert_anneal_start, expert_anneal_end
        self.expert_buffer = None
        if expert_buffer_path and expert_ratio > 0.0:
            assert self.symmetry is None, "FlashSACDemo does not support symmetry augmentation"
            rb = self.replay_buffer
            self.expert_buffer = ExpertBuffer(
                expert_buffer_path, expert_buffer_device or rb._device, self.n_steps, self.gamma, sorted(rb._store_groups),
                mmap=expert_buffer_mmap,
                shard=(self.gpu_global_rank, self.gpu_world_size) if expert_buffer_shard and self.gpu_world_size > 1 else None,
            )
            self.num_expert = int(round(expert_ratio * rb._sample_batch_size))
            print(
                f"[FlashSACDemo] rank {self.gpu_global_rank}/{self.gpu_world_size}"
                f" expert buffer {expert_buffer_path}: {self.expert_buffer.size} transitions,"
                f" success {self.expert_buffer.metadata.get('success_rate')}, {self.num_expert}/"
                f"{rb._sample_batch_size} per mini-batch"
                + (f" -> {self.expert_ratio_final} over iterations {expert_anneal_start}-{expert_anneal_end}"
                   if self.expert_ratio_final != expert_ratio else "")
            )

    def current_expert_ratio(self) -> float:
        """Linear anneal expert_ratio -> expert_ratio_final over [expert_anneal_start, expert_anneal_end] iterations,
        counted in update steps (restored on resume)."""
        it = self.update_step / (self.num_learning_epochs * self.num_mini_batches)
        span = self.expert_anneal_end - self.expert_anneal_start
        frac = float(it >= self.expert_anneal_end) if span == 0 else min(max((it - self.expert_anneal_start) / span, 0.0), 1.0)
        return self.expert_ratio + (self.expert_ratio_final - self.expert_ratio) * frac

    def set_expert_envs(self, policy, env_ids: torch.Tensor, action_scale: torch.Tensor, action_bias: torch.Tensor) -> None:
        """``policy`` (env-unit actions from the ``expert_obs_group`` obs) drives ``env_ids`` from now on."""
        self.expert_policy = policy
        self.expert_env_ids = env_ids.to(self.device, torch.long)
        self.expert_action_scale = action_scale.to(self.device)
        self.expert_action_bias = action_bias.to(self.device)

    def _apply_expert_envs(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Overwrite the expert envs' actions with the expert's (deterministic, no exploration noise) in the
        policy's [-1, 1] space and record them as the executed actions; a no-op for obs without the expert group
        (e.g. FlashSACRGB's encoded features)."""
        if self.expert_policy is None or self.expert_obs_group not in obs.keys():
            return actions
        from .expert_envs import to_policy_space

        with torch.no_grad():
            env_actions = self.expert_policy(obs.select(self.expert_obs_group)[self.expert_env_ids])
            expert_actions = to_policy_space(env_actions, self.expert_action_scale, self.expert_action_bias)
            unclipped = (env_actions - self.expert_action_bias) / self.expert_action_scale
            self.expert_clip_fraction = (unclipped != expert_actions).float().mean()  # tensor: no per-step sync
        actions = actions.clone()
        actions[self.expert_env_ids] = expert_actions.to(actions.dtype)
        self._last_actions = actions
        return actions

    def act(self, obs: TensorDict) -> torch.Tensor:
        return self._apply_expert_envs(obs, super().act(obs))

    def act_random(self, obs: TensorDict) -> torch.Tensor:
        return self._apply_expert_envs(obs, super().act_random(obs))

    def _sample_batch(self) -> dict:
        if self.expert_buffer is None:
            return super()._sample_batch()
        online = self.replay_buffer.sample()
        self.num_expert = int(round(self.current_expert_ratio() * self.replay_buffer._sample_batch_size))
        if self.num_expert == 0:
            return online
        n_online = self.replay_buffer._sample_batch_size - self.num_expert
        expert = self.expert_buffer.sample(self.num_expert)
        return {key: torch.cat([online[key][:n_online], expert[key].to(online[key].device)], dim=0) for key in online}


@configclass
class FlashSACDemoAlgorithmCfg(FlashSACAlgorithmCfg):
    class_name: str = "uwlab_rl.flashsac:FlashSACDemo"

    expert_buffer_path: str = ""
    """Expert buffer from ``scripts/tools/collect_flashsac_expert_buffer.py`` (empty: no demo mixing)."""

    expert_ratio: float = 0.2
    """Share of every mini-batch drawn from the expert buffer (at the start of the anneal)."""

    expert_ratio_final: float = -1.0
    """Expert share after the anneal (negative = no anneal)."""

    expert_anneal_start: int = 0
    """Iteration where the linear anneal expert_ratio -> expert_ratio_final starts."""

    expert_anneal_end: int = 0
    """Iteration where the anneal ends; the share then holds expert_ratio_final."""

    expert_buffer_device: str = ""
    """Device of the expert buffer ("" = the replay buffer's); on "cpu" each sample is moved to the GPU."""

    expert_buffer_mmap: bool = True
    """Memory-map a CPU expert buffer (low RSS) instead of reading it into RAM; use False on network filesystems."""

    expert_buffer_shard: bool = False
    """Multi-GPU: each rank keeps every world_size-th expert transition (1/world_size of the memory)."""

    expert_policy_checkpoint: str = ""
    """Expert envs: rsl_rl checkpoint of a state expert whose deterministic actions the reset event's expert envs
    (``reset_from_reset_states.params.expert_env_every``) execute; their transitions enter the replay buffer
    ("" = off). Attached by ``uwlab_rl.flashsac.expert_envs.attach_expert_envs`` in train.py."""

    expert_policy_agent: str = "rsl_rl_cfg_entry_point"
    """Task registry entry point of the expert's rsl_rl runner cfg."""

    expert_obs_group: str = "policy"
    """Obs group the expert reads (privileged state; never stored in the replay buffer)."""
