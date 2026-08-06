# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.
#
# Copyright (c) 2019-2023 Antonin Raffin, Ashley Hill, Anssi Kanervisto,
# Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from __future__ import annotations

from telnetlib import theNULL
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from tensordict import TensorDict
from gymnasium import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class AsymmetricReplayBufferSamples(NamedTuple):
    policy_observations: th.Tensor
    critic_observations: th.Tensor
    actions: th.Tensor
    next_policy_observations: th.Tensor
    next_critic_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    effective_n_steps: th.Tensor
    # Optional proprioception stream, present only when the task defines a "proprio" obs group
    # (e.g. the grayscale/depth reaching tasks). Stays None for state-only tasks.
    proprio_observations: th.Tensor | None = None
    next_proprio_observations: th.Tensor | None = None

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples | RolloutBufferSamples | AsymmetricReplayBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

class AsymmetricReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    policy_observations: th.tensor
    critic_observations: th.tensor
    next_policy_observations: th.tensor
    next_critic_observations: th.tensor
    actions: th.tensor
    rewards: th.tensor
    terminations: th.tensor
    truncations: th.tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        sample_device: th.device | str | None = None,
        share_policy_critic_obs: bool = False,
        store_next_obs: bool = True,
        truncation_capacity_ratio: float = 0.05,
        pin_memory: bool = True,
    ):
        """
        :param share_policy_critic_obs: store ONE image stream instead of separate policy/critic
            copies. Only valid when the two groups are bitwise identical every step (true when they
            wrap the same obs terms and no noise/corruption actually differs between them). Verified
            on the first add(), which raises if they diverge. Halves image storage.
        :param store_next_obs: if False, do not materialize next_* streams; recover them at sample
            time as observations[t+1]. Halves image storage again. See the note below on why a small
            terminal-observation side store is still required for exactness.
        :param truncation_capacity_ratio: size of that side store, as a fraction of the buffer's
            transition capacity. Only used when store_next_obs=False.
        :param pin_memory: stage CPU->GPU sample transfers through reusable page-locked buffers,
            avoiding the driver's bounce-buffer copy. Costs only batch_size worth of pinned memory,
            not the whole buffer -- pinning the buffer would be pointless, since indexing it yields a
            fresh pageable tensor anyway.
        """
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.sample_device = get_device(sample_device) if sample_device is not None else self.device

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        policy_dim = int(np.array(self.obs_shape["policy"]).prod())
        critic_dim = int(np.array(self.obs_shape["critic"]).prod())
        self.share_policy_critic_obs = share_policy_critic_obs
        self.store_next_obs = store_next_obs
        self._shared_obs_verified = not share_policy_critic_obs
        self.pin_memory = pin_memory
        self._pinned_staging: dict[int, th.Tensor] = {}

        if share_policy_critic_obs and policy_dim != critic_dim:
            raise ValueError(
                f"share_policy_critic_obs=True needs matching policy/critic obs dims, got "
                f"{policy_dim} vs {critic_dim}."
            )

        self.policy_observations = th.zeros((self.buffer_size, self.n_envs, policy_dim), device=device)
        # Alias, not a copy: both names point at one allocation.
        self.critic_observations = (
            self.policy_observations
            if share_policy_critic_obs
            else th.zeros((self.buffer_size, self.n_envs, critic_dim), device=device)
        )

        if store_next_obs:
            self.next_policy_observations = th.zeros((self.buffer_size, self.n_envs, policy_dim), device=device)
            self.next_critic_observations = (
                self.next_policy_observations
                if share_policy_critic_obs
                else th.zeros((self.buffer_size, self.n_envs, critic_dim), device=device)
            )
        else:
            # next_obs[t] == observations[t+1] EXCEPT at truncations. The caller substitutes the true
            # terminal observation into next_obs on truncated steps, while observations[t+1] holds the
            # post-reset observation -- and truncations are NOT masked out of bootstrapping here
            # (`dones` returns terminations only), so those targets would be silently wrong if we just
            # indexed t+1. Terminations need no such care: the env auto-resets so next_obs is already
            # observations[t+1], and bootstrap is zero for them anyway.
            #
            # Hence a compact side store holding terminal observations for truncated steps only --
            # roughly one entry per episode per env, i.e. a few percent of a full next_* stream.
            capacity = max(1, int(self.buffer_size * self.n_envs * truncation_capacity_ratio))
            self.trunc_capacity = capacity
            self.trunc_obs_policy = th.zeros((capacity, policy_dim), device=device)
            self.trunc_obs_critic = (
                self.trunc_obs_policy if share_policy_critic_obs else th.zeros((capacity, critic_dim), device=device)
            )
            # slot index per (step, env), -1 when that step was not a truncation
            self.trunc_slot = th.full((self.buffer_size, self.n_envs), -1, dtype=th.long, device=device)
            # which (step, env) each slot was written for, so a slot recycled by the ring can be
            # detected as stale instead of silently returning another transition's observation
            self.trunc_owner = th.full((capacity,), -1, dtype=th.long, device=device)
            self.trunc_ptr = 0
            self.trunc_live = 0          # truncations currently referenced by the main ring
            self.trunc_stale_reads = 0   # backstop counter; should stay 0 now that the store grows
            self.next_policy_observations = None
            self.next_critic_observations = None

        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32, device=device
        )

        # Optional proprioception stream: allocated only if the observation space declares a
        # "proprio" group, so state-only tasks are unaffected and cost no extra memory.
        self.has_proprio = isinstance(self.obs_shape, dict) and "proprio" in self.obs_shape
        if self.has_proprio:
            proprio_dim = np.array(self.obs_shape["proprio"]).prod()
            self.proprio_observations = th.zeros((self.buffer_size, self.n_envs, proprio_dim), device=device)
            self.next_proprio_observations = th.zeros((self.buffer_size, self.n_envs, proprio_dim), device=device)

        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        self.terminations = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.truncations = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=device)
        self.n_steps = n_steps
        self.gamma = gamma

    def _grow_truncation_store(self, needed: int) -> None:
        """Enlarge the terminal-observation store in place, preserving existing slot indices.

        Capacity cannot be derived at construction time -- it depends on episode length, which the
        buffer never sees -- so rather than make the user guess a ratio (and silently lose exactness
        when they guess low), the store doubles on demand. Existing indices stay valid because this
        only appends.
        """
        new_cap = max(int(needed), self.trunc_capacity * 2)
        shared = self.share_policy_critic_obs
        dev = self.trunc_slot.device

        # Compact, don't merely extend. Extending leaves live entries scattered while trunc_ptr
        # wraps modulo capacity, so the allocator can circle back onto occupied slots even though
        # free ones exist elsewhere. The owner check then rejects the recycled slot and falls back
        # to observations[t+1] -- wrong for exactly the truncated transitions this store exists to
        # get right. Re-packing live entries into [0, L) restores the single contiguous free region
        # the ring allocator assumes.
        live_mask = self.trunc_slot >= 0
        pos_env = live_mask.nonzero(as_tuple=False)  # (L, 2): rows of (step, env)
        old_slots = self.trunc_slot[live_mask]
        n_live = int(old_slots.numel())

        def _repacked(src: th.Tensor) -> th.Tensor:
            out = th.zeros((new_cap, src.shape[1]), dtype=src.dtype, device=src.device)
            if n_live:
                out[:n_live] = src[old_slots]
            return out

        self.trunc_obs_policy = _repacked(self.trunc_obs_policy)
        self.trunc_obs_critic = self.trunc_obs_policy if shared else _repacked(self.trunc_obs_critic)

        owner = th.full((new_cap,), -1, dtype=th.long, device=dev)
        if n_live:
            new_slots = th.arange(n_live, device=dev)
            owner[new_slots] = pos_env[:, 0] * self.n_envs + pos_env[:, 1]
            self.trunc_slot[live_mask] = new_slots
        self.trunc_owner = owner

        self.trunc_capacity = new_cap
        self.trunc_ptr = n_live
        self.trunc_live = n_live

    def add(
        self,
        obs: TensorDict,
        next_obs: TensorDict,
        action: th.tensor,
        reward: th.tensor,
        termination: th.tensor,
        truncation: th.tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            print("[WARNING] Discrete obs not handled well in AsymmetricReplayBuffer")

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference; flatten to match storage shape and move to
        # the buffer's storage device (may differ from the device incoming tensors live on).
        # Every stream below is a device->host copy when the buffer lives on CPU, and for image
        # observations that copy dominates add(). So fetch only what is actually stored: with
        # share_policy_critic_obs + store_next_obs=False that is one stream instead of four, plus a
        # handful of rows for truncations.
        def _to_buf(t: th.Tensor) -> th.Tensor:
            return t.detach().reshape(self.n_envs, -1).to(self.device)

        pol = _to_buf(obs["policy"])

        if not self._shared_obs_verified:
            # One-time check: sharing the allocation is only sound if the two groups really are
            # identical. Fail loudly here rather than train on silently-wrong critic observations.
            # Costs one extra full transfer, once.
            if not (
                th.equal(pol, _to_buf(obs["critic"]))
                and th.equal(_to_buf(next_obs["policy"]), _to_buf(next_obs["critic"]))
            ):
                raise ValueError(
                    "share_policy_critic_obs=True but the policy and critic observation groups "
                    "differ. They must be bitwise identical (same terms, no differing corruption)."
                )
            self._shared_obs_verified = True

        self.policy_observations[self.pos] = pol
        if not self.share_policy_critic_obs:
            self.critic_observations[self.pos] = _to_buf(obs["critic"])

        if self.store_next_obs:
            self.next_policy_observations[self.pos] = _to_buf(next_obs["policy"])
            if not self.share_policy_critic_obs:
                self.next_critic_observations[self.pos] = _to_buf(next_obs["critic"])
        else:
            # This ring position is being overwritten, so any terminal obs recorded for it is dead
            # and its slots are free again.
            self.trunc_live -= int((self.trunc_slot[self.pos] >= 0).sum())
            self.trunc_slot[self.pos] = -1
            trunc_src = th.nonzero(truncation.detach().reshape(-1).bool(), as_tuple=False).flatten()
            if trunc_src.numel() > 0:
                n = int(trunc_src.numel())
                # Slice on the source device first: only the truncated rows cross the bus, rather
                # than a full next_obs stream every step for the sake of a handful of episodes.
                sel_pol = next_obs["policy"].detach().reshape(self.n_envs, -1)[trunc_src].to(self.device)
                sel_cri = (
                    None
                    if self.share_policy_critic_obs
                    else next_obs["critic"].detach().reshape(self.n_envs, -1)[trunc_src].to(self.device)
                )
                trunc_envs = trunc_src.to(self.device)
                if self.trunc_live + n > self.trunc_capacity:
                    self._grow_truncation_store(self.trunc_live + n)
                slots = (th.arange(n, device=self.device) + self.trunc_ptr) % self.trunc_capacity
                self.trunc_obs_policy[slots] = sel_pol
                if not self.share_policy_critic_obs:
                    self.trunc_obs_critic[slots] = sel_cri
                self.trunc_slot[self.pos, trunc_envs] = slots
                self.trunc_owner[slots] = self.pos * self.n_envs + trunc_envs
                self.trunc_ptr = (self.trunc_ptr + n) % self.trunc_capacity
                self.trunc_live += n

        if self.has_proprio:
            self.proprio_observations[self.pos] = obs["proprio"].detach().reshape(self.n_envs, -1).to(self.device)
            self.next_proprio_observations[self.pos] = (
                next_obs["proprio"].detach().reshape(self.n_envs, -1).to(self.device)
            )

        self.actions[self.pos] = action.detach().to(self.device)
        self.rewards[self.pos] = reward.detach().to(self.device)
        self.terminations[self.pos] = termination.detach().to(self.device)
        self.truncations[self.pos] = truncation.detach().to(self.device)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> AsymmetricReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if self.full:
            # FIRST SET self.pos-1 to truncated
            upper_bound = self.buffer_size
            current_pos = self.pos % self.buffer_size
            current_truncations = self.truncations[current_pos - 1, :]
            self.truncations[current_pos - 1, :] = th.logical_not(self.terminations[current_pos - 1, :])
        else:
            # sample only up to self.pos - self.n_steps + 1
            upper_bound = self.pos - self.n_steps + 1

        if self.store_next_obs:
            batch_inds = th.randint(0, upper_bound, (batch_size, ), device=self.device) % self.buffer_size
        else:
            # Reconstructing next_obs as observations[t+1] means index self.pos is poison: when full
            # it holds the oldest entry (a lap behind, not t+1); before that it is uninitialized. The
            # n-step window can reach batch_inds + n_steps, so keep that whole span clear of it.
            if self.full:
                span = max(self.buffer_size - self.n_steps - 1, 1)
                batch_inds = (
                    self.pos + 1 + th.randint(0, span, (batch_size,), device=self.device)
                ) % self.buffer_size
            else:
                batch_inds = th.randint(
                    0, max(upper_bound - 1, 1), (batch_size,), device=self.device
                ) % self.buffer_size
        samples = self._get_samples(batch_inds)
        if self.full:
            self.truncations[self.pos - 1, :] = current_truncations
        return samples

    def _get_samples(self, batch_inds: th.tensor) -> AsymmetricReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = th.randint(0, high=self.n_envs, size=(len(batch_inds),), device=self.device)
        offsets = th.arange(self.n_steps, device=self.device).view(-1, 1)
        seq_inds = (batch_inds + offsets) % self.buffer_size

        n_step_rewards = self.rewards[seq_inds, env_indices]
        n_step_terminations = self.terminations[seq_inds, env_indices].int().bool()
        n_step_truncations = self.truncations[seq_inds, env_indices].int().bool()
        n_step_dones = n_step_terminations | n_step_truncations

        n_step_dones_shifted = th.concat([th.zeros_like(n_step_dones[0].unsqueeze(0)), n_step_dones[:-1, :]])
        done_mask = th.cumprod(1 - n_step_dones_shifted.int(), dim=0)
        effective_n_steps = done_mask.sum(dim=0)
        masked_rewards = done_mask * n_step_rewards
        discounts = th.pow(self.gamma, th.arange(self.n_steps, device=self.device))

        discounted_rewards = masked_rewards * discounts.view(-1, 1)
        n_step_rewards = discounted_rewards.sum(dim=0)

        first_done = th.argmax(n_step_dones.int(), dim=0)
        no_done = n_step_dones.sum(dim=0) == 0
        first_done = th.where(no_done, self.n_steps - 1, first_done)

        next_inds = (batch_inds + first_done) % self.buffer_size

        if self.store_next_obs:
            next_pol = self.next_policy_observations[next_inds, env_indices]
            next_cri = self.next_critic_observations[next_inds, env_indices]
        else:
            # next_obs[t] = observations[t+1], with truncated steps patched from the side store.
            plus1 = (next_inds + 1) % self.buffer_size
            next_pol = self.policy_observations[plus1, env_indices]
            next_cri = next_pol if self.share_policy_critic_obs else self.critic_observations[plus1, env_indices]

            slot = self.trunc_slot[next_inds, env_indices]
            safe = slot.clamp(min=0)
            # A slot is usable only if it still belongs to this exact (step, env): the ring may have
            # recycled it for a newer truncation, in which case fall back to observations[t+1].
            valid = (slot >= 0) & (self.trunc_owner[safe] == next_inds * self.n_envs + env_indices)
            if bool(valid.any()):
                mask = valid.unsqueeze(-1)
                next_pol = th.where(mask, self.trunc_obs_policy[safe], next_pol)
                # Re-alias rather than running an identical where(): th.where allocates, which would
                # otherwise split the shared stream back into two tensors and cost a second transfer.
                next_cri = (
                    next_pol
                    if self.share_policy_critic_obs
                    else th.where(mask, self.trunc_obs_critic[safe], next_cri)
                )
            stale = int((slot >= 0).sum()) - int(valid.sum())
            if stale:
                self.trunc_stale_reads += stale

        # When the two streams are aliased, gather once and hand back the same tensor for both:
        # otherwise the identical rows are gathered twice and cross the bus twice.
        pol_batch = self.policy_observations[batch_inds, env_indices]
        cri_batch = pol_batch if self.share_policy_critic_obs else self.critic_observations[batch_inds, env_indices]

        data = (
            pol_batch,
            cri_batch,
            self.actions[batch_inds, env_indices],
            next_pol,
            next_cri,
            self.terminations[next_inds, env_indices],
            n_step_rewards,
            effective_n_steps
        )
        if self.has_proprio:
            # Appended last so the positional AsymmetricReplayBufferSamples(*data) construction keeps
            # working unchanged for tasks without a proprio group (fields default to None).
            data = data + (
                self.proprio_observations[batch_inds, env_indices],
                self.next_proprio_observations[next_inds, env_indices],
            )
        if self.sample_device != self.device:
            data = self._to_sample_device(data)
        return AsymmetricReplayBufferSamples(*data)

    def _to_sample_device(self, tensors: tuple) -> tuple:
        """Move a gathered batch to the sample device, staging through pinned host memory.

        Pinning the buffer storage itself would achieve nothing: advanced indexing allocates a fresh
        pageable tensor, so it is the *gather output* that crosses the bus. A pageable copy makes the
        driver stage through an internal bounce buffer; copying into our own reusable page-locked
        staging tensor first lets the DMA read host memory directly.

        Transfers are issued blocking on purpose. Reusing one staging tensor per stream is only safe
        if the previous DMA has finished before the next copy_ overwrites it, and a host-side write
        is not ordered against an async copy already queued on the stream.
        """
        seen: dict[int, th.Tensor] = {}
        out = []
        for t in tensors:
            if t is None:
                out.append(None)
                continue
            if id(t) in seen:  # aliased streams (shared policy/critic) move once
                out.append(seen[id(t)])
                continue
            moved = self._move_to_sample_device(t, len(out))
            seen[id(t)] = moved
            out.append(moved)
        return tuple(out)

    def _move_to_sample_device(self, t: th.Tensor, key: int) -> th.Tensor:
        if not self.pin_memory or t.device.type != "cpu" or self.sample_device.type != "cuda":
            return t.to(self.sample_device)
        staging = self._pinned_staging.get(key)
        if staging is None or staging.shape != t.shape or staging.dtype != t.dtype:
            try:
                staging = th.empty(t.shape, dtype=t.dtype, pin_memory=True)
            except RuntimeError:
                # No CUDA host allocator available -- fall back permanently rather than retry.
                self.pin_memory = False
                return t.to(self.sample_device)
            self._pinned_staging[key] = staging
        staging.copy_(t)
        return staging.to(self.sample_device)


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


def load_expert_replay_buffer(
    path: str,
    device: th.device | str = "cpu",
    sample_device: th.device | str | None = None,
    n_steps: int | None = None,
    gamma: float | None = None,
) -> "AsymmetricReplayBuffer":
    """Rebuild an AsymmetricReplayBuffer from a payload written by collect_expert_replay_buffer.py.

    Restores whichever layout it was saved in (shared policy/critic, dense vs reconstructed next_*,
    proprio present or not) and, for the compact layout, rebuilds the truncation side store at the
    capacity it grew to during collection -- saved slot indices would otherwise point past the end of
    a default-sized store. ``n_steps``/``gamma`` override the recorded values so an expert buffer can
    be replayed at the consumer's n-step horizon rather than the collector's.
    """
    # Always land the payload on the HOST, never on `device`. Loading straight to the GPU means the
    # th.load payload and the buffer it is copied into are both resident at once -- a 9.7 GiB file
    # peaks near 19 GiB of VRAM, which is what put both-buffers-on-GPU out of reach on a 44 GiB
    # l40s despite a steady-state footprint that fits comfortably. Staged through host memory the
    # peak is just the buffer itself; the copies below go straight from CPU into the device tensors.
    payload = th.load(path, map_location="cpu", weights_only=False)
    meta, tensors = payload["metadata"], payload["buffer_tensors"]

    obs_space = spaces.Dict(
        {
            "policy": spaces.Box(low=-np.inf, high=np.inf, shape=(meta["n_obs"],)),
            "critic": spaces.Box(low=-np.inf, high=np.inf, shape=(meta["n_critic_obs"],)),
        }
    )
    if meta.get("n_proprio"):
        obs_space.spaces["proprio"] = spaces.Box(low=-np.inf, high=np.inf, shape=(meta["n_proprio"],))
    action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(meta["n_act"],))

    share = meta.get("share_policy_critic_obs", False)
    store_next = meta.get("store_next_obs", True)
    rb = AsymmetricReplayBuffer(
        meta["buffer_size"] * meta["n_envs"],
        obs_space,
        action_space,
        device,
        n_envs=meta["n_envs"],
        n_steps=meta["n_steps"] if n_steps is None else n_steps,
        gamma=meta["gamma"] if gamma is None else gamma,
        sample_device=sample_device,
        share_policy_critic_obs=share,
        store_next_obs=store_next,
    )

    names = ["policy_observations", "actions", "rewards", "terminations", "truncations"]
    if not share:
        names.append("critic_observations")
    if meta.get("n_proprio"):
        names += ["proprio_observations", "next_proprio_observations"]
    if store_next:
        names.append("next_policy_observations")
        if not share:
            names.append("next_critic_observations")
    for name in names:
        getattr(rb, name).copy_(tensors[name])          # CPU -> device, no GPU temporary

    if not store_next:
        cap = int(tensors["trunc_capacity"])
        if cap != rb.trunc_capacity:
            rb._grow_truncation_store(cap)
        rb.trunc_obs_policy[:cap].copy_(tensors["trunc_obs_policy"])
        if not share:
            rb.trunc_obs_critic[:cap].copy_(tensors["trunc_obs_critic"])
        rb.trunc_slot.copy_(tensors["trunc_slot"])
        rb.trunc_owner[:cap].copy_(tensors["trunc_owner"])
        rb.trunc_ptr = int(tensors["trunc_ptr"])
        rb.trunc_live = int(tensors["trunc_live"])

    rb.pos = tensors["pos"]
    rb.full = tensors["full"]
    rb._shared_obs_verified = True  # restored buffers never call add()
    return rb


def cat_samples(a: AsymmetricReplayBufferSamples, b: AsymmetricReplayBufferSamples):
    """Concatenate two sample batches field-wise, preserving None for absent optional streams."""
    return AsymmetricReplayBufferSamples(
        *[None if x is None or y is None else th.cat([x, y], dim=0) for x, y in zip(a, b)]
    )
