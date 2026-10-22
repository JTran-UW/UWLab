"""Episodic image replay buffer, copied from ManiSkill-opt-all reinforcement_learning/utils/buffers.py (FlashSAC-RGB)."""

from typing import NamedTuple

import torch as th


class ImageReplayBufferSamples(NamedTuple):
    observations: dict
    actions: th.Tensor
    next_observations: dict
    dones: th.Tensor
    rewards: th.Tensor
    effective_n_steps: th.Tensor


def cat_image_samples(a: ImageReplayBufferSamples, b: ImageReplayBufferSamples) -> ImageReplayBufferSamples:
    """Concatenate two sampled batches along the batch dim (dict fields per key)."""
    return ImageReplayBufferSamples(*[
        {k: th.cat([x[k], y[k]]) for k in x} if isinstance(x, dict) else th.cat([x, y])
        for x, y in zip(a, b)
    ])


class ImageReplayBuffer:
    def __init__(
        self, pixels_obs_space, state_obs_dim, act_dim, size, horizon, num_envs, n_frame_stack, n_steps, gamma, device="cuda"
    ):
        assert (
            size % horizon == 0
        ), f"{self.__class__.__name__} needs size divisble by horizon"

        num_episodes = size // horizon
        assert (
            num_episodes % num_envs == 0
        ), f"{self.__class__.__name__} needs num_episodes divisible by num_envs"

        ## init buffers as numpy arrays
        self.pixel_obs_buf = th.zeros(
            num_episodes, horizon + n_frame_stack, *pixels_obs_space[1:],
            dtype=th.uint8,
            device=device
        )
        to = list(range(len(pixels_obs_space) + 1))
        to[0] = 1
        to[1] = 0
        self.transpose_order = tuple(to)

        self.state_obs_buf = th.zeros(
            num_episodes, horizon, state_obs_dim, dtype=th.float32, device=device
        )
        self.state_next_obs_buf = th.zeros(
            num_episodes, horizon, state_obs_dim, dtype=th.float32, device=device
        )
        self.acts_buf = th.zeros(num_episodes, horizon, act_dim, dtype=th.float32, device=device)
        self.rews_buf = th.zeros(num_episodes, horizon, dtype=th.float32, device=device)
        self.done_buf = th.zeros(num_episodes, horizon, dtype=th.float32, device=device)

        self.idx_to_coord = th.zeros(size, 2, dtype=th.long, device=device)

        self.max_size = size
        self.num_episodes = num_episodes
        self.horizon = horizon
        self.num_envs = num_envs
        self.frame_stack = n_frame_stack

        self.current_size = 0
        self.batch_start_episode = 0
        self.step_num = 0

        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device

    def add(
        self,
        obs,
        next_obs,
        act,
        rew,
        term,
    ):
        bs, be = self.batch_start_episode, self.batch_start_episode + self.num_envs
        sn = self.step_num
        fs = self.frame_stack

        pixel_obs = obs["rgb"]
        pixel_next_obs = next_obs["rgb"]
        state_obs = obs["state"][:, -1]
        state_next_obs = next_obs["state"][:, -1]

        self.pixel_obs_buf[bs:be, sn : sn + fs] = pixel_obs
        self.pixel_obs_buf[bs:be, sn + 1 : sn + fs + 1] = pixel_next_obs

        self.state_obs_buf[bs:be, sn] = state_obs
        self.state_next_obs_buf[bs:be, sn] = state_next_obs
        self.acts_buf[bs:be, sn] = act
        self.rews_buf[bs:be, sn] = rew
        self.done_buf[bs:be, sn] = term

        if self.current_size < self.max_size:
            self.idx_to_coord[self.current_size : self.current_size + self.num_envs] = (
                th.stack([th.arange(bs, be, device=self.device), th.full((be - bs,), sn, device=self.device)], dim=-1)
            )

        self.step_num += 1
        self.current_size = min(self.current_size + self.num_envs, self.max_size)

        if self.step_num == self.horizon:
            self.step_num = 0
            self.batch_start_episode += self.num_envs
        if self.batch_start_episode == self.num_episodes:
            self.batch_start_episode = 0

    def sample(self, batch_size):
        idxs = th.randint(0, self.current_size, (batch_size,), device=self.device)
        ep_nums, step_nums = self.idx_to_coord[idxs].unbind(-1)
        ep = ep_nums[:, None]

        # steps after t that hold this episode's data: up to the horizon, or only up to step_num for the rows being filled
        limit = th.full_like(step_nums, self.horizon)
        bs, be = self.batch_start_episode, self.batch_start_episode + self.num_envs
        filling = (ep_nums >= bs) & (ep_nums < be) & (step_nums < self.step_num)
        limit = th.where(filling, th.full_like(step_nums, self.step_num), limit)

        i = th.arange(self.n_steps, device=self.device)
        steps = step_nums[:, None] + i
        valid = (steps < limit[:, None]).float()
        steps_c = steps.clamp(max=self.horizon - 1)

        rews = self.rews_buf[ep, steps_c] * valid
        terms = self.done_buf[ep, steps_c] * valid
        # a step counts if it is valid and no termination happened earlier in the window
        alive = th.cumprod(th.cat([th.ones_like(terms[:, :1]), 1 - terms[:, :-1]], dim=1), dim=1) * valid
        effective_n_steps = alive.sum(1).long()

        returns = (rews * alive * self.gamma ** i).sum(1)
        dones = (terms * alive).amax(1)
        last = step_nums + effective_n_steps - 1

        offs = th.arange(self.frame_stack, device=self.device)
        obs = {
            "rgb": self.pixel_obs_buf[ep, step_nums[:, None] + offs],
            "state": self.state_obs_buf[ep_nums, step_nums],
        }
        next_obs = {
            "rgb": self.pixel_obs_buf[ep, (last + 1)[:, None] + offs],
            "state": self.state_next_obs_buf[ep_nums, last],
        }

        return ImageReplayBufferSamples(
            observations=obs,
            actions=self.acts_buf[ep_nums, step_nums],
            next_observations=next_obs,
            dones=dones,
            rewards=returns,
            effective_n_steps=effective_n_steps,
        )

    def __len__(self):
        return self.current_size

    def state_dict(self, metadata=None):
        return {
            "pixel_obs_buf": self.pixel_obs_buf.cpu(),
            "state_obs_buf": self.state_obs_buf.cpu(),
            "state_next_obs_buf": self.state_next_obs_buf.cpu(),
            "acts_buf": self.acts_buf.cpu(),
            "rews_buf": self.rews_buf.cpu(),
            "done_buf": self.done_buf.cpu(),
            "idx_to_coord": self.idx_to_coord.cpu(),
            "max_size": self.max_size,
            "horizon": self.horizon,
            "num_envs": self.num_envs,
            "frame_stack": self.frame_stack,
            "current_size": self.current_size,
            "batch_start_episode": self.batch_start_episode,
            "step_num": self.step_num,
            "metadata": metadata or {},
        }

    @classmethod
    def from_state_dict(cls, d, n_steps, gamma, device="cuda"):
        frame_shape = tuple(d["pixel_obs_buf"].shape[2:])
        buf = cls(
            pixels_obs_space=(d["frame_stack"], *frame_shape),
            state_obs_dim=d["state_obs_buf"].shape[-1],
            act_dim=d["acts_buf"].shape[-1],
            size=d["max_size"],
            horizon=d["horizon"],
            num_envs=d["num_envs"],
            n_frame_stack=d["frame_stack"],
            n_steps=n_steps,
            gamma=gamma,
            device=device,
        )
        for k in ("pixel_obs_buf", "state_obs_buf", "state_next_obs_buf", "acts_buf", "rews_buf", "done_buf", "idx_to_coord"):
            getattr(buf, k).copy_(d[k])
        buf.current_size = d["current_size"]
        buf.batch_start_episode = d["batch_start_episode"]
        buf.step_num = d["step_num"]
        buf.metadata = d.get("metadata", {})
        return buf
