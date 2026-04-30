#!/usr/bin/env python3
# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare two replay buffers (e.g. PPO reaching vs FastSAC reaching)."""

import argparse
import torch
from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer


def load_into_simple_replay_buffer(path: str, device: str = "cpu") -> SimpleReplayBuffer:
    payload = torch.load(path, map_location=device, weights_only=False)
    tensors = payload["buffer_tensors"]
    meta    = payload["metadata"]

    rb = SimpleReplayBuffer(
        n_env=meta["n_env"],
        buffer_size=meta["buffer_size"],
        n_obs=meta["n_obs"],
        n_act=meta["n_act"],
        n_critic_obs=meta["n_critic_obs"],
        n_steps=1,
        gamma=0.99,
        device=device,
    )
    rb.observations.copy_(tensors["observations"])
    rb.actions.copy_(tensors["actions"])
    rb.rewards.copy_(tensors["rewards"])
    rb.dones.copy_(tensors["dones"])
    rb.truncations.copy_(tensors["truncations"])
    rb.next_observations.copy_(tensors["next_observations"])
    rb.critic_observations.copy_(tensors["critic_observations"])
    rb.next_critic_observations.copy_(tensors["next_critic_observations"])
    rb.ptr = int(tensors["ptr"])
    return rb


def main():
    parser = argparse.ArgumentParser(description="Compare two replay buffers.")
    parser.add_argument("ppo_buffer", type=str, help="Path to PPO replay buffer .pt file.")
    parser.add_argument("fastsac_buffer", type=str, help="Path to FastSAC replay buffer .pt file.")
    args = parser.parse_args()

    print(f"Loading PPO buffer:     {args.ppo_buffer}")
    ppo_rb = load_into_simple_replay_buffer(args.ppo_buffer)
    print(f"  n_env={ppo_rb.n_env}  buffer_size={ppo_rb.buffer_size}  n_obs={ppo_rb.n_obs}  n_act={ppo_rb.n_act}  n_critic_obs={ppo_rb.n_critic_obs}")

    print(f"Loading FastSAC buffer: {args.fastsac_buffer}")
    fastsac_rb = load_into_simple_replay_buffer(args.fastsac_buffer)
    print(f"  n_env={fastsac_rb.n_env}  buffer_size={fastsac_rb.buffer_size}  n_obs={fastsac_rb.n_obs}  n_act={fastsac_rb.n_act}  n_critic_obs={fastsac_rb.n_critic_obs}")

    # TODO: compare


if __name__ == "__main__":
    main()
