# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

from dataclasses import dataclass
from typing import List
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cleanrl_utils.utils import EmpiricalNormalization

from vecenv_wrapper import IsaacLabVectorEnv

@dataclass
class Args:
    ""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False"""
    env_id: str = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Reaching-OffPolicy-v0"
    """id of env"""
    num_envs: int = 1
    """number of envs"""
    num_episodes: int = 1
    """number of episodes"""
    cuda: bool = True
    """use or do not use cuda"""
    checkpoint: str = ""
    """checkpoint path"""
    algorithm: str = "PPO"
    """what algo to eval"""
    gamma: float = 0.99
    """discount factor for computing true returns"""


import sac_continuous_action as sac
import ppo_continuous_action as ppo


def load_actor_critic(envs, algo, checkpoint, device):
    checkpoint = torch.load(checkpoint, map_location=device)
    if algo == "SAC":
        actor_net = sac.Actor(envs).to(device)
        critic = sac.SoftQNetwork(envs).to(device)
        actor_net.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["qf1"])
        actor_net.eval()
        critic.eval()
        actor = lambda obs: actor_net.get_action(obs)[2]

        actor_obs_normalizer = EmpiricalNormalization(shape=envs.single_observation_space["policy"].shape, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=envs.single_observation_space["critic"].shape, device=device)
        actor_obs_normalizer.load_state_dict(checkpoint["actor_obs_normalizer"])
        critic_obs_normalizer.load_state_dict(checkpoint["critic_obs_normalizer"])
    elif algo == "PPO":
        agent = ppo.Agent(envs).to(device)
        agent.load_state_dict(checkpoint)
        actor = agent.actor_mean
        critic = agent.critic
        actor.eval()
        critic.eval()
        actor_obs_normalizer = None
        critic_obs_normalizer = None
    else:
        raise ValueError(f"Algorithm '{algo}' not found")

    return actor, critic, actor_obs_normalizer, critic_obs_normalizer


if __name__ == "__main__":

    args, launcher_args = tyro.cli(Args, return_unknown_args=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load checkpoints
    envs = IsaacLabVectorEnv(args.env_id, args.num_envs, launcher_args=launcher_args)
    actor, critic, actor_obs_normalizer, critic_obs_normalizer = load_actor_critic(envs, args.algorithm, checkpoint=args.checkpoint, device=device)
    
    obs, _ = envs.reset(seed=args.seed)
    ep = 0
    v_hist = []
    ret_hist = []
    rew_hist = []
    r_hist = []
    while ep < args.num_episodes:
        with torch.no_grad():
            policy_obs = obs["policy"]
            if actor_obs_normalizer:
                policy_obs = actor_obs_normalizer(policy_obs, update=False)
            actions = actor(policy_obs.to(device))
        
        # Eval the action
        with torch.no_grad():
            critic_obs = obs["critic"]
            if critic_obs_normalizer:
                critic_obs = critic_obs_normalizer(critic_obs, update=False)

            if args.algorithm == "SAC":
                values = critic(critic_obs, actions)
            elif args.algorithm == "PPO":
                values = critic(critic_obs)
            else:
                raise ValueError(f"Unknown algorithm '{args.algorithm}'")

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        done = torch.sum(terminations or truncations).item()
        r_hist.append(rewards.item())
        rew_hist.append(rewards.item())
        v_hist.append(values.item())
    
        # Compute true returns
        if done:
            tmp_ret_hist = []
            for i in reversed(range(len(r_hist))):
                ret = r_hist[i]
                if len(tmp_ret_hist) > 0:
                    ret += args.gamma * tmp_ret_hist[0]
                tmp_ret_hist.insert(0, ret)
            ret_hist += tmp_ret_hist
            r_hist = []
        
        obs = next_obs
        ep += done

    plt.plot(ret_hist, label="True return")
    plt.plot(rew_hist, label="True reward")
    plt.plot(v_hist, label="Value estimate")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    envs.close()
