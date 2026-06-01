# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--expert_checkpoint", type=str, default=None, help="Path to an RSL-RL (PPO) checkpoint .pt file. If set, loads an OnPolicyRunner from that checkpoint and passes its policy and critic to FastSACAgent as BC regularization targets.")
parser.add_argument("--fastsac_expert_checkpoint", type=str, default=None, help="Path to a FastSAC checkpoint .pt file to use as expert Q function for critic comparison.")
parser.add_argument("--plot_rewards", action="store_true", default=False, help="Plot individual reward components over time.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import tqdm
from tensordict import TensorDict

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from uwlab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx
from vecenv_wrapper import HolosomaVecEnvWrapper
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.agents.rsl_rl_cfg import Base_PPORunnerCfg

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create runner and load checkpoint
    runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.setup()
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    critic = runner.get_inference_critic(device=env.unwrapped.device)
    actor_obs_keys = agent_cfg.actor_obs_keys
    critic_obs_keys = agent_cfg.critic_obs_keys

    expert_policy = None
    expert_critic = None
    expert_is_fastsac = False

    if args_cli.fastsac_expert_checkpoint:
        fastsac_expert_resume_path = retrieve_file_path(args_cli.fastsac_expert_checkpoint)
        fastsac_expert_runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        fastsac_expert_runner.setup()
        fastsac_expert_runner.load(fastsac_expert_resume_path)
        expert_policy = fastsac_expert_runner.get_inference_policy(device=env.unwrapped.device)
        expert_critic = fastsac_expert_runner.get_inference_critic(device=env.unwrapped.device)
        expert_is_fastsac = True
        print(f"[INFO] Loaded FastSAC expert checkpoint from: {fastsac_expert_resume_path}")
    elif args_cli.expert_checkpoint:
        expert_rsl_env = RslRlVecEnvWrapper(env.env, clip_actions=agent_cfg.clip_actions)
        expert_ppo_cfg = cli_args.sanitize_rsl_rl_cfg(Base_PPORunnerCfg())
        expert_resume_path = retrieve_file_path(args_cli.expert_checkpoint)
        expert_runner = OnPolicyRunner(expert_rsl_env, expert_ppo_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        expert_runner.load(expert_resume_path)
        expert_policy = expert_runner.get_inference_policy(device=agent_cfg.device)
        expert_critic = expert_runner.get_inference_critic(device=agent_cfg.device)

    dt = env.unwrapped.step_dt

    # reset environment — use reset() so RecordVideo initializes its recorder
    obs, _ = env.reset()
    timestep = 0
    # simulate environment

    expert_critic_values = []
    critic_values = []
    true_rewards = []

    reward_components: dict[str, list[float]] = {}
    if args_cli.plot_rewards:
        rm = env.unwrapped.reward_manager
        for name in rm._term_names:
            reward_components[name] = []

    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=-1)
            if expert_is_fastsac:
                actions = expert_policy({"actor_obs": actor_obs})
            elif expert_policy is not None:
                actions = expert_policy(obs)
            else:
                actions = policy({"actor_obs": actor_obs})
            new_obs, rew, dones, extras = env.step(actions)

            # Get policy critic on current obs/actions
            critic_obs = torch.cat([obs[k] for k in critic_obs_keys], dim=-1)
            critic_output = critic(critic_obs, actions)
            critic_value = critic_output.mean()
            critic_values.append(critic_value.cpu().item())

            # Get expert critic value on current obs/actions (if available)
            if expert_critic is not None:
                expert_critic_obs = torch.cat([new_obs[k] for k in critic_obs_keys], dim=-1)
                if expert_is_fastsac:
                    expert_critic_output = expert_critic(expert_critic_obs, actions)
                else:
                    expert_critic_output = rew + 0.99 * expert_critic(expert_critic_obs)
                expert_critic_values.append(expert_critic_output.mean().cpu().item())

            true_rewards.append(rew.mean().cpu().item())

            if args_cli.plot_rewards:
                step_rew = rm._step_reward * dt
                for idx, name in enumerate(rm._term_names):
                    reward_components[name].append(step_rew[:, idx].mean().cpu().item())

            obs = new_obs

        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video; don't break on dones
            if timestep == args_cli.video_length:
                break
        elif dones.any():
            break

    # Monte Carlo discounted return: G_t = sum_{k=0}^{N-t-1} gamma^k * r_{t+k}
    GAMMA = 0.99
    mc_returns = [0.0] * len(true_rewards)
    running = 0.0
    for i in range(len(true_rewards) - 1, -1, -1):
        running = true_rewards[i] + GAMMA * running
        mc_returns[i] = running

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if expert_critic_values:
        ax.plot(expert_critic_values, label="Expert Critic")
    ax.plot(critic_values, label="Policy Critic")
    ax.plot(true_rewards, label="True Reward (mean across envs)", linestyle="--")
    ax.plot(mc_returns, label=f"MC Return (γ={GAMMA})", linestyle=":")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.legend()
    fig.savefig("critic_comparison.png")

    if args_cli.plot_rewards and reward_components:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for name, values in reward_components.items():
            ax2.plot(values, label=name)
        ax2.plot(true_rewards, label="Total Reward", linewidth=2, color="black", linestyle="--")
        ax2.set_xlabel("step")
        ax2.set_ylabel("reward")
        ax2.set_title("Reward Components Over Time")
        ax2.legend(fontsize="small", loc="best")
        fig2.tight_layout()
        fig2.savefig("reward_components.png")
        print(f"[INFO] Saved reward components plot to: reward_components.png")

        # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
