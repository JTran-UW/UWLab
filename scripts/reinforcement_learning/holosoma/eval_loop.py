# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("remote_ckpt_path", type=str, help="Path to checkpoint directory.")
parser.add_argument("--local_ckpt_path", type=str, default=f"./checkpoints/eval_loop/{time.time()}")
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
import subprocess
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

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config


from pathlib import Path
def get_latest_checkpoint(n_curr):
    n_latest_ckpt = n_curr

    while n_latest_ckpt <= n_curr:
        print("[INFO] Polling for the latest checkpoint...")
        time.sleep(5)
        out = subprocess.check_output(
            ["ssh", "klone-login",
            f"ls {args_cli.remote_ckpt_path}/model_*.pt 2>/dev/null | sed -n 's/.*model_\\([0-9]\\+\\)\\.pt$/\\1/p' | sort -n | tail -1"],
            text=True,
        ).strip()
        n_latest_ckpt = int(out) if out else n_curr
    
    print(f"[INFO] Found a new checkpoint at step {n_latest_ckpt}, rsyncing...")
    subprocess.run([
        "rsync", "-avzP",
        "--include=model_*.pt",
        "--exclude=*",
        f"klone-login:{args_cli.remote_ckpt_path}",
        args_cli.local_ckpt_path,
    ], check=True)
    return n_latest_ckpt, f"{args_cli.local_ckpt_path}/model_{n_latest_ckpt}.pt"

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Instantiate the task
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Check for the path
    ckpt_path = args_cli.remote_ckpt_path
    exists = subprocess.run(["ssh", "klone-login", f"test -d {ckpt_path}"]).returncode == 0
    while not exists:
        print(f"[ERROR] Could not find directory {ckpt_path}.")
        time.sleep(5)
        exists = subprocess.run(["ssh", "klone-login", f"test -d {ckpt_path}"]).returncode == 0
    

    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    n_curr_ckpt = 0
    n_curr_ckpt, latest_ckpt_path = get_latest_checkpoint(n_curr_ckpt)

    # create runner and load checkpoint
    print(f"[INFO]: Loading initial model checkpoint from: {latest_ckpt_path}")
    runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.setup()
    runner.load(latest_ckpt_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    actor_obs_keys = agent_cfg.actor_obs_keys
    critic_obs_keys = agent_cfg.critic_obs_keys
    dt = env.unwrapped.step_dt

    while True:
        # Look for the latest checkpoint
        n_curr_ckpt, latest_ckpt_path = get_latest_checkpoint(n_curr_ckpt)
        print(f"Loading latest checkpoint: {latest_ckpt_path}")

        env.reset()
        



    # # reset environment
    # obs = env.get_observations()

    # timestep = 0
    # # simulate environment
    # while simulation_app.is_running():
    #     start_time = time.time()
    #     # run everything in inference mode
    #     with torch.inference_mode():
    #         # FastSACAgent policy expects {"actor_obs": tensor}
    #         actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
    #         actions = policy({"actor_obs": actor_obs})
    #         obs, _, dones, _ = env.step(actions)
        
    #     if args_cli.video:
    #         timestep += 1
    #         # Exit the play loop after recording one video
    #         if timestep == args_cli.video_length:
    #             break
        
    # # close the simulator
    # env.close()




if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
