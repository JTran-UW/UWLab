# Copyright (c) 2024-2025, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect ~N episodes of sim gripper joint poses from the debug obs group.

Saves: /tmp/sim_gripper_episodes.npy  — list of (T_i, 6) arrays, one per episode.

Usage::

    python scripts/reinforcement_learning/rsl_rl/collect_sim_gripper.py \
        --teacher teachers/seed22_sysidenv.pt \
        --num_episodes 6 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PCTeacher-FinetuneEval-v0"

parser = argparse.ArgumentParser()
parser.add_argument("--teacher", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import uwlab_tasks     # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rgb_dagger_cfg import (
    Ur5eRobotiq2f85PCTeacherFinetuneEvalCfg,
)


def main():
    env_cfg = Ur5eRobotiq2f85PCTeacherFinetuneEvalCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed

    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    device = env.unwrapped.device
    teacher = torch.jit.load(os.path.abspath(args_cli.teacher), map_location=device)
    teacher.eval()

    obs_td = env.get_observations()
    obs = obs_td["policy"]

    episodes_done = 0
    current_ep = []        # list of (6,) gripper joint arrays for current episode
    all_episodes = []      # list of completed episode arrays

    print(f"[collect] Running until {args_cli.num_episodes} episodes complete...")
    while simulation_app.is_running():
        with torch.inference_mode():
            out = teacher(obs)
            actions = out[0] if isinstance(out, (tuple, list)) else out
            obs_td, rewards, dones, _ = env.step(actions)
            obs = obs_td["policy"]

            # debug: (1, 12) joint_pos; last 6 = gripper
            joint_pos = obs_td["debug"].cpu().numpy()[0]   # (12,)
            current_ep.append(joint_pos[6:].copy())        # (6,)

            if dones[0]:
                all_episodes.append(np.array(current_ep))  # (T, 6)
                print(f"[collect] Episode {episodes_done + 1} done — {len(current_ep)} steps")
                current_ep = []
                episodes_done += 1

        if episodes_done >= args_cli.num_episodes:
            break

    out_path = "/tmp/sim_gripper_episodes.npy"
    np.save(out_path, np.array(all_episodes, dtype=object))
    print(f"[collect] Saved {len(all_episodes)} episodes to {out_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
