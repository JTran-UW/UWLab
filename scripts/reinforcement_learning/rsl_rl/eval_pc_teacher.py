# Copyright (c) 2024-2025, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a JIT ScenePC teacher on the PCTeacher-FinetuneEval environment.

Usage::

    python scripts/reinforcement_learning/rsl_rl/eval_pc_teacher.py \\
        --teacher teachers/seed22_sysidenv.pt \\
        --num_envs 32 --num_episodes 200 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PCTeacher-FinetuneEval-v0"

parser = argparse.ArgumentParser(description="Evaluate a JIT PC teacher.")
parser.add_argument("--teacher", type=str, required=True, help="Path to JIT teacher .pt file.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=200, help="Stop after this many completed episodes.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# clear hydra args
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is up."""

import os
import sys as _sys
import numpy as np
# Force unbuffered stdout so debug prints survive a C++ crash
_sys.stdout.reconfigure(line_buffering=True)

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rgb_dagger_cfg import (
    Ur5eRobotiq2f85PCTeacherFinetuneEvalCfg,
    DebugEvalCfg
)


def main():
    print("[EVAL] Building env config", flush=True)
    env_cfg = Ur5eRobotiq2f85PCTeacherFinetuneEvalCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    print("[EVAL] gym.make ...", flush=True)
    env = gym.make(TASK, cfg=env_cfg)
    print("[EVAL] gym.make done — wrapping", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    print("[EVAL] wrap done (reset executed)", flush=True)

    device = env.unwrapped.device
    teacher_path = os.path.abspath(args_cli.teacher)
    print(f"[EVAL] Loading teacher from: {teacher_path}", flush=True)
    teacher = torch.jit.load(teacher_path, map_location=device)
    teacher.eval()
    print("[EVAL] Teacher loaded", flush=True)

    print("[EVAL] Getting initial obs", flush=True)
    obs_td = env.get_observations()
    print(f"[EVAL] obs keys: {list(obs_td.keys())}", flush=True)
    obs = obs_td["policy"]
    print(f"[EVAL] obs shape: {obs.shape}", flush=True)

    num_episodes = 0
    num_successes = 0

    # per-env accumulators for debug joint_pos (12-dim)
    num_envs = env.unwrapped.num_envs
    current_eps = [[] for _ in range(num_envs)]
    all_episodes = []   # list of (T, 12) arrays

    out_path = os.path.join(os.path.dirname(os.path.abspath(args_cli.teacher)),
                            "debug_joint_pos_episodes.npy")

    while simulation_app.is_running():
        with torch.inference_mode():
            out = teacher(obs)
            actions = out[0] if isinstance(out, (tuple, list)) else out
            obs_td, rewards, dones, _ = env.step(actions)
            obs = obs_td["policy"]

            # debug: (num_envs, 12) joint_pos
            joint_pos = obs_td["debug"].cpu().numpy()  # (num_envs, 12)
            print(joint_pos)
            for i in range(num_envs):
                current_eps[i].append(joint_pos[i].copy())

            if dones.any():
                for i in range(num_envs):
                    if dones[i]:
                        all_episodes.append(np.array(current_eps[i]))  # (T, 12)
                        current_eps[i] = []
                num_episodes += int(dones.sum().item())
                num_successes += int(torch.logical_and(rewards > 0.1, dones).sum().item())

        if num_episodes >= args_cli.num_episodes:
            break

    np.save(out_path, np.array(all_episodes, dtype=object))
    print(f"Saved {len(all_episodes)} episodes of debug joint_pos to: {out_path}")
    print(f"Episodes : {num_episodes}")
    print(f"Successes: {num_successes}")
    if num_episodes:
        print(f"Success rate: {num_successes / num_episodes:.2%}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
