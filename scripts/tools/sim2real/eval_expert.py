# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quick eval of a JIT expert in an env: roll it out and report success rate.

Feeds the expert the concatenation of the named observation groups (in order),
steps the env with the expert's action mean, and logs the episode success rate
(``progress_context.success``). Use it to verify an expert actually solves a task
in a given env before trusting it for data collection.

Example (expert's native ZeroG env)::

    ./_isaaclab/IsaacLab/isaaclab.sh -p scripts/tools/sim2real/eval_expert.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-FullDR-v0 \\
        --obs_groups proprio pointcloud \\
        --expert teachers/patrick_jit_expert.pt --num_envs 32 --max_iters 700 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval a JIT expert's success rate in an env.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--expert", type=str, default="teachers/patrick_jit_expert.pt")
parser.add_argument("--obs_groups", type=str, nargs="+", default=["policy"],
                    help="Obs group(s) concatenated (in order) to form the expert input.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_iters", type=int, default=700)
parser.add_argument("--log_every", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401


def main():
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    device = env.unwrapped.device

    expert = torch.jit.load(args_cli.expert, map_location=device).eval()
    for p in expert.parameters():
        p.requires_grad_(False)
    success_ref = env.unwrapped.reward_manager.get_term_cfg("progress_context").func.success

    obs, _ = env.reset()
    dims = {g: tuple(v.shape) for g, v in obs.items()}
    print(f"[eval] obs group dims: {dims}", flush=True)
    teacher_in = torch.cat([obs[g] for g in args_cli.obs_groups], dim=-1)
    print(f"[eval] expert input from groups {args_cli.obs_groups} -> dim {teacher_in.shape[-1]}", flush=True)

    n_att = 0
    n_succ = 0
    last_log = 0
    for it in range(args_cli.max_iters):
        teacher_in = torch.cat([obs[g] for g in args_cli.obs_groups], dim=-1)
        with torch.no_grad():
            out = expert(teacher_in)
        action = out[0] if isinstance(out, (tuple, list)) else out
        obs, _, terminated, truncated, _ = env.step(action)
        done = (terminated | truncated)
        succ = success_ref
        nd = int(done.sum())
        if nd > 0:
            n_att += nd
            n_succ += int((succ & done).sum())
            if n_att - last_log >= args_cli.log_every:
                last_log = n_att
                print(f"[eval] episodes={n_att}  success={n_succ}/{n_att} "
                      f"({n_succ / max(n_att, 1) * 100:.1f}%)", flush=True)
    print(f"[eval] FINAL: expert success {n_succ}/{n_att} ({n_succ / max(n_att, 1) * 100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
