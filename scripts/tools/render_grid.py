# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact sheet of a camera task's policy images: one tile per env and camera after a few zero-action steps, to
judge a look or a visual randomization across envs."""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--columns", type=int, default=4, help="Envs per row (each env shows its cameras side by side).")
parser.add_argument("--scale", type=int, default=2, help="Integer upscaling of the tiles.")
parser.add_argument("--sheets", type=int, default=1, help="Number of sheets, `--steps` apart.")
parser.add_argument("--out", type=str, required=True, help="Output prefix: <out>_<k>.png and <out>.npz.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "flashsac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    cameras = {n: s for n, s in base.scene.sensors.items() if "rgb" in getattr(s.data, "output", {})}
    zero = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
    dump = {}
    for sheet in range(args_cli.sheets):
        start = time.perf_counter()
        for _ in range(args_cli.steps):
            env.step(zero)
        step_ms = 1000.0 * (time.perf_counter() - start) / max(args_cli.steps, 1)
        images = []
        for name, sensor in cameras.items():
            rgb = sensor.data.output["rgb"]
            images.append((rgb.torch if hasattr(rgb, "torch") else rgb)[..., :3].to(torch.uint8).cpu().numpy())
            dump[f"{name}_{sheet}"] = images[-1]
        tiles = np.concatenate(images, axis=2)  # (env, H, cameras * W, 3)
        tiles = tiles.repeat(args_cli.scale, axis=1).repeat(args_cli.scale, axis=2)
        rows = [
            np.concatenate(list(tiles[i : i + args_cli.columns]), axis=1)
            for i in range(0, len(tiles) - args_cli.columns + 1, args_cli.columns)
        ]
        Image.fromarray(np.concatenate(rows, axis=0)).save(f"{args_cli.out}_{sheet}.png")
        stats = " ".join(f"{n}: mean {dump[f'{n}_{sheet}'].mean():.1f}" for n in cameras)
        print(f"[render_grid] sheet {sheet}: cameras {list(cameras)} | {stats} | {step_ms:.1f} ms/env step", flush=True)
    np.savez_compressed(f"{args_cli.out}.npz", **dump)
    print(f"[render_grid] wrote {args_cli.out}_*.png", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
