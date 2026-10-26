# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer smoke test: step a camera task a few times, print per-camera image statistics and save the frames
(a blank renderer gives all-zero / constant images without raising)."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--out", type=str, required=True, help="Output .png.")
parser.add_argument("--every", type=int, default=0, help="Also save env 0's first camera every N steps as a strip.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os

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
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    strip = []
    for k in range(args_cli.steps):
        env.step(zero)
        if args_cli.every and k % args_cli.every == 0:
            sensor = next(s for s in env.scene.sensors.values() if "rgb" in getattr(s.data, "output", {}))
            strip.append(sensor.data.output["rgb"][0, ..., :3].clamp(0, 255).byte().cpu().numpy())
    if strip:
        Image.fromarray(np.concatenate(strip, axis=1)).save(args_cli.out.replace(".png", "_strip.png"))
    tiles = []
    for name, sensor in env.scene.sensors.items():
        if "rgb" not in getattr(sensor.data, "output", {}):
            continue
        rgb = sensor.data.output["rgb"][..., :3].float()
        print(f"[render_check] {name}: shape {tuple(rgb.shape)} mean {rgb.mean():.1f} std {rgb.std():.1f} "
              f"min {rgb.min():.0f} max {rgb.max():.0f}", flush=True)
        tiles.append(rgb[0].clamp(0, 255).byte().cpu().numpy())
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(args_cli.out)
    print(f"[render_check] saved {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
