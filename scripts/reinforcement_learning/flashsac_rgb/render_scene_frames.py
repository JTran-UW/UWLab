# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render one frame of the vision task (overview + policy cameras, RGB and grayscale) to an .npz for comparisons."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render vision-task frames.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-v0")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=3, help="Zero-action steps after reset before capturing.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--no_overview", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, remaining_args = parser.parse_known_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import copy

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if not args.no_overview:
      env_cfg.scene.overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/overview_camera", update_period=0, height=360, width=480,
        offset=CameraCfg.OffsetCfg(pos=(1.0770121, -0.1679045, 0.4486344), rot=(0.46613815, 0.25072644, 0.47107948, 0.70564552), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=13.20), renderer_cfg=copy.deepcopy(env_cfg.scene.front_camera.renderer_cfg),
    )
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.steps):
        obs, *_ = env.step(zero)
    out = {}
    for name, sensor in env.scene.sensors.items():
        if "rgb" in getattr(sensor.data, "output", {}):
            rgb = sensor.data.output["rgb"]
            out[f"rgb_{name}"] = (rgb.torch if hasattr(rgb, "torch") else rgb)[..., :3].to(torch.uint8).cpu().numpy()
    for key, img in obs["camera"].items():
        out[f"gray_{key}"] = (img * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()[..., 0]
    np.savez_compressed(args.out, **out)
    print(f"[render] wrote {args.out}: " + ", ".join(f"{k} {v.shape}" for k, v in out.items()), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
