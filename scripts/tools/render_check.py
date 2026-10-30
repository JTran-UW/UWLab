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
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--policy_obs", action="store_true", help="Also save env 0's policy 'image' obs (per camera).")
parser.add_argument("--dump_obs", type=str, default="", help="Save env 0's policy image obs at every step to this .npy (needs --policy_obs).")
parser.add_argument("--count_black", action="store_true", help="Report, per camera, how many envs have a black (mean < 5) policy image.")
parser.add_argument("--rtx_aa", type=str, default="", help="Override RTX anti-aliasing after startup (Off, FXAA, DLSS, TAA, DLAA).")
parser.add_argument("--no_dlssg", action="store_true", help="Disable DLSS frame generation after startup.")
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
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    try:
        import carb

        cs = carb.settings.get_settings()
        if args_cli.rtx_aa:
            import omni.replicator.core as rep

            rep.settings.set_render_rtx_realtime(antialiasing=args_cli.rtx_aa)
        if args_cli.no_dlssg:
            cs.set("/rtx-transient/dlssg/enabled", False)
        keys = ("/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx-transient/dlssg/enabled", "/rtx/rendermode")
        print("[render_check] rtx settings " + " ".join(f"{k}={cs.get(k)}" for k in keys), flush=True)
    except ImportError:  # kit-less renderer
        pass
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    strip = []
    dump = []
    for k in range(args_cli.steps):
        env.step(zero)
        if args_cli.dump_obs:
            dump.append(env.observation_manager.compute_group("image")[0].clamp(0, 255).byte().cpu().numpy())
        if args_cli.every and k % args_cli.every == 0:
            sensor = next(s for s in env.scene.sensors.values() if "rgb" in getattr(s.data, "output", {}))
            strip.append(sensor.data.output["rgb"][0, ..., :3].clamp(0, 255).byte().cpu().numpy())
    if dump:
        np.save(args_cli.dump_obs, np.stack(dump))
    if strip:
        Image.fromarray(np.concatenate(strip, axis=1)).save(args_cli.out.replace(".png", "_strip.png"))
    if args_cli.count_black:
        img = env.observation_manager.compute_group("image").float()  # (N, C, H, W)
        names = [c.name for c in env.cfg.observations.image.image.params["sensor_cfgs"]]
        per = img.shape[1] // len(names)
        for i, name in enumerate(names):
            black = img[:, i * per:(i + 1) * per].mean((1, 2, 3)) < 5
            ids = torch.nonzero(black).view(-1).tolist()
            print(f"[render_check] {name}: black in {len(ids)}/{img.shape[0]} envs; first ids {ids[:12]}", flush=True)
    if args_cli.policy_obs:
        img = env.observation_manager.compute_group("image")[0]  # (C, H, W) in 0..255
        names = [c.name for c in env.cfg.observations.image.image.params["sensor_cfgs"]]
        per = img.shape[0] // len(names)
        views = [img[i * per:(i + 1) * per].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy() for i in range(len(names))]
        views = [np.repeat(v, 3, axis=-1) if v.shape[-1] == 1 else v for v in views]
        Image.fromarray(np.concatenate(views, axis=1)).save(args_cli.out.replace(".png", "_policy.png"))
        print(f"[render_check] policy obs {tuple(img.shape)} cameras {names}", flush=True)
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
