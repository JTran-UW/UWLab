# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic eval video of a flashsac_rgb.py checkpoint: overview camera + the policy's own camera inputs."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FlashSAC-RGB eval video.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rows", type=int, default=4)
parser.add_argument("--horizon", type=int, default=160)
parser.add_argument("--action_scale", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args, remaining_args = parser.parse_known_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from isaaclab_newton.renderers import NewtonWarpRendererCfg

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks.flashsac_actor import FlashSACActor  # noqa: E402
from networks.flashsac_utils import FlashSACEncoder  # noqa: E402
from vision_env import IMAGE_KEYS, UWLabVisionVecEnv  # noqa: E402


@hydra_task_compose(args.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.scene.overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/overview_camera", update_period=0, height=240, width=320,
        offset=CameraCfg.OffsetCfg(pos=(1.0770121, -0.1679045, 0.4486344), rot=(0.46613815, 0.25072644, 0.47107948, 0.70564552), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=13.20), renderer_cfg=NewtonWarpRendererCfg(),
    )
    envs = UWLabVisionVecEnv(gym.make(args.task, cfg=env_cfg), args.horizon, args.action_scale, "vision_state")
    device = envs.device
    act_shape = envs.single_action_space.shape
    embed_dim = cfg["image_embed_dim"] + cfg["state_embed_dim"]
    actor = FlashSACActor(envs, input_dim=embed_dim, action_scale=torch.ones(act_shape), action_bias=torch.zeros(act_shape), use_tanh=True).to(device)
    actor.load_state_dict(ckpt["actor"])
    key = "actor_encoder" if "actor_encoder" in ckpt else "encoder"
    encoder = FlashSACEncoder(envs, image_embed_dim=cfg["image_embed_dim"], state_embed_dim=cfg["state_embed_dim"]).to(device)
    encoder.load_state_dict(ckpt[key])
    actor.eval()
    print(f"[eval_video] {args.checkpoint} (step {ckpt.get('global_step')}, {key}, state_group {cfg.get('state_group')})", flush=True)

    overview = envs.env.scene.sensors["overview_camera"]
    obs, _ = envs.reset()
    frames = []
    with torch.no_grad():
        for t in range(args.horizon):
            actions = actor.get_action(encoder(obs), training=False)[2]
            obs_next, _, _, _, infos = envs.step(actions)
            shown = infos["final_observation"] if "final_info" in infos else obs_next
            ov = overview.data.output["rgb"]
            ov = (ov.torch if hasattr(ov, "torch") else ov)[: args.rows, ..., :3].cpu().numpy().astype(np.uint8)
            rgb = shown["rgb"][: args.rows, 0].cpu().numpy()
            succ = (envs.ep_success_once if "final_info" not in infos else infos["final_info"]["episode"]["success_once"].bool())[: args.rows].tolist()
            rows = []
            for i in range(args.rows):
                tiles = [ov[i]] + [np.repeat(np.kron(rgb[i, ..., c], np.ones((4, 4), dtype=np.uint8))[..., None], 3, -1)[:240, :240] for c in range(len(IMAGE_KEYS))]
                row = Image.fromarray(np.concatenate(tiles, axis=1))
                ImageDraw.Draw(row).text((6, 6), f"env {i}  step {t + 1}  {'SUCCESS' if succ[i] else ''}", fill=(255, 40, 40))
                rows.append(np.asarray(row))
            frames.append(np.concatenate(rows, axis=0))
            if "final_info" in infos:
                ep = infos["final_info"]["episode"]
                print(f"[eval_video] episode success_once {ep['success_once'].mean().item():.3f} over {args.num_envs} envs; shown rows {ep['success_once'][: args.rows].tolist()}", flush=True)
            obs = obs_next
    imageio.mimwrite(args.out, frames, fps=10, macro_block_size=1)
    print(f"[eval_video] wrote {args.out} ({len(frames)} frames)", flush=True)
    envs.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
