# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic N-episode success benchmark of a flashsac_rgb.py checkpoint, overall and per reset type (95% CI)."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FlashSAC-RGB eval video.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=10, help="Episodes per env.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--horizon", type=int, default=160)
parser.add_argument("--action_scale", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args, remaining_args = parser.parse_known_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch



import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks.flashsac_actor import FlashSACActor  # noqa: E402
from networks.flashsac_utils import FlashSACEncoder  # noqa: E402
from vision_env import UWLabVisionVecEnv  # noqa: E402


@hydra_task_compose(args.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
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
    print(f"[eval] {args.checkpoint} (step {ckpt.get('global_step')}, {key}, state_group {cfg.get('state_group')})", flush=True)

    import math

    reset_types = env_cfg.events.reset_from_reset_states.params["reset_types"]
    succ, tids = [], []
    obs, _ = envs.reset()
    with torch.no_grad():
        while len(succ) < args.episodes:
            for _ in range(args.horizon):
                actions = actor.get_action(encoder(obs), training=False)[2]
                obs, _, _, _, infos = envs.step(actions)
            ep = infos["final_info"]["episode"]
            succ.append(ep["success_once"].cpu())
            tids.append(infos["final_info"]["task_id"].cpu())
            print(f"[eval] episode {len(succ)}/{args.episodes}: success_once {ep['success_once'].mean().item():.3f}", flush=True)
    s, t = torch.cat(succ), torch.cat(tids)
    n = len(s)
    p = s.mean().item()
    print(f"[eval] {args.checkpoint}: success_once {p:.3f} +- {1.96 * math.sqrt(p * (1 - p) / n):.3f} (95% CI) over {n} episodes", flush=True)
    for k, name in enumerate(reset_types):
        m = t == k
        if m.any():
            pk = s[m].mean().item()
            print(f"[eval]   {name:34s} {pk:.3f} +- {1.96 * math.sqrt(pk * (1 - pk) / int(m.sum())):.3f} (n={int(m.sum())})", flush=True)
    envs.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
