# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One deterministic episode on many envs; write videos of the first N successful and N failed episodes, played back to back."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FlashSAC-RGB success/failure eval videos.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out_prefix", type=str, required=True, help="Writes <prefix>_success.mp4 and <prefix>_failure.mp4.")
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--n_show", type=int, default=10)
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
from vision_env import UWLabVisionVecEnv  # noqa: E402

OV_H, OV_W = 360, 480  # overview left; policy cams stacked right, each OV_H // n_cams tall


@hydra_task_compose(args.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.scene.overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/overview_camera", update_period=0, height=OV_H, width=OV_W,
        offset=CameraCfg.OffsetCfg(pos=(1.0770121, -0.1679045, 0.4486344), rot=(0.46613815, 0.25072644, 0.47107948, 0.70564552), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=13.20), renderer_cfg=NewtonWarpRendererCfg(),
    )
    reset_types = env_cfg.events.reset_from_reset_states.params["reset_types"]
    envs = UWLabVisionVecEnv(gym.make(args.task, cfg=env_cfg), args.horizon, args.action_scale, "vision_state")
    device = envs.device
    act_shape = envs.single_action_space.shape
    actor = FlashSACActor(envs, input_dim=cfg["image_embed_dim"] + cfg["state_embed_dim"], action_scale=torch.ones(act_shape),
                          action_bias=torch.zeros(act_shape), use_tanh=True).to(device)
    actor.load_state_dict(ckpt["actor"])
    encoder = FlashSACEncoder(envs, image_embed_dim=cfg["image_embed_dim"], state_embed_dim=cfg["state_embed_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.eval()
    print(f"[split_video] {args.checkpoint} (step {ckpt.get('global_step')})", flush=True)

    overview = envs.env.scene.sensors["overview_camera"]
    obs, _ = envs.reset()
    task_id = envs.ep_task_id.cpu().numpy()
    ov_frames, cam_frames, succ_t = [], [], np.full(args.num_envs, -1)
    with torch.no_grad():
        for t in range(args.horizon):
            actions = actor.get_action(encoder(obs), training=False)[2]
            obs_next, _, _, _, infos = envs.step(actions)
            shown = infos["final_observation"] if "final_info" in infos else obs_next
            ov = overview.data.output["rgb"]
            ov_frames.append((ov.torch if hasattr(ov, "torch") else ov)[..., :3].to(torch.uint8).cpu().numpy())
            cam_frames.append(shown["rgb"][:, 0].cpu().numpy())
            once = (infos["final_info"]["episode"]["success_once"].bool() if "final_info" in infos else envs.ep_success_once).cpu().numpy()
            succ_t[(succ_t < 0) & once] = t
            obs = obs_next
    success = succ_t >= 0
    names = [rt.replace("Object", "").replace("EEGrasped", "") for rt in reset_types]
    for k, rt in enumerate(reset_types):
        m = task_id == k
        if m.any():
            print(f"[split_video] {rt}: {success[m].sum()}/{m.sum()} succeeded", flush=True)
    print(f"[split_video] overall {success.mean():.3f} over {args.num_envs} envs", flush=True)

    for label, pick in (("success", np.flatnonzero(success)), ("failure", np.flatnonzero(~success))):
        pick = pick[: args.n_show]
        if len(pick) == 0:
            print(f"[split_video] no {label} episodes", flush=True)
            continue
        cam_keys = envs.image_keys
        CAM = OV_H // len(cam_keys)
        frames = []
        for n, i in enumerate(pick):
            title = Image.new("RGB", (OV_W + CAM, OV_H))
            d = ImageDraw.Draw(title)
            d.text((20, OV_H // 2 - 20), f"{label} {n + 1}/{len(pick)}: env {i}, reset {names[task_id[i]]}", fill=(255, 255, 255))
            if success[i]:
                d.text((20, OV_H // 2), f"success at t={succ_t[i] + 1}", fill=(0, 255, 0))
            frames += [np.asarray(title)] * 10
            for t in range(args.horizon):
                cams = [np.repeat(np.asarray(Image.fromarray(cam_frames[t][i, ..., c]).resize((CAM, CAM), Image.NEAREST))[..., None], 3, -1)
                        for c in range(cam_frames[t].shape[-1])]
                frame = Image.fromarray(np.concatenate([ov_frames[t][i], np.concatenate(cams, axis=0)[:OV_H]], axis=1))
                d = ImageDraw.Draw(frame)
                d.text((6, 6), f"{label} {n + 1}/{len(pick)}  env {i}  {names[task_id[i]]}  t={t + 1}", fill=(255, 255, 0))
                if 0 <= succ_t[i] <= t:
                    d.text((6, 20), f"SUCCESS @ t={succ_t[i] + 1}", fill=(0, 255, 0))
                for c, key in enumerate(cam_keys):
                    d.text((OV_W + 4, c * CAM + 4), f"{key} (policy)", fill=(255, 255, 0))
                frames.append(np.asarray(frame))
        out = f"{args.out_prefix}_{label}.mp4"
        imageio.mimwrite(out, frames, fps=10, macro_block_size=1)
        print(f"[split_video] wrote {out}: envs {pick.tolist()} types {[names[task_id[i]] for i in pick]}", flush=True)
    envs.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
