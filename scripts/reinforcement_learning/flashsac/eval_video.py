# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One deterministic episode per env of an rsl_rl FlashSAC-RGB checkpoint, played back to back: an overview camera on the
left, the policy's own camera inputs on the right, and a title card per episode."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sequential eval video for an rsl_rl FlashSAC-RGB checkpoint.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-v0")
parser.add_argument("--agent", type=str, default="flashsac_rgb_4096_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--expert", action="store_true", help="--checkpoint is the rsl_rl PPO expert (rsl_rl_cfg_entry_point); actions replicate expert-buffer collection: clamp(a / bound, -1, 1).")
parser.add_argument("--out_prefix", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--no_video", action="store_true", help="Only write the per-episode results .json (no frames).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
from rsl_rl_flashsac.runners import OffPolicyRunner
import importlib.metadata as metadata
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_rl.flashsac import make_flashsac_env
from uwlab_rl.flashsac.rgb import _PixelPolicy
from uwlab_tasks.utils.hydra import hydra_task_config

OV_H, OV_W = 360, 480


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.scene.overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/overview_camera", update_period=0, height=OV_H, width=OV_W,
        offset=CameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344), rot=(0.46613815, 0.25072644, 0.47107948, 0.70564552), convention="opengl"
        ),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
        renderer_cfg=(env_cfg.scene.front_camera or env_cfg.scene.wrist_camera).renderer_cfg,
    )
    image_term = getattr(getattr(env_cfg.observations, "image", None), "image", None)
    cam_names = [c.name for c in image_term.params["sensor_cfgs"]] if image_term is not None else ["front_camera", "wrist_camera"]
    # eval only: no expert buffer, a token replay buffer
    agent_cfg.algorithm.expert_buffer_path = ""
    agent_cfg.algorithm.replay_buffer_size = 4 * args_cli.num_envs
    agent_cfg.algorithm.buffer_min_length = args_cli.num_envs
    env = make_flashsac_env(
        gym.make(args_cli.task, cfg=env_cfg), FlashSACVecEnvWrapper, agent_cfg.clip_actions, agent_cfg.action_bound,
        agent_cfg.action_bound_scale,
    )
    if args_cli.expert:
        expert_cfg = handle_deprecated_rsl_rl_cfg(
            load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
        )
        runner = OnPolicyRunner(env, expert_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        expert = runner.get_inference_policy(device=agent_cfg.device)

        def policy(o):
            return (expert(o) / env.action_scale).clamp(-1.0, 1.0)
    else:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
        actor = runner.alg.get_policy().eval()
        # the wrapper expects [-1, 1] and applies bias + scale * a; FlashSACActor already returns that,
        # the pixel agent's _PixelPolicy returns env units
        env_units = isinstance(actor, _PixelPolicy)

        def policy(o):
            a = actor(o)
            return (a - env.action_bias) / env.action_scale if env_units else a

    base = env.unwrapped
    n, horizon = base.num_envs, int(base.max_episode_length)
    cmd = base.command_manager.get_term("task_command")
    overview = base.scene.sensors["overview_camera"]
    reset_cfg = base.event_manager.get_term_cfg("reset_from_reset_states")
    obs = env.get_observations()
    names = [rt.replace("Object", "").replace("EEGrasped", "") for rt in reset_cfg.params["reset_types"]]
    task_id = reset_cfg.func.task_id.cpu().numpy().copy()

    image_group = getattr(agent_cfg.algorithm, "image_group", None)
    policy_cams = [base.scene.sensors[k] for k in cam_names]

    def policy_views(o):
        """(N, cams, H, W, 3) uint8: the pixel agent's own image obs, else the rendered RGB (e.g. DINO input)."""
        if image_group is not None and image_group in o.keys():
            return o[image_group].to(torch.uint8).unsqueeze(-1).expand(-1, -1, -1, -1, 3).cpu().numpy()
        return torch.stack([c.data.output["rgb"][..., :3].to(torch.uint8) for c in policy_cams], dim=1).cpu().numpy()
    succ_t = np.full(n, -1)
    done_once = torch.zeros(n, dtype=torch.bool, device=base.device)
    ov_frames, cam_frames, live = [], [], []
    with torch.no_grad():
        for t in range(horizon):
            if not args_cli.no_video:
                cam_frames.append(policy_views(obs))
            obs, _, dones, _ = env.step(policy(obs))
            # envs that terminate on this step were already auto-reset: their state and frame belong to the next episode
            live_now = ~(done_once | dones.bool())
            live.append(live_now.cpu().numpy())
            ov = overview.data.output["rgb"]
            ov_frames.append(None if args_cli.no_video else (ov.torch if hasattr(ov, "torch") else ov)[..., :3].to(torch.uint8).cpu().numpy())
            aligned = (cmd.orientation_aligned & cmd.position_aligned & live_now).cpu().numpy()
            succ_t[(succ_t < 0) & aligned] = t
            done_once |= dones.bool()
            if bool(done_once.all()):
                break
    steps = len(ov_frames)
    success = succ_t >= 0
    for k, rt in enumerate(names):
        m = task_id == k
        if m.any():
            print(f"[eval_video] {rt}: {success[m].sum()}/{m.sum()} succeeded", flush=True)
    print(f"[eval_video] overall {success.sum()}/{n} over {steps} steps", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_prefix)), exist_ok=True)
    json.dump({"checkpoint": args_cli.checkpoint, "steps": steps, "success": int(success.sum()), "episodes": [
        {"env": i, "reset_type": names[task_id[i]], "success": bool(success[i]), "success_t": int(succ_t[i] + 1)}
        for i in range(n)]}, open(f"{args_cli.out_prefix}.json", "w"), indent=1)
    if args_cli.no_video:
        env.close()
        return
    cams = cam_frames[0].shape[1]
    # a fixed-width camera column (2 cameras fill it; 3 stack at OV_H / 3) so different camera sets stack aligned
    col_w = OV_H // 2
    cam_px = min(col_w, OV_H // cams)
    frames = []
    for i in range(n):
        title = Image.new("RGB", (OV_W + col_w, OV_H))
        d = ImageDraw.Draw(title)
        d.text((20, OV_H // 2 - 20), f"episode {i + 1}/{n}: reset {names[task_id[i]]}", fill=(255, 255, 255))
        d.text((20, OV_H // 2), f"success at t={succ_t[i] + 1}" if success[i] else "no success",
               fill=(0, 255, 0) if success[i] else (255, 80, 80))
        frames += [np.asarray(title)] * 10
        last = 0
        for t in range(steps):
            # an env that terminated early auto-resets; hold its last live frame
            last = t if live[t][i] else last
            views = [np.asarray(Image.fromarray(cam_frames[last][i, c]).resize((cam_px, cam_px), Image.NEAREST))
                     for c in range(cams)]
            column = np.zeros((OV_H, col_w, 3), dtype=np.uint8)
            stack = np.concatenate(views, axis=0)[:OV_H]
            x0 = (col_w - cam_px) // 2
            column[: stack.shape[0], x0: x0 + cam_px] = stack
            frame = Image.fromarray(np.concatenate([ov_frames[last][i], column], axis=1))
            d = ImageDraw.Draw(frame)
            d.text((6, 6), f"episode {i + 1}/{n}  {names[task_id[i]]}  t={t + 1}", fill=(255, 255, 0))
            if 0 <= succ_t[i] <= t:
                d.text((6, 20), f"SUCCESS @ t={succ_t[i] + 1}", fill=(0, 255, 0))
            for c, label in enumerate(cam_names[:cams]):
                d.text((OV_W + x0 + 4, c * cam_px + 4), f"{label.replace('_camera', '')} (policy)", fill=(255, 255, 0))
            frames.append(np.asarray(frame))
    imageio.mimwrite(f"{args_cli.out_prefix}.mp4", frames, fps=10, macro_block_size=1)
    print(f"[eval_video] wrote {args_cli.out_prefix}.mp4", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
