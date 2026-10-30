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
parser.add_argument("--policy", type=str, default="mean", choices=["mean", "sample", "train_act"],
                    help="mean: deterministic tanh(mean); sample: fresh Gaussian noise each step; train_act: the "
                    "algorithm's training-rollout act() (compiled actor, temporally correlated noise).")
parser.add_argument("--video_envs", type=int, default=0, help="Film only the first K envs (0 = all); stats still cover every env.")
parser.add_argument("--overview_scale", type=float, default=1.0, help="Render the overview camera at this fraction of 480x360 (upscaled in the video); 0 = no overview camera (videos show the policy cameras only, e.g. to keep training-time render memory).")
parser.add_argument("--record_states", type=str, default="", help="Save the filmed envs' per-step robot joints and object poses (.pt).")
parser.add_argument("--replay_states", type=str, default="", help="Kinematically replay a --record_states file (no physics) and film it.")
parser.add_argument("--use_ema", action="store_true", help="Film the checkpoint's EMA policy (FlashSACRGB policy_ema).")
parser.add_argument("--warmup_renders", type=int, default=0, help="Render N frames before the first action (RTX loads materials late in the SIF).")
parser.add_argument("--rtx_aa", type=str, default="", help="Override RTX anti-aliasing after startup (Off, FXAA, DLSS, TAA, DLAA).")
parser.add_argument("--image_cams", type=str, default="", help="Comma list of cameras for the image obs (e.g. side_camera), so one camera can be rendered at a different resolution; only for --expert / viewing.")
parser.add_argument("--per_env_videos", type=str, default="", help="Write one mp4 per filmed env into this dir (plus index.csv) instead of one long video.")
parser.add_argument("--tf32", action="store_true", help="Use train.py's TF32 / matmul-precision settings.")
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
from isaaclab.managers import SceneEntityCfg
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
if args_cli.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.overview_scale > 0:
        env_cfg.scene.overview_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/overview_camera", update_period=0, height=int(OV_H * args_cli.overview_scale), width=int(OV_W * args_cli.overview_scale),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0770121, -0.1679045, 0.4486344), rot=(0.46613815, 0.25072644, 0.47107948, 0.70564552), convention="opengl"
            ),
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
            renderer_cfg=(env_cfg.scene.front_camera or env_cfg.scene.wrist_camera).renderer_cfg,
        )
    image_term = getattr(getattr(env_cfg.observations, "image", None), "image", None)
    if args_cli.image_cams and image_term is not None:
        image_term.params["sensor_cfgs"] = [SceneEntityCfg(c) for c in args_cli.image_cams.split(",")]
    dino_term = getattr(getattr(env_cfg.observations, "dino", None), "features", None)
    cam_term = image_term if image_term is not None else dino_term
    cam_names = [c.name for c in cam_term.params["sensor_cfgs"]] if cam_term is not None else ["front_camera", "wrist_camera"]
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
        if args_cli.use_ema:
            ckpt = torch.load(args_cli.checkpoint, map_location=agent_cfg.device, weights_only=False)
            runner.alg.actor.load_state_dict(ckpt["ema_actor_state_dict"])
            runner.alg.encoder.load_state_dict(ckpt["ema_encoder_state_dict"])
            print("[eval_video] filming the EMA policy", flush=True)
        actor = runner.alg.get_policy().eval()
        # the wrapper expects [-1, 1] and applies bias + scale * a; FlashSACActor already returns that,
        # the pixel agent's _PixelPolicy returns env units
        env_units = isinstance(actor, _PixelPolicy)

        def policy(o):
            if args_cli.policy == "train_act":
                return runner.alg.act(o)
            if args_cli.policy == "sample":
                if env_units:
                    mean, std = runner.alg.actor.get_mean_and_std(runner.alg._encode(o, runner.alg.encoder), training=False)
                    return torch.tanh(mean + std * torch.randn_like(mean))
                return actor(o, stochastic_output=True)
            a = actor(o)
            return (a - env.action_bias) / env.action_scale if env_units else a

    base = env.unwrapped
    if args_cli.rtx_aa:
        import omni.replicator.core as rep

        rep.settings.set_render_rtx_realtime(antialiasing=args_cli.rtx_aa)
    n, horizon = base.num_envs, int(base.max_episode_length)
    n_vid = min(args_cli.video_envs or n, n)
    cmd = base.command_manager.get_term("task_command")
    overview = base.scene.sensors["overview_camera"] if args_cli.overview_scale > 0 else None
    extra_cams = [overview] if overview is not None else []
    reset_cfg = base.event_manager.get_term_cfg("reset_from_reset_states")
    obs = env.get_observations()
    for _ in range(args_cli.warmup_renders):
        base.sim.render()
        for c in [base.scene.sensors[k] for k in cam_names] + extra_cams:
            c.update(base.physics_dt, force_recompute=True)
    if args_cli.warmup_renders:
        obs = env.get_observations()
    names = [rt.replace("Object", "").replace("EEGrasped", "") for rt in reset_cfg.params["reset_types"]]
    task_id = reset_cfg.func.task_id.cpu().numpy().copy()

    image_group = getattr(agent_cfg.algorithm, "image_group", None)
    policy_cams = [base.scene.sensors[k] for k in cam_names]

    def policy_views(o):
        """(N, cams, H, W, 3) uint8: the pixel agent's own image obs, else the rendered RGB (e.g. DINO input)."""
        if image_group is not None and image_group in o.keys():
            img = o[image_group][:n_vid].to(torch.uint8)
            if img.shape[1] == 3 * len(cam_names):  # RGB: 3 channels per camera
                return img.view(img.shape[0], len(cam_names), 3, *img.shape[2:]).permute(0, 1, 3, 4, 2).cpu().numpy()
            return img.unsqueeze(-1).expand(-1, -1, -1, -1, 3).cpu().numpy()
        return torch.stack([c.data.output["rgb"][:n_vid, ..., :3].to(torch.uint8) for c in policy_cams], dim=1).cpu().numpy()
    succ_t = np.full(n, -1)
    done_once = torch.zeros(n, dtype=torch.bool, device=base.device)
    ov_frames, cam_frames, live = [], [], []
    robot, peg, hole = (base.scene[k] for k in ("robot", "insertive_object", "receptive_object"))

    def _t(x):
        return x.torch if hasattr(x, "torch") else x

    def object_state():
        return {"joint_pos": _t(robot.data.joint_pos)[:n_vid].clone(), "peg": _t(peg.data.root_pose_w)[:n_vid].clone(),
                "hole": _t(hole.data.root_pose_w)[:n_vid].clone()}

    if args_cli.replay_states:
        rec = torch.load(args_cli.replay_states, map_location=base.device, weights_only=False)
        ids = torch.arange(n_vid, device=base.device)
        cams = policy_cams + extra_cams

        def show(st):
            robot.write_joint_state_to_sim(st["joint_pos"], torch.zeros_like(st["joint_pos"]), env_ids=ids)
            peg.write_root_pose_to_sim(st["peg"], env_ids=ids)
            hole.write_root_pose_to_sim(st["hole"], env_ids=ids)
            base.sim.forward()
            base.scene.update(0.0)
            base.sim.render()
            for c in cams:
                c.update(base.physics_dt, force_recompute=True)

        with torch.no_grad():
            for _ in range(30):  # let RTX finish loading materials before the first recorded frame
                show(rec["states"][0])
            for st in rec["states"]:
                show(st)
                cam_frames.append(policy_views({image_group: base.observation_manager.compute_group(image_group)}))
                ov_frames.append(_t(overview.data.output["rgb"])[:n_vid, ..., :3].to(torch.uint8).cpu().numpy() if overview is not None else None)
                live.append(np.ones(n, dtype=bool))
                if "interval" in base.event_manager.available_modes:  # texture / HDRI redraws on the recorded schedule
                    base.event_manager.apply(mode="interval", dt=base.step_dt)
        succ_t[:], task_id[:] = rec["succ_t"], rec["task_id"]
    recorded = []
    with torch.no_grad():
        for t in range(0 if args_cli.replay_states else horizon):
            if args_cli.record_states:
                recorded.append(object_state())
            if not args_cli.no_video:
                cam_frames.append(policy_views(obs))
            obs, _, dones, _ = env.step(policy(obs))
            # envs that terminate on this step were already auto-reset: their state and frame belong to the next episode
            live_now = ~(done_once | dones.bool())
            live.append(live_now.cpu().numpy())
            ov_frames.append(None if args_cli.no_video or overview is None else _t(overview.data.output["rgb"])[:n_vid, ..., :3].to(torch.uint8).cpu().numpy())
            aligned = (cmd.orientation_aligned & cmd.position_aligned & live_now).cpu().numpy()
            succ_t[(succ_t < 0) & aligned] = t
            done_once |= dones.bool()
            if bool(done_once.all()):
                break
    steps = len(ov_frames)
    success = succ_t >= 0
    if args_cli.record_states:
        torch.save({"states": recorded, "succ_t": succ_t, "task_id": task_id}, args_cli.record_states)
    for k, rt in enumerate(names):
        m = task_id == k
        if m.any():
            print(f"[eval_video] {rt}: {success[m].sum()}/{m.sum()} succeeded", flush=True)
    print(f"[eval_video] overall {success.sum()}/{n} over {steps} steps", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_prefix)), exist_ok=True)
    json.dump({"checkpoint": args_cli.checkpoint, "policy": args_cli.policy, "tf32": args_cli.tf32, "steps": steps, "success": int(success.sum()), "episodes": [
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
    if args_cli.per_env_videos:
        os.makedirs(args_cli.per_env_videos, exist_ok=True)
        index = ["file,env,reset_type,success,success_t"]
    for i in range(n_vid):
        env_frames = []
        title = Image.new("RGB", (OV_W + col_w, OV_H))
        d = ImageDraw.Draw(title)
        d.text((20, OV_H // 2 - 20), f"episode {i + 1}/{n_vid}: reset {names[task_id[i]]}", fill=(255, 255, 255))
        d.text((20, OV_H // 2), f"success at t={succ_t[i] + 1}" if success[i] else "no success",
               fill=(0, 255, 0) if success[i] else (255, 80, 80))
        env_frames += [np.asarray(title)] * 10
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
            if ov_frames[last] is not None:
                ov_img = np.asarray(Image.fromarray(ov_frames[last][i]).resize((OV_W, OV_H), Image.BILINEAR))
            else:  # no overview camera: the first policy camera, enlarged
                ov_img = np.zeros((OV_H, OV_W, 3), dtype=np.uint8)
                ov_img[:, (OV_W - OV_H) // 2: (OV_W + OV_H) // 2] = np.asarray(Image.fromarray(cam_frames[last][i, 0]).resize((OV_H, OV_H), Image.NEAREST))
            frame = Image.fromarray(np.concatenate([ov_img, column], axis=1))
            d = ImageDraw.Draw(frame)
            d.text((6, 6), f"episode {i + 1}/{n_vid}  {names[task_id[i]]}  t={t + 1}", fill=(255, 255, 0))
            if 0 <= succ_t[i] <= t:
                d.text((6, 20), f"SUCCESS @ t={succ_t[i] + 1}", fill=(0, 255, 0))
            for c, label in enumerate(cam_names[:cams]):
                d.text((OV_W + x0 + 4, c * cam_px + 4), f"{label.replace('_camera', '')} (policy)", fill=(255, 255, 0))
            env_frames.append(np.asarray(frame))
        if args_cli.per_env_videos:
            name = f"env{i:04d}_{names[task_id[i]]}_{'success' if success[i] else 'fail'}.mp4"
            imageio.mimwrite(os.path.join(args_cli.per_env_videos, name), env_frames, fps=10, macro_block_size=1)
            index.append(f"{name},{i},{names[task_id[i]]},{int(success[i])},{succ_t[i] + 1 if success[i] else ''}")
        else:
            frames += env_frames
    if args_cli.per_env_videos:
        open(os.path.join(args_cli.per_env_videos, "index.csv"), "w").write("\n".join(index) + "\n")
        print(f"[eval_video] wrote {n_vid} videos to {args_cli.per_env_videos}", flush=True)
    else:
        imageio.mimwrite(f"{args_cli.out_prefix}.mp4", frames, fps=10, macro_block_size=1)
        print(f"[eval_video] wrote {args_cli.out_prefix}.mp4", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
