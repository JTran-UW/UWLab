# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline vision-probe dataset: state-expert rollouts (per-env action noise) on a randomized camera task, recording
side + wrist RGB with ground-truth peg / hole geometry and per-frame appearance labels, in .pt shards.

Every ``--record_every`` steps, all envs: ``side`` / ``wrist`` uint8 (N, H, W, 3); ``aux_state`` (the aux-loss targets);
world-frame (env-origin relative) peg / hole / wrist poses; arm + gripper joints; ``aligned`` (task success now);
episode step, episode index, env id, reset type, action-noise std; and for the peg, hole and table: sRGB albedo, texture
id (-1 = none), roughness, metallic, specular and texture scale. ``meta.json`` describes the layout.

usage: run_in_sif.sh scripts/tools/collect_vision_probe_dataset.py --task <camera task> --checkpoint <rsl_rl expert>
       --output_dir <dir> [--num_envs 512 --num_steps 800 --record_every 4 --image_size 224] <hydra overrides>
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, help="rsl_rl state-expert checkpoint")
parser.add_argument("--expert_agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_steps", type=int, default=800)
parser.add_argument("--record_every", type=int, default=4)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--noise_levels", default="0,0.1,0.3", help="action-noise std (policy space), assigned per env round robin")
parser.add_argument("--shard_frames", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_images", action="store_true", help="labels, camera poses, visibility and brightness only")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import json
import os

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
from uwlab_rl.flashsac import make_flashsac_env
from uwlab_rl.flashsac.expert_envs import load_expert_policy, to_policy_space
from uwlab_tasks.utils.hydra import hydra_task_config

SURFACES = ("insertive_object", "receptive_object", "table")
HALF = {"peg": (0.015, 0.015, 0.030), "hole": (0.0349, 0.0349, 0.0203)}  # bounding-box half extents (m)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


@hydra_task_config(args.task, "flashsac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    for cam in (env_cfg.scene.side_camera, env_cfg.scene.wrist_camera):
        cam.height = cam.width = args.image_size
    env = make_flashsac_env(gym.make(args.task, cfg=env_cfg), FlashSACVecEnvWrapper, agent_cfg.clip_actions,
                            agent_cfg.action_bound, agent_cfg.action_bound_scale)
    base = env.unwrapped
    dev, n = base.device, base.num_envs
    expert = load_expert_policy(env, args.task, args.expert_agent, args.checkpoint, dev)
    levels = torch.tensor([float(v) for v in args.noise_levels.split(",")], device=dev)
    noise = levels[torch.arange(n, device=dev) % len(levels)]
    robot, peg, hole = base.scene["robot"], base.scene["insertive_object"], base.scene["receptive_object"]
    wrist_id = robot.find_bodies("wrist_3_link")[0][0]
    side, wrist = base.scene.sensors["side_camera"], base.scene.sensors["wrist_camera"]
    cmd = base.command_manager.get_term("task_command")
    reset_cfg = base.event_manager.get_term_cfg("reset_from_reset_states")
    reset_term = reset_cfg.func
    look = base.event_manager.get_term_cfg("rtx_look").func
    om = base.observation_manager
    aux_names, aux_dims = om.active_terms["aux_state"], [d[0] for d in om.group_obs_term_dim["aux_state"]]
    os.makedirs(args.output_dir, exist_ok=True)
    origins = base.scene.env_origins
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)] + [[0, 0, 0]], device=dev, dtype=torch.float32)

    def in_view(cam, asset, half, centre_z=0.0):
        """Fraction of the asset's bounding-box corners + centre that project inside the camera image (no occlusion)."""
        pose = _t(asset.data.root_pose_w)
        pts = signs * torch.tensor(half, device=dev) + torch.tensor([0.0, 0.0, centre_z], device=dev)
        world = pose[:, None, :3] + quat_apply(pose[:, None, 3:7].expand(-1, pts.shape[0], -1), pts[None].expand(pose.shape[0], -1, -1))
        q = _t(cam.data.quat_w_ros)[:, None].expand(-1, pts.shape[0], -1)
        local = quat_apply_inverse(q, world - _t(cam.data.pos_w)[:, None])
        uvw = torch.einsum("nij,npj->npi", _t(cam.data.intrinsic_matrices), local)
        u, v, z = uvw[..., 0] / uvw[..., 2], uvw[..., 1] / uvw[..., 2], local[..., 2]
        h, w = cam.image_shape if hasattr(cam, "image_shape") else (args.image_size, args.image_size)
        return ((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)).float().mean(1)

    def appearance():
        """Per env and surface: albedo (3), texture id, roughness, metallic, specular, texture scale (first shape)."""
        from isaaclab_newton.physics.newton_manager import NewtonManager

        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.rtx_look import RTX_SURFACES

        if look.shapes is None:
            return None
        keys = list(RTX_SURFACES)
        colors = wp.to_torch(NewtonManager.get_model().shape_color)
        ctx = look.renderers[0]._context
        tex = wp.to_torch(ctx.shape_texture_ids)
        mats = ctx.materials.numpy()
        out = {}
        for name in SURFACES:
            ids = look.shapes[keys.index(name)][:, 0]
            m = mats[ids.cpu().numpy()]
            out[name] = torch.cat([colors[ids].float().cpu(), tex[ids].float().cpu()[:, None],
                                   torch.from_numpy(np.stack([m["roughness"], m["metallic"], m["specular"], m["uv_scale"]], 1)).float()], 1)
        return out

    shard, buf, frames = 0, {}, 0
    episode = torch.zeros(n, dtype=torch.long, device=dev)

    def flush():
        nonlocal shard, buf
        if not buf:
            return
        data = {k: torch.cat(v) for k, v in buf.items()}
        path = os.path.join(args.output_dir, f"shard_{shard:04d}.pt")
        torch.save(data, path)
        print(f"[probe-data] wrote {path}: {data['episode'].shape[0]} frames", flush=True)
        shard, buf = shard + 1, {}

    obs = env.get_observations()
    with torch.no_grad():
        for step in range(args.num_steps):
            if step % args.record_every == 0:
                pose = lambda p: torch.cat([p[:, :3] - origins, p[:, 3:7]], 1)
                side_rgb = _t(side.data.output["rgb"])[..., :3]
                wrist_rgb = _t(wrist.data.output["rgb"])[..., :3]
                rec = {} if args.no_images else {"side": side_rgb.to(torch.uint8).cpu(), "wrist": wrist_rgb.to(torch.uint8).cpu()}
                rec.update({
                    "side_mean": side_rgb.float().mean((1, 2, 3)).cpu(), "wrist_mean": wrist_rgb.float().mean((1, 2, 3)).cpu(),
                    "side_std": side_rgb.float().std((1, 2, 3)).cpu(), "wrist_std": wrist_rgb.float().std((1, 2, 3)).cpu(),
                    "side_cam_pose": torch.cat([_t(side.data.pos_w) - origins, _t(side.data.quat_w_ros)], 1).cpu(),
                    "wrist_cam_pose": torch.cat([_t(wrist.data.pos_w) - origins, _t(wrist.data.quat_w_ros)], 1).cpu(),
                    "side_intrinsics": _t(side.data.intrinsic_matrices).cpu(), "wrist_intrinsics": _t(wrist.data.intrinsic_matrices).cpu(),
                    "peg_in_side": in_view(side, peg, HALF["peg"]).cpu(), "peg_in_wrist": in_view(wrist, peg, HALF["peg"]).cpu(),
                    "hole_in_side": in_view(side, hole, HALF["hole"]).cpu(), "hole_in_wrist": in_view(wrist, hole, HALF["hole"]).cpu(),
                    "aux_state": obs["aux_state"].float().cpu(),
                    "peg_pose": pose(_t(peg.data.root_pose_w)).cpu(),
                    "hole_pose": pose(_t(hole.data.root_pose_w)).cpu(),
                    "wrist_pose": pose(_t(robot.data.body_link_pose_w)[:, wrist_id]).cpu(),
                    "joint_pos": _t(robot.data.joint_pos).float().cpu(),
                    "aligned": (cmd.position_aligned & cmd.orientation_aligned).cpu(),
                    "episode_step": base.episode_length_buf.cpu().clone(),
                    "episode": episode.cpu().clone(),
                    "env_id": torch.arange(n),
                    "reset_type": reset_term.task_id.cpu().clone(),
                    "noise_std": noise.cpu(),
                })
                app_ = appearance()
                if app_ is not None:
                    for name, v in app_.items():
                        rec[f"appearance_{name}"] = v
                for k, v in rec.items():
                    buf.setdefault(k, []).append(v)
                frames += n
                if sum(x.shape[0] for x in buf["episode"]) >= args.shard_frames:
                    flush()
            act = to_policy_space(expert(obs.select("policy")), env.action_scale, env.action_bias)
            act = (act + noise[:, None] * torch.randn_like(act)).clamp(-1.0, 1.0)
            obs, _, dones, _ = env.step(act)
            episode += dones.long().to(dev)
            if step % 100 == 0:
                print(f"[probe-data] step {step}/{args.num_steps} frames {frames} episodes done {int(episode.sum())}", flush=True)
    flush()
    meta = {
        "task": args.task, "checkpoint": args.checkpoint, "num_envs": n, "num_steps": args.num_steps,
        "record_every": args.record_every, "image_size": args.image_size, "noise_levels": args.noise_levels,
        "frames": frames, "shards": shard, "aux_terms": list(zip(aux_names, aux_dims)),
        "reset_types": list(reset_cfg.params["reset_types"]),
        "appearance_columns": ["r", "g", "b", "texture_id", "roughness", "metallic", "specular", "uv_scale"],
        "appearance_surfaces": list(SURFACES), "pose_layout": "x y z (env frame, m) + quat x y z w",
        "hydra_overrides": hydra_args, "seed": args.seed, "images": not args.no_images,
        "camera_pose_layout": "x y z (env frame, m) + quat (isaaclab order) of the ROS camera frame (z forward, x right, y down)",
        "in_view": "fraction of 8 bounding-box corners + centre inside the image, no occlusion; half extents " + str(HALF),
    }
    json.dump(meta, open(os.path.join(args.output_dir, "meta.json"), "w"), indent=1)
    print(f"[probe-data] done: {frames} frames in {shard} shards", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
