# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the OptiX twin in the scenes of a ``record_or_lighting.py`` RTX reference: the reference's robot / peg / hole
states are written kinematically into ``--keep`` envs and the side + wrist cameras rendered for each sky entry of the
reference (map, intensity, orientation) under the given look (``--looks``: ``ambient:sky_scale:exposure`` triples;
sky_scale multiplies the reference sky intensity / 1000). ``--full`` renders the twin's own randomization instead.
Writes ``<out>.npz`` (images [L, R, keep, 2, H, W, 3]) and prints the error against the reference per look.
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="flashsac_cfg_entry_point")
parser.add_argument("--ref", required=True, help="record_or_lighting.py output prefix")
parser.add_argument("--looks", default="7:1:0.45", help="Comma-separated ambient:sky_scale:exposure looks.")
parser.add_argument("--hdri_yaml", default="")
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--sky_turns", type=float, default=-0.25, help="Map yaw [turns]; -0.25 is the RTX dome convention.")
parser.add_argument("--full", action="store_true", help="The twin's own full randomization: --rounds interval redraws.")
parser.add_argument("--rounds", type=int, default=12)
parser.add_argument("--keep", type=int, default=8, help="--full: envs to film.")
parser.add_argument("--look_params", default="", help="JSON merged into the rtx_look term params (e.g. dome, sky ranges).")
parser.add_argument("--seed", type=int, default=-1, help="Env seed (paired --full renders).")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os

import gymnasium as gym
import numpy as np
import torch

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


def _t(x):
    return x.torch if hasattr(x, "torch") else x


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if args_cli.seed >= 0:
        env_cfg.seed = args_cli.seed
    if args_cli.look_params:
        env_cfg.events.rtx_look.params.update(json.loads(args_cli.look_params))
        print(f"[twin-light] rtx_look params: {env_cfg.events.rtx_look.params}", flush=True)
    if args_cli.full:
        return full(env_cfg)
    ref = np.load(args_cli.ref + ".npz")
    meta = json.load(open(args_cli.ref + ".json"))
    k = int(ref["joint_pos"].shape[0])
    env_cfg.scene.num_envs = k
    if args_cli.hdri_yaml:  # load the sky bank (index i of the yaml = map i + 1); surfaces keep their materials
        env_cfg.events.rtx_look.params.update(randomize=True, randomize_appearance=False, hdri_config_path=args_cli.hdri_yaml)
    else:
        env_cfg.events.rtx_look.params["randomize"] = False
    for name in [n for n in list(vars(env_cfg.events)) if n.startswith("randomize_") or n.endswith("_interval")]:
        setattr(env_cfg.events, name, None)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    dev = base.device
    origins = base.scene.env_origins
    robot, peg, hole = base.scene["robot"], base.scene["insertive_object"], base.scene["receptive_object"]
    ids = torch.arange(k, device=dev)
    look = base.event_manager.get_term_cfg("rtx_look").func
    pose = lambda a: torch.cat([torch.as_tensor(a[:, :3], device=dev) + origins, torch.as_tensor(a[:, 3:], device=dev)], 1)
    robot.write_root_pose_to_sim(pose(ref["robot_root"]), env_ids=ids)
    jp = torch.as_tensor(ref["joint_pos"], device=dev)
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
    peg.write_root_pose_to_sim(pose(ref["peg_root"]), env_ids=ids)
    hole.write_root_pose_to_sim(pose(ref["hole_root"]), env_ids=ids)
    base.sim.forward()
    base.scene.update(0.0)
    cams = [base.scene.sensors["side_camera"], base.scene.sensors["wrist_camera"]]
    out = []
    for spec in args_cli.looks.split(","):
        amb, sky_scale, expo = (float(v) for v in spec.split(":"))
        frames = []
        for s in meta["skies"]:
            for r in look.renderers:
                r.set_environment(ids, ambient=torch.full((k, 3), amb, device=dev),
                                  map_ids=torch.full((k,), (s["index"] + 1) if args_cli.hdri_yaml else 0, device=dev, dtype=torch.int32),
                                  intensity=torch.full((k,), sky_scale * s["intensity"] / 1000.0, device=dev),
                                  rotation=torch.full((k,), args_cli.sky_turns, device=dev),
                                  orientation=torch.as_tensor([s["quat_xyzw"]] * k, device=dev, dtype=torch.float32))
                r.set_exposure(ids, expo)
            for _ in range(args_cli.warmup):
                base.sim.render()
                for c in cams:
                    c.update(base.physics_dt, force_recompute=True)
            frames.append(np.stack([_t(c.data.output["rgb"])[:k, ..., :3].to(torch.uint8).cpu().numpy() for c in cams], 1))
        frames = np.stack(frames)
        out.append(frames)
        rimg = ref["images"].astype(np.float32)
        f = rimg.shape[-2] // frames.shape[-2]  # reference rendered larger: area-average it down
        rimg = rimg.reshape(*rimg.shape[:3], rimg.shape[3] // f, f, rimg.shape[4] // f, f, 3).mean((4, 6))
        err = np.abs(frames.astype(np.float32) - rimg).mean()
        print(f"[twin-light] look ambient {amb} sky {sky_scale} exposure {expo}: mean side {frames[:, :, 0].mean():.1f} wrist {frames[:, :, 1].mean():.1f} "
              f"(ref {rimg[:, :, 0].mean():.1f} / {rimg[:, :, 1].mean():.1f}); mean |twin - ref| {err:.1f}", flush=True)
    np.savez_compressed(args_cli.out + ".npz", images=np.stack(out), looks=np.array(args_cli.looks.split(",")))
    print(f"[twin-light] wrote {args_cli.out}.npz", flush=True)
    os._exit(0)


def full(env_cfg):
    """The twin's own randomization (look from the task / overrides): --rounds interval redraws of --keep envs."""
    env_cfg.scene.num_envs = max(args_cli.keep, 8)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    zero = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
    for _ in range(10):
        env.step(zero)
    cams = [base.scene.sensors["side_camera"], base.scene.sensors["wrist_camera"]]
    k, frames = args_cli.keep, []
    for r in range(args_cli.rounds):
        base.event_manager.apply(mode="interval", dt=4.01)
        for _ in range(args_cli.warmup):
            base.sim.render()
            for c in cams:
                c.update(base.physics_dt, force_recompute=True)
        frames.append(np.stack([_t(c.data.output["rgb"])[:k, ..., :3].to(torch.uint8).cpu().numpy() for c in cams], 1))
    frames = np.stack(frames)[None]
    np.savez_compressed(args_cli.out + ".npz", images=frames, looks=np.array(["full"]))
    print(f"[twin-light] full: {frames.shape}; mean side {frames[0, :, :, 0].mean():.1f} wrist {frames[0, :, :, 1].mean():.1f}; wrote {args_cli.out}.npz", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
