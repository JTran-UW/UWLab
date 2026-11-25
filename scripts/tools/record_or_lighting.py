# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX references with the OmniReset RGB lighting: per-env robot dome lights (from the UR5e USD) + the /World/skyLight
HDRI dome (intensity and a uniformly random 3D orientation, as ``randomize_hdri``), OmniReset render settings.

Default (lighting only): visual randomization off; for each requested HDRI the sky texture / intensity / orientation are
set explicitly, the scene re-rendered, and the side + wrist images of the first ``--keep`` envs saved. ``--full``: keep
the OmniReset appearance / sky / camera randomization and save ``--rounds`` redraws (the sky of each round is recorded).
Writes ``<out>.npz`` (images [R, keep, 2, H, W, 3] uint8, sky params, scene states) and ``<out>_sheet.png``.

usage (native RTX on the workstation): run_native_vision.sh record_or_lighting.py --task <RTX RealCam task> --out <prefix>
"""

import argparse
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=32, help="OmniReset collection used 32 envs (one robot dome each).")
parser.add_argument("--keep", type=int, default=8)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--steps", type=int, default=10, help="Zero-action steps before rendering.")
parser.add_argument("--hdri_yaml", required=True)
parser.add_argument("--hdri_indices", default="", help="Comma-separated indices into the HDRI list (lighting-only mode).")
parser.add_argument("--only", type=int, default=-1, help="Render only this one of --hdri_indices (same random draws, no earlier map in the denoiser history).")
parser.add_argument("--intensity", type=str, default="2500", help="Sky intensity, or 'random' for 1000-4000 per HDRI.")
parser.add_argument("--orientation", default="random", choices=["random", "identity"])
parser.add_argument("--full", action="store_true")
parser.add_argument("--rounds", type=int, default=12)
parser.add_argument("--warmup", type=int, default=24, help="Renders after each lighting change (denoiser history).")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json
import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation

import isaaclab_tasks  # noqa: F401
import omni.usd
import uwlab_tasks  # noqa: F401
from isaaclab.sensors import CameraCfg
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as mdp_utils
from uwlab_tasks.utils.hydra import hydra_task_config

VISUAL_RAND = re.compile(r"^randomize_.*(appearance|sky_light|camera)")
OR_RENDER = dict(enable_dlssg=False, enable_ambient_occlusion=True, enable_reflections=True, enable_dl_denoiser=True,
                 antialiasing_mode="DLAA")


def _t(x):
    return x.torch if hasattr(x, "torch") else x


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if os.environ.get("UWLAB_UR5E_DISABLE_DOMELIGHT") == "1":
        print("[or-light] robot domes DISABLED (control run)", flush=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if not args_cli.full:
        for n in [n for n in list(vars(env_cfg.events)) if VISUAL_RAND.match(n)]:
            setattr(env_cfg.events, n, None)
    else:
        env_cfg.events.randomize_sky_light.params["hdri_config_path"] = args_cli.hdri_yaml
    for c in vars(env_cfg.scene).values():
        if isinstance(c, CameraCfg) and getattr(c.renderer_cfg, "global_settings", None) is not None:
            c.renderer_cfg = c.renderer_cfg.replace(global_settings=c.renderer_cfg.global_settings.replace(**OR_RENDER))
    ev = getattr(env_cfg.events, "render_settings", None)
    if ev is not None and "settings" in ev.params:
        ev.params["settings"] = ev.params["settings"].replace(**OR_RENDER)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    zero = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
    for _ in range(args_cli.steps):
        env.step(zero)
    k = args_cli.keep
    robot, peg, hole = base.scene["robot"], base.scene["insertive_object"], base.scene["receptive_object"]
    origins = base.scene.env_origins
    states = {"joint_pos": _t(robot.data.joint_pos)[:k].cpu().numpy(),
              "robot_root": (_t(robot.data.root_pose_w)[:k] - torch.cat([origins[:k], torch.zeros(k, 4, device=origins.device)], 1)).cpu().numpy(),
              "peg_root": (_t(peg.data.root_pose_w)[:k] - torch.cat([origins[:k], torch.zeros(k, 4, device=origins.device)], 1)).cpu().numpy(),
              "hole_root": (_t(hole.data.root_pose_w)[:k] - torch.cat([origins[:k], torch.zeros(k, 4, device=origins.device)], 1)).cpu().numpy()}
    cams = [base.scene.sensors["side_camera"], base.scene.sensors["wrist_camera"]]
    stage = omni.usd.get_context().get_stage()
    sky = stage.GetPrimAtPath("/World/skyLight")
    paths = mdp_utils.load_asset_paths_from_config(args_cli.hdri_yaml, cache_subdir="hdris", skip_validation=False)
    rng = np.random.default_rng(args_cli.seed)

    def render():
        for _ in range(args_cli.warmup):
            base.sim.render()
            for c in cams:
                c.update(base.physics_dt, force_recompute=True)
        return np.stack([_t(c.data.output["rgb"])[:k, ..., :3].to(torch.uint8).cpu().numpy() for c in cams], 1)

    def read_sky():
        q = UsdGeom.Xformable(sky).GetLocalTransformation().ExtractRotationQuat()
        file = sky.GetAttribute("inputs:texture:file").Get()
        file = file.path if hasattr(file, "path") else str(file)
        return {"file": file, "index": paths.index(file) if file in paths else -1,
                "intensity": float(sky.GetAttribute("inputs:intensity").Get()),
                "quat_xyzw": [q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()]}

    images, skies = [], []
    if not args_cli.full:
        for idx in [int(i) for i in args_cli.hdri_indices.split(",")]:
            inten = float(rng.uniform(1000.0, 4000.0)) if args_cli.intensity == "random" else float(args_cli.intensity)
            q = Rotation.random(random_state=int(rng.integers(1 << 31))).as_quat() if args_cli.orientation == "random" else np.array([0, 0, 0, 1.0])
            if args_cli.only >= 0 and idx != args_cli.only:
                continue
            sky.GetAttribute("inputs:texture:file").Set(paths[idx])
            sky.GetAttribute("inputs:intensity").Set(inten)
            xf = UsdGeom.Xformable(sky)
            xf.ClearXformOpOrder()
            xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2]))))
            images.append(render())
            skies.append(read_sky())
            print(f"[or-light] hdri {idx} {os.path.basename(paths[idx])} intensity {inten:.0f} quat {np.round(q, 3).tolist()}", flush=True)
    else:
        for r in range(args_cli.rounds):
            base.event_manager.apply(mode="interval", dt=4.01)
            images.append(render())
            skies.append(read_sky())
            print(f"[or-light] round {r} sky {skies[-1]['index']} {os.path.basename(skies[-1]['file'])} intensity {skies[-1]['intensity']:.0f}", flush=True)
    images = np.stack(images)
    np.savez_compressed(args_cli.out + ".npz", images=images, **states)
    json.dump({"task": args_cli.task, "num_envs": args_cli.num_envs, "keep": k, "seed": args_cli.seed, "steps": args_cli.steps,
               "full": args_cli.full, "skies": skies, "hdri_yaml": args_cli.hdri_yaml, "render": OR_RENDER,
               "robot_domes": True}, open(args_cli.out + ".json", "w"), indent=1)
    rows = []
    for r in range(images.shape[0]):
        row = np.concatenate([np.concatenate([images[r, e, 0], images[r, e, 1]], 1) for e in range(min(k, 4))], 1)
        rows.append(row)
    sheet = Image.fromarray(np.concatenate(rows, 0))
    sheet.save(args_cli.out + "_sheet.png")
    print(f"[or-light] wrote {args_cli.out}.npz {images.shape}; mean brightness side {images[:, :, 0].mean():.1f} wrist {images[:, :, 1].mean():.1f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
