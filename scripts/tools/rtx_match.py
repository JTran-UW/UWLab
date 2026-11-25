# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Look development against RTX references: the scenes of ``record_norand_compare.py`` (fixed seed, zero actions, no
visual randomization) on the UWLab Newton renderer.

``--save_states`` runs the reference's env count / seed / steps and stores the first ``--keep`` envs' scene state.
``--load_states`` replays those states in a small env. ``--serve DIR`` then executes ``DIR/requests/*.py`` snippets
in-process with ``H`` (this module's helpers) in scope, so looks are re-rendered without restarting the simulator.
"""

import argparse
import dataclasses
import glob
import os
import re
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-WristSide-Sysid-Aux-RGB-VisRand-v0"
)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--steps", type=int, default=30, help="Zero-action env steps before the states are taken.")
parser.add_argument("--keep", type=int, default=16, help="Envs whose state is saved / rendered.")
parser.add_argument("--save_states", type=str, default="")
parser.add_argument("--load_states", type=str, default="")
parser.add_argument("--cmp_cams", type=str, default="front_camera,side_camera,wrist_camera")
parser.add_argument("--cmp_res", type=str, default="320x240")
parser.add_argument("--policy_res", type=int, default=96)
parser.add_argument("--environments", type=str, default="", help="Comma-separated environment maps to preload.")
parser.add_argument("--textures", type=str, default="", help="Comma-separated albedo textures to preload.")
parser.add_argument("--serve", type=str, default="", help="Directory to serve request snippets from.")
parser.add_argument("--path_samples", type=int, default=0, help="Path-lighting samples per pixel (0: off).")
parser.add_argument("--path_aa", type=int, default=4, help="Primary samples per pixel for edge anti-aliasing.")
parser.add_argument("--exposure", type=float, default=0.45, help="Tone-mapping exposure [stops].")
parser.add_argument("--ambient", type=float, default=7.0, help="Uniform dome radiance (RTX dome intensity / 1000).")
parser.add_argument("--sky", type=float, default=1.0, help="Environment-map scale (RTX sky intensity / 1000).")
parser.add_argument("--sky_id", type=int, default=1, help="Index of the environment map used as the sky.")
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import warp as wp
from isaaclab.sensors import CameraCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton._src.sensors.warp_raytrace.types import ClearData
from PIL import Image

import uwlab_tasks  # noqa: F401
from uwlab.utils.newton_warp_renderer_cfg import UWLabNewtonWarpRendererCfg
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.data_collection_rgb_cfg import (
    DataCollectionRGBObjectSceneCfg,
)
from uwlab_tasks.utils.hydra import hydra_task_config

VISUAL_RAND = re.compile(r"^randomize_.*(appearance|sky_light|camera)")
OUTPUTS = dict(
    color=("color_image", wp.uint32),
    hdr=("hdr_color_image", wp.vec3f),
    depth=("depth_image", wp.float32),
    shape=("shape_index_image", wp.uint32),
    normal=("normal_image", wp.vec3f),
    albedo=("albedo_image", wp.uint32),
    light=("light_image", wp.vec3f),
    gloss=("gloss_image", wp.vec3f),
    tint=("tint_image", wp.vec3f),
    cover=("cover_image", wp.float32),
)


class Harness:
    """Helpers handed to request snippets as ``H``."""

    def __init__(self, env, keep, out):
        self.env, self.keep, self.out = env, keep, out
        self.model = NewtonManager.get_model()
        self.cameras = {n: s for n, s in env.scene.sensors.items() if "rgb" in getattr(s.data, "output", {})}
        self.renderer = next(iter(self.cameras.values()))._renderer
        self.np, self.torch, self.wp, self.Image = np, torch, wp, Image

    @property
    def ctx(self):
        return self.renderer._context

    @property
    def config(self):
        return self.renderer.newton_sensor.default_render_config

    def render(self, camera, outputs=("hdr", "albedo", "shape", "normal", "depth"), backend=None, frames=1, **overrides):
        """Render ``camera`` with the default config plus ``overrides``; numpy arrays of the first ``keep`` envs.
        ``frames`` > 1 averages the float outputs over that many sample seeds."""
        sensor = self.cameras[camera]
        renderer, rd = sensor._renderer, sensor._render_data
        ctx, model, state = renderer._context, self.model, NewtonManager.get_state_0()
        config = dataclasses.replace(self.config, **overrides)
        shape = (model.world_count, 1, rd.height, rd.width)
        out = {OUTPUTS[k][0]: wp.zeros(shape, dtype=OUTPUTS[k][1], device=model.device) for k in outputs}
        floats = [k for k in outputs if OUTPUTS[k][1] is wp.vec3f and k != "normal"]
        total = {k: torch.zeros((self.keep, rd.height, rd.width, 3), device=str(model.device)) for k in floats}
        previous, seed = ctx.backend, ctx.frame_seed
        ctx.backend = backend or previous
        try:
            ctx.camera_intrinsics, ctx.previous_camera_transforms = rd._intrinsics, rd._previous_transforms
            if ctx.backend == "warp":
                model.bvh_refit_shapes(state)
            for frame in range(frames):
                ctx.frame_seed = seed + frame
                ctx.render(
                    model,
                    state,
                    camera_transforms=rd.camera_transforms,
                    camera_rays=rd.camera_rays,
                    clear_data=ClearData(clear_color=rd.clear_color),
                    config=config,
                    **out,
                )
                for k in floats:
                    total[k] += wp.to_torch(out[OUTPUTS[k][0]])[: self.keep, 0]
        finally:
            ctx.backend, ctx.frame_seed = previous, seed
        wp.synchronize_device(model.device)
        res = {}
        for k in outputs:
            if k in floats:
                res[k] = (total[k] / frames).cpu().numpy()
                continue
            a = out[OUTPUTS[k][0]].numpy()[: self.keep, 0]
            if k in ("color", "albedo"):
                a = a.view(np.uint8).reshape(*a.shape, 4)[..., :3]
            res[k] = a
        return res

    def shapes(self, pattern, env=None):
        """Shape ids whose label matches the regex ``pattern`` (optionally within one env)."""
        root = None if env is None else f"/World/envs/env_{env}/"
        return [
            i
            for i, label in enumerate(self.model.shape_label)
            if re.search(pattern, label or "") and (root is None or (label or "").startswith(root))
        ]

    def reload(self):
        """Re-import the render package and rebuild the context (after editing kernel code)."""
        import importlib

        import uwlab.render.context as c
        import uwlab.render.ibl as i
        import uwlab.render.kernel as k
        import uwlab.render.optix as o
        import uwlab.render.postprocess as p
        import uwlab.render.shading as s
        import uwlab.render.types as t

        for module in (t, s, i, p, o, k, c):
            importlib.reload(module)
        shapes = getattr(self.model, "_uwlab_optix_shapes", None)
        if shapes is not None:
            shapes.pipelines.clear()
        self.renderer._install_context()
        return self.ctx


def _write_state(base, state, ids):
    """``InteractiveScene.reset_to`` for the kept envs (relative to their origins), then forward kinematics."""
    scene = base.scene
    for name, asset in scene._articulations.items():
        if name not in state["articulation"]:
            continue
        s = state["articulation"][name]
        pose = s["root_pose"].clone()
        pose[:, :3] += scene.env_origins[ids]
        asset.write_root_pose_to_sim(pose, env_ids=ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(s["root_velocity"]), env_ids=ids)
        asset.write_joint_state_to_sim(s["joint_position"].clone(), torch.zeros_like(s["joint_velocity"]), env_ids=ids)
    for name, asset in scene._rigid_objects.items():
        if name not in state["rigid_object"]:
            continue
        s = state["rigid_object"][name]
        pose = s["root_pose"].clone()
        pose[:, :3] += scene.env_origins[ids]
        asset.write_root_pose_to_sim(pose, env_ids=ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(s["root_velocity"]), env_ids=ids)
    base.sim.forward()
    scene.update(0.0)
    base.sim.render()
    for sensor in scene.sensors.values():
        if hasattr(sensor, "_render_data"):
            sensor.update(base.physics_dt, force_recompute=True)


def _slice(state, n):
    return {
        kind: {name: {k: v[:n].clone().cpu() for k, v in s.items()} for name, s in assets.items()}
        for kind, assets in state.items()
        if kind in ("articulation", "rigid_object")
    }


@hydra_task_config(args_cli.task, "flashsac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.keep if args_cli.load_states else args_cli.num_envs
    env_cfg.seed = args_cli.seed
    off = [n for n in list(vars(env_cfg.events)) if VISUAL_RAND.match(n) and getattr(env_cfg.events, n) is not None]
    for n in off:
        setattr(env_cfg.events, n, None)
    print(f"[rtx_match] disabled: {off}", flush=True)
    # the real-fitted policy cameras (apply_real_camera_calibration, without its jitter events)
    scene = env_cfg.scene
    scene.side_camera.offset = CameraCfg.OffsetCfg(
        pos=(0.8263, 0.5318, 0.2900), rot=(-0.32400, -0.77359, 0.50469, 0.20460), convention="ros"
    )
    scene.side_camera.spawn.focal_length = 27.053
    scene.wrist_camera.offset = CameraCfg.OffsetCfg(
        pos=(0.0320, -0.0062, -0.0583), rot=(0.35130, 0.35514, 0.61136, 0.61376), convention="ros"
    )
    scene.wrist_camera.spawn.focal_length = 28.03
    renderer_cfg = UWLabNewtonWarpRendererCfg(
        enable_patches=True,
        enable_textures=True,
        enable_shadows=True,
        enable_pbr=True,
        create_default_light=False,
        tone_mapper="aces",
        tone_blur=0.5,
        path_samples=args_cli.path_samples,
        path_aa=args_cli.path_aa,
        path_sky_clamp=5.0,
        environment_paths=tuple(p for p in args_cli.environments.split(",") if p),
        texture_paths=tuple(p for p in args_cli.textures.split(",") if p),
    )
    for name in ("side_camera", "wrist_camera"):
        camera = getattr(scene, name)
        camera.renderer_cfg = renderer_cfg
        camera.width = camera.height = args_cli.policy_res
        camera.update_latest_camera_pose = True
    ref = DataCollectionRGBObjectSceneCfg(num_envs=1)
    width, height = (int(v) for v in args_cli.cmp_res.lower().split("x"))
    for name in (c for c in args_cli.cmp_cams.split(",") if c):
        c = getattr(ref, name)
        setattr(
            scene,
            f"cmp_{name}",
            c.replace(prim_path=c.prim_path + "_cmp", renderer_cfg=renderer_cfg, width=width, height=height),
        )
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    os.makedirs(args_cli.out, exist_ok=True)
    renderers = {id(s._renderer): s._renderer for s in base.scene.sensors.values() if hasattr(s, "_render_data")}
    for renderer in renderers.values():
        renderer.set_environment(
            slice(None),
            ambient=[[args_cli.ambient] * 3],
            map_ids=args_cli.sky_id if args_cli.environments else -1,
            intensity=args_cli.sky,
        )
        renderer.set_lights(None, None, [[False] * 3] * base.num_envs)
        renderer.set_exposure(slice(None), args_cli.exposure)
    ids = torch.arange(args_cli.keep, device=base.device)
    if args_cli.load_states:
        state = torch.load(args_cli.load_states, map_location=base.device, weights_only=False)
        _write_state(base, state, ids)
        print(f"[rtx_match] replayed {args_cli.load_states} into {args_cli.keep} envs", flush=True)
    else:
        zero = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
        for _ in range(args_cli.steps):
            env.step(zero)
        if args_cli.save_states:
            torch.save(_slice(base.scene.get_state(is_relative=True), args_cli.keep), args_cli.save_states)
            print(f"[rtx_match] saved {args_cli.save_states}", flush=True)
    H = Harness(base, args_cli.keep, args_cli.out)
    for name, sensor in H.cameras.items():
        rgb = sensor.data.output["rgb"]
        rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[: min(4, args_cli.keep), ..., :3].to(torch.uint8).cpu().numpy()
        Image.fromarray(np.concatenate(list(rgb), axis=0)).save(f"{args_cli.out}/env_{name}.png")
    print(f"[rtx_match] cameras {list(H.cameras)} | {H.model.world_count} worlds, {H.model.shape_count} shapes", flush=True)
    print("[rtx_match] READY", flush=True)
    if args_cli.serve:
        requests = os.path.join(args_cli.serve, "requests")
        os.makedirs(requests, exist_ok=True)
        done = set()
        while not os.path.exists(os.path.join(args_cli.serve, "quit")):
            pending = sorted(p for p in glob.glob(os.path.join(requests, "*.py")) if p not in done)
            if not pending:
                time.sleep(0.5)
                continue
            path = pending[0]
            done.add(path)
            start = time.time()
            try:
                with open(path) as f:
                    exec(compile(f.read(), path, "exec"), {"H": H, "np": np, "torch": torch, "wp": wp, "Image": Image})
                status = "ok"
            except Exception:
                status = "error\n" + traceback.format_exc()
            with open(path[:-3] + ".done", "w") as f:
                f.write(f"{status}\n{time.time() - start:.1f} s\n")
            print(f"[rtx_match] {os.path.basename(path)}: {status.splitlines()[0]} ({time.time() - start:.1f} s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
