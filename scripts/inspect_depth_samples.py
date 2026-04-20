"""Dump a few depth samples from the Depth-DAgger env to investigate camera placement.

Saves per-env PNGs of raw ``distance_to_camera`` (meters, visualized) and the
processed obs the student sees (``[0,1]`` after inf/clip/normalize). Also prints
pixel-value statistics (min/max/mean/frac_at_max/inf_count) so we can judge
whether the working volume occupies enough of the dynamic range.

Usage:
    python scripts/inspect_depth_samples.py --num_envs 4 --num_steps 12 \\
        --out_dir depth_samples --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-v0 \\
        env.scene.insertive_object=peg env.scene.receptive_object=peghole
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump depth samples from Depth-DAgger env.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=12, help="Env steps to run before dumping.")
parser.add_argument("--save_every", type=int, default=4, help="Save frames every N steps.")
parser.add_argument("--out_dir", type=str, default="depth_samples")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-v0",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--modality", choices=["depth", "rgb", "both"], default="both",
    help="What to render and dump: depth-only, rgb-only, or both at 240x320 for comparison.",
)
parser.add_argument("--num_samples_per_step", type=int, default=4, help="Save at most N env samples per saved step.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from isaaclab_tasks.utils import parse_env_cfg

import uwlab_tasks  # noqa: F401  (register envs)


def to_uint8(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip((x - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def stats(name: str, x: torch.Tensor, hi: float) -> None:
    finite = x[torch.isfinite(x)]
    n_inf = int(torch.isinf(x).sum().item())
    n_nan = int(torch.isnan(x).sum().item())
    if finite.numel() == 0:
        print(f"  [{name}] ALL non-finite: inf={n_inf} nan={n_nan}")
        return
    frac_at_max = float((finite >= hi - 1e-6).float().mean().item()) if hi else 0.0
    print(
        f"  [{name}] min={finite.min():.4f} max={finite.max():.4f} mean={finite.mean():.4f} "
        f"std={finite.std():.4f} frac>=max({hi})={frac_at_max:.3f} inf={n_inf} nan={n_nan}"
    )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    render_rgb = args_cli.modality in ("rgb", "both")
    render_depth = args_cli.modality in ("depth", "both")
    data_types = []
    if render_depth:
        data_types.append("distance_to_camera")
    if render_rgb:
        data_types.append("rgb")
    env_cfg.scene.side_camera.data_types = data_types
    if args_cli.modality == "both":
        env_cfg.scene.side_camera.height = 240
        env_cfg.scene.side_camera.width = 320
    # Apply same override to front_camera if the 2-cam scene has one.
    if hasattr(env_cfg.scene, "front_camera") and env_cfg.scene.front_camera is not None:
        env_cfg.scene.front_camera.data_types = data_types
        if args_cli.modality == "both":
            env_cfg.scene.front_camera.height = 240
            env_cfg.scene.front_camera.width = 320

    # If rgb-only, swap the obs term's data_type so the obs manager is happy.
    if args_cli.modality == "rgb":
        if hasattr(env_cfg.observations, "side_depth"):
            env_cfg.observations.side_depth.image.params["data_type"] = "rgb"
            env_cfg.observations.side_depth.image.params["process_image"] = True
        if hasattr(env_cfg.observations, "front_depth"):
            env_cfg.observations.front_depth.image.params["data_type"] = "rgb"
            env_cfg.observations.front_depth.image.params["process_image"] = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.unwrapped.seed(args_cli.seed)

    obs, _ = env.reset()
    print(f"obs keys: {list(obs.keys())}")

    os.makedirs(args_cli.out_dir, exist_ok=True)

    # Auto-detect all TiledCameras in the scene so 2-cam envs render front+side.
    cam_names = [n for n in ("side_camera", "front_camera") if n in env.unwrapped.scene.sensors]
    print(f"Active cameras: {cam_names}")
    depth_clip_hi = 2.0
    RGB_H, RGB_W = 224, 224
    n_save = min(args_cli.num_samples_per_step, args_cli.num_envs)

    for step in range(args_cli.num_steps):
        actions = torch.zeros(
            args_cli.num_envs,
            env.unwrapped.action_space.shape[1] if env.unwrapped.action_space.shape is not None else 7,
            device=env.unwrapped.device,
        )
        obs, *_ = env.step(actions)

        if step % args_cli.save_every != 0:
            continue

        print(f"\n=== step {step} ===")

        for cam_name in cam_names:
            short = cam_name.split("_", 1)[0]  # "side" / "front"
            sensor = env.unwrapped.scene.sensors[cam_name]

            if render_depth:
                raw = sensor.data.output["distance_to_camera"]
                if raw.ndim == 4 and raw.shape[-1] == 1:
                    raw = raw.squeeze(-1)
                stats(f"{short} raw depth (m)", raw, hi=depth_clip_hi)
                for i in range(n_save):
                    raw_np = raw[i].detach().cpu().float().numpy()
                    raw_vis = np.where(np.isfinite(raw_np), raw_np, depth_clip_hi)
                    Image.fromarray(to_uint8(raw_vis, 0.0, depth_clip_hi)).save(
                        os.path.join(args_cli.out_dir, f"{short}_depth_env{i}_step{step:02d}.png")
                    )

            if render_rgb:
                rgb = sensor.data.output["rgb"]
                stats(f"{short} rgb", rgb.float(), hi=255.0)
                for i in range(n_save):
                    rgb_np = rgb[i].detach().cpu().numpy()
                    if rgb_np.dtype != np.uint8:
                        rgb_np = (
                            np.clip(rgb_np, 0.0, 255.0).astype(np.uint8)
                            if rgb_np.max() > 1.5
                            else (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
                        )
                    rgb_img = Image.fromarray(rgb_np[..., :3])
                    rgb_img.save(
                        os.path.join(args_cli.out_dir, f"{short}_rgb_env{i}_step{step:02d}_full.png")
                    )
                    rgb_img.resize((RGB_W, RGB_H), Image.BILINEAR).save(
                        os.path.join(args_cli.out_dir, f"{short}_rgb_env{i}_step{step:02d}_224.png")
                    )

    print(f"\nSaved PNGs to {args_cli.out_dir}/")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
