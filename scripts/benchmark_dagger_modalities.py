"""Benchmark memory + throughput for RGB vs depth DAgger.

Creates the Depth-DAgger env at a specified num_envs + modality (depth/rgb/both),
builds a matching DepthCNN encoder, and runs ``--bench_steps`` training-like
iterations (env.step + CNN forward + MSE backward). Reports peak VRAM, env
step throughput (env-steps/s), and wall-time per iteration so we can compare
the cost of a depth student vs an rgb student at the same env count.

Usage:
    python scripts/benchmark_dagger_modalities.py --num_envs 256 --modality depth
    python scripts/benchmark_dagger_modalities.py --num_envs 256 --modality rgb
    python scripts/benchmark_dagger_modalities.py --num_envs 256 --modality rgbd
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark RGB vs depth DAgger modality.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--modality", choices=["depth", "rgb", "rgbd"], default="depth")
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--bench_steps", type=int, default=60)
parser.add_argument("--img_h", type=int, default=224)
parser.add_argument("--img_w", type=int, default=224)
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab_tasks.utils import parse_env_cfg

import uwlab_tasks  # noqa: F401


def _conv_out(hw, k, s):
    return ((hw[0] - k) // s + 1, (hw[1] - k) // s + 1)


class DepthCNN(nn.Module):
    """Same architecture as source/uwlab_rl/.../student_teacher_vision.py."""

    def __init__(self, in_channels: int, height: int, width: int, embed_dim: int = 128):
        super().__init__()
        h1, w1 = _conv_out((height, width), 6, 2)
        h2, w2 = _conv_out((h1, w1), 4, 2)
        h3, w3 = _conv_out((h2, w2), 4, 2)
        h4, w4 = _conv_out((h3, w3), 4, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.LayerNorm([16, h1, w1]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm([32, h2, w2]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm([64, h3, w3]),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm([128, h4, w4]),
        )
        self.flat_dim = 128 * h4 * w4
        self.head = nn.Sequential(nn.Linear(self.flat_dim, embed_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


def main():
    device = torch.device(args_cli.device)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Modality-specific override: set camera data_types and render resolution.
    if args_cli.modality == "depth":
        data_types = ["distance_to_camera"]
        in_channels = 1
    elif args_cli.modality == "rgb":
        data_types = ["rgb"]
        in_channels = 3
    else:
        data_types = ["distance_to_camera", "rgb"]
        in_channels = 4
    env_cfg.scene.side_camera.data_types = data_types
    env_cfg.scene.side_camera.height = args_cli.img_h
    env_cfg.scene.side_camera.width = args_cli.img_w

    # Obs manager's side_depth term reads data_type="distance_to_camera"; swap it
    # to "rgb" for rgb-only so the obs group stays satisfied (we still read the
    # sensor output directly below — this just keeps the manager from crashing).
    if args_cli.modality == "rgb":
        env_cfg.observations.side_depth.image.params["data_type"] = "rgb"
        env_cfg.observations.side_depth.image.params["process_image"] = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    torch.cuda.reset_peak_memory_stats(device)
    mem_env_built = torch.cuda.memory_allocated(device) / 1e9

    encoder = DepthCNN(in_channels, args_cli.img_h, args_cli.img_w, embed_dim=128).to(device)
    num_actions = env.unwrapped.action_space.shape[1]
    action_head = nn.Sequential(nn.Linear(128 + 21, 256), nn.ReLU(), nn.Linear(256, num_actions)).to(device)
    optim = torch.optim.Adam(list(encoder.parameters()) + list(action_head.parameters()), lr=1e-4)

    sensor = env.unwrapped.scene.sensors["side_camera"]

    def get_image():
        if args_cli.modality == "depth":
            img = sensor.data.output["distance_to_camera"].clone()
            if img.ndim == 4 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            img = torch.nan_to_num(img, nan=2.0, posinf=2.0, neginf=2.0).clamp(0.01, 2.0)
            img = (img - 0.01) / (2.0 - 0.01)
            return img.unsqueeze(1)  # (N, 1, H, W)
        elif args_cli.modality == "rgb":
            rgb = sensor.data.output["rgb"].float()  # (N, H, W, 3)
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            return rgb.permute(0, 3, 1, 2)[:, :3]  # (N, 3, H, W)
        else:
            rgb = sensor.data.output["rgb"].float()
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            rgb = rgb.permute(0, 3, 1, 2)[:, :3]
            d = sensor.data.output["distance_to_camera"].clone()
            if d.ndim == 4 and d.shape[-1] == 1:
                d = d.squeeze(-1)
            d = torch.nan_to_num(d, nan=2.0, posinf=2.0, neginf=2.0).clamp(0.01, 2.0)
            d = (d - 0.01) / (2.0 - 0.01)
            return torch.cat([rgb, d.unsqueeze(1)], dim=1)  # (N, 4, H, W)

    obs, _ = env.reset()
    proprio_dim = obs["proprio"].shape[-1]
    if proprio_dim != 21:
        action_head = nn.Sequential(nn.Linear(128 + proprio_dim, 256), nn.ReLU(), nn.Linear(256, num_actions)).to(device)
        optim = torch.optim.Adam(list(encoder.parameters()) + list(action_head.parameters()), lr=1e-4)

    # Warmup.
    for _ in range(args_cli.warmup_steps):
        actions = torch.zeros(args_cli.num_envs, num_actions, device=device)
        obs, *_ = env.step(actions)
        img = get_image()
        feat = encoder(img)
        pred = action_head(torch.cat([feat, obs["proprio"]], dim=-1))
        target = torch.randn_like(pred)
        loss = F.mse_loss(pred, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.cuda.synchronize()

    # Benchmark.
    my_pid = os.getpid()

    def process_vram_gb() -> float:
        """Total VRAM held by this process on GPU 0 (includes IsaacSim native allocs)."""
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        )
        total_mib = 0
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(parts[0]) == my_pid:
                total_mib += int(parts[1])
        return total_mib / 1024.0

    torch.cuda.reset_peak_memory_stats(device)
    peak_process_vram = process_vram_gb()
    t0 = time.perf_counter()
    for step in range(args_cli.bench_steps):
        actions = torch.zeros(args_cli.num_envs, num_actions, device=device)
        obs, *_ = env.step(actions)
        img = get_image()
        feat = encoder(img)
        pred = action_head(torch.cat([feat, obs["proprio"]], dim=-1))
        target = torch.randn_like(pred)
        loss = F.mse_loss(pred, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 5 == 0:
            peak_process_vram = max(peak_process_vram, process_vram_gb())
    torch.cuda.synchronize()
    peak_process_vram = max(peak_process_vram, process_vram_gb())
    t1 = time.perf_counter()

    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
    total_time = t1 - t0
    iters_per_s = args_cli.bench_steps / total_time
    env_steps_per_s = iters_per_s * args_cli.num_envs

    print("\n" + "=" * 60)
    print(f"  modality       = {args_cli.modality}  ({in_channels} ch)")
    print(f"  num_envs       = {args_cli.num_envs}")
    print(f"  image          = {args_cli.img_h}x{args_cli.img_w}")
    print(f"  bench_steps    = {args_cli.bench_steps}")
    print(f"  peak VRAM (torch)   = {peak_vram:.2f} GB")
    print(f"  peak VRAM (process) = {peak_process_vram:.2f} GB")
    print(f"  wall time      = {total_time:.2f} s")
    print(f"  iters/s        = {iters_per_s:.2f}")
    print(f"  env-steps/s    = {env_steps_per_s:.0f}")
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
