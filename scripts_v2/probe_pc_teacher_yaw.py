"""Probe: does a ScenePC peg teacher produce unimodal or multimodal actions across peg yaw?

Reconstruct the actor (PC encoder + MLP + normalizers) from a training checkpoint,
then sweep peg yaw at otherwise-fixed scene context and plot action vs yaw.

Multimodal hypothesis: encoder learned point-identity -> yaw, so action snaps to the
closest of 4 (or 8) canonical yaw modes -> discontinuous jumps at canonical boundaries.

Unimodal hypothesis: encoder is yaw-equivariant (SO(2) symmetric), so action varies
smoothly (or stays roughly constant) as peg yaw rotates.

Usage:
    conda activate env_uwlab
    python scripts_v2/probe_pc_teacher_yaw.py \
        --checkpoint checkpoints/pc_teacher/peg_pc_prevact.pt \
        --proprio-dim 25 \
        --out scripts/probe_yaw_prevact.png

    python scripts_v2/probe_pc_teacher_yaw.py \
        --checkpoint checkpoints/pc_teacher/peg_pc_noprevact.pt \
        --proprio-dim 18 \
        --out scripts/probe_yaw_noprevact.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

NUM_POINTS = 512
NUM_ACTIONS = 7

# Approximate split (training prints "robot=N_R, insertive=N_I, receptive=N_RE");
# FPS-driven so it varies, but typical for 512-pt scene-pc is roughly 200/156/156.
N_ROBOT = 200
N_INSERTIVE = 156
N_RECEPTIVE = NUM_POINTS - N_ROBOT - N_INSERTIVE  # 156


def build_mlp(state_dict: dict, prefix: str, activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    """Reconstruct Linear+activation Sequential from rsl_rl-style indexed state dict."""
    idx_re = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    indices = sorted(int(m.group(1)) for k in state_dict if (m := idx_re.match(k)))
    assert indices, f"no Linear weights found under prefix '{prefix}'"
    layers: list[nn.Module] = []
    for i, idx in enumerate(indices):
        w = state_dict[f"{prefix}.{idx}.weight"]
        b = state_dict[f"{prefix}.{idx}.bias"]
        out_dim, in_dim = w.shape
        lin = nn.Linear(in_dim, out_dim)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(indices) - 1:
            layers.append(activation())
    return nn.Sequential(*layers)


class TeacherActor(nn.Module):
    """Reconstructed teacher: normalize(proprio) + normalize(pc) -> encoder -> concat -> actor MLP -> mean."""

    def __init__(self, state_dict: dict, proprio_dim: int, pc_dim: int = NUM_POINTS * 3):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.pc_dim = pc_dim

        # Normalizers (EmpiricalNormalization stores per-feature mean/std).
        self.register_buffer("proprio_mean", state_dict["_group_normalizers.proprio._mean"])
        self.register_buffer("proprio_std", state_dict["_group_normalizers.proprio._std"])
        self.register_buffer("pc_mean", state_dict["_group_normalizers.pointcloud._mean"])
        self.register_buffer("pc_std", state_dict["_group_normalizers.pointcloud._std"])

        # Encoder: 1536 -> 256 -> 128 -> 32
        self.pc_encoder = build_mlp(state_dict, "actor_encoders.pointcloud")
        # Actor: (proprio_dim + 32) -> 512 -> 256 -> 128 -> 64 -> 7
        self.actor = build_mlp(state_dict, "actor")
        self.eps = 1e-2

    def forward(self, proprio: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        # Normalize.
        proprio_n = (proprio - self.proprio_mean) / (self.proprio_std + self.eps)
        pc_n = (pc - self.pc_mean) / (self.pc_std + self.eps)
        # Encode PC.
        pc_enc = self.pc_encoder(pc_n)  # (B, 32)
        # Concat and run actor.
        x = torch.cat([proprio_n, pc_enc], dim=-1)
        return self.actor(x)


def sample_cylinder_points(n: int, radius: float, length: float, seed: int = 0) -> torch.Tensor:
    """Sample n points uniformly on a cylinder surface (no caps) of given radius and length.

    Cylinder axis is local z-axis, centered at origin.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-length / 2, length / 2, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return torch.tensor(np.stack([x, y, z], axis=-1), dtype=torch.float32)


def sample_box_with_hole_points(n: int, box_size: float, hole_radius: float, depth: float, seed: int = 1) -> torch.Tensor:
    """Sample n points on a flat plate (box top surface) with a circular hole in the middle.

    Plate is at z=0, extends [-box_size/2, box_size/2] x [-box_size/2, box_size/2].
    Hole carved out at radius hole_radius around origin.
    """
    rng = np.random.default_rng(seed)
    pts = []
    # Top surface (z=0) outside hole
    while len(pts) < n // 2:
        x = rng.uniform(-box_size / 2, box_size / 2)
        y = rng.uniform(-box_size / 2, box_size / 2)
        if x * x + y * y > hole_radius * hole_radius:
            pts.append([x, y, 0.0])
    # Inner wall of hole, going down to -depth
    for _ in range(n - len(pts)):
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-depth, 0)
        pts.append([hole_radius * np.cos(theta), hole_radius * np.sin(theta), z])
    return torch.tensor(pts[:n], dtype=torch.float32)


def sample_robot_points(n: int, seed: int = 2) -> torch.Tensor:
    """Synthetic robot wrist-region points: cluster around (0.4, 0.0, 0.2) with some spread.

    These represent the robot arm + gripper in robot base frame.
    """
    rng = np.random.default_rng(seed)
    center = np.array([0.4, 0.0, 0.2])
    pts = rng.normal(center, scale=[0.15, 0.1, 0.15], size=(n, 3))
    return torch.tensor(pts, dtype=torch.float32)


def build_scene_pc(peg_yaw_rad: float, peg_xy: tuple[float, float] = (0.5, 0.0), peg_z: float = 0.15,
                   hole_xy: tuple[float, float] = (0.5, 0.0), hole_z: float = 0.10,
                   seed: int = 0) -> torch.Tensor:
    """Build one scene PC at given peg yaw.

    Returns (1, NUM_POINTS * 3) flattened to match teacher input.

    All coordinates in robot base frame.
    """
    # Peg in local frame (axis = z)
    peg_local = sample_cylinder_points(N_INSERTIVE, radius=0.005, length=0.04, seed=seed)
    # Rotate around z by peg_yaw, then translate to peg_xy/peg_z
    c, s = np.cos(peg_yaw_rad), np.sin(peg_yaw_rad)
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    peg_world = peg_local @ R.T
    peg_world[:, 0] += peg_xy[0]
    peg_world[:, 1] += peg_xy[1]
    peg_world[:, 2] += peg_z

    # Hole at hole_xy/hole_z
    hole_world = sample_box_with_hole_points(N_RECEPTIVE, box_size=0.08, hole_radius=0.006, depth=0.04, seed=seed + 100)
    hole_world[:, 0] += hole_xy[0]
    hole_world[:, 1] += hole_xy[1]
    hole_world[:, 2] += hole_z

    # Robot points (fixed across yaw sweep)
    robot_world = sample_robot_points(N_ROBOT, seed=seed + 200)

    # Concatenate: order is robot, ins, rec (matches ScenePointCloud._SRC_* enum).
    pc = torch.cat([robot_world, peg_world, hole_world], dim=0)  # (NUM_POINTS, 3)
    return pc.flatten().unsqueeze(0)  # (1, NUM_POINTS*3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--proprio-dim", type=int, required=True, help="25 for prev_act variant, 18 for noprevact")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n-yaws", type=int, default=181, help="Number of yaw samples in [0, 2pi)")
    p.add_argument("--n-scenes", type=int, default=8, help="Number of distinct (hole_pos, robot_pos) scenes to average over")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Sanity: verify proprio dim matches the actor input.
    actor_in_dim = sd["actor.0.weight"].shape[1]
    expected = args.proprio_dim + 32
    assert actor_in_dim == expected, (
        f"actor.0.weight in_dim={actor_in_dim} but expected proprio({args.proprio_dim}) + encoded(32) = {expected}"
    )
    print(f"[ok] checkpoint loaded, iter={ckpt.get('iter', '?')}")
    print(f"     proprio_dim={args.proprio_dim}, actor_in_dim={actor_in_dim}")

    actor = TeacherActor(sd, proprio_dim=args.proprio_dim)
    actor.eval()

    # Use proprio = normalizer mean -> normalized to zero. Most "typical" input.
    proprio = actor.proprio_mean.clone()

    yaws = np.linspace(0, 2 * np.pi, args.n_yaws, endpoint=False)
    actions = np.zeros((args.n_scenes, args.n_yaws, NUM_ACTIONS))

    with torch.no_grad():
        for scene_idx in range(args.n_scenes):
            for yaw_idx, yaw in enumerate(yaws):
                pc = build_scene_pc(peg_yaw_rad=float(yaw), seed=scene_idx)
                a = actor(proprio, pc)  # (1, 7)
                actions[scene_idx, yaw_idx] = a.squeeze(0).numpy()

    # Per-component variation analysis.
    yaw_std = actions.std(axis=1)  # (n_scenes, NUM_ACTIONS) — std across yaw within each scene
    scene_std = actions.std(axis=0)  # (n_yaws, NUM_ACTIONS) — std across scene within each yaw
    yaw_std_mean = yaw_std.mean(axis=0)  # (NUM_ACTIONS,)
    scene_std_mean = scene_std.mean(axis=0)  # (NUM_ACTIONS,)
    ratio = scene_std_mean / np.maximum(yaw_std_mean, 1e-6)

    print()
    print("Per-action-component variation:")
    print(f"  {'comp':<10s} {'yaw_std':>10s} {'scene_std':>10s} {'ratio':>10s}")
    action_names = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
    for i, name in enumerate(action_names):
        print(f"  {name:<10s} {yaw_std_mean[i]:>10.4f} {scene_std_mean[i]:>10.4f} {ratio[i]:>9.1f}x")
    print(f"  {'OVERALL':<10s} {yaw_std_mean.mean():>10.4f} {scene_std_mean.mean():>10.4f} {(scene_std_mean.mean()/yaw_std_mean.mean()):>9.1f}x")

    yaws_deg = np.rad2deg(yaws)

    # Plot: 7 subplots. For each, plot mean action across scenes (bold) +/- across-scene std band.
    fig, axes = plt.subplots(NUM_ACTIONS, 1, figsize=(12, 14), sharex=True)
    for i in range(NUM_ACTIONS):
        ax = axes[i]
        per_yaw_mean = actions[:, :, i].mean(axis=0)
        per_yaw_std = actions[:, :, i].std(axis=0)
        # Across-scene-std band shows what variation looks like at *fixed* yaw — i.e., scene noise.
        ax.fill_between(yaws_deg, per_yaw_mean - per_yaw_std, per_yaw_mean + per_yaw_std,
                        color="gray", alpha=0.25, label="±std across scene (the noise floor)")
        # Each scene as faint line (showing yaw curve at fixed scene)
        for s in range(args.n_scenes):
            ax.plot(yaws_deg, actions[s, :, i], color="tab:blue", alpha=0.4, linewidth=0.8)
        # Mean across scenes (the apparent yaw response)
        ax.plot(yaws_deg, per_yaw_mean, color="black", linewidth=2, label="mean across scenes")
        # Canonical boundaries
        for x in [0, 90, 180, 270]:
            ax.axvline(x, color="red", linestyle="--", alpha=0.3, linewidth=1)
        for x in [45, 135, 225, 315]:
            ax.axvline(x, color="orange", linestyle="--", alpha=0.2, linewidth=1)
        # Annotate per-component ratio
        ax.text(0.99, 0.95, f"yaw_std={yaw_std_mean[i]:.3f}, scene_std={scene_std_mean[i]:.2f} ({ratio[i]:.0f}x)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_ylabel(f"action[{i}]\n{action_names[i]}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Peg yaw (deg)")
    axes[0].legend(loc="upper left", fontsize=9)

    overall_ratio = scene_std_mean.mean() / yaw_std_mean.mean()
    verdict = "UNIMODAL (yaw-invariant)" if overall_ratio > 10 else "AMBIGUOUS / multimodal candidate"
    fig.suptitle(
        f"Teacher action vs peg yaw — {Path(args.checkpoint).name} (iter={ckpt.get('iter','?')})\n"
        f"{args.n_scenes} synthetic scenes × {args.n_yaws} yaws, proprio=normalizer_mean\n"
        f"red dashed = 4-canonical boundaries, orange dashed = 8-canonical boundaries\n"
        f"Overall: scene_std/yaw_std = {overall_ratio:.0f}x → {verdict}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"[ok] saved {args.out}")


if __name__ == "__main__":
    main()
