#!/usr/bin/env python3
# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect a replay buffer and print statistics about observations, actions, and rewards.

Supported formats:
  1. PPO play.py recorded transitions (has "buffer_tensors" + "metadata" keys,
     tensors shaped [n_env, buffer_size, dim])
  2. rsl_rl ReplayBuffer (TensorDict-based, tensors shaped [capacity, n_env, dim],
     observations/next_observations are TensorDicts)
  3. FastSAC SimpleReplayBuffer (raw module or state_dict with flat tensor attributes)
"""

import argparse
import torch
from tensordict import TensorDict

# Per-step obs layout for known tasks.
# UR5e Robotiq 2F-85: joint_pos=12 (6 arm + 6 gripper linkage), confirmed from obs_dim=155=31*5, critic=32=31+1
_JOINT_LABELS = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
    "finger", "r_outer_finger", "l_inner_knuckle", "r_inner_knuckle", "l_inner_finger", "r_inner_finger",
]
_EE_LABELS = ["ee_pos_x", "ee_pos_y", "ee_pos_z", "ee_aa_x", "ee_aa_y", "ee_aa_z"]

_TASK_OBS_META = {
    # Reaching: prev_actions(7) + joint_pos(12) + ee_pose(6) + target_pose(6) = 31 per step, history 5
    "Reaching": {
        "per_step": 31, "history": 5,
        "joint_offset": 7,  "joint_len": 12, "joint_labels": _JOINT_LABELS,
        "ee_offset":    19, "ee_len":     6, "ee_labels":    _EE_LABELS,
    },
    # Insertion: prev_actions(7) + joint_pos(12) + ee_pose(6) + ins(6) + rec(6) + ins_in_rec(6) = 43, history 5
    "OmniReset": {
        "per_step": 43, "history": 5,
        "joint_offset": 7,  "joint_len": 12, "joint_labels": _JOINT_LABELS,
        "ee_offset":    19, "ee_len":     6, "ee_labels":    _EE_LABELS,
    },
}


def _get_obs_meta(task: str | None) -> dict | None:
    if task is None:
        return None
    for key, meta in _TASK_OBS_META.items():
        if key in task:
            return meta
    return None


def print_tensor_stats(name: str, t: torch.Tensor, per_dim: bool = True, indent: int = 2) -> None:
    """Print mean, std, min, max, and optionally per-dimension stats."""
    pad = " " * indent
    flat = t.reshape(-1, t.shape[-1]) if t.dim() > 1 else t.unsqueeze(-1)
    n_dim = flat.shape[-1]

    print(f"{pad}{name}  shape={tuple(t.shape)}  dtype={t.dtype}")
    print(f"{pad}  global : mean={flat.mean():.4f}  std={flat.std():.4f}  "
          f"min={flat.min():.4f}  max={flat.max():.4f}")

    if per_dim and n_dim <= 50:
        mean = flat.mean(dim=0)
        std = flat.std(dim=0)
        abs_max = flat.abs().max(dim=0).values
        print(f"{pad}  per-dim mean : [{', '.join(f'{v:.3f}' for v in mean.tolist())}]")
        print(f"{pad}  per-dim std  : [{', '.join(f'{v:.3f}' for v in std.tolist())}]")
        print(f"{pad}  per-dim |max|: [{', '.join(f'{v:.3f}' for v in abs_max.tolist())}]")
    elif per_dim:
        # Too many dims for inline display; just show summary
        mean = flat.mean(dim=0)
        std = flat.std(dim=0)
        print(f"{pad}  per-dim mean: min={mean.min():.3f}  max={mean.max():.3f}")
        print(f"{pad}  per-dim std : min={std.min():.3f}  max={std.max():.3f}")


def print_reward_stats(rewards: torch.Tensor, indent: int = 2) -> None:
    pad = " " * indent
    flat = rewards.reshape(-1).float()
    print(f"{pad}global : mean={flat.mean():.4f}  std={flat.std():.4f}  "
          f"min={flat.min():.4f}  max={flat.max():.4f}")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    qs = torch.quantile(flat, torch.tensor([p / 100.0 for p in percentiles]))
    print(f"{pad}percentiles: " + "  ".join(f"p{p}={v:.4f}" for p, v in zip(percentiles, qs.tolist())))


def print_norm_stats(name: str, t: torch.Tensor, indent: int = 2) -> None:
    pad = " " * indent
    flat = t.reshape(-1, t.shape[-1])
    norms = flat.norm(dim=-1)
    print(f"{pad}{name}: mean={norms.mean():.4f}  std={norms.std():.4f}  "
          f"min={norms.min():.4f}  max={norms.max():.4f}")


def print_done_stats(dones: torch.Tensor, truncations: torch.Tensor | None, indent: int = 2) -> None:
    pad = " " * indent
    total = dones.numel()
    n_done = int(dones.sum().item())
    n_trunc = int(truncations.sum().item()) if truncations is not None else 0
    print(f"{pad}total transitions : {total}")
    print(f"{pad}dones             : {n_done}  ({100*n_done/total:.2f}%)")
    if truncations is not None:
        print(f"{pad}truncations       : {n_trunc}  ({100*n_trunc/total:.2f}%)")


# ---------------------------------------------------------------------------
# Format parsers — each returns a normalized dict of sliced tensors + metadata
# ---------------------------------------------------------------------------

def parse_ppo_play_format(data: dict, limit: int | None) -> dict:
    """PPO play.py recorded transitions: {'buffer_tensors': {...}, 'metadata': {...}}"""
    tensors = data["buffer_tensors"]
    meta = data["metadata"]

    ptr = int(tensors.get("ptr", meta.get("buffer_size", -1)))
    n_env = meta["n_env"]
    buffer_size = meta["buffer_size"]
    filled = min(ptr, buffer_size) if ptr >= 0 else buffer_size
    lim = min(limit, filled) if limit else filled

    return {
        "format": "PPO play.py (SimpleReplayBuffer)",
        "n_env": n_env,
        "buffer_size": buffer_size,
        "ptr": ptr,
        "filled": filled,
        "limit": lim,
        "task": meta.get("task"),
        "actor_obs_keys": meta.get("actor_obs_keys"),
        "critic_obs_keys": meta.get("critic_obs_keys"),
        # Tensors shaped [n_env, buffer_size, dim] → slice on axis 1
        "observations": tensors["observations"][:, :lim],
        "critic_observations": tensors.get("critic_observations", tensors["observations"])[:, :lim],
        "actions": tensors["actions"][:, :lim],
        "rewards": tensors["rewards"][:, :lim],
        "dones": tensors.get("dones", None),
        "truncations": tensors.get("truncations", None),
    }


def parse_rsl_rl_replay_buffer(data: dict, limit: int | None) -> dict:
    """rsl_rl ReplayBuffer: TensorDict observations, tensors shaped [capacity, n_env, dim]."""
    # observations is a TensorDict with obs-group keys
    obs_td = data["observations"]
    if isinstance(obs_td, TensorDict):
        # Flatten all obs group tensors into a single [capacity, n_env, total_dim]
        obs_keys = list(obs_td.keys())
        capacity, n_env = obs_td[obs_keys[0]].shape[:2]
        obs_cat = torch.cat([obs_td[k] for k in obs_keys], dim=-1)
    else:
        obs_cat = obs_td
        capacity, n_env = obs_cat.shape[:2]
        obs_keys = None

    pos = int(data.get("_pos", data.get("ptr", capacity)))
    size = int(data.get("_size", min(pos, capacity)))
    lim = min(limit, size) if limit else size

    # Critic obs: same as obs for on-policy (critic groups share the obs TensorDict)
    actions = data["actions"][:lim]    # [capacity, n_env, act_dim]
    rewards = data["rewards"][:lim]    # [capacity, n_env, 1]
    dones = data.get("dones", None)

    return {
        "format": "rsl_rl ReplayBuffer (TensorDict)",
        "n_env": n_env,
        "buffer_size": capacity,
        "ptr": pos,
        "filled": size,
        "limit": lim,
        "task": None,
        "actor_obs_keys": obs_keys,
        "critic_obs_keys": obs_keys,
        # Transpose to [n_env, limit, dim] for consistency
        "observations": obs_cat[:lim].transpose(0, 1),
        "critic_observations": obs_cat[:lim].transpose(0, 1),
        "actions": actions.transpose(0, 1),
        "rewards": rewards.transpose(0, 1),
        "dones": dones[:lim].transpose(0, 1) if dones is not None else None,
        "truncations": None,
    }


def parse_simple_replay_buffer(data: dict, limit: int | None) -> dict:
    """FastSAC SimpleReplayBuffer: flat tensors shaped [n_env, buffer_size, dim]."""
    obs = data["observations"]
    n_env, buffer_size = obs.shape[:2]
    ptr = int(data.get("ptr", buffer_size))
    filled = min(ptr, buffer_size) if ptr >= 0 else buffer_size
    lim = min(limit, filled) if limit else filled

    return {
        "format": "FastSAC SimpleReplayBuffer",
        "n_env": n_env,
        "buffer_size": buffer_size,
        "ptr": ptr,
        "filled": filled,
        "limit": lim,
        "task": None,
        "actor_obs_keys": None,
        "critic_obs_keys": None,
        "observations": obs[:, :lim],
        "critic_observations": data.get("critic_observations", obs)[:, :lim],
        "actions": data["actions"][:, :lim],
        "rewards": data["rewards"][:, :lim],
        "dones": data.get("dones", None),
        "truncations": data.get("truncations", None),
    }


def detect_and_parse(data, limit: int | None) -> dict:
    """Auto-detect buffer format and parse into a normalized dict."""
    if isinstance(data, dict):
        # Format 1: PPO play.py recorded transitions
        if "buffer_tensors" in data and "metadata" in data:
            return parse_ppo_play_format(data, limit)

        # Format 2: rsl_rl ReplayBuffer (TensorDict observations)
        if "observations" in data and isinstance(data["observations"], TensorDict):
            return parse_rsl_rl_replay_buffer(data, limit)

        # Format 3: FastSAC SimpleReplayBuffer (flat tensors)
        if "observations" in data and isinstance(data["observations"], torch.Tensor):
            return parse_simple_replay_buffer(data, limit)

    # Format 3b: module object with tensor attributes
    if hasattr(data, "observations") and isinstance(data.observations, torch.Tensor):
        return parse_simple_replay_buffer(data.__dict__, limit)

    available = list(data.keys()) if isinstance(data, dict) else dir(data)
    raise ValueError(f"Unrecognized replay buffer format. Available keys/attrs: {available}")


def _plot_obs_slice(obs: torch.Tensor, title: str, offset: int, length: int,
                    labels: list[str], per_step: int, history: int, nrows: int = 1) -> None:
    """Extract a named slice from the most-recent history frame and show per-dim histograms."""
    try:
        import math
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping histogram.")
        return

    obs_dim = obs.shape[-1]
    expected = per_step * history
    if obs_dim != expected:
        print(f"[WARN] {title}: obs_dim={obs_dim} but expected {per_step}*{history}={expected}. Skipping.")
        return

    last_frame_start = (history - 1) * per_step
    start = last_frame_start + offset
    end   = start + length

    flat   = obs.reshape(-1, obs_dim)
    values = flat[:, start:end].float().cpu()  # [N, length]

    nrows = max(1, nrows)
    ncols = math.ceil(length / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for i, lbl in enumerate(labels):
        ax = axes_flat[i]
        ax.hist(values[:, i].numpy(), bins=60, edgecolor="none")
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    for ax in axes_flat[length:]:  # hide unused subplots
        ax.set_visible(False)
    fig.suptitle(f"{title} (most-recent frame)", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_ee_pose_histograms(obs: torch.Tensor, task: str | None,
                            per_step: int | None = None, history: int | None = None,
                            ee_offset: int | None = None, ee_len: int | None = None,
                            ee_labels: list[str] | None = None) -> None:
    meta = _get_obs_meta(task)
    per_step  = per_step  or (meta["per_step"]  if meta else None)
    history   = history   or (meta["history"]   if meta else None)
    offset    = ee_offset or (meta["ee_offset"] if meta else None)
    length    = ee_len    or (meta["ee_len"]     if meta else None)
    labels    = ee_labels or (meta["ee_labels"]  if meta else None)
    if any(v is None for v in [per_step, history, offset, length]):
        print("[WARN] Cannot determine obs layout for EE pose. Pass --per_step/--history/--ee_offset/--ee_len.")
        return
    if labels is None:
        labels = [f"ee_{i}" for i in range(length)]
    _plot_obs_slice(obs, f"EE Pose Histograms  task={task or 'unknown'}", offset, length, labels, per_step, history)


def plot_joint_pos_histograms(obs: torch.Tensor, task: str | None,
                              per_step: int | None = None, history: int | None = None,
                              joint_offset: int | None = None, joint_len: int | None = None,
                              joint_labels: list[str] | None = None) -> None:
    meta = _get_obs_meta(task)
    per_step     = per_step     or (meta["per_step"]      if meta else None)
    history      = history      or (meta["history"]       if meta else None)
    offset       = joint_offset or (meta["joint_offset"]  if meta else None)
    length       = joint_len    or (meta["joint_len"]     if meta else None)
    labels       = joint_labels or (meta["joint_labels"]  if meta else None)
    if any(v is None for v in [per_step, history, offset, length]):
        print("[WARN] Cannot determine obs layout for joint pos. Pass --per_step/--history/--joint_offset/--joint_len.")
        return
    if labels is None:
        labels = [f"joint_{i}" for i in range(length)]
    _plot_obs_slice(obs, f"Joint Pos Histograms  task={task or 'unknown'}", offset, length, labels, per_step, history, nrows=2)


def main():
    parser = argparse.ArgumentParser(description="Inspect a replay buffer (.pt file).")
    parser.add_argument("buffer", type=str, help="Path to replay buffer .pt file.")
    parser.add_argument("--task", type=str, default=None, help="Task name (informational, overrides metadata).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max per-env steps to inspect (default: all filled entries).")
    parser.add_argument("--no_per_dim", action="store_true",
                        help="Skip per-dimension stats.")
    parser.add_argument("--plot_ee_pose", action="store_true",
                        help="Plot per-dimension histograms of the EE pose from actor obs.")
    parser.add_argument("--plot_joint_pos", action="store_true",
                        help="Plot per-dimension histograms of joint positions from actor obs.")
    # Manual obs-layout overrides (only needed if task auto-detection fails)
    parser.add_argument("--per_step", type=int, default=None, help="Obs dims per timestep.")
    parser.add_argument("--history", type=int, default=None, help="History length.")
    parser.add_argument("--ee_offset", type=int, default=None, help="EE pose start dim within one frame.")
    parser.add_argument("--ee_len", type=int, default=None, help="Number of EE pose dims.")
    parser.add_argument("--joint_offset", type=int, default=None, help="Joint pos start dim within one frame.")
    parser.add_argument("--joint_len", type=int, default=None, help="Number of joint pos dims.")
    args = parser.parse_args()

    print(f"Loading: {args.buffer}")
    data = torch.load(args.buffer, map_location="cpu", weights_only=False)
    parsed = detect_and_parse(data, args.max_samples)

    task = args.task or parsed.get("task")
    per_dim = not args.no_per_dim

    # Header
    print(f"\n{'='*60}")
    print(f"Buffer info")
    print(f"{'='*60}")
    print(f"  format      : {parsed['format']}")
    if task:
        print(f"  task        : {task}")
    print(f"  n_env       : {parsed['n_env']}")
    print(f"  buffer_size : {parsed['buffer_size']}")
    print(f"  ptr         : {parsed['ptr']}")
    print(f"  filled      : {parsed['filled']}  (inspecting {parsed['limit']})")
    obs = parsed["observations"]
    cobs = parsed["critic_observations"]
    acts = parsed["actions"]
    print(f"  obs_dim     : {obs.shape[-1]}")
    print(f"  critic_dim  : {cobs.shape[-1]}")
    print(f"  act_dim     : {acts.shape[-1]}")
    if parsed.get("actor_obs_keys"):
        print(f"  actor_keys  : {parsed['actor_obs_keys']}")
    if parsed.get("critic_obs_keys"):
        print(f"  critic_keys : {parsed['critic_obs_keys']}")

    # Observations
    print(f"\n{'='*60}")
    print("Observations (actor)")
    print(f"{'='*60}")
    print_tensor_stats("obs", obs, per_dim=per_dim)

    print(f"\n{'='*60}")
    print("Observations (critic)")
    print(f"{'='*60}")
    print_tensor_stats("critic_obs", cobs, per_dim=per_dim)

    # Actions
    print(f"\n{'='*60}")
    print("Actions")
    print(f"{'='*60}")
    print_tensor_stats("actions", acts, per_dim=per_dim)

    # Rewards
    print(f"\n{'='*60}")
    print("Rewards")
    print(f"{'='*60}")
    print_reward_stats(parsed["rewards"])

    # Dones / truncations
    if parsed["dones"] is not None:
        print(f"\n{'='*60}")
        print("Episode terminations")
        print(f"{'='*60}")
        dones_s = parsed["dones"][:, :parsed["limit"]] if parsed["dones"].dim() > 1 else parsed["dones"]
        trunc_s = parsed["truncations"]
        if trunc_s is not None and trunc_s.dim() > 1:
            trunc_s = trunc_s[:, :parsed["limit"]]
        print_done_stats(dones_s, trunc_s)

    # L2 norms
    print(f"\n{'='*60}")
    print("L2 norms per step")
    print(f"{'='*60}")
    print_norm_stats("actor obs ", obs)
    print_norm_stats("critic obs", cobs)
    print_norm_stats("actions   ", acts)

    # EE pose histograms
    if args.plot_ee_pose:
        plot_ee_pose_histograms(
            obs, task,
            per_step=args.per_step,
            history=args.history,
            ee_offset=args.ee_offset,
            ee_len=args.ee_len,
        )

    # Joint pos histograms
    if args.plot_joint_pos:
        plot_joint_pos_histograms(
            obs, task,
            per_step=args.per_step,
            history=args.history,
            joint_offset=args.joint_offset,
            joint_len=args.joint_len,
        )


if __name__ == "__main__":
    main()
