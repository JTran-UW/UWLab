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
import os
import torch
from tensordict import TensorDict

# Isaac Lab ObservationManager.compute_group() builds a dict keyed by term name in dict
# insertion order (NOT the declaration order in ObservationsCfg), then concatenates
# list(group_obs.values()). For each term, history is flattened oldest→newest, so the
# newest frame of a term is the LAST (term_dim) entries of that term's block.
#
# The dict insertion order observed at runtime for OmniReset PolicyCfg:
#   insertive_asset_in_receptive_asset_frame (6) | prev_actions (7) | joint_pos (12)
#   | end_effector_pose (6) | insertive_asset_pose (6) | receptive_asset_pose (6)
# History=5 → 30 | 35 | 60 | 30 | 30 | 30 = 215 dims.
_JOINT_LABELS = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
    "finger", "r_outer_finger", "l_inner_knuckle", "r_inner_knuckle", "l_inner_finger", "r_inner_finger",
]
_EE_LABELS = ["ee_pos_x", "ee_pos_y", "ee_pos_z", "ee_aa_x", "ee_aa_y", "ee_aa_z"]

# Task → ordered list of (term_name, per_frame_dim) matching Isaac Lab's concat order.
_TASK_TERM_ORDER: dict[str, list[tuple[str, int]]] = {
    "OmniReset": [
        ("insertive_asset_in_receptive_asset_frame", 6),
        ("prev_actions", 7),
        ("joint_pos", 12),
        ("end_effector_pose", 6),
        ("insertive_asset_pose", 6),
        ("receptive_asset_pose", 6),
    ],
}


def _term_offsets(task: str | None, obs_dim: int) -> tuple[dict[str, tuple[int, int]], int] | None:
    """Return {term_name: (block_start, block_end)} and history, inferred from task + obs_dim.

    history is computed as obs_dim / sum(per_frame_dims) and asserted to be a positive integer.
    """
    if task is None:
        return None
    terms = None
    for key, t in _TASK_TERM_ORDER.items():
        if key in task:
            terms = t
            break
    if terms is None:
        return None
    per_frame_total = sum(d for _, d in terms)
    if obs_dim % per_frame_total != 0:
        print(f"[WARN] obs_dim={obs_dim} is not a multiple of per-frame total {per_frame_total} for task '{task}'.")
        return None
    history = obs_dim // per_frame_total
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, dim in terms:
        block_size = dim * history
        offsets[name] = (cursor, cursor + block_size)
        cursor += block_size
    return offsets, history


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


def _plot_newest_frame(obs: torch.Tensor, title: str,
                       block_start: int, block_end: int, term_dim: int,
                       labels: list[str], nrows: int = 1,
                       xlim_per_label: dict[str, tuple[float, float]] | None = None) -> None:
    """Plot per-dim histograms of the newest frame of a term block.

    The newest frame is the LAST term_dim entries of [block_start:block_end].
    `xlim_per_label` maps label → (xmin, xmax) to fix the x-axis for specific dims.
    """
    try:
        import math
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping histogram.")
        return

    start = block_end - term_dim
    end   = block_end

    obs_dim = obs.shape[-1]
    if end > obs_dim or start < block_start:
        print(f"[WARN] {title}: slice [{start}:{end}] out of range for block [{block_start}:{block_end}] / obs_dim={obs_dim}.")
        return

    flat   = obs.reshape(-1, obs_dim)
    values = flat[:, start:end].float().cpu()

    nrows = max(1, nrows)
    ncols = math.ceil(term_dim / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for i, lbl in enumerate(labels):
        ax = axes_flat[i]
        ax.hist(values[:, i].numpy(), bins=60, edgecolor="none")
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        if xlim_per_label is not None and lbl in xlim_per_label:
            ax.set_xlim(*xlim_per_label[lbl])
    for ax in axes_flat[term_dim:]:
        ax.set_visible(False)
    fig.suptitle(f"{title} (newest frame, dims {start}:{end} of block {block_start}:{block_end})", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_episode_end_breakdown(dones: torch.Tensor, truncations: torch.Tensor | None,
                               out_path: str | None = None, title_extra: str = "") -> None:
    """Bar chart: what fraction of stored transitions are episode ends, split by cause.

    The `dones` convention differs by writer. FastSAC stores the vec-env's dones, which are
    terminated|truncated, so a true termination is `dones & ~truncations`. Other collectors store
    dones already excluding truncations. Detect which by testing whether every truncation is also
    flagged done -- getting this backwards would swap the two bars.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping episode-end plot.")
        return

    d = dones.reshape(-1).bool()
    total = d.numel()
    if truncations is None:
        term = d
        trunc = torch.zeros_like(d)
        convention = "no truncation field; all dones counted as terminations"
    else:
        t = truncations.reshape(-1).bool()
        if t.numel() != total:
            print(f"[WARN] dones ({total}) and truncations ({t.numel()}) differ in size; skipping.")
            return
        union = bool((t & ~d).sum() == 0)  # truncations imply dones -> dones is the union
        if union:
            term, trunc = d & ~t, t
            convention = "dones = terminated|truncated"
        else:
            term, trunc = d, t
            convention = "dones excludes truncations"

    n_term, n_trunc = int(term.sum()), int(trunc.sum())
    n_ends = n_term + n_trunc
    n_ongoing = total - n_ends
    pct = lambda n: 100.0 * n / total if total else 0.0

    print(f"\n  episode-end breakdown ({convention})")
    print(f"    termination : {n_term:>10}  ({pct(n_term):.3f}% of samples)")
    print(f"    truncation  : {n_trunc:>10}  ({pct(n_trunc):.3f}%)")
    print(f"    any end     : {n_ends:>10}  ({pct(n_ends):.3f}%)")
    print(f"    mean episode length ~ {total / n_ends:.1f} steps" if n_ends else "    no episode ends recorded")

    labels = ["termination", "truncation", "non-terminal"]
    counts = [n_term, n_trunc, n_ongoing]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, [pct(c) for c in counts], color=["tab:green", "tab:orange", "tab:gray"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{c}\n{pct(c):.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of stored transitions")
    ax.set_title(f"Episode ends in replay buffer{title_extra}\n"
                 f"{total} transitions — {convention}", fontsize=10)
    ax.set_ylim(0, max(pct(c) for c in counts) * 1.25 or 1)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"    saved plot -> {out_path}")
    plt.show()


def plot_ee_pose_histograms(obs: torch.Tensor, task: str | None,
                            block_override: tuple[int, int] | None = None) -> None:
    if block_override is not None:
        block_start, block_end = block_override
    else:
        result = _term_offsets(task, obs.shape[-1])
        if result is None or "end_effector_pose" not in result[0]:
            print("[WARN] Cannot determine end_effector_pose offsets. Pass --ee_block_start/--ee_block_end.")
            return
        block_start, block_end = result[0]["end_effector_pose"]
    _plot_newest_frame(
        obs, f"EE Pose Histograms  task={task or 'unknown'}",
        block_start, block_end, term_dim=6, labels=_EE_LABELS,
        xlim_per_label={"ee_pos_x": (-1.0, 1.0), "ee_pos_y": (-1.0, 1.0), "ee_pos_z": (-1.0, 1.0)},
    )


def plot_joint_pos_histograms(obs: torch.Tensor, task: str | None,
                              block_override: tuple[int, int] | None = None) -> None:
    if block_override is not None:
        block_start, block_end = block_override
    else:
        result = _term_offsets(task, obs.shape[-1])
        if result is None or "joint_pos" not in result[0]:
            print("[WARN] Cannot determine joint_pos offsets. Pass --joint_block_start/--joint_block_end.")
            return
        block_start, block_end = result[0]["joint_pos"]
    _plot_newest_frame(
        obs, f"Joint Pos Histograms  task={task or 'unknown'}",
        block_start, block_end, term_dim=12, labels=_JOINT_LABELS, nrows=2,
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect a replay buffer (.pt file).")
    parser.add_argument("buffer", type=str, help="Path to replay buffer .pt file.")
    parser.add_argument("--task", type=str, default=None, help="Task name (informational, overrides metadata).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max per-env steps to inspect (default: all filled entries).")
    parser.add_argument("--no_per_dim", action="store_true",
                        help="Skip per-dimension stats.")
    parser.add_argument("--plot_episode_ends", action="store_true",
                        help="Bar chart of what %% of stored transitions are terminations vs "
                             "truncations vs non-terminal. Saved next to the buffer file.")
    parser.add_argument("--plot_ee_pose", action="store_true",
                        help="Plot per-dimension histograms of the EE pose from actor obs.")
    parser.add_argument("--plot_joint_pos", action="store_true",
                        help="Plot per-dimension histograms of joint positions from actor obs.")
    # Manual block overrides (only needed if task auto-detection fails)
    parser.add_argument("--ee_block_start", type=int, default=None, help="EE pose block start dim.")
    parser.add_argument("--ee_block_end", type=int, default=None, help="EE pose block end dim (exclusive).")
    parser.add_argument("--joint_block_start", type=int, default=None, help="Joint pos block start dim.")
    parser.add_argument("--joint_block_end", type=int, default=None, help="Joint pos block end dim (exclusive).")
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

    # Episode-end breakdown
    if args.plot_episode_ends:
        if parsed["dones"] is None:
            print("[WARN] buffer has no `dones` field; skipping --plot_episode_ends.")
        else:
            d = parsed["dones"][:, :parsed["limit"]] if parsed["dones"].dim() > 1 else parsed["dones"]
            t = parsed["truncations"]
            if t is not None and t.dim() > 1:
                t = t[:, :parsed["limit"]]
            out = os.path.splitext(args.buffer)[0] + "_episode_ends.png"
            plot_episode_end_breakdown(d, t, out_path=out, title_extra=f"  ({os.path.basename(args.buffer)})")

    # EE pose histograms
    if args.plot_ee_pose:
        ee_override = (
            (args.ee_block_start, args.ee_block_end)
            if args.ee_block_start is not None and args.ee_block_end is not None
            else None
        )
        plot_ee_pose_histograms(obs, task, block_override=ee_override)

    # Joint pos histograms
    if args.plot_joint_pos:
        joint_override = (
            (args.joint_block_start, args.joint_block_end)
            if args.joint_block_start is not None and args.joint_block_end is not None
            else None
        )
        plot_joint_pos_histograms(obs, task, block_override=joint_override)


if __name__ == "__main__":
    main()
