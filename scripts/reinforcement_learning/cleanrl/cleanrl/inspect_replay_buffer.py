#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect a saved replay buffer: numeric statistics, or image grids / videos for camera buffers.

Needs no Isaac/simulation import, so it starts instantly, and mmaps the file so a 32 GiB buffer
costs page reads rather than 32 GiB of RAM.

Subcommands
-----------
* ``stats`` -- print global / per-dim statistics of observations, actions and rewards, episode-end
  counts and L2 norms. Optional plots: episode-end breakdown, end-effector-pose and joint-position
  histograms (the newest history frame of the OmniReset obs layout).
* ``grid``  -- a (num_cameras x num_samples) grid of random transitions from an image buffer; each
  camera gets a row and each column is one sampled transition.
* ``video`` -- a contiguous run of steps from ONE env, written as one video per camera. Rows of the
  stored tensor are chronological per env, so a slice ``[start : start+N]`` at a fixed env is a real
  trajectory. Episode boundaries are drawn as a coloured border and counted in the overlay.

Supported on-disk formats (auto-detected)
-----------------------------------------
1. ``collect_expert_replay_buffer.py`` payload -- ``{"buffer_tensors", "metadata"}`` with
   ``policy_observations`` shaped ``[steps, n_env, dim]``, ``pos``/``full``, ``terminations``/``truncations``.
2. holosoma ``play.py`` recorder payload -- ``{"buffer_tensors", "metadata"}`` with ``observations``
   shaped ``[n_env, steps, dim]``, ``ptr``, ``dones``/``truncations``.
3. rsl_rl ``ReplayBuffer`` -- TensorDict observations shaped ``[capacity, n_env, dim]``, ``_pos``/``_size``.
4. FastSAC ``SimpleReplayBuffer`` -- raw module or state_dict with flat tensors ``[n_env, steps, dim]``, ``ptr``.

Everything is normalised to a steps-first ``[steps, n_env, ...]`` view before any subcommand runs.

Image layout
------------
The policy stream of a camera buffer is stored flat; for the three-camera tasks each row unpacks as
(history, camera, H, W) = (3, 3, 84, 84), with history index 0 oldest and -1 most recent
(isaaclab CircularBuffer.buffer puts the newest entry last), and cameras in the order the task cfg
lists them: front, side, wrist. Two modalities are auto-detected from the metadata:

* ``grayscale`` -- rgb collapsed to luma, values scaled to [0, 1].
* ``depth``     -- ``distance_to_camera`` in raw metres. ``process_image`` maps inf (nothing hit within
  the clipping range) to exactly 0.0, so 0.0 means "no return", NOT "very close". Those pixels are
  masked out of the colour scale and drawn in red.

Examples
--------
    python inspect_replay_buffer.py stats fast_sac_transitions.pt --task OmniReset-...-OffPolicy-v0
    python inspect_replay_buffer.py stats expert_rb/peg.pt --plot_episode_ends --plot_ee_pose
    python inspect_replay_buffer.py grid  expert_rb/peg_grayscale_asym_seed42.pt --shared_scale
    python inspect_replay_buffer.py video expert_rb/peg_grayscale_asym_seed42.pt --video_steps 160
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

INVALID_COLOR = "red"
INVALID_RGB = (255, 0, 0)
BOUNDARY_RGB = (255, 215, 0)

# ---------------------------------------------------------------------------
# OmniReset observation layout (used by the stats histograms)
# ---------------------------------------------------------------------------
# Isaac Lab ObservationManager.compute_group() builds a dict keyed by term name in dict insertion
# order (NOT the declaration order in ObservationsCfg), then concatenates list(group_obs.values()).
# For each term, history is flattened oldest -> newest, so the newest frame of a term is the LAST
# (term_dim) entries of that term's block.
#
# Insertion order observed at runtime for OmniReset PolicyCfg:
#   insertive_asset_in_receptive_asset_frame (6) | prev_actions (7) | joint_pos (12)
#   | end_effector_pose (6) | insertive_asset_pose (6) | receptive_asset_pose (6)
# History=5 -> 30 | 35 | 60 | 30 | 30 | 30 = 215 dims.
_JOINT_LABELS = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
    "finger", "r_outer_finger", "l_inner_knuckle", "r_inner_knuckle", "l_inner_finger", "r_inner_finger",
]
_EE_LABELS = ["ee_pos_x", "ee_pos_y", "ee_pos_z", "ee_aa_x", "ee_aa_y", "ee_aa_z"]

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


# ---------------------------------------------------------------------------
# Canonical view + loader
# ---------------------------------------------------------------------------
@dataclass
class BufferView:
    """Every supported format normalised to steps-first tensors.

    ``streams`` holds every stored observation-like tensor (already steps-first) so the image modes
    can pick e.g. ``next_policy_observations``; ``obs`` / ``critic_obs`` are the defaults.
    """

    format: str
    n_env: int
    buffer_size: int
    n_valid: int              # number of filled steps (per env)
    ptr: int                  # write pointer (pos / ptr / _pos, whichever the writer used)
    wrapped: bool             # ring buffer wrapped and stopped mid-ring -> rows not chronological
    obs: torch.Tensor         # [steps, n_env, obs_dim]
    critic_obs: torch.Tensor  # [steps, n_env, critic_dim]
    actions: torch.Tensor     # [steps, n_env, act_dim]
    rewards: torch.Tensor     # [steps, n_env]
    dones: torch.Tensor | None          # [steps, n_env] bool -- writer's "done" flag (convention varies)
    truncations: torch.Tensor | None    # [steps, n_env] bool
    task: str | None = None
    actor_obs_keys: list[str] | None = None
    critic_obs_keys: list[str] | None = None
    meta: dict = field(default_factory=dict)
    streams: dict[str, torch.Tensor] = field(default_factory=dict)

    def sliced(self, limit: int | None) -> "BufferView":
        """Restrict to the first ``limit`` valid steps (no copy for mmapped tensors)."""
        lim = self.n_valid if limit is None else min(limit, self.n_valid)
        cut = lambda t: None if t is None else t[:lim]
        return BufferView(
            format=self.format, n_env=self.n_env, buffer_size=self.buffer_size, n_valid=lim, ptr=self.ptr,
            wrapped=self.wrapped, obs=cut(self.obs), critic_obs=cut(self.critic_obs), actions=cut(self.actions),
            rewards=cut(self.rewards), dones=cut(self.dones), truncations=cut(self.truncations), task=self.task,
            actor_obs_keys=self.actor_obs_keys, critic_obs_keys=self.critic_obs_keys, meta=self.meta,
            streams={k: v[:lim] for k, v in self.streams.items()},
        )

    def episode_ends(self) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
        """Return (terminations, truncations, convention) with the writer's done convention resolved.

        FastSAC-style writers store the vec-env's dones, which are terminated|truncated, so a true
        termination is ``dones & ~truncations``. The collector stores pure terminations. Detect which
        by testing whether every truncation is also flagged done -- getting this backwards would
        swap the two categories.
        """
        if self.dones is None:
            return None, None, "no done field"
        d = self.dones.bool()
        if self.truncations is None:
            return d, torch.zeros_like(d), "no truncation field; all dones counted as terminations"
        t = self.truncations.bool()
        if t.shape != d.shape:
            return d, None, f"dones {tuple(d.shape)} and truncations {tuple(t.shape)} differ in shape"
        if bool((t & ~d).sum() == 0) and bool(t.any()):
            return d & ~t, t, "dones = terminated|truncated"
        return d, t, "dones excludes truncations"


def load_payload(path: str):
    """Load with mmap when possible so a huge buffer costs a few page reads, not a full read."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError) as exc:  # not zipfile-serialized, or torch too old
        print(f"[warn] mmap load failed ({exc}); falling back to a full read -- this may take a while.")
        return torch.load(path, map_location="cpu", weights_only=False)


def _as_bool(t: torch.Tensor | None) -> torch.Tensor | None:
    return None if t is None else t.bool()


def _parse_collector(payload: dict) -> BufferView:
    """collect_expert_replay_buffer.py / AsymmetricReplayBuffer: steps-first, pos/full."""
    t, meta = payload["buffer_tensors"], payload["metadata"]
    obs = t["policy_observations"]
    buffer_size, n_env = obs.shape[:2]
    pos, full = int(t.get("pos", buffer_size)), bool(t.get("full", True))
    n_valid = buffer_size if full else pos
    critic = t.get("critic_observations", obs)
    streams = {k: v for k, v in t.items() if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[:2] == obs.shape[:2]}
    return BufferView(
        format="collect_expert_replay_buffer (AsymmetricReplayBuffer)",
        n_env=n_env, buffer_size=buffer_size, n_valid=n_valid, ptr=pos, wrapped=bool(full and pos != 0),
        obs=obs, critic_obs=critic, actions=t["actions"], rewards=t["rewards"].reshape(buffer_size, n_env),
        dones=_as_bool(t.get("terminations")), truncations=_as_bool(t.get("truncations")),
        task=meta.get("task"), actor_obs_keys=meta.get("actor_obs_keys"), critic_obs_keys=meta.get("critic_obs_keys"),
        meta=meta, streams=streams,
    )


def _parse_play_recorder(payload: dict) -> BufferView:
    """holosoma play.py recorder: env-first [n_env, steps, dim], ptr, dones."""
    t, meta = payload["buffer_tensors"], payload["metadata"]
    obs = t["observations"]
    n_env, buffer_size = obs.shape[:2]
    ptr = int(t.get("ptr", meta.get("buffer_size", buffer_size)))
    n_valid = min(ptr, buffer_size) if ptr >= 0 else buffer_size
    sw = lambda x: None if x is None else x.transpose(0, 1)
    critic = t.get("critic_observations", obs)
    streams = {k: sw(v) for k, v in t.items() if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[:2] == obs.shape[:2]}
    return BufferView(
        format="holosoma play.py recorder",
        n_env=n_env, buffer_size=buffer_size, n_valid=n_valid, ptr=ptr, wrapped=False,
        obs=sw(obs), critic_obs=sw(critic), actions=sw(t["actions"]),
        rewards=sw(t["rewards"].reshape(n_env, buffer_size)),
        dones=_as_bool(sw(t.get("dones"))), truncations=_as_bool(sw(t.get("truncations"))),
        task=meta.get("task"), actor_obs_keys=meta.get("actor_obs_keys"), critic_obs_keys=meta.get("critic_obs_keys"),
        meta=meta, streams=streams,
    )


def _parse_rsl_rl(payload: dict) -> BufferView:
    """rsl_rl ReplayBuffer: TensorDict observations, steps-first [capacity, n_env, dim], _pos/_size."""
    obs_td = payload["observations"]
    obs_keys = list(obs_td.keys())
    obs = torch.cat([obs_td[k] for k in obs_keys], dim=-1)
    capacity, n_env = obs.shape[:2]
    pos = int(payload.get("_pos", payload.get("ptr", capacity)))
    size = int(payload.get("_size", min(pos, capacity)))
    rewards = payload["rewards"]
    return BufferView(
        format="rsl_rl ReplayBuffer (TensorDict)",
        n_env=n_env, buffer_size=capacity, n_valid=size, ptr=pos, wrapped=bool(size == capacity and pos != 0),
        obs=obs, critic_obs=obs, actions=payload["actions"], rewards=rewards.reshape(capacity, n_env),
        dones=_as_bool(payload.get("dones")), truncations=_as_bool(payload.get("truncations")),
        actor_obs_keys=obs_keys, critic_obs_keys=obs_keys, streams={"observations": obs},
    )


def _parse_simple(d: dict) -> BufferView:
    """FastSAC SimpleReplayBuffer: env-first [n_env, steps, dim], ptr."""
    obs = d["observations"]
    n_env, buffer_size = obs.shape[:2]
    ptr = int(d.get("ptr", buffer_size))
    n_valid = min(ptr, buffer_size) if ptr >= 0 else buffer_size
    sw = lambda x: None if x is None else x.transpose(0, 1)
    critic = d.get("critic_observations", obs)
    return BufferView(
        format="FastSAC SimpleReplayBuffer",
        n_env=n_env, buffer_size=buffer_size, n_valid=n_valid, ptr=ptr, wrapped=False,
        obs=sw(obs), critic_obs=sw(critic), actions=sw(d["actions"]),
        rewards=sw(d["rewards"].reshape(n_env, buffer_size)),
        dones=_as_bool(sw(d.get("dones"))), truncations=_as_bool(sw(d.get("truncations"))),
        streams={"observations": sw(obs), "critic_observations": sw(critic)},
    )


def load_buffer(path: str) -> BufferView:
    """Load a replay buffer of any supported format into the canonical steps-first view."""
    payload = load_payload(path)
    if isinstance(payload, dict):
        if "buffer_tensors" in payload and "metadata" in payload:
            t = payload["buffer_tensors"]
            if "policy_observations" in t:
                return _parse_collector(payload)
            if "observations" in t:
                return _parse_play_recorder(payload)
            raise ValueError(f"buffer_tensors has neither policy_observations nor observations; keys={sorted(t)}")
        obs = payload.get("observations")
        if obs is not None and not isinstance(obs, torch.Tensor) and hasattr(obs, "keys"):  # TensorDict
            return _parse_rsl_rl(payload)
        if isinstance(obs, torch.Tensor):
            return _parse_simple(payload)
    if hasattr(payload, "observations") and isinstance(payload.observations, torch.Tensor):
        return _parse_simple(payload.__dict__)
    available = sorted(payload.keys()) if isinstance(payload, dict) else dir(payload)
    raise ValueError(f"Unrecognized replay buffer format. Available keys/attrs: {available}")


def print_header(path: str, b: BufferView) -> None:
    print(f"[rb] {path}")
    print(f"     format     : {b.format}")
    if b.task:
        print(f"     task       : {b.task}")
    print(f"     n_env      : {b.n_env}")
    print(f"     buffer_size: {b.buffer_size}   ptr={b.ptr}   valid steps={b.n_valid}"
          f"   transitions={b.n_valid * b.n_env:,}" + ("   [wrapped mid-ring]" if b.wrapped else ""))
    print(f"     obs_dim    : {b.obs.shape[-1]}   critic_dim={b.critic_obs.shape[-1]}   act_dim={b.actions.shape[-1]}")
    if b.actor_obs_keys:
        print(f"     actor_keys : {b.actor_obs_keys}")
    if b.critic_obs_keys:
        print(f"     critic_keys: {b.critic_obs_keys}")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def _term_offsets(task: str | None, obs_dim: int) -> tuple[dict[str, tuple[int, int]], int] | None:
    """Return ({term_name: (block_start, block_end)}, history) inferred from task + obs_dim."""
    if task is None:
        return None
    terms = next((t for key, t in _TASK_TERM_ORDER.items() if key in task), None)
    if terms is None:
        return None
    per_frame_total = sum(d for _, d in terms)
    if obs_dim % per_frame_total != 0:
        print(f"[WARN] obs_dim={obs_dim} is not a multiple of per-frame total {per_frame_total} for task '{task}'.")
        return None
    history = obs_dim // per_frame_total
    offsets, cursor = {}, 0
    for name, dim in terms:
        offsets[name] = (cursor, cursor + dim * history)
        cursor += dim * history
    return offsets, history


def print_tensor_stats(name: str, t: torch.Tensor, per_dim: bool = True, indent: int = 2) -> None:
    pad = " " * indent
    flat = t.reshape(-1, t.shape[-1]).float()
    n_dim = flat.shape[-1]
    print(f"{pad}{name}  shape={tuple(t.shape)}  dtype={t.dtype}")
    print(f"{pad}  global : mean={flat.mean():.4f}  std={flat.std():.4f}  min={flat.min():.4f}  max={flat.max():.4f}")
    if not per_dim:
        return
    mean, std = flat.mean(dim=0), flat.std(dim=0)
    if n_dim <= 50:
        abs_max = flat.abs().max(dim=0).values
        print(f"{pad}  per-dim mean : [{', '.join(f'{v:.3f}' for v in mean.tolist())}]")
        print(f"{pad}  per-dim std  : [{', '.join(f'{v:.3f}' for v in std.tolist())}]")
        print(f"{pad}  per-dim |max|: [{', '.join(f'{v:.3f}' for v in abs_max.tolist())}]")
    else:
        print(f"{pad}  per-dim mean: min={mean.min():.3f}  max={mean.max():.3f}")
        print(f"{pad}  per-dim std : min={std.min():.3f}  max={std.max():.3f}")


def print_reward_stats(rewards: torch.Tensor, indent: int = 2) -> None:
    pad = " " * indent
    flat = rewards.reshape(-1).float()
    print(f"{pad}global : mean={flat.mean():.4f}  std={flat.std():.4f}  min={flat.min():.4f}  max={flat.max():.4f}")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    qs = torch.quantile(flat, torch.tensor([p / 100.0 for p in percentiles]))
    print(f"{pad}percentiles: " + "  ".join(f"p{p}={v:.4f}" for p, v in zip(percentiles, qs.tolist())))


def print_norm_stats(name: str, t: torch.Tensor, indent: int = 2) -> None:
    pad = " " * indent
    norms = t.reshape(-1, t.shape[-1]).float().norm(dim=-1)
    print(f"{pad}{name}: mean={norms.mean():.4f}  std={norms.std():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")


def print_episode_end_stats(b: BufferView, indent: int = 2) -> None:
    pad = " " * indent
    term, trunc, convention = b.episode_ends()
    if term is None:
        print(f"{pad}(no done field)")
        return
    total = term.numel()
    n_term = int(term.sum())
    n_trunc = int(trunc.sum()) if trunc is not None else 0
    n_ends = n_term + n_trunc
    pct = lambda n: 100.0 * n / total if total else 0.0
    print(f"{pad}convention        : {convention}")
    print(f"{pad}total transitions : {total}")
    print(f"{pad}terminations      : {n_term:>10}  ({pct(n_term):.3f}%)")
    if trunc is not None:
        print(f"{pad}truncations       : {n_trunc:>10}  ({pct(n_trunc):.3f}%)")
    print(f"{pad}any end           : {n_ends:>10}  ({pct(n_ends):.3f}%)")
    print(f"{pad}mean episode length ~ {total / n_ends:.1f} steps" if n_ends else f"{pad}no episode ends recorded")


def plot_episode_end_breakdown(b: BufferView, out_path: str, title_extra: str = "") -> None:
    """Bar chart: what fraction of stored transitions are episode ends, split by cause."""
    import matplotlib.pyplot as plt

    term, trunc, convention = b.episode_ends()
    if term is None:
        print("[WARN] buffer has no done field; skipping --plot_episode_ends.")
        return
    if trunc is None:
        trunc = torch.zeros_like(term)
    total = term.numel()
    n_term, n_trunc = int(term.sum()), int(trunc.sum())
    counts = [n_term, n_trunc, total - n_term - n_trunc]
    pct = lambda n: 100.0 * n / total if total else 0.0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["termination", "truncation", "non-terminal"], [pct(c) for c in counts],
                  color=["tab:green", "tab:orange", "tab:gray"])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{c}\n{pct(c):.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of stored transitions")
    ax.set_title(f"Episode ends in replay buffer{title_extra}\n{total} transitions -- {convention}", fontsize=10)
    ax.set_ylim(0, max(pct(c) for c in counts) * 1.25 or 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")


def plot_newest_frame_hist(obs: torch.Tensor, title: str, block: tuple[int, int], term_dim: int,
                           labels: list[str], out_path: str, nrows: int = 1,
                           xlim_per_label: dict[str, tuple[float, float]] | None = None) -> None:
    """Per-dim histograms of the newest history frame of a term block (its LAST term_dim entries)."""
    import matplotlib.pyplot as plt

    block_start, block_end = block
    start, end = block_end - term_dim, block_end
    obs_dim = obs.shape[-1]
    if end > obs_dim or start < block_start:
        print(f"[WARN] {title}: slice [{start}:{end}] out of range for block {block} / obs_dim={obs_dim}.")
        return
    values = obs.reshape(-1, obs_dim)[:, start:end].float().cpu()
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
        if xlim_per_label and lbl in xlim_per_label:
            ax.set_xlim(*xlim_per_label[lbl])
    for ax in axes_flat[term_dim:]:
        ax.set_visible(False)
    fig.suptitle(f"{title} (newest frame, dims {start}:{end} of block {block_start}:{block_end})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")


def _resolve_block(b: BufferView, term: str, override: tuple[int | None, int | None]) -> tuple[int, int] | None:
    if override[0] is not None and override[1] is not None:
        return int(override[0]), int(override[1])
    result = _term_offsets(b.task, b.obs.shape[-1])
    if result is None or term not in result[0]:
        print(f"[WARN] Cannot determine {term} offsets for task={b.task!r}; pass --{term.split('_')[0]}_block_start/_end.")
        return None
    return result[0][term]


def cmd_stats(args: argparse.Namespace) -> None:
    b = load_buffer(args.rb_path)
    if args.task:
        b.task = args.task
    b = b.sliced(args.max_samples)
    per_dim = not args.no_per_dim
    print_header(args.rb_path, b)
    sep = "=" * 60

    print(f"\n{sep}\nObservations (actor)\n{sep}")
    print_tensor_stats("obs", b.obs, per_dim=per_dim)
    print(f"\n{sep}\nObservations (critic)\n{sep}")
    print_tensor_stats("critic_obs", b.critic_obs, per_dim=per_dim)
    print(f"\n{sep}\nActions\n{sep}")
    print_tensor_stats("actions", b.actions, per_dim=per_dim)
    print(f"\n{sep}\nRewards\n{sep}")
    print_reward_stats(b.rewards)
    print(f"\n{sep}\nEpisode ends\n{sep}")
    print_episode_end_stats(b)
    print(f"\n{sep}\nL2 norms per step\n{sep}")
    print_norm_stats("actor obs ", b.obs)
    print_norm_stats("critic obs", b.critic_obs)
    print_norm_stats("actions   ", b.actions)

    stem = os.path.splitext(args.rb_path)[0]
    base = os.path.basename(args.rb_path)
    if args.plot_episode_ends:
        plot_episode_end_breakdown(b, f"{stem}_episode_ends.png", title_extra=f"  ({base})")
    if args.plot_ee_pose:
        block = _resolve_block(b, "end_effector_pose", (args.ee_block_start, args.ee_block_end))
        if block is not None:
            plot_newest_frame_hist(
                b.obs, f"EE Pose Histograms  task={b.task or 'unknown'}", block, 6, _EE_LABELS,
                f"{stem}_ee_pose.png",
                xlim_per_label={"ee_pos_x": (-1.0, 1.0), "ee_pos_y": (-1.0, 1.0), "ee_pos_z": (-1.0, 1.0)},
            )
    if args.plot_joint_pos:
        block = _resolve_block(b, "joint_pos", (args.joint_block_start, args.joint_block_end))
        if block is not None:
            plot_newest_frame_hist(
                b.obs, f"Joint Pos Histograms  task={b.task or 'unknown'}", block, 12, _JOINT_LABELS,
                f"{stem}_joint_pos.png", nrows=2,
            )


# ---------------------------------------------------------------------------
# grid / video (image buffers)
# ---------------------------------------------------------------------------
def detect_modality(meta: dict, sample: np.ndarray) -> tuple[str, str]:
    """Return (modality, how_it_was_decided). Metadata is authoritative; the value-range fallback is
    deliberately conservative -- depth in metres routinely exceeds 1.0, whereas rgb is clamped to [0, 1]."""
    keys = [str(k).lower() for k in (meta.get("actor_obs_keys") or [])]
    if any("depth" in k for k in keys):
        return "depth", f"metadata actor_obs_keys={keys}"
    if any("gray" in k or "rgb" in k for k in keys):
        return "grayscale", f"metadata actor_obs_keys={keys}"
    if float(np.nanmax(sample)) > 1.001:
        return "depth", "value-range guess (max > 1.0)"
    return "grayscale", "value-range guess (values within [0,1])"


def colorize(img: np.ndarray, cmap, lo: float, hi: float, is_depth: bool) -> np.ndarray:
    """Map an HxW float image to an HxWx3 uint8 RGB frame, painting invalid depth pixels red."""
    hi = hi if hi > lo else lo + 1e-6
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    if is_depth:
        rgb[img == 0.0] = INVALID_RGB
    return rgb


@dataclass
class ImageView:
    """An image buffer resolved to a (steps, n_env, hist, cam, H, W) layout plus display settings."""

    b: BufferView
    obs: torch.Tensor  # the chosen stream, [steps, n_env, obs_dim]
    stream: str
    hist: int
    cam: int
    h: int
    w: int
    frame: int
    camera_names: list[str]
    modality: str
    cmap: object
    cmap_name: str

    @property
    def is_depth(self) -> bool:
        return self.modality == "depth"

    def frame_at(self, step: int, env: int) -> np.ndarray:
        """(cam, H, W) for one transition at the chosen history slot."""
        return self.obs[step, env].reshape(self.hist, self.cam, self.h, self.w)[self.frame].float().numpy()


def resolve_image_view(args: argparse.Namespace) -> ImageView:
    import matplotlib

    b = load_buffer(args.rb_path)
    if b.n_valid == 0:
        raise SystemExit("buffer is empty (no valid steps)")
    if args.stream is None:
        obs, stream = b.obs, "observations"
    elif args.stream in b.streams:
        obs, stream = b.streams[args.stream], args.stream
    else:
        raise SystemExit(f"stream '{args.stream}' not in buffer; available: {sorted(b.streams)}")

    hist, cam, (h, w) = args.history, args.num_cameras, args.image_size
    expected = hist * cam * h * w
    if expected != obs.shape[-1]:
        raise SystemExit(
            f"layout {hist}x{cam}x{h}x{w} = {expected} != stored obs dim {obs.shape[-1]}.\n"
            f"Pass --history/--num_cameras/--image_size to match this buffer."
        )
    if len(args.camera_names) != cam:
        raise SystemExit(f"got {len(args.camera_names)} camera names for {cam} cameras")

    probe = obs[0, 0].reshape(hist, cam, h, w)[args.frame].float().numpy()
    modality, why = detect_modality(b.meta, probe)
    if args.modality != "auto":
        modality, why = args.modality, "forced by --modality"
    cmap_name = args.cmap or ("viridis" if modality == "depth" else "gray")
    cmap = matplotlib.colormaps[cmap_name]

    print_header(args.rb_path, b)
    print(
        f"     stream     : {stream}  shape={tuple(obs.shape)}\n"
        f"     layout     : ({hist} history, {cam} cameras, {h}, {w}), showing frame {args.frame} "
        f"({'most recent' if args.frame in (-1, hist - 1) else 'index ' + str(args.frame)})\n"
        f"     modality   : {modality}  [{why}]  cmap={cmap_name}"
    )
    return ImageView(b, obs, stream, hist, cam, h, w, args.frame, list(args.camera_names), modality, cmap, cmap_name)


def cmd_grid(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = resolve_image_view(args)
    b = v.b
    rng = np.random.default_rng(args.seed)
    n = args.num_samples

    def draw_sample() -> tuple[int, int]:
        return int(rng.integers(0, b.n_valid)), int(rng.integers(0, b.n_env))

    shared = [draw_sample() for _ in range(n)]
    picks_per_cam = [([draw_sample() for _ in range(n)] if args.independent else shared) for _ in range(v.cam)]
    grid = [[v.frame_at(s, e)[r] for (s, e) in picks_per_cam[r]] for r in range(v.cam)]
    all_px = np.stack([img for row in grid for img in row])

    def valid_mask(a: np.ndarray) -> np.ndarray:
        return a != 0.0 if v.is_depth else np.ones_like(a, dtype=bool)

    finite = all_px[valid_mask(all_px)]
    if finite.size == 0:
        print("[warn] every sampled pixel is invalid (all zero) -- nothing to scale against.")
        finite = all_px
    unit = " m" if v.is_depth else ""
    print(
        f"     values     : min={finite.min():.3f}{unit} max={finite.max():.3f}{unit} "
        f"mean={finite.mean():.3f}{unit} std={finite.std():.3f}{unit}"
        + (f"  invalid(==0): {100 * (all_px == 0).mean():.1f}% of pixels" if v.is_depth else "")
    )
    print(f"     {'camera':<8}{'min':>9}{'max':>9}{'mean':>9}" + (f"{'invalid%':>10}" if v.is_depth else ""))
    for r, name in enumerate(v.camera_names):
        px = np.stack(grid[r])
        vals = px[valid_mask(px)]
        vals = vals if vals.size else px
        line = f"     {name:<8}{vals.min():9.3f}{vals.max():9.3f}{vals.mean():9.3f}"
        if v.is_depth:
            line += f"{100 * (px == 0).mean():10.1f}"
        print(line)

    if args.shared_scale:
        lo = args.vmin if args.vmin is not None else (float(finite.min()) if v.is_depth else 0.0)
        hi = args.vmax if args.vmax is not None else (float(finite.max()) if v.is_depth else 1.0)
    else:
        lo = hi = None

    mpl_cmap = v.cmap.with_extremes(bad=INVALID_COLOR)
    fig, axes = plt.subplots(v.cam, n, figsize=(1.5 * n, 1.65 * v.cam))
    axes = np.atleast_2d(axes)
    im = None
    for r in range(v.cam):
        for c in range(n):
            img = grid[r][c]
            shown = np.ma.masked_where(~valid_mask(img), img) if v.is_depth else img
            if lo is not None:
                vmin, vmax = lo, hi
            else:
                vals = img[valid_mask(img)]
                vmin, vmax = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
                if vmax <= vmin:
                    vmax = vmin + 1e-6
            ax = axes[r, c]
            im = ax.imshow(shown, cmap=mpl_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                step, env = picks_per_cam[r][c]
                ax.set_title(f"s{step}/e{env}", fontsize=6)
        axes[r, 0].set_ylabel(v.camera_names[r], fontsize=10, rotation=0, ha="right", va="center", labelpad=18)

    scale_desc = f"shared {lo:.2f}-{hi:.2f}{unit}" if args.shared_scale else "per-image min-max"
    sampling = "independent per camera" if args.independent else "shared across cameras"
    title = (
        f"{os.path.basename(args.rb_path)} - {v.stream} [{v.modality}], "
        f"history frame {v.frame} ({sampling}, {scale_desc})"
    )
    if v.is_depth:
        title += f"\ninvalid / no-return pixels in {INVALID_COLOR}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99 if not v.is_depth else 0.94))
    if args.shared_scale and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
        cbar.set_label("metres" if v.is_depth else "intensity", fontsize=9)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.rb_path)),
        f"{os.path.splitext(os.path.basename(args.rb_path))[0]}_samples.png",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


def cmd_video(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    v = resolve_image_view(args)
    b = v.b
    if b.wrapped:
        print(
            f"[warn] buffer wrapped and stopped mid-ring (ptr={b.ptr}); rows are NOT chronological "
            f"across index {b.ptr}. A window spanning it will jump in time."
        )

    rng = np.random.default_rng(args.seed)
    n_steps = args.video_steps
    if n_steps > b.n_valid:
        raise SystemExit(f"--video_steps {n_steps} exceeds the {b.n_valid} valid steps in this buffer")
    env = args.env if args.env is not None else int(rng.integers(0, b.n_env))
    if not 0 <= env < b.n_env:
        raise SystemExit(f"--env {env} out of range (buffer has {b.n_env} envs)")
    max_start = b.n_valid - n_steps
    start = args.start if args.start is not None else int(rng.integers(0, max_start + 1))
    if not 0 <= start <= max_start:
        raise SystemExit(f"--start {start} out of range; must be within [0, {max_start}] for {n_steps} steps")
    stop = start + n_steps

    # (n_steps, cameras, H, W) -- one frame per step, taking the chosen history slot.
    seq = v.obs[start:stop, env].reshape(n_steps, v.hist, v.cam, v.h, v.w)[:, v.frame].float().numpy()

    # Episode boundaries: a 160-step window of an expert buffer spans several episodes, and without
    # marking them the jump between episodes looks like a glitch.
    term, trunc, _ = b.episode_ends()
    dones = np.zeros(n_steps, dtype=bool)
    for t in (term, trunc):
        if t is not None:
            dones |= t[start:stop, env].numpy().reshape(n_steps)
    n_boundaries = int(dones.sum())

    # One fixed scale for the whole clip. Per-frame normalization would make the video flicker and
    # would also destroy any sense of absolute depth.
    valid = seq[seq != 0.0] if v.is_depth else seq
    valid = valid if valid.size else seq
    lo = args.vmin if args.vmin is not None else (float(valid.min()) if v.is_depth else 0.0)
    hi = args.vmax if args.vmax is not None else (float(valid.max()) if v.is_depth else 1.0)
    unit = " m" if v.is_depth else ""
    print(
        f"     video      : env {env}, steps [{start}, {stop}) = {n_steps} frames @ {args.fps} fps "
        f"({n_steps / args.fps:.1f} s)\n"
        f"     scale      : fixed {lo:.3f}{unit} - {hi:.3f}{unit} across all frames\n"
        f"     episodes   : {n_boundaries} boundary/boundaries inside the window"
        + (f"  |  invalid pixels: {100 * (seq == 0).mean():.1f}%" if v.is_depth else "")
    )

    out_dir = args.out or os.path.dirname(os.path.abspath(args.rb_path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.rb_path))[0]
    zoom = max(1, args.upscale)
    episode_of = np.cumsum(np.concatenate([[0], dones[:-1]]))  # frame -> episode ordinal in-window

    for c in range(v.cam):
        name = v.camera_names[c]
        frames = []
        for i in range(n_steps):
            rgb = colorize(seq[i, c], v.cmap, lo, hi, v.is_depth)
            im = Image.fromarray(rgb).resize((v.w * zoom, v.h * zoom), Image.NEAREST)
            if not args.no_overlay:
                d = ImageDraw.Draw(im)
                d.text((4, 2), f"{name}  s{start + i}  ep{int(episode_of[i])}", fill=(255, 255, 255))
                if dones[i]:
                    d.rectangle([0, 0, im.width - 1, im.height - 1], outline=BOUNDARY_RGB, width=3)
                    d.text((4, 14), "episode end", fill=BOUNDARY_RGB)
            frames.append(np.asarray(im))
        out = os.path.join(out_dir, f"{base}_{name}_e{env}_s{start}-{stop}.mp4")
        # macro_block_size=1 keeps the exact pixel size instead of padding to a multiple of 16.
        imageio.mimsave(out, frames, fps=args.fps, macro_block_size=1)
        px = seq[:, c]
        vals = px[px != 0.0] if v.is_depth else px
        vals = vals if vals.size else px
        extra = f"  invalid {100 * (px == 0).mean():5.1f}%" if v.is_depth else ""
        print(f"[saved] {out}   range {vals.min():.3f}-{vals.max():.3f}{unit}{extra}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_image_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stream", default=None,
                   help="which stored tensor to visualize (default: the actor obs; e.g. next_policy_observations)")
    p.add_argument("--frame", type=int, default=-1, help="which history frame to show; -1 = most recent, 0 = oldest")
    p.add_argument("--history", type=int, default=3, help="history length stored per obs")
    p.add_argument("--num_cameras", type=int, default=3)
    p.add_argument("--image_size", type=int, nargs=2, default=(84, 84), metavar=("H", "W"))
    p.add_argument("--camera_names", nargs="*", default=["front", "side", "wrist"],
                   help="row labels, in the order the task cfg lists sensor_cfgs")
    p.add_argument("--modality", choices=("auto", "grayscale", "depth"), default="auto",
                   help="auto reads metadata['actor_obs_keys'] and falls back to a value-range guess")
    p.add_argument("--cmap", default=None, help="matplotlib colormap (default: gray / viridis for depth)")
    p.add_argument("--vmin", type=float, default=None, help="explicit lower limit for the shared scale")
    p.add_argument("--vmax", type=float, default=None, help="explicit upper limit for the shared scale")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output file (grid) or directory (video)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="print observation / action / reward statistics")
    s.add_argument("rb_path", help="path to the saved replay buffer .pt")
    s.add_argument("--task", default=None, help="task name (overrides metadata; selects the obs term layout)")
    s.add_argument("--max_samples", type=int, default=None, help="max per-env steps to inspect (default: all filled)")
    s.add_argument("--no_per_dim", action="store_true", help="skip per-dimension stats")
    s.add_argument("--plot_episode_ends", action="store_true",
                   help="bar chart of terminations vs truncations vs non-terminal, saved next to the buffer")
    s.add_argument("--plot_ee_pose", action="store_true", help="per-dim histograms of the EE pose (newest frame)")
    s.add_argument("--plot_joint_pos", action="store_true", help="per-dim histograms of joint positions (newest frame)")
    s.add_argument("--ee_block_start", type=int, default=None, help="EE pose block start dim (override auto-detect)")
    s.add_argument("--ee_block_end", type=int, default=None, help="EE pose block end dim, exclusive")
    s.add_argument("--joint_block_start", type=int, default=None, help="joint pos block start dim (override)")
    s.add_argument("--joint_block_end", type=int, default=None, help="joint pos block end dim, exclusive")
    s.set_defaults(func=cmd_stats)

    g = sub.add_parser("grid", help="render a cameras x samples grid from an image buffer")
    g.add_argument("rb_path", help="path to the saved replay buffer .pt")
    _add_image_args(g)
    g.add_argument("--num_samples", type=int, default=10, help="images per camera (columns)")
    g.add_argument("--independent", action="store_true",
                   help="sample each camera's images independently instead of sharing one set of transitions "
                        "(shared is the default so a column shows the same instant from all views)")
    g.add_argument("--shared_scale", action="store_true",
                   help="use ONE colour scale for every panel instead of per-image min-max; for depth this makes "
                        "panels metrically comparable and adds a colourbar in metres, for grayscale it pins [0,1]")
    g.set_defaults(func=cmd_grid)

    vd = sub.add_parser("video", help="write one video per camera from a contiguous run of one env")
    vd.add_argument("rb_path", help="path to the saved replay buffer .pt")
    _add_image_args(vd)
    vd.add_argument("--video_steps", type=int, default=160, help="length of the contiguous run")
    vd.add_argument("--start", type=int, default=None, help="first step index (default: random)")
    vd.add_argument("--env", type=int, default=None, help="which env to follow (default: random)")
    vd.add_argument("--fps", type=int, default=20)
    vd.add_argument("--upscale", type=int, default=4, help="nearest-neighbour zoom for legibility")
    vd.add_argument("--no_overlay", action="store_true", help="omit the step/episode text and episode-boundary border")
    vd.set_defaults(func=cmd_video)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
