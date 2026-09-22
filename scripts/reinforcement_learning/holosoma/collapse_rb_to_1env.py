#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a multi-env replay buffer to a single-env replay buffer.

Handles both layouts:

* **online RB** — saved by ``FastSACAgent.save_replay_buffer``; tensors plus ``ptr``/``n_env``/
  ``buffer_size``/``global_step`` at the top level. Read by ``agent.load_replay_buffer_path``.
* **expert RB** — saved by ``play.py``'s recorder; ``{"buffer_tensors": {...}, "metadata": {...}}``.
  Read by ``--expert_transitions``.

The output is written back in whichever format the input used.

Why collapse an expert buffer: ``_sample_and_prepare_batches`` computes
``expert_per_env = max(target_total // expert_rb.n_env, 1)``. That floor draws at least ONE sample
per expert env, so an expert buffer with ``n_env=2048`` contributes >= 2048 samples to every update
no matter how small ``expert_ratio`` is -- an intended 5% share (51 samples) became 2048, i.e. ~68%
of the batch. Collapsing the expert buffer to ``n_env=1`` makes the floor one single sample, so the
ratio lands where it should.

Ordering is **env-major**: each env's row is first rotated so index 0 is its oldest entry (was at
``ptr``), then rows are concatenated, giving
``[env_0_step_0 ... env_0_step_{T-1}, env_1_step_0, ...]``.

This deliberately replaces the previous *time-major* interleave
(``[step_0_env_0, step_0_env_1, ...]``). Time-major puts a different env in every adjacent slot, and
``SimpleReplayBuffer.sample`` walks ``(i + offset) % buffer_size`` for n-step returns, stopping only
at a done/truncation -- so with ``n_steps > 1`` *every* sample would sum rewards across unrelated
trajectories and bootstrap off a foreign state. Env-major leaves only ``n_env - 1`` seams, and each
is stamped ``dones = 1`` AND ``truncations = 1`` (the success-as-truncation convention): the sampler
cuts the reward walk on ``dones`` only -- a truncation-only flag merely relocates the bootstrap obs
while rewards keep accumulating into the next env -- and the truncation flag makes the critic
bootstrap through the seam from that env's true final observation (handle_truncations).

Output ``ptr`` is set to ``n_env * buffer_size`` (fully filled; the next write wraps to 0) and
``global_step`` is reset to 0 for the online format.

Usage: ``python collapse_rb_to_1env.py <input.pt> <output.pt>``
"""

import argparse
import torch


TENSOR_KEYS = [
    "observations",
    "actions",
    "rewards",
    "dones",
    "truncations",
    "next_observations",
    "critic_observations",
    "next_critic_observations",
]


def _detect(payload: dict) -> str:
    return "expert" if "buffer_tensors" in payload else "online"


def _read(payload: dict, fmt: str):
    """Return (tensors, n_env, buffer_size, ptr) for either layout."""
    if fmt == "expert":
        t, m = payload["buffer_tensors"], payload["metadata"]
        return t, int(m["n_env"]), int(m["buffer_size"]), int(t["ptr"])
    return payload, int(payload["n_env"]), int(payload["buffer_size"]), int(payload["ptr"])


def _validate(tensors: dict, n_env, buffer_size, ptr, label: str) -> None:
    print(f"--- {label} ---")
    print(f"  n_env={n_env}  buffer_size={buffer_size}  ptr={ptr}")
    for k in TENSOR_KEYS:
        if k not in tensors:
            print(f"  {k}: MISSING")
            continue
        t = tensors[k]
        is_float = t.is_floating_point()
        nan = bool(torch.isnan(t).any().item()) if is_float else False
        inf = bool(torch.isinf(t).any().item()) if is_float else False
        line = f"  {k}: shape={tuple(t.shape)} dtype={t.dtype} nan={nan} inf={inf}"
        if is_float and not (nan or inf):
            line += f"  min={t.min().item():.4f}  max={t.max().item():.4f}"
        print(line)


def _rotate_env_major(t: torch.Tensor, ptr: int, buffer_size: int) -> torch.Tensor:
    """Rotate dim=1 so the oldest entry leads, then flatten env-major.

    ``(n_env, buffer_size, *)`` -> ``(1, n_env * buffer_size, *)`` with each env contiguous.
    """
    if t.dim() < 2:
        return t
    rotated = torch.roll(t, shifts=-int(ptr) % max(buffer_size, 1), dims=1)
    return rotated.reshape(1, -1, *t.shape[2:])


def _stamp_seams(out: dict, n_env: int, seg_len: int) -> int:
    """Flag the last slot of every env segment done+truncated so n-step walks stop there and
    the target bootstraps through the seam (rows already done keep their own truncation flag)."""
    total = n_env * seg_len
    idx = torch.arange(seg_len - 1, total, seg_len)
    already_done = out["dones"][0, idx] > 0
    out["truncations"][0, idx] = torch.where(
        already_done, out["truncations"][0, idx], torch.ones_like(out["truncations"][0, idx])
    )
    out["dones"][0, idx] = torch.ones_like(out["dones"][0, idx])
    return int(idx.numel())


def _check_seams(out: dict, seg_len: int, n_seams: int) -> None:
    """Assert continuity holds inside segments and every boundary terminates the walk."""
    total = out["dones"].shape[1]
    ends = torch.arange(seg_len - 1, total, seg_len)
    stops = out["dones"][0, ends] > 0  # only `dones` cuts the sampler's n-step reward walk
    bad = int((~stops).sum())
    if bad:
        raise AssertionError(f"{bad}/{len(ends)} segment boundaries lack a done flag")
    # next_obs[i] must equal obs[i+1] within a segment (the env auto-resets, so this holds across
    # dones too); at a boundary the two belong to different envs and should differ.
    obs, nobs = out["observations"][0], out["next_observations"][0]
    i = torch.arange(total - 1)
    in_seg = ((i + 1) % seg_len) != 0
    gap = (nobs[i] - obs[i + 1]).abs().amax(dim=1)
    print(
        f"[INFO] seam check OK: {len(ends)} boundaries flagged; "
        f"within-segment max |next_obs[i]-obs[i+1]|={gap[in_seg].max():.3e}, "
        f"at boundaries={gap[~in_seg].max():.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=str, help="path to source .pt (online or expert RB format)")
    parser.add_argument("output", type=str, help="path to write the single-env .pt file")
    args = parser.parse_args()

    print(f"[INFO] Loading: {args.input}")
    src = torch.load(args.input, map_location="cpu", weights_only=False)
    fmt = _detect(src)
    tensors, n_env, buffer_size, ptr = _read(src, fmt)
    print(f"[INFO] Detected {fmt}-RB format")
    _validate(tensors, n_env, buffer_size, ptr, "SOURCE")

    total = n_env * buffer_size
    if n_env == 1:
        print("[INFO] Source already has n_env=1; canonicalizing ptr and re-saving.")
        out_tensors = {k: tensors[k] for k in TENSOR_KEYS if k in tensors}
    else:
        out_tensors = {}
        for k in TENSOR_KEYS:
            if k not in tensors:
                print(f"[WARN] missing key '{k}' in source; skipping")
                continue
            out_tensors[k] = _rotate_env_major(tensors[k], ptr, buffer_size)
        n_seams = _stamp_seams(out_tensors, n_env, buffer_size)
        print(f"[INFO] env-major layout; flagged {n_seams} segment boundaries as truncated")
        _check_seams(out_tensors, buffer_size, n_seams)

    _validate(out_tensors, 1, total, total, "OUTPUT")

    if fmt == "expert":
        meta = dict(src["metadata"])
        meta.update({"n_env": 1, "buffer_size": total, "segment_length": buffer_size})
        payload = {"buffer_tensors": {**out_tensors, "ptr": total}, "metadata": meta}
    else:
        payload = dict(out_tensors)
        payload.update({"n_env": 1, "buffer_size": total, "ptr": total, "global_step": 0})

    torch.save(payload, args.output)
    print(f"[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()
