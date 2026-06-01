#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a multi-env online replay buffer to a single-env replay buffer.

Reads a `.pt` saved by ``FastSACAgent.save_replay_buffer`` (top-level keys
``observations``/``actions``/``rewards``/``dones``/``truncations``/``next_observations``/
``critic_observations``/``next_critic_observations``/``ptr``/``n_env``/``buffer_size``/
``global_step``), reshapes each tensor from ``(n_env, buffer_size, ...)`` to
``(1, n_env * buffer_size, ...)`` and writes it back in the same online-RB format.

Entries are emitted in **chronological time-major** order: each env's row is first
rotated so that index 0 is the oldest entry (was at ``ptr``), then the result is
transposed to ``(buffer_size, n_env, ...)`` and flattened. So the output is:
``[step_0_env_0, step_0_env_1, ..., step_0_env_{N-1}, step_1_env_0, ...]``.

Output ``ptr`` is set to ``n_env * buffer_size`` (treats the output buffer as fully
filled — the next write wraps to index 0), and ``global_step`` is reset to 0.

If the source already has ``n_env == 1`` the script just canonicalizes
``ptr``/``global_step`` and re-saves.

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


def _validate(payload: dict, label: str) -> None:
    print(f"--- {label} ---")
    print(
        f"  n_env={payload.get('n_env')}  buffer_size={payload.get('buffer_size')}  "
        f"ptr={payload.get('ptr')}  global_step={payload.get('global_step')}"
    )
    for k in TENSOR_KEYS:
        if k not in payload:
            print(f"  {k}: MISSING")
            continue
        t = payload[k]
        is_float = t.is_floating_point()
        nan = bool(torch.isnan(t).any().item()) if is_float else False
        inf = bool(torch.isinf(t).any().item()) if is_float else False
        line = f"  {k}: shape={tuple(t.shape)} dtype={t.dtype} nan={nan} inf={inf}"
        if is_float and not (nan or inf):
            line += f"  min={t.min().item():.4f}  max={t.max().item():.4f}"
        print(line)


def _rotate_then_time_major(t: torch.Tensor, ptr: int, buffer_size: int) -> torch.Tensor:
    """Rotate dim=1 so oldest (index ``ptr``) lands at index 0, then time-major flatten.

    Input shape ``(n_env, buffer_size, *)`` → output shape ``(1, n_env*buffer_size, *)``.
    """
    if t.dim() < 2:
        return t
    rotated = torch.roll(t, shifts=-int(ptr), dims=1)  # (n_env, buffer_size, *)
    flat = rotated.transpose(0, 1).reshape(-1, *t.shape[2:])  # (buffer_size*n_env, *)
    return flat.unsqueeze(0)  # (1, buffer_size*n_env, *)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=str, help="path to source .pt (online RB format)")
    parser.add_argument("output", type=str, help="path to write the single-env .pt file")
    args = parser.parse_args()

    print(f"[INFO] Loading: {args.input}")
    src = torch.load(args.input, map_location="cpu", weights_only=False)

    n_env = int(src["n_env"])
    buffer_size = int(src["buffer_size"])
    ptr = int(src["ptr"]) % buffer_size  # normalize in case ptr == buffer_size

    _validate(src, "SOURCE")

    if n_env == 1:
        print("[INFO] Source already has n_env=1; canonicalizing ptr/global_step and re-saving.")
        out = dict(src)
        out["ptr"] = buffer_size
        out["global_step"] = 0
        _validate(out, "OUTPUT")
        torch.save(out, args.output)
        print(f"[INFO] Saved: {args.output}")
        return

    out: dict = {}
    for k in TENSOR_KEYS:
        if k not in src:
            print(f"[WARN] missing key '{k}' in source; skipping")
            continue
        out[k] = _rotate_then_time_major(src[k], ptr, buffer_size)

    out["n_env"] = 1
    out["buffer_size"] = n_env * buffer_size
    out["ptr"] = n_env * buffer_size  # fully-filled; next write wraps to 0
    out["global_step"] = 0

    _validate(out, "OUTPUT")
    torch.save(out, args.output)
    print(f"[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()
