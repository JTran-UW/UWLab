#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shrink the env dimension of a saved replay buffer.

Reads a `.pt` saved by ``FastSACAgent.save_replay_buffer`` (top-level keys
``observations``/``actions``/``rewards``/``dones``/``truncations``/``next_observations``/
``critic_observations``/``next_critic_observations``/``ptr``/``n_env``/``buffer_size``/
``global_step``) and keeps only the first ``--n_env`` rows of every tensor:
``(n_env_src, buffer_size, ...) -> (n_env_target, buffer_size, ...)``.

The buffer-length dimension is untouched, so ``buffer_size``, ``ptr`` and
``global_step`` carry over unchanged.

Usage: ``python slice_rb_envs.py <input.pt> <output.pt> --n_env 256``
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=str, help="path to source .pt (online RB format)")
    parser.add_argument("output", type=str, help="path to write the sliced .pt file")
    parser.add_argument("--n_env", type=int, required=True, help="number of envs to keep (takes the first N)")
    args = parser.parse_args()

    print(f"[INFO] Loading: {args.input}")
    src = torch.load(args.input, map_location="cpu", weights_only=False)

    n_env_src = int(src["n_env"])
    n_env_tgt = int(args.n_env)
    if n_env_tgt > n_env_src:
        raise ValueError(f"--n_env={n_env_tgt} exceeds source n_env={n_env_src}")

    _validate(src, "SOURCE")

    out = dict(src)
    for k in TENSOR_KEYS:
        if k not in src:
            print(f"[WARN] missing key '{k}' in source; skipping")
            continue
        out[k] = src[k][:n_env_tgt].clone()

    out["n_env"] = n_env_tgt

    _validate(out, "OUTPUT")
    torch.save(out, args.output)
    print(f"[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()
