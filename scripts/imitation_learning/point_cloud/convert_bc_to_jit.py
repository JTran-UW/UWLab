# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JIT-compile a trained BC point-cloud policy for fast real-eval inference.

Takes a Lightning checkpoint written by ``train_point_net.py`` and traces it into a
self-contained ``torch.jit`` module whose ``forward(points, proprio) -> action`` has the
proprio z-scoring and action de-normalization BAKED IN (so deploy needs only torch, no
Lightning / config). A ``<output>.meta.json`` sidecar records the expected input layout
(point_dim, num_points, proprio_dim, and the per-prim layout if any) so the eval harness
can assemble inputs. Reload with ``bc_utils.load_bc_jit`` -- it drops straight into
``bc_utils.bc_actions`` (which feeds raw inputs when ``jit=True``).

Usage::

    python scripts/imitation_learning/point_cloud/convert_bc_to_jit.py \
        --checkpoint logs/pc_bc/<run>/point-net-...ckpt \
        --output teachers/<name>_bc_jit.pt

    # benchmark eager vs JIT forward throughput
    python scripts/imitation_learning/point_cloud/convert_bc_to_jit.py \
        --checkpoint <ckpt> --output <out> --benchmark
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_utils import export_bc_jit, load_bc_jit, load_bc_pointnet  # noqa: E402


def _benchmark(eager_bc: dict, jit_bc: dict, device: str, iters: int = 200) -> None:
    """Time eager vs JIT forward (points, proprio) -> action at batch=1 (real-eval shape)."""
    hp = eager_bc["hp"]
    n, pd, prd = int(hp["num_points"]), eager_bc["point_dim"], eager_bc["proprio_dim"]
    pts = torch.randn(1, n, pd, device=device)
    prop = torch.randn(1, prd, device=device)
    eager, jit = eager_bc["model"], jit_bc["model"]

    def run(fn, raw):
        with torch.no_grad():
            for _ in range(10):  # warmup
                fn(pts, prop)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            import time
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(pts, prop)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3  # ms/call

    # eager forward must replicate bc norm to be comparable; jit bakes it in.
    pm, ps = eager_bc["proprio_mean"], eager_bc["proprio_std"]
    am, ac = eager_bc["action_mean"], eager_bc["action_std"]
    predict_std = bool(hp.get("predict_std", False))

    def eager_call(p, pr):
        prn = (pr - pm) / ps
        out = eager(p, prn)
        mean = out[0] if predict_std else out
        return mean * ac + am

    ms_eager = run(eager_call, False)
    ms_jit = run(jit, True)
    print(f"[benchmark] batch=1 forward: eager {ms_eager:.3f} ms  |  jit {ms_jit:.3f} ms  "
          f"|  speedup {ms_eager / max(ms_jit, 1e-9):.2f}x  ({device})")


def main():
    p = argparse.ArgumentParser(description="JIT-compile a BC point-cloud policy for fast real eval.")
    p.add_argument("--checkpoint", type=str, required=True, help="Lightning .ckpt from train_point_net.py")
    p.add_argument("--output", type=str, required=True, help="Output JIT .pt path (sidecar <out>.meta.json too)")
    p.add_argument("--device", type=str, default="cpu", help="cpu (default; matches most real-eval rigs) or cuda")
    p.add_argument("--benchmark", action="store_true", help="Time eager vs JIT forward after export.")
    args = p.parse_args()

    bc = load_bc_pointnet(args.checkpoint, args.device)
    print(f"[convert] loaded {args.checkpoint} | arch={bc['hp'].get('architecture', 'point_net')} "
          f"point_dim={bc['point_dim']} num_points={bc['hp']['num_points']} proprio_dim={bc['proprio_dim']}"
          + (f" | per-prim parts={bc['pc_parts']}" if bc.get("pc_all_prim_names") else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    export_bc_jit(bc, args.output, device=args.device)

    # End-to-end check: reload the JIT and confirm it matches the eager policy's action.
    jit_bc = load_bc_jit(args.output, args.device)
    n, pd, prd = int(bc["hp"]["num_points"]), bc["point_dim"], bc["proprio_dim"]
    pts, prop = torch.randn(2, n, pd, device=args.device), torch.randn(2, prd, device=args.device)
    with torch.no_grad():
        prn = (prop - bc["proprio_mean"]) / bc["proprio_std"]
        out = bc["model"](pts, prn)
        mean = out[0] if bool(bc["hp"].get("predict_std", False)) else out
        eager_action = mean * bc["action_std"] + bc["action_mean"]
        jit_action = jit_bc["model"](pts, prop)
    diff = (eager_action - jit_action).abs().max().item()
    print(f"[convert] reloaded-JIT vs eager action max diff: {diff:.2e}")
    assert diff < 1e-3, f"reloaded JIT diverges from eager by {diff}"  # loose: op-fusion perturbs ~1e-5

    if args.benchmark:
        _benchmark(bc, jit_bc, args.device)
    print(f"[ok] {args.output}")


if __name__ == "__main__":
    main()
