# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the point-cloud coordinate distribution of a collected demo dataset.

Randomly samples ``--num_clouds`` point clouds (one per randomly chosen demo timestep)
from a ``collect_pc_demos.py`` HDF5 file, then reports the per-axis (x, y, z) and
point-norm distributions: percentile tables + histograms. Use it to decide whether the
sim2real augmentation is injecting wacky fliers that should be cropped/filtered before
training.

Usage (inside the uw-lab container, where the dataset lives)::

    /isaac-sim/python.sh scripts/tools/sim2real/analyze_pc_dataset.py \\
        --dataset eval_ckpts/.../bench_16k_100k.hdf5 --num_clouds 4000 \\
        --out scripts/tools/sim2real/pc_dist.png
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PCTILES = [0.0, 0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9, 100.0]


def parse_args():
    p = argparse.ArgumentParser(description="Analyze PC coordinate distribution of a demo dataset.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--pc_key", type=str, default="scene_pc", help="obs term holding the (T, N, 3) cloud.")
    p.add_argument("--num_clouds", type=int, default=4000, help="Number of clouds (random demo timesteps) to sample.")
    p.add_argument("--out", type=str, default="scripts/tools/sim2real/pc_dist.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
        lengths = np.array([data[d]["actions"].shape[0] for d in demos])
        n_total = int(lengths.sum())
        print(f"[analyze] {len(demos)} demos, {n_total} timesteps; sampling {args.num_clouds} clouds")

        # Sample distinct (demo, t) pairs proportional to demo length, then read one cloud each.
        k = min(args.num_clouds, n_total)
        flat = rng.choice(n_total, size=k, replace=False)
        starts = np.concatenate([[0], np.cumsum(lengths)])
        demo_idx = np.searchsorted(starts, flat, side="right") - 1
        t_idx = flat - starts[demo_idx]

        clouds = []
        for di, ti in zip(demo_idx, t_idx):
            clouds.append(data[demos[di]]["obs"][args.pc_key][int(ti)])  # (N, 3)
        pts = np.concatenate(clouds, axis=0).astype(np.float32)  # (k*N, 3)

    norm = np.linalg.norm(pts, axis=1)
    cols = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "norm": norm}

    # ---- percentile table ----
    print(f"\n[analyze] {pts.shape[0]:,} points from {len(clouds)} clouds\n")
    hdr = "        " + "".join(f"{p:>9}%" for p in _PCTILES)
    print(hdr)
    for name, v in cols.items():
        qs = np.percentile(v, _PCTILES)
        print(f"{name:>6}  " + "".join(f"{q:>10.3f}" for q in qs) + f"   (mean {v.mean():.3f}, std {v.std():.3f})")

    # Outlier fractions: points whose norm sits far in the tail (likely fliers).
    print("\n[analyze] norm-tail fractions (candidate fliers):")
    for thr in (1.5, 2.0, 3.0, 5.0):
        frac = float((norm > thr).mean())
        print(f"  |p| > {thr:>3.1f} m : {frac * 100:7.4f}%  ({int((norm > thr).sum()):,} pts)")

    # ---- histograms ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, v) in zip(axes.ravel(), cols.items()):
        lo, hi = np.percentile(v, [0.1, 99.9])
        ax.hist(v, bins=200, range=(lo, hi), color="steelblue", alpha=0.85)
        ax.axvline(np.median(v), color="k", ls="--", lw=1, label=f"median {np.median(v):.3f}")
        ax.set_title(f"{name}  [{v.min():.2f}, {v.max():.2f}]  (hist clipped to 0.1-99.9 pctile)")
        ax.set_xlabel(name)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle(f"PC coord distribution -- {args.dataset}  ({pts.shape[0]:,} pts / {len(clouds)} clouds)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\n[analyze] wrote histogram figure -> {args.out}")


if __name__ == "__main__":
    main()
