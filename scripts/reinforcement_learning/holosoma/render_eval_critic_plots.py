# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the per-reset figures from ``eval_critic.py --defer_plots`` rollout dumps, in parallel.

    python render_eval_critic_plots.py <plots_dir> [<plots_dir> ...] [--workers N] [--peg_xlim ..] [--peg_ylim ..]

Each ``rollout*.npz`` under a given directory yields iter####_{q_vs_mc,peg_xy,q_over_traj}.png next to it.
"""

import argparse
import glob
import os
from multiprocessing import Pool

from eval_critic_plots import render_rollout_npz


def _job(args):
    path, xlim, ylim = args
    return path, render_rollout_npz(path, xlim, ylim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="plots dirs (searched recursively for rollout*.npz)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--peg_xlim", type=float, nargs=2, default=[0.0, 0.9])
    p.add_argument("--peg_ylim", type=float, nargs=2, default=[-0.4, 0.7])
    a = p.parse_args()
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "**", "rollout*.npz"), recursive=True))
    print(f"[render] {len(files)} rollout files, {a.workers} workers")
    with Pool(a.workers) as pool:
        for path, n in pool.imap_unordered(_job, [(f, a.peg_xlim, a.peg_ylim) for f in files]):
            print(f"[render] {path}: {n} groups")


if __name__ == "__main__":
    main()
