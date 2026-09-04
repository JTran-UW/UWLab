# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concatenate the failure clips of a record_failure_videos.py run into one sequential video (+ a 4x one).

Reads ``<run_dir>/results.json`` and plays every recorded clip back to back, each with a header
(clip number, env, episode, outcome, length). No simulator needed.
"""

import argparse
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np

parser = argparse.ArgumentParser(description="Sequentially compose failure clips.")
parser.add_argument("run_dir", type=str, help="Output dir of record_failure_videos.py (contains results.json and clips/).")
parser.add_argument("--speedups", type=int, nargs="+", default=[1, 4], help="Playback multipliers to write (one file each).")
parser.add_argument("--name", type=str, default="failures_sequential")
parser.add_argument("--hold_frames", type=int, default=5, help="Frames to hold the (dimmed, tagged) last frame of each clip.")
args = parser.parse_args()

with open(os.path.join(args.run_dir, "results.json")) as f:
    meta = json.load(f)
fps = int(meta["fps"])
recorded = [r for r in meta["episodes"] if r.get("clip")]
if not recorded:
    raise SystemExit("no recorded clips in results.json")
print(f"[INFO] {len(recorded)} clips, fps={fps}, speedups={args.speedups}")

out_paths = {k: os.path.join(args.run_dir, args.name + ("" if k == 1 else f"_{k}x") + ".mp4") for k in args.speedups}
writers = {k: imageio.get_writer(out_paths[k], fps=fps * k, codec="libx264", quality=9, macro_block_size=1)
           for k in args.speedups}


def label(frame: np.ndarray, top: str, tag: str | None, dim: bool) -> np.ndarray:
    f = frame.copy()
    if dim:
        f = (f * 0.45).astype(np.uint8)
    bar_h = 24
    cv2.rectangle(f, (0, 0), (f.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(f, top, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if tag:
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x, y = (f.shape[1] - tw) // 2, f.shape[0] // 2
        cv2.rectangle(f, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.putText(f, tag, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
    return f


total = 0
for n, r in enumerate(recorded, 1):
    path = r["clip"] if os.path.isabs(r["clip"]) else os.path.join(os.getcwd(), r["clip"])
    if not os.path.exists(path):
        path = os.path.join(args.run_dir, "clips", os.path.basename(r["clip"]))
    frames = [np.asarray(x) for x in imageio.mimread(path, memtest=False)]
    top = f"{n}/{len(recorded)}  env{r['env_id']} ep{r['episode']}  {r['outcome']}  {r['steps']} steps"
    for fr in frames:
        out = label(fr, top, None, dim=False)
        for w in writers.values():
            w.append_data(out)
    last = label(frames[-1], top, f"done: {r['outcome']}", dim=True)
    for _ in range(args.hold_frames):
        for w in writers.values():
            w.append_data(last)
    total += len(frames) + args.hold_frames
for w in writers.values():
    w.close()
for k in args.speedups:
    print(f"[INFO] wrote {out_paths[k]}  ({total / (fps * k):.0f} s)")
