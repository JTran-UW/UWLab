"""Recompose a side-by-side gap video from an existing results.json + clips (no simulator needed).

Usage: python compose_gap_video.py <results.json> <out.mp4> [--order "label1" "label2" ...]
Panels follow --order (default: results.json order). Same padding/labels/badges as play_peg_mass_video.py.
"""
import argparse
import json
import os

import cv2
import imageio
import numpy as np


def _draw_label(frame, top, bottom, dim, success=False, show_mass=None):
    f = frame.copy()
    if dim:
        f = (f * 0.45).astype(np.uint8)
    bar_h = 36
    cv2.rectangle(f, (0, 0), (f.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(f, top, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if success:
        txt = "SUCCESS"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x, y = f.shape[1] - tw - 16, bar_h + th + 14
        cv2.rectangle(f, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 110, 0), -1)
        cv2.putText(f, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 255, 90), 3, cv2.LINE_AA)
    if bottom:
        (tw, th), _ = cv2.getTextSize(bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x, y = (f.shape[1] - tw) // 2, f.shape[0] // 2
        cv2.rectangle(f, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(f, bottom, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 80) if "success" in bottom else (80, 80, 255), 2, cv2.LINE_AA)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--order", nargs="+", default=None)
    ap.add_argument("--show_mass", action="store_true")
    a = ap.parse_args()
    r = json.load(open(a.results))
    fps = r["fps"]
    order = a.order or r["conditions"]
    missing = [c for c in order if c not in r["results"]]
    assert not missing, f"unknown conditions {missing}; available: {list(r['results'])}"
    n = len(next(iter(r["results"].values())))
    out4x = os.path.splitext(a.out)[0] + "_4x.mp4"
    w = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=9, macro_block_size=1)
    w4 = imageio.get_writer(out4x, fps=fps * 4, codec="libx264", quality=9, macro_block_size=1)
    for k in range(n):
        per = []
        for label in order:
            info = r["results"][label][k]
            fr = [np.asarray(x) for x in imageio.mimread(info["clip"], memtest=False)]
            per.append((label, info, fr))
        L = max(len(fr) for _, _, fr in per)
        H, W = per[0][2][0].shape[:2]
        for t in range(L):
            panels = []
            for label, info, fr in per:
                live = t < len(fr)
                frame = fr[t] if live else fr[-1]
                if frame.shape[:2] != (H, W):
                    frame = cv2.resize(frame, (W, H))
                top = f"{label}  |  s_{k}" + (f"  |  m={info['peg_mass']:.2f} kg" if a.show_mass else "")
                tag = None if live else ("done: " + ("success" if info["success"] else ("abnormal" if info["outcome"] == "abnormal" else "timeout")))
                fs = info["first_success_step"]
                succ = bool(info["success"]) and fs is not None and t >= fs
                panels.append(_draw_label(frame, top, tag, dim=not live, success=succ))
            row = np.concatenate(panels, axis=1)
            w.append_data(row)
            w4.append_data(row)
    w.close()
    w4.close()
    print(f"wrote {a.out} and {out4x}")


if __name__ == "__main__":
    main()
