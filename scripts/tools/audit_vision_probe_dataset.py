# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit a collect_vision_probe_dataset.py output: label-only file, statistics and image sample sheets.

Writes into ``<dataset>/audit/``: ``labels_all.pt`` (every non-image field, all shards, plus per-frame image brightness),
``stats.json``, and sheets ``random.png``, ``dark_wrist.png``, ``aligned.png``, ``near_miss.png`` (side | wrist per tile,
labelled reset type / noise / peg-hole distance / aligned).

usage: python audit_vision_probe_dataset.py <dataset dir>
"""

import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

D = sys.argv[1]
OUT = os.path.join(D, "audit")
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(os.path.join(D, "meta.json")))
names = [r.replace("Object", "").replace("EEGrasped", "") for r in meta["reset_types"]]
rng = np.random.default_rng(0)


def qrot_inv_xyzw(q, v):
    x, y, z, w = q.unbind(-1)
    u = torch.stack([-x, -y, -z], -1)
    t = 2 * torch.cross(u, v, dim=-1)
    return v + w[..., None] * t + torch.cross(u, t, dim=-1)


labels, samples, label_err = {}, {k: [] for k in ("random", "dark_wrist", "aligned", "near_miss")}, 0.0
prev_last = None
dup = 0
for path in sorted(glob.glob(os.path.join(D, "shard_*.pt"))):
    d = torch.load(path)
    side, wrist = d.pop("side"), d.pop("wrist")
    n = side.shape[0]
    d["side_mean"] = side.float().mean((1, 2, 3))
    d["wrist_mean"] = wrist.float().mean((1, 2, 3))
    d["side_std"] = side.float().std((1, 2, 3))
    d["wrist_std"] = wrist.float().std((1, 2, 3))
    d["shard"] = torch.full((n,), int(os.path.basename(path)[6:10]))
    ph = qrot_inv_xyzw(d["wrist_pose"][:, 3:], d["peg_pose"][:, :3] - d["hole_pose"][:, :3])
    label_err = max(label_err, float((ph - d["aux_state"][:, -3:]).abs().max()))
    # exact duplicate images within a shard (same env, consecutive records)
    flat = wrist.view(n, -1)[:, ::97]
    dup += int((flat[1:] == flat[:-1]).all(1).sum())
    for k, v in d.items():
        labels.setdefault(k, []).append(v)
    dist = (d["peg_pose"][:, :3] - d["hole_pose"][:, :3]).norm(dim=1)
    pick = {
        "random": rng.choice(n, 6, replace=False),
        "dark_wrist": torch.argsort(d["wrist_mean"])[:3].numpy(),
        "aligned": np.flatnonzero(d["aligned"].numpy())[:3],
        "near_miss": np.flatnonzero(((dist < 0.05) & ~d["aligned"]).numpy())[:3],
    }
    for k, idx in pick.items():
        for i in idx:
            tag = (f"{names[int(d['reset_type'][i])]} n{float(d['noise_std'][i]):.1f} "
                   f"d{float(dist[i]) * 1000:.0f}mm {'ALIGNED' if bool(d['aligned'][i]) else ''}")
            samples[k].append((np.concatenate([side[i].numpy(), wrist[i].numpy()], 1), tag))
    print(f"[audit] {path}: {n} frames", flush=True)
labels = {k: torch.cat(v) for k, v in labels.items()}
torch.save(labels, os.path.join(OUT, "labels_all.pt"))


def sheet(items, path, cols=4, scale=0.75):
    if not items:
        return
    h, w = items[0][0].shape[:2]
    th, tw = int(h * scale), int(w * scale)
    rows = (len(items) + cols - 1) // cols
    img = Image.new("RGB", (cols * tw, rows * (th + 14)), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    for k, (arr, tag) in enumerate(items):
        x, y = (k % cols) * tw, (k // cols) * (th + 14)
        img.paste(Image.fromarray(arr).resize((tw, th)), (x, y + 14))
        draw.text((x + 3, y + 1), tag, fill=(255, 255, 0))
    img.save(path)


for k, items in samples.items():
    sel = [items[i] for i in rng.choice(len(items), min(24, len(items)), replace=False)] if items else []
    if k == "dark_wrist":
        sel = items[:24]
    sheet(sel, os.path.join(OUT, f"{k}.png"))

N = labels["side_mean"].numel()
dist = (labels["peg_pose"][:, :3] - labels["hole_pose"][:, :3]).norm(dim=1)
ep_key = labels["env_id"] * 100000 + labels["episode"]
app = {s: labels[f"appearance_{s}"] for s in meta["appearance_surfaces"] if f"appearance_{s}" in labels}
pct = lambda x, q=(1, 10, 50, 90, 99): {str(p): round(float(np.percentile(x.numpy(), p)), 3) for p in q}
stats = {
    "frames": N, "episodes": int(ep_key.unique().numel()), "image_size": meta["image_size"],
    "label_check_max_abs_err_mm": round(label_err * 1000, 4), "exact_duplicate_wrist_frames": dup,
    "frames_per_reset_type": {names[i]: int((labels["reset_type"] == i).sum()) for i in range(len(names))},
    "frames_per_noise": {str(round(float(v), 2)): int((labels["noise_std"] == v).sum()) for v in labels["noise_std"].unique()},
    "aligned_fraction": round(float(labels["aligned"].float().mean()), 4),
    "aligned_fraction_by_noise": {str(round(float(v), 2)): round(float(labels["aligned"][labels["noise_std"] == v].float().mean()), 4) for v in labels["noise_std"].unique()},
    "peg_hole_distance_mm_percentiles": {k: round(v * 1000, 1) for k, v in pct(dist).items()},
    "frames_within_5cm_not_aligned": int(((dist < 0.05) & ~labels["aligned"]).sum()),
    "frames_within_2cm_not_aligned": int(((dist < 0.02) & ~labels["aligned"]).sum()),
    "side_brightness_percentiles": pct(labels["side_mean"]), "wrist_brightness_percentiles": pct(labels["wrist_mean"]),
    "near_black_frames": {"side": int((labels["side_mean"] < 15).sum()), "wrist": int((labels["wrist_mean"] < 15).sum())},
    "near_white_frames": {"side": int((labels["side_mean"] > 240).sum()), "wrist": int((labels["wrist_mean"] > 240).sum())},
    "flat_frames_std_lt_5": {"side": int((labels["side_std"] < 5).sum()), "wrist": int((labels["wrist_std"] < 5).sum())},
    "appearance": {
        s: {"textured_fraction": round(float((a[:, 3] >= 0).float().mean()), 3),
            "distinct_textures": int(a[a[:, 3] >= 0, 3].unique().numel()),
            "roughness_range": [round(float(a[:, 4].min()), 2), round(float(a[:, 4].max()), 2)],
            "metallic_range": [round(float(a[:, 5].min()), 2), round(float(a[:, 5].max()), 2)],
            "metallic_gt_0.7_fraction": round(float((a[:, 5] > 0.7).float().mean()), 3)}
        for s, a in app.items()},
}
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
