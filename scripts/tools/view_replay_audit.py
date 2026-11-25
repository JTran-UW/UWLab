# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render FlashSACRGB replay-audit dumps: per-env stored episodes as sequential videos, update batches as sheets."""

import argparse, glob, os
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("audit_dir")
p.add_argument("--out", default="")
p.add_argument("--scale", type=int, default=3)
args = p.parse_args()
out = args.out or os.path.join(args.audit_dir, "rendered")
os.makedirs(out, exist_ok=True)


def to_hwc(img):  # (C,H,W) cams*3 channels -> cameras side by side, uint8
    a = img.float().clamp(0, 255).to(torch.uint8).numpy() if img.dtype != torch.uint8 else img.numpy()
    if a.ndim == 3 and a.shape[0] % 3 == 0 and a.shape[0] <= 12:
        a = np.concatenate([a[3 * c:3 * c + 3].transpose(1, 2, 0) for c in range(a.shape[0] // 3)], axis=1)
    return a


def big(a, s):
    return np.asarray(Image.fromarray(a).resize((a.shape[1] * s, a.shape[0] * s), Image.NEAREST))


for f in sorted(glob.glob(os.path.join(args.audit_dir, "audit_call*.pt"))):
    d = torch.load(f, weights_only=False); tag = os.path.basename(f)[:-3]
    w = imageio.get_writer(os.path.join(out, f"{tag}_replay_episodes.mp4"), fps=10, macro_block_size=1)
    for e, s in d.get("series", {}).items():
        kind = "EXPERT env" if s["expert_env"] else "policy env"
        for t in range(s["image"].shape[0]):
            fr = big(to_hwc(s["image"][t]), args.scale)
            c = Image.new("RGB", (fr.shape[1], fr.shape[0] + 26)); c.paste(Image.fromarray(fr), (0, 26))
            mark = " DONE" if s["terminated"][t] > 0 else (" TRUNC" if s["truncated"][t] > 0 else "")
            ImageDraw.Draw(c).text((4, 6), f"{tag} {kind} {e} stored step {t} r={float(s['reward'][t]):.2f}{mark} (side|wrist)", fill=(255, 255, 0))
            w.append_data(np.asarray(c))
    w.close()
    for key in ("batch_obs_image", "batch_next_image"):
        imgs = d[key][:32]
        tiles = [big(to_hwc(i), 2) for i in imgs]
        cols = 4; rows = (len(tiles) + cols - 1) // cols; th, tw = tiles[0].shape[:2]
        sheet = Image.new("RGB", (cols * tw, rows * th + 24))
        for k, t in enumerate(tiles): sheet.paste(Image.fromarray(t), ((k % cols) * tw, 24 + (k // cols) * th))
        ImageDraw.Draw(sheet).text((4, 4), f"{tag} {key} (update batch, as fed to the encoder; dtype {d[key].dtype})", fill=(255, 255, 0))
        sheet.save(os.path.join(out, f"{tag}_{key}.png"))
    print("rendered", tag)
