"""Render replay/expert-buffer image obs (uint8, (cams*3, H, W) per row) as sequential labeled mp4s, pixels as stored.

usage: python render_replay_images.py --dump <replay_dump.pt> [--expert <expert_buffer.pt>] --out <dir>
"""

import argparse
import os
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("--dump", default=None, help="FLASHSAC_DUMP_REPLAY output")
p.add_argument("--expert", default=None, help="expert buffer from collect_flashsac_expert_buffer.py")
p.add_argument("--out", required=True)
p.add_argument("--envs", type=int, default=8)
p.add_argument("--expert_steps", type=int, default=160)
p.add_argument("--scale", type=int, default=5)
p.add_argument("--fps", type=int, default=10)
p.add_argument("--full_episodes", type=int, default=0, help="also write N complete online episodes that start after --warmup steps")
p.add_argument("--warmup", type=int, default=12, help="random-action warm-up steps at the start of the dump")
p.add_argument("--tag", default="")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
CAMS = ["side", "wrist"]


def frame(img, label):
    """img: (cams*3, H, W) uint8 -> cameras side by side, nearest-upscaled, label bar underneath."""
    img = np.asarray(img)
    cams = [img[3 * c : 3 * c + 3].transpose(1, 2, 0) for c in range(img.shape[0] // 3)]
    h, w = cams[0].shape[:2]
    s = args.scale
    canvas = Image.new("RGB", (len(cams) * w * s + (len(cams) - 1) * 8, h * s + 48), (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    for c, im in enumerate(cams):
        x = c * (w * s + 8)
        canvas.paste(Image.fromarray(im).resize((w * s, h * s), Image.NEAREST), (x, 0))
        d.text((x + 4, 4), f"{CAMS[c] if c < len(CAMS) else c} {w}x{h}", font=FONT, fill=(255, 255, 0))
    d.text((6, h * s + 14), label, font=FONT, fill=(255, 255, 255))
    return np.asarray(canvas)


def write(frames, path, fps):
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path],
                            stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {path} ({len(frames)} frames)")


def series_frames(img, rew, term, trunc, envs, tag):
    """img: (T, E, C, H, W); one env after another, each over its full stored time series.

    Rows hold n-step transitions, so an episode end flags the last n rows before it: the new episode starts after a flag run.
    """
    frames = []
    for e in range(envs):
        ep, k = 0, 0
        flag = (term[:, e] + trunc[:, e]) > 0
        for t in range(img.shape[0]):
            if t > 0 and flag[t - 1] and not flag[t]:
                ep, k = ep + 1, 0
            mark = "  NEW EPISODE" if k == 0 and t > 0 else ("  (episode ends within n steps)" if flag[t] else "")
            frames.append(frame(img[t, e], f"{tag}  env{e}  ep{ep} step{k}  r={float(rew[t, e]):+.2f}{mark}"))
            k += 1
    return frames


if args.dump:
    d = torch.load(args.dump, map_location="cpu", weights_only=False)
    print(f"dump: {d['steps']} steps x {d['num_envs']} envs, storage dtype {d['dtype']}")
    for k, img in d["obs"].items():
        u8 = img if img.dtype == torch.uint8 else img.clamp(0, 255).to(torch.uint8)
        print(f"  {k}: {tuple(img.shape)} {img.dtype} min {int(img.min())} max {int(img.max())} mean {img.float().mean():.1f}")
        write(series_frames(u8.numpy(), d["reward"].numpy(), d["terminated"].numpy(), d["truncated"].numpy(),
                            min(args.envs, u8.shape[1]), "online replay"), f"{args.out}/online_replay_{k}_sequential.mp4", args.fps)
        if args.full_episodes:
            flag = ((d["terminated"] + d["truncated"]) > 0).numpy()
            rew, img_np = d["reward"].numpy(), u8.numpy()
            eps = []
            for e in range(flag.shape[1]):
                starts = [t for t in range(1, flag.shape[0]) if flag[t - 1, e] and not flag[t, e]]
                for s0 in starts:
                    ends = [t for t in range(s0, flag.shape[0] - 1) if flag[t, e] and not flag[t + 1, e]]
                    if s0 >= args.warmup and ends:
                        eps.append((e, s0, ends[0]))
                        break
            frames = []
            for e, s0, s1 in eps[: args.full_episodes]:
                term = bool(d["terminated"][s1, e] > 0)
                for t in range(s0, s1 + 1):
                    frames.append(frame(img_np[t, e], f"{args.tag}online replay  env{e}  step {t - s0}/{s1 - s0}  r={float(rew[t, e]):+.2f}  "
                                                      f"{'ends: TERMINATED' if term else 'ends: TRUNCATED'}"))
                print(f"  episode env{e} rows {s0}-{s1} ({s1 - s0 + 1} steps, {'terminated' if term else 'truncated'})")
            if frames:
                write(frames, f"{args.out}/online_replay_{k}_full_episodes_sequential.mp4", args.fps)
        b = d["batch"][k].clamp(0, 255).to(torch.uint8).numpy()
        frames = [frame(b[i], f"train mini-batch sample {i}  (online)") for i in range(len(b))]
        if d.get("batch_expert"):
            be = d["batch_expert"][k].clamp(0, 255).to(torch.uint8).numpy()
            frames += [frame(be[i], f"train mini-batch sample {i}  (expert)") for i in range(len(be))]
        write(frames, f"{args.out}/train_minibatch_{k}_sequential.mp4", 2)

if args.expert:
    x = torch.load(args.expert, map_location="cpu", weights_only=False, mmap=True)
    ne = int(x["metadata"]["num_envs"])
    t = min(args.expert_steps, x["num_in_buffer"] // ne)
    view = lambda v: v[: t * ne].view(t, ne, *v.shape[1:])[:, : args.envs]
    for k, v in x["observation"].items():
        if v.dim() < 4:
            continue
        img = view(v).clone()
        print(f"expert {k}: {tuple(img.shape)} {img.dtype} min {int(img.min())} max {int(img.max())} mean {img.float().mean():.1f}")
        write(series_frames(img.numpy(), view(x["reward"]).numpy(), view(x["terminated"]).numpy(), view(x["truncated"]).numpy(),
                            args.envs, "expert buffer"), f"{args.out}/expert_buffer_{k}_sequential.mp4", args.fps)
