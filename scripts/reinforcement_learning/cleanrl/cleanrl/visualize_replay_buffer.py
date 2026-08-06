"""Visualize a saved (expert) replay buffer -- as a sampled grid, or as per-camera videos.

Reads the payload written by ``collect_expert_replay_buffer.py``. Needs no Isaac/simulation
import, so it starts instantly, and mmaps the file so a 32 GiB buffer costs page reads rather
than 32 GiB of RAM.

Two modes:

* ``--mode grid``  (default) -- an (num_cameras x num_samples) grid of random transitions; each
  camera gets a row and each column is one sampled transition.
* ``--mode video`` -- takes a contiguous run of steps from ONE env and writes one video per
  camera. Rows of the stored tensor are chronological per env (the collector appends one
  rollout step at a time), so a slice ``[start : start+N]`` at a fixed env is a real trajectory.

The policy stream is stored flat; for the three-camera tasks each row unpacks as
(history, camera, H, W) = (3, 3, 84, 84), with history index 0 oldest and -1 most recent
(isaaclab CircularBuffer.buffer puts the newest entry last), and cameras in the order the
task cfg lists them: front, side, wrist.

Two modalities are supported and auto-detected from the buffer metadata:

* ``grayscale`` -- rgb collapsed to luma, values scaled to [0, 1].
* ``depth``     -- ``distance_to_camera``, values are raw metres. ``process_image`` maps inf
  (nothing hit within the camera's clipping range) to exactly 0.0, so 0.0 means "no return",
  NOT "very close". Those pixels are masked out of the colour scale and drawn in red.

Note episodes are short in expert buffers (~30 steps), so a 160-step window spans several of
them. Episode boundaries are drawn as a coloured border and counted in the overlay.

Examples:
    python visualize_replay_buffer.py --rb_path expert_rb/peg_grayscale_asym_seed42.pt
    python visualize_replay_buffer.py --rb_path expert_rb/peg_grayscale_asym_seed42.pt \
        --mode video --video_steps 160
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

INVALID_COLOR = "red"
INVALID_RGB = (255, 0, 0)
BOUNDARY_RGB = (255, 215, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rb_path", required=True, help="path to the saved replay buffer .pt")
    p.add_argument("--mode", choices=("grid", "video"), default="grid")
    p.add_argument("--out", default=None, help="output file (grid) or directory (video)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="which history frame to show; -1 = most recent, 0 = oldest",
    )
    p.add_argument("--history", type=int, default=3, help="history length stored per obs")
    p.add_argument("--num_cameras", type=int, default=3)
    p.add_argument("--image_size", type=int, nargs=2, default=(84, 84), metavar=("H", "W"))
    p.add_argument(
        "--camera_names",
        nargs="*",
        default=["front", "side", "wrist"],
        help="row labels, in the order the task cfg lists sensor_cfgs",
    )
    p.add_argument(
        "--stream",
        default="policy_observations",
        help="which tensor to visualize (e.g. policy_observations, next_policy_observations)",
    )
    p.add_argument(
        "--modality",
        choices=("auto", "grayscale", "depth"),
        default="auto",
        help="auto reads metadata['actor_obs_keys'] and falls back to a value-range guess",
    )
    p.add_argument("--cmap", default=None, help="matplotlib colormap (default: gray / viridis for depth)")
    p.add_argument("--vmin", type=float, default=None, help="explicit lower limit for the shared scale")
    p.add_argument("--vmax", type=float, default=None, help="explicit upper limit for the shared scale")

    g = p.add_argument_group("grid mode")
    g.add_argument("--num_samples", type=int, default=10, help="images per camera (columns)")
    g.add_argument(
        "--independent",
        action="store_true",
        help="sample each camera's images independently instead of sharing one set of transitions "
        "(shared is the default so a column shows the same instant from all three views)",
    )
    g.add_argument(
        "--shared_scale",
        action="store_true",
        help="use ONE colour scale for every panel instead of per-image min-max. For depth this "
        "makes panels metrically comparable and adds a colourbar in metres; for grayscale it "
        "pins the scale to [0,1] so you see true brightness.",
    )
    g.add_argument("--raw_scale", dest="shared_scale", action="store_true", help=argparse.SUPPRESS)

    v = p.add_argument_group("video mode")
    v.add_argument("--video_steps", type=int, default=160, help="length of the contiguous run")
    v.add_argument("--start", type=int, default=None, help="first step index (default: random)")
    v.add_argument("--env", type=int, default=None, help="which env to follow (default: random)")
    v.add_argument("--fps", type=int, default=20)
    v.add_argument("--upscale", type=int, default=4, help="nearest-neighbour zoom for legibility")
    v.add_argument(
        "--no_overlay",
        action="store_true",
        help="omit the step/episode text and episode-boundary border",
    )
    return p.parse_args()


def load_payload(path: str) -> dict:
    """Load with mmap when possible so a 32 GiB buffer costs a few page reads, not 32 GiB of RAM."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError) as exc:  # not zipfile-serialized, or torch too old
        print(f"[warn] mmap load failed ({exc}); falling back to a full read -- this may take a while.")
        return torch.load(path, map_location="cpu", weights_only=False)


def detect_modality(meta: dict, sample: np.ndarray) -> tuple[str, str]:
    """Return (modality, how_it_was_decided).

    Metadata is authoritative: the collector records which observation group was recorded, and the
    task cfgs name them 'depth' / 'grayscale'. The value-range fallback only fires for buffers
    written before those keys existed, and is deliberately conservative -- depth in metres routinely
    exceeds 1.0, whereas rgb is clamped to [0, 1].
    """
    keys = [str(k).lower() for k in (meta.get("actor_obs_keys") or [])]
    if any("depth" in k for k in keys):
        return "depth", f"metadata actor_obs_keys={keys}"
    if any("gray" in k or "rgb" in k for k in keys):
        return "grayscale", f"metadata actor_obs_keys={keys}"
    if float(np.nanmax(sample)) > 1.001:
        return "depth", "value-range guess (max > 1.0)"
    return "grayscale", "value-range guess (values within [0,1])"


def colorize(img: np.ndarray, cmap, lo: float, hi: float, is_depth: bool) -> np.ndarray:
    """Map a HxW float image to an HxWx3 uint8 RGB frame, painting invalid depth pixels red."""
    hi = hi if hi > lo else lo + 1e-6
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    if is_depth:
        rgb[img == 0.0] = INVALID_RGB
    return rgb


def write_videos(args, obs, tensors, meta, modality, cmap, dims) -> None:
    import imageio.v2 as imageio

    hist, cam, h, w = dims
    is_depth = modality == "depth"
    buffer_size, n_envs = obs.shape[0], obs.shape[1]
    pos, full = int(tensors.get("pos", buffer_size)), bool(tensors.get("full", True))
    n_valid = buffer_size if full else pos

    rng = np.random.default_rng(args.seed)
    n_steps = args.video_steps
    if n_steps > n_valid:
        raise SystemExit(f"--video_steps {n_steps} exceeds the {n_valid} valid steps in this buffer")
    env = args.env if args.env is not None else int(rng.integers(0, n_envs))
    if not 0 <= env < n_envs:
        raise SystemExit(f"--env {env} out of range (buffer has {n_envs} envs)")
    max_start = n_valid - n_steps
    start = args.start if args.start is not None else int(rng.integers(0, max_start + 1))
    if not 0 <= start <= max_start:
        raise SystemExit(f"--start {start} out of range; must be within [0, {max_start}] for {n_steps} steps")
    stop = start + n_steps

    # (n_steps, cameras, H, W) -- one frame per step, taking the chosen history slot.
    seq = obs[start:stop, env].reshape(n_steps, hist, cam, h, w)[:, args.frame].float().numpy()

    # Episode boundaries: a 160-step window of an expert buffer spans several episodes, and without
    # marking them the jump between episodes looks like a glitch.
    dones = np.zeros(n_steps, dtype=bool)
    for key in ("terminations", "truncations"):
        if key in tensors:
            dones |= tensors[key][start:stop, env].bool().numpy().reshape(n_steps)
    n_boundaries = int(dones.sum())

    # One fixed scale for the whole clip. Per-frame normalization would make the video flicker and
    # would also destroy any sense of absolute depth.
    valid = seq[seq != 0.0] if is_depth else seq
    valid = valid if valid.size else seq
    lo = args.vmin if args.vmin is not None else (float(valid.min()) if is_depth else 0.0)
    hi = args.vmax if args.vmax is not None else (float(valid.max()) if is_depth else 1.0)

    unit = " m" if is_depth else ""
    print(
        f"     video     : env {env}, steps [{start}, {stop}) = {n_steps} frames @ {args.fps} fps "
        f"({n_steps / args.fps:.1f} s)\n"
        f"     scale     : fixed {lo:.3f}{unit} - {hi:.3f}{unit} across all frames\n"
        f"     episodes  : {n_boundaries} boundary/boundaries inside the window"
        + (f"  |  invalid pixels: {100 * (seq == 0).mean():.1f}%" if is_depth else "")
    )

    out_dir = args.out or os.path.dirname(os.path.abspath(args.rb_path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.rb_path))[0]

    zoom = max(1, args.upscale)
    episode_of = np.cumsum(np.concatenate([[0], dones[:-1]]))  # frame -> episode ordinal in-window

    for c in range(cam):
        name = args.camera_names[c]
        frames = []
        for i in range(n_steps):
            rgb = colorize(seq[i, c], cmap, lo, hi, is_depth)
            im = Image.fromarray(rgb).resize((w * zoom, h * zoom), Image.NEAREST)
            if not args.no_overlay:
                d = ImageDraw.Draw(im)
                d.text((4, 2), f"{name}  s{start + i}  ep{int(episode_of[i])}", fill=(255, 255, 255))
                if dones[i]:
                    d.rectangle([0, 0, im.width - 1, im.height - 1], outline=BOUNDARY_RGB, width=3)
                    d.text((4, 14), "episode end", fill=BOUNDARY_RGB)
            frames.append(np.asarray(im))

        out = os.path.join(out_dir, f"{base}_{name}_e{env}_s{start}-{stop}.mp4")
        # macro_block_size=1 keeps the exact pixel size instead of padding to a multiple of 16.
        imageio.mimsave(out, frames, fps=args.fps, macro_block_size=1)
        px = seq[:, c]
        v = px[px != 0.0] if is_depth else px
        v = v if v.size else px
        extra = f"  invalid {100 * (px == 0).mean():5.1f}%" if is_depth else ""
        print(f"[saved] {out}   range {v.min():.3f}-{v.max():.3f}{unit}{extra}")


def render_grid(args, obs, tensors, meta, modality, cmap, dims) -> None:
    hist, cam, h, w = dims
    is_depth = modality == "depth"
    buffer_size, n_envs = obs.shape[0], obs.shape[1]
    pos, full = int(tensors.get("pos", buffer_size)), bool(tensors.get("full", True))
    n_valid = buffer_size if full else pos

    rng = np.random.default_rng(args.seed)
    n = args.num_samples

    def draw_sample() -> tuple[int, int]:
        return int(rng.integers(0, n_valid)), int(rng.integers(0, n_envs))

    def frame_at(step: int, env: int, c: int) -> np.ndarray:
        return obs[step, env].reshape(hist, cam, h, w)[args.frame, c].float().numpy()

    shared = [draw_sample() for _ in range(n)]
    picks_per_cam = [([draw_sample() for _ in range(n)] if args.independent else shared) for _ in range(cam)]
    grid = [[frame_at(s, e, r) for (s, e) in picks_per_cam[r]] for r in range(cam)]
    all_px = np.stack([img for row in grid for img in row])

    def valid_mask(a: np.ndarray) -> np.ndarray:
        return a != 0.0 if is_depth else np.ones_like(a, dtype=bool)

    finite = all_px[valid_mask(all_px)]
    if finite.size == 0:
        print("[warn] every sampled pixel is invalid (all zero) -- nothing to scale against.")
        finite = all_px
    unit = " m" if is_depth else ""
    print(
        f"     values    : min={finite.min():.3f}{unit} max={finite.max():.3f}{unit} "
        f"mean={finite.mean():.3f}{unit} std={finite.std():.3f}{unit}"
        + (f"  invalid(==0): {100 * (all_px == 0).mean():.1f}% of pixels" if is_depth else "")
    )
    print(f"     {'camera':<8}{'min':>9}{'max':>9}{'mean':>9}" + (f"{'invalid%':>10}" if is_depth else ""))
    for r, name in enumerate(args.camera_names):
        px = np.stack(grid[r])
        v = px[valid_mask(px)]
        v = v if v.size else px
        line = f"     {name:<8}{v.min():9.3f}{v.max():9.3f}{v.mean():9.3f}"
        if is_depth:
            line += f"{100 * (px == 0).mean():10.1f}"
        print(line)

    if args.shared_scale:
        lo = args.vmin if args.vmin is not None else (float(finite.min()) if is_depth else 0.0)
        hi = args.vmax if args.vmax is not None else (float(finite.max()) if is_depth else 1.0)
    else:
        lo = hi = None

    mpl_cmap = cmap.with_extremes(bad=INVALID_COLOR)
    fig, axes = plt.subplots(cam, n, figsize=(1.5 * n, 1.65 * cam))
    axes = np.atleast_2d(axes)
    im = None
    for r in range(cam):
        for c in range(n):
            img = grid[r][c]
            shown = np.ma.masked_where(~valid_mask(img), img) if is_depth else img
            if lo is not None:
                vmin, vmax = lo, hi
            else:
                v = img[valid_mask(img)]
                vmin, vmax = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
                if vmax <= vmin:
                    vmax = vmin + 1e-6
            ax = axes[r, c]
            im = ax.imshow(shown, cmap=mpl_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                step, env = picks_per_cam[r][c]
                ax.set_title(f"s{step}/e{env}", fontsize=6)
        axes[r, 0].set_ylabel(args.camera_names[r], fontsize=10, rotation=0, ha="right", va="center", labelpad=18)

    scale_desc = f"shared {lo:.2f}-{hi:.2f}{unit}" if args.shared_scale else "per-image min-max"
    sampling = "independent per camera" if args.independent else "shared across cameras"
    title = (
        f"{os.path.basename(args.rb_path)} - {args.stream} [{modality}], "
        f"history frame {args.frame} ({sampling}, {scale_desc})"
    )
    if is_depth:
        title += f"\ninvalid / no-return pixels in {INVALID_COLOR}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99 if not is_depth else 0.94))

    if args.shared_scale and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
        cbar.set_label("metres" if is_depth else "intensity", fontsize=9)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.rb_path)),
        f"{os.path.splitext(os.path.basename(args.rb_path))[0]}_samples.png",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


def main() -> None:
    args = parse_args()

    payload = load_payload(args.rb_path)
    tensors, meta = payload["buffer_tensors"], payload["metadata"]

    if args.stream not in tensors:
        raise SystemExit(f"stream '{args.stream}' not in buffer; available: {sorted(tensors.keys())}")
    obs = tensors[args.stream]  # (buffer_size, n_envs, obs_dim)
    buffer_size, n_envs, obs_dim = obs.shape

    hist, cam, (h, w) = args.history, args.num_cameras, args.image_size
    expected = hist * cam * h * w
    if expected != obs_dim:
        raise SystemExit(
            f"layout {hist}x{cam}x{h}x{w} = {expected} != stored obs dim {obs_dim}.\n"
            f"Pass --history/--num_cameras/--image_size to match this buffer."
        )
    if len(args.camera_names) != cam:
        raise SystemExit(f"got {len(args.camera_names)} camera names for {cam} cameras")

    pos, full = int(tensors.get("pos", buffer_size)), bool(tensors.get("full", True))
    n_valid = buffer_size if full else pos
    if n_valid == 0:
        raise SystemExit("buffer is empty (pos=0, full=False)")
    # A wrapped ring buffer is only chronological when it stopped exactly at the wrap point.
    if full and pos != 0 and args.mode == "video":
        print(
            f"[warn] buffer wrapped and stopped mid-ring (pos={pos}); rows are NOT chronological "
            f"across index {pos}. A window spanning it will jump in time."
        )

    probe = obs[0, 0].reshape(hist, cam, h, w)[args.frame].float().numpy()
    modality, why = detect_modality(meta, probe)
    if args.modality != "auto":
        modality, why = args.modality, "forced by --modality"
    cmap_name = args.cmap or ("viridis" if modality == "depth" else "gray")
    cmap = matplotlib.colormaps[cmap_name]

    print(
        f"[rb] {args.rb_path}\n"
        f"     task      : {meta.get('task', '?')}\n"
        f"     transitions: {n_valid * n_envs:,} valid ({n_valid} steps x {n_envs} envs, full={full})\n"
        f"     stream    : {args.stream}  shape={tuple(obs.shape)}\n"
        f"     layout    : ({hist} history, {cam} cameras, {h}, {w}), showing frame {args.frame} "
        f"({'most recent' if args.frame in (-1, hist - 1) else 'index ' + str(args.frame)})\n"
        f"     modality  : {modality}  [{why}]  cmap={cmap_name}"
    )

    dims = (hist, cam, h, w)
    if args.mode == "video":
        write_videos(args, obs, tensors, meta, modality, cmap, dims)
    else:
        render_grid(args, obs, tensors, meta, modality, cmap, dims)


if __name__ == "__main__":
    main()
