# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a collected point-cloud episode to a video, recolored by point origin.

Reads one (or more) demos from a sim2real PC collection HDF5 (e.g.
``demos/contact_seg_100k.hdf5``) and writes an MP4 per episode showing the
``scene_pc`` cloud evolve over the full trajectory. Points are recolored by their
source class, read from the 4th ``scene_pc`` channel written by the segmented
collection (``robot=0``, ``insertive=-1``, ``receptive=+1``).

Pure offline -- no Isaac Sim. Runs in the host ``patlab`` conda (matplotlib +
imageio[ffmpeg]).

Usage::

    /home/ubuntu/miniforge3/envs/patlab/bin/python \
        scripts/imitation_learning/point_cloud/visualize_pc_episode.py \
        --dataset demos/contact_seg_100k.hdf5 --num_episodes 5 --out_dir /tmp/pc_vids
"""

from __future__ import annotations

import argparse
import os

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # headless render to RGB buffers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# scene_pc channel-4 segmentation codes (see OccludedScenePointCloud /
# DataCollectionPCContactCfg.segmentation_labels) -> a display color + name.
_CLASS_SPEC = {
    0.0: ("robot", "#7f7f7f"),  # gray
    -1.0: ("insertive", "#d62728"),  # red
    1.0: ("receptive", "#1f77b4"),  # blue
}
_UNKNOWN_COLOR = "#2ca02c"  # green: any label not in _CLASS_SPEC
_UNKNOWN_VALUE = 999.0  # sentinel seg code -> rendered with _UNKNOWN_COLOR


def _parse_episodes(args, n_demos: int) -> list[int]:
    """Resolve --episodes / --num_episodes into a concrete, in-range index list."""
    if args.episodes:
        out = []
        for tok in args.episodes.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-")
                out.extend(range(int(a), int(b) + 1))
            elif tok:
                out.append(int(tok))
    else:
        out = list(range(args.num_episodes))
    return [i for i in out if 0 <= i < n_demos]


def _colors_for(labels: np.ndarray) -> np.ndarray:
    """Map per-point segmentation codes -> RGBA colors. Robot is dimmed (low alpha)
    so the sparse task objects (peg/hole) stand out against the dominant arm cloud."""
    rounded = np.round(labels, 3)
    cols = np.empty((labels.shape[0], 4), dtype=np.float32)
    cols[:] = matplotlib.colors.to_rgba(_UNKNOWN_COLOR)
    for code, (name, hexc) in _CLASS_SPEC.items():
        rgba = list(matplotlib.colors.to_rgba(hexc))
        if name == "robot":
            rgba[3] = 0.30  # dim the background arm
        cols[rounded == code] = rgba
    return cols


def _sizes_for(labels: np.ndarray, base: float) -> np.ndarray:
    """Per-point marker size: enlarge peg/hole (small, occluded) vs the robot."""
    rounded = np.round(labels, 3)
    sizes = np.full(labels.shape[0], base, dtype=np.float32)
    sizes[rounded == 0.0] = base * 0.7  # robot smaller
    sizes[rounded == -1.0] = base * 3.5  # insertive (peg) larger
    sizes[rounded == 1.0] = base * 3.5  # receptive (hole) larger
    return sizes


def _remap_optical_to_view(xyz: np.ndarray) -> np.ndarray:
    """Camera optical frame (x right, y up, -z forward) -> plot frame with z = up.

    plot_x = x (right), plot_y = -z (depth into scene), plot_z = y (height). Makes
    the table roughly horizontal and the arm vertical for a natural 3D view.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([x, -z, y], axis=-1)


def _block_labels(class_counts: list[int], num_points: int) -> np.ndarray:
    """Per-point origin labels for a cloud with NO segmentation channel.

    ``ScenePointCloud`` emits points in a fixed index order -- ``[robot, insertive,
    receptive]`` blocks sized by ``class_ratios`` (e.g. (0.5,0.25,0.25)*512 ->
    (256,128,128)) -- and the per-index source is constant for the whole episode. So
    when only xyz is stored (e.g. clean_scenepc), we can recover the seg codes
    (robot=0, insertive=-1, receptive=+1) purely from the index blocks.
    """
    codes = (0.0, -1.0, 1.0)  # robot, insertive, receptive -- matches _CLASS_SPEC
    labels = np.full(num_points, _UNKNOWN_VALUE, dtype=np.float32)
    start = 0
    for code, n in zip(codes, class_counts):
        labels[start:start + n] = code
        start += n
    return labels


def render_episode(pc: np.ndarray, out_path: str, fps: int, elev: float, azim: float, point_size: float,
                   block_labels: np.ndarray | None = None):
    """pc: (T, P, C>=3). Channel 3 (if present) is the origin label; otherwise
    ``block_labels`` (P,) supplies a fixed per-index label. Writes an MP4."""
    T = pc.shape[0]
    has_label = pc.shape[2] >= 4 or block_labels is not None
    xyz_all = _remap_optical_to_view(pc[..., :3])  # (T, P, 3)

    # Fixed limits over the whole episode so the cloud doesn't jump frame to frame.
    flat = xyz_all.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    center = (lo + hi) / 2.0
    radius = (hi - lo).max() / 2.0 * 1.05 + 1e-6
    lims = np.stack([center - radius, center + radius], axis=1)  # (3, 2), equal aspect

    def _counts(lbl):
        u, c = np.unique(np.round(lbl, 3), return_counts=True)
        return {next((n for k, (n, _) in _CLASS_SPEC.items() if k == uu), f"{uu:g}"): int(cc)
                for uu, cc in zip(u.tolist(), c.tolist())}

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=None)
    try:
        for t in range(T):
            ax.clear()
            xyz = xyz_all[t]
            # Re-derive colors/sizes/counts PER FRAME: the occluded cloud is
            # re-augmented (resampled) each step, so point i is a different source
            # at each t -- a single frame-0 mapping would mis-color later frames.
            if has_label:
                lbl = pc[t, :, 3] if pc.shape[2] >= 4 else block_labels
                colors = _colors_for(lbl)
                sizes = _sizes_for(lbl, point_size)
                counts = _counts(lbl)
            else:
                colors = np.tile(matplotlib.colors.to_rgba(_UNKNOWN_COLOR), (pc.shape[1], 1))
                sizes = np.full(pc.shape[1], point_size, dtype=np.float32)
                counts = {"points": pc.shape[1]}
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=sizes,
                       depthshade=False, linewidths=0)
            ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("x (right)"); ax.set_ylabel("depth"); ax.set_zlabel("up")
            legend = "  ".join(f"{n}:{counts.get(n, 0)}" for n in ("robot", "insertive", "receptive")
                               if n in counts) or "  ".join(f"{k}:{v}" for k, v in counts.items())
            ax.set_title(f"{os.path.basename(out_path)}   t={t+1}/{T}   {legend}", fontsize=10)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        writer.close()
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Visualize collected PC episodes as recolored videos.")
    p.add_argument("--dataset", default="demos/contact_seg_100k.hdf5")
    p.add_argument("--num_episodes", type=int, default=5, help="First N demos (if --episodes unset).")
    p.add_argument("--episodes", type=str, default="", help="Explicit indices, e.g. '0,3,7' or '0-4'.")
    p.add_argument("--out_dir", default="/tmp/pc_vids")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-72.0)
    p.add_argument("--point_size", type=float, default=6.0)
    p.add_argument("--class_counts", type=str, default="",
                   help="For xyz-only clouds (no seg channel): color by index blocks "
                        "'robot,insertive,receptive', e.g. '256,128,128' for clean_scenepc.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        demo_keys = sorted((k for k in data.keys() if k.startswith("demo_")),
                           key=lambda s: int(s.split("_")[1]))
        eps = _parse_episodes(args, len(demo_keys))
        pc_point_dim = int(data.attrs.get("pc_point_dim", 0))
        print(f"[viz] {args.dataset}: {len(demo_keys)} demos, pc_point_dim={pc_point_dim}; "
              f"rendering {eps}")
        class_counts = [int(c) for c in args.class_counts.split(",")] if args.class_counts else None
        for i in eps:
            pc = data[f"demo_{i}"]["obs"]["scene_pc"][:]  # (T, P, C)
            # Synthesize per-index origin labels when the cloud is xyz-only (no seg channel).
            block_labels = None
            if pc.shape[2] < 4 and class_counts is not None:
                if sum(class_counts) != pc.shape[1]:
                    raise ValueError(f"--class_counts {class_counts} sum != num_points {pc.shape[1]}")
                block_labels = _block_labels(class_counts, pc.shape[1])
            out_path = os.path.join(args.out_dir, f"demo_{i}.mp4")
            print(f"[viz] demo_{i}: scene_pc {pc.shape} -> {out_path}", flush=True)
            render_episode(pc, out_path, args.fps, args.elev, args.azim, args.point_size, block_labels)
    print(f"[viz] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
