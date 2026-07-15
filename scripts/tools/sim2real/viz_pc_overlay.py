# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlay the (occluded) scene_pc observation on an RGB render of the physical scene.

Spawns an RGB camera at the SAME calibrated extrinsics the point-cloud obs term uses
(and the same pinhole intrinsics: focal 13.20mm / aperture 20.955mm / 640x480), disables
the term's extrinsics DR, switches the cloud output to the CAMERA frame, and projects the
points onto the image, colored by class (robot / insertive / receptive). One panel per
(reset x env) -- use it to eyeball that the right prims receive points.

Run in the isaac-sim container::

    python scripts/tools/sim2real/viz_pc_overlay.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-OccGrip4Path-v0 \
        --num_envs 2 --num_resets 3 --out /tmp/pc_overlay.png --headless --enable_cameras
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Overlay scene_pc obs on an RGB render.")
parser.add_argument("--task", type=str,
                    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-OccFingers4Path-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--num_resets", type=int, default=3, help="Panels = num_resets x num_envs.")
parser.add_argument("--settle_steps", type=int, default=10, help="Zero-action steps after each reset.")
parser.add_argument("--env_spacing", type=float, default=12.0,
                    help="Override env spacing so neighbouring envs don't leak into the render.")
parser.add_argument("--occluder_splat_radius", type=int, default=None,
                    help="Override the term's occluder splat radius (0 = reproduce the pre-fix leak).")
parser.add_argument("--occluder_points", type=int, default=None,
                    help="Override the term's occluder-only point budget.")
parser.add_argument("--clean", action="store_true",
                    help="Disable the noise stages (edge bleed / surface bias / dropout / fliers) so "
                         "every behind-surface violation is a genuine occlusion leak.")
parser.add_argument("--obs_group", type=str, default="data_collect")
parser.add_argument("--out", type=str, default="/tmp/pc_overlay.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.sim2real_pc_cfg import (  # noqa: E402
    _SEG_LABELS,
    CALIBRATED_CAMERA_OFFSET_POS,
    CALIBRATED_CAMERA_OFFSET_QUAT,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.pc_sim2real import AugParams  # noqa: E402

# Same pinhole model the PC term's frustum/z-buffer uses (AugParams defaults).
FOCAL_MM, APERTURE_MM, W, H = 13.20, 20.955, 640, 480
FX = FOCAL_MM / APERTURE_MM * W  # square pixels -> fy == fx
CX, CY = W / 2.0, H / 2.0

# seg label value -> (name, colour); labels follow _SEG_LABELS (robot=0, ins=-1, rec=+1)
CLASS_SPEC = {0.0: ("robot", "#00e5ff"), -1.0: ("insertive", "#ff3b30"), 1.0: ("receptive", "#7cfc00")}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.env_spacing = args_cli.env_spacing

    # The obs term: camera-frame output (no ref frame), no extrinsics DR, seg labels ON so
    # points can be coloured by class. Geometry is otherwise identical to the training obs.
    pc_params = getattr(env_cfg.observations, args_cli.obs_group).scene_pc.params
    pc_params["ref_cfg"] = None
    pc_params["camera_offset_pos_range"] = (0.0, 0.0, 0.0)  # explicit None breaks torch.tensor
    pc_params["camera_offset_rot_range_deg"] = 0.0
    pc_params["include_segmentation"] = True
    pc_params["segmentation_labels"] = dict(_SEG_LABELS)
    if args_cli.occluder_splat_radius is not None:
        pc_params["occluder_splat_radius"] = args_cli.occluder_splat_radius
    if args_cli.occluder_points is not None:
        pc_params["occluder_points"] = args_cli.occluder_points
    if args_cli.clean:
        pc_params["aug_params"] = AugParams(
            enable_edge_bleed=False, enable_surface_bias=False, enable_dropout=False, enable_flier=False
        )

    # RGB camera at the SAME calibrated extrinsics (offset is relative to the robot root
    # link, opengl convention -- exactly how the term interprets it).
    env_cfg.scene.viz_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/viz_cam",
        offset=CameraCfg.OffsetCfg(
            pos=CALIBRATED_CAMERA_OFFSET_POS, rot=CALIBRATED_CAMERA_OFFSET_QUAT, convention="opengl"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FOCAL_MM, horizontal_aperture=APERTURE_MM, clipping_range=(0.02, 6.0)
        ),
        width=W, height=H, data_types=["rgb", "distance_to_image_plane"],
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    zero_action = torch.zeros((args_cli.num_envs,) + env.unwrapped.single_action_space.shape,
                              device=env.unwrapped.device)

    # The scene_pc term instance: its _camera_pose_w() is THE pose occlusion is computed
    # from -- the render camera is set to exactly that pose (no prim-hierarchy inference).
    om = env.unwrapped.observation_manager
    pc_term = None
    for name, tcfg in zip(om._group_obs_term_names[args_cli.obs_group], om._group_obs_term_cfgs[args_cli.obs_group]):
        if name == "scene_pc":
            pc_term = tcfg.func
    assert pc_term is not None, f"no scene_pc term in group {args_cli.obs_group}"
    cam = env.unwrapped.scene["viz_cam"]

    panels = []  # (rgb HxWx3 uint8, depth HxW, pts (P,3) cam frame, labels (P,))
    for r in range(args_cli.num_resets):
        obs, _ = env.reset()
        for _ in range(args_cli.settle_steps):
            obs, *_ = env.step(zero_action)
        # The cloud is already in the TERM's camera frame; the render camera sits at the
        # same pose (same calibrated offset on the same parent). Alignment is PROVEN per
        # panel by the on-surface fraction printed below (|z - rendered depth| <= 1cm) --
        # a viewpoint error of even ~20mm would collapse it.
        cloud_t = obs[args_cli.obs_group]["scene_pc"].reshape(args_cli.num_envs, -1, 4)
        pad_t = cloud_t[..., :3].abs().sum(-1) == 0.0  # zero-padded fully-occluded slots
        cloud = cloud_t.cpu().numpy()
        pad_np = pad_t.cpu().numpy()
        rgb = cam.data.output["rgb"].cpu().numpy()
        depth = cam.data.output["distance_to_image_plane"].cpu().numpy()
        for e in range(args_cli.num_envs):
            panels.append((rgb[e][..., :3].astype(np.uint8), depth[e].reshape(H, W),
                           cloud[e, :, :3], cloud[e, :, 3], pad_np[e]))

    ncol = args_cli.num_envs
    nrow = args_cli.num_resets
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.8 * nrow), squeeze=False)
    for i, (rgb, depth, pts, lab, pad) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(rgb)
        z = pts[:, 2]
        valid = (~pad) & (z > 1e-4)
        u = FX * pts[:, 0] / np.clip(z, 1e-4, None) + CX
        v = FX * pts[:, 1] / np.clip(z, 1e-4, None) + CY
        # Leak check: a point is a VIOLATION if it sits >3cm BEHIND the rendered surface
        # at its pixel -- i.e. it is visible "through" something. Edge-bleed skirts and
        # fliers legitimately place a few points off-surface; the arm-leak signal is
        # violations clustered on the arm silhouette.
        ui = np.clip(u.astype(int), 0, W - 1)
        vi = np.clip(v.astype(int), 0, H - 1)
        d_at = depth[vi, ui]
        viol = valid & np.isfinite(d_at) & (z > d_at + 0.03)
        on_surf = valid & np.isfinite(d_at) & (np.abs(z - d_at) <= 0.01)
        for code, (name, color) in CLASS_SPEC.items():
            m = valid & ~viol & (lab == code)
            ax.scatter(u[m], v[m], s=1.5, c=color, alpha=0.8, label=f"{name} ({int(m.sum())})")
        if viol.any():
            ax.scatter(u[viol], v[viol], s=2.5, c="#ff00ff", alpha=0.9,
                       label=f"VIOLATION ({int(viol.sum())})")
        n_pad = int(pad.sum())
        title = f"reset {i // ncol}, env {i % ncol}  [{int(viol.sum())} behind-surface pts]"
        if n_pad:
            title += f"  [{n_pad} zero-padded]"
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right", fontsize=8, markerscale=6)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        print(f"[viz] panel reset {i // ncol} env {i % ncol}: {int(viol.sum())} violations "
              f"/ {int(valid.sum())} valid pts; on-surface {int(on_surf.sum())} "
              f"({100.0 * on_surf.sum() / max(int(valid.sum()), 1):.1f}% = alignment proof)")
    fig.tight_layout()
    fig.savefig(args_cli.out, dpi=110)
    print(f"[viz] wrote {len(panels)} panels -> {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
