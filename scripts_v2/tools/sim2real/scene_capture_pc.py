# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlay a real point cloud (.ply) on a sim reset scene, rendered from a chosen camera.

A point-cloud variant of scene_capture.py: it resets the sim scene to a specific
OmniReset reset state (by index), loads a real point cloud (.npz with a 'points'
key, or .ply), places it in sim world space, drops the points into the scene as
marker spheres, adds some lights, renders the camera, and saves the picture.  By
default it then holds the sim open so you can interact with the scene (use --no-hold
to exit immediately).

Use it to eyeball how a real capture lines up with the sim scene for a given reset.

The cloud is assumed to be in the *robot-base* frame (the default), and is placed
via the current robot root pose (this matches the cloud scene_capture.py saves).
Pass --pc_frame 'ros'/'opengl' to instead treat the cloud as a camera optical /
opengl frame cloud (placed via the sim camera pose).  Colors in the .npz/.ply are
ignored for now.

Usage:
    python scripts_v2/tools/sim2real/scene_capture_pc.py \
        --enable_cameras \
        --camera orbbec \
        --idx 0 \
        --pc /home/yandabao/diffusion_policy/pc_debug_orbbec/perception_test.npz
"""

import argparse

from isaaclab.app import AppLauncher

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

parser = argparse.ArgumentParser(description="Overlay a real point cloud on a sim reset scene.")
parser.add_argument(
    "--camera",
    type=str,
    default="orbbec",
    choices=["front_camera", "side_camera", "wrist_camera", "orbbec"],
    help="Which camera the .ply was captured from / to render ('orbbec' -> front_orbbec scene camera)",
)
parser.add_argument(
    "--pc",
    type=str,
    default="/home/yandabao/diffusion_policy/pc_debug_orbbec/perception_test.npz",
    help="Path to the real point cloud (.npz with a 'points' key, or .ply), in robot-base frame by default",
)
parser.add_argument(
    "--pc_frame",
    type=str,
    default="base",
    choices=["base", "ros", "opengl"],
    help="Frame the .pc points live in: 'base'=robot-root frame (default, placed via the robot root pose), "
    "'ros'=camera optical (z fwd) / 'opengl'=camera frame (both placed via the sim camera pose)",
)
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEAnywhere", help="Reset dataset name")
parser.add_argument("--idx", type=int, default=0, help="Index of the reset state to load from the dataset")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
    help="Root directory of the OmniReset dataset",
)
parser.add_argument("--marker_radius", type=float, default=0.002, help="Radius of each point marker sphere (m)")
parser.add_argument(
    "--color", type=float, nargs=3, default=[1.0, 0.55, 0.0], help="RGB color (0-1) for the point markers"
)
parser.add_argument("--stride", type=int, default=1, help="Subsample the cloud by this stride")
parser.add_argument("--max_points", type=int, default=80000, help="Cap on rendered points (uniform subsample)")
parser.add_argument("--output", type=str, default=None, help="Output image path (default: scene_capture_pc_<cam>_idx<idx>.png)")
parser.add_argument(
    "--hold",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="After capturing, keep the sim open (render-only, no physics) so you can interact with the scene. "
    "Use --no-hold to exit immediately.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.camera_align_cfg import (  # noqa: E402
    CameraAlignEnvCfg,
)

# ---- CLI alias → scene sensor name ----
CAMERA_ALIASES = {"orbbec": "front_orbbec"}

# Arm joints whose angles map to the real UR5e (first 6 joints).
ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def load_reset_state(env, dataset_dir, reset_type, idx):
    """Load a single reset state (batched to 1 env) from the OmniReset dataset."""
    # Derive the object-pair directory from the scene's objects (same as MultiResetManager).
    pair = task_mdp.compute_pair_dir(
        env.scene["insertive_object"].cfg.spawn.usd_path,
        env.scene["receptive_object"].cfg.spawn.usd_path,
    )
    dataset_file = f"{dataset_dir}/Resets/{pair}/resets_{reset_type}.pt"
    local_path = task_mdp.safe_retrieve_file_path(dataset_file)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Dataset file {dataset_file} could not be accessed or downloaded.")

    dataset = torch.load(local_path, map_location="cpu")
    num_states = len(dataset["initial_state"]["articulation"]["robot"]["joint_position"])
    if not -num_states <= idx < num_states:
        raise IndexError(f"idx {idx} out of range for {reset_type} (has {num_states} states)")

    # sample_state_data_set stacks per-index list entries into a (1, ...) batched tensor dict.
    idx_t = torch.tensor([idx], device=env.device)
    return task_mdp.sample_state_data_set(dataset["initial_state"], idx_t, env.device)


def read_points_file(path):
    """Read an (N, 3) float32 array of points from a .npz ('points' key) or .ply file.

    Colors (npz 'colors' key / ply vertex colors) are intentionally ignored for now.
    """
    if os.path.splitext(path)[1].lower() == ".npz":
        return np.asarray(np.load(path, allow_pickle=True)["points"], dtype=np.float32)
    return np.asarray(trimesh.load(path, process=False).vertices, dtype=np.float32)


def load_points(path, stride, max_points, device):
    """Load points from a .npz/.ply as an (N, 3) tensor, subsampled to <= max_points."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    pts = torch.from_numpy(read_points_file(path)).to(device)
    if stride > 1:
        pts = pts[::stride]
    if pts.shape[0] > max_points:
        sel = torch.linspace(0, pts.shape[0] - 1, max_points, device=device).long()
        pts = pts[sel]
    return pts


def points_to_world(points, frame, camera, env):
    """Transform the loaded points into sim world space.

    'base':  robot-root (base) frame -> placed via the current robot root pose.
    'ros'/'opengl': camera optical / opengl frame -> placed via the sim camera pose.
    """
    n = points.shape[0]
    if frame == "base":
        robot = env.scene["robot"]
        root_pos = robot.data.root_pos_w[0]
        root_quat = robot.data.root_quat_w[0]
        return math_utils.quat_apply(root_quat.unsqueeze(0).expand(n, -1), points) + root_pos
    cam_pos = camera.data.pos_w[0]
    cam_quat = camera.data.quat_w_ros[0] if frame == "ros" else camera.data.quat_w_opengl[0]
    return math_utils.quat_apply(cam_quat.unsqueeze(0).expand(n, -1), points) + cam_pos


def add_lights():
    """Add a key + fill distant light so the scene reads better than the dome alone."""
    key = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.98, 0.95))
    key.func("/World/CaptureKeyLight", key)  # identity orientation -> shines straight down (-Z)
    fill = sim_utils.DistantLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0))
    # rotate ~45 deg about world Y so the fill comes in from the front and down
    fill.func("/World/CaptureFillLight", fill, orientation=(0.92388, 0.0, 0.38268, 0.0))


def apply_reset(env, state):
    """Re-apply the reset state and re-render cameras (no physics stepping)."""
    return env.reset_to(state, env_ids=torch.tensor([0], device=env.device), is_relative=True)


def main():
    camera_key = CAMERA_ALIASES.get(args_cli.camera, args_cli.camera)

    env = gym.make("OmniReset-Ur5eRobotiq2f85-CameraAlign-v0", cfg=CameraAlignEnvCfg()).unwrapped
    env.reset()

    # Build a full scene state, then overlay the dataset assets (robot + objects).
    # env.reset_to() requires every scene asset present, but the dataset only holds
    # the robot and the two objects -- the rest keep their current (default) state.
    state = env.scene.get_state(is_relative=True)
    reset_state = load_reset_state(env, args_cli.dataset_dir, args_cli.reset_type, args_cli.idx)
    for category, assets in reset_state.items():
        for asset_name, asset_state in assets.items():
            state[category][asset_name] = asset_state

    # First apply: place the scene + populate the camera pose used for the transform.
    apply_reset(env, state)
    camera = env.scene.sensors[camera_key]

    # Real cloud -> sim world, then drop in as marker spheres.
    points_raw = load_points(args_cli.pc, args_cli.stride, args_cli.max_points, env.device)
    points_world = points_to_world(points_raw, args_cli.pc_frame, camera, env)
    print(f"Loaded {points_world.shape[0]} points from {args_cli.pc} (frame={args_cli.pc_frame})")

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/scene_pc",
        markers={
            "pt": sim_utils.SphereCfg(
                radius=args_cli.marker_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(args_cli.color)),
            )
        },
    )
    markers = VisualizationMarkers(marker_cfg)
    markers.visualize(translations=points_world)

    add_lights()

    # Second apply: re-render the camera now that markers + lights are in the scene.
    apply_reset(env, state)
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()  # (H, W, 3) uint8
    out_path = args_cli.output or f"scene_capture_pc_{camera_key}_idx{args_cli.idx}.png"
    plt.imsave(out_path, rgb)
    print(f"Saved point-cloud overlay to {out_path}")

    # ---- print the first 6 robot joint angles (handy for driving the real robot) ----
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos[0]
    joint_names = list(robot.data.joint_names)
    degs = [torch.rad2deg(joint_pos[joint_names.index(n)]).item() for n in ARM_JOINT_NAMES]
    print(f"Reset: {args_cli.reset_type}  idx={args_cli.idx}  camera={args_cli.camera}")
    print("joint_angles (deg): " + " ".join(f"{d:.4f}" for d in degs))

    # Keep the sim open for interaction. Render-only (no env.step) so the floating
    # reset objects stay put instead of falling under gravity.
    if args_cli.hold:
        print("Holding sim open for interaction -- close the window to exit.")
        while simulation_app.is_running():
            env.sim.render()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
