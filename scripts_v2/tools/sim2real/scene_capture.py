# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture a sim scene from an OmniReset reset state, to align the real scene to it.

The "reverse" of align_cameras.py: instead of moving the sim camera to match a
real photo, this resets the sim scene to a specific reset state (object + EE
positions) from the OmniReset dataset, renders the chosen camera, and prints the
first 6 robot joint angles.  Drive the real robot to those joints and move the
real objects until the real camera view matches the saved sim image.

Mirrors the sim2real tooling pattern:
  scripts_v2/tools/sim2real/align_cameras.py    →  align sim camera to real photo
  scripts_v2/tools/sim2real/scene_capture.py    →  align real scene to sim photo

It also dumps a dense robot point cloud (.npy) sampled with the same area-weighted
mesh sampler the Sim2RealPC env uses.  Overlay this fixed sim cloud on a real
camera capture and ICP/eyeball-fit the camera extrinsics to it (the saved cloud is
the alignment target; cf. overlay_orbbec_pc.py).  Defaults: robot-only, robot-base
frame.  Pass --no-save_pc to skip, --pc_frame world / --pc_include_objects to vary.

Usage:
    python scripts_v2/tools/sim2real/scene_capture.py \
        --enable_cameras \
        --camera front_camera \
        --idx 0
"""

import argparse

from isaaclab.app import AppLauncher

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

parser = argparse.ArgumentParser(description="Capture a sim scene from an OmniReset reset state.")
parser.add_argument(
    "--camera",
    type=str,
    default="front_camera",
    choices=["front_camera", "side_camera", "wrist_camera", "orbbec"],
    help="Which camera to render ('orbbec' maps to the front_orbbec scene camera)",
)
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEAnywhere", help="Reset dataset name")
parser.add_argument("--idx", type=int, default=0, help="Index of the reset state to load from the dataset")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
    help="Root directory of the OmniReset dataset",
)
parser.add_argument("--output", type=str, default=None, help="Output image path (default: scene_capture_<cam>_idx<idx>.png)")
# ---- robot point-cloud dump (for sim2real extrinsics matching) ----
parser.add_argument(
    "--save_pc",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Also sample a dense robot point cloud and save it as .npy (use --no-save_pc to skip).",
)
parser.add_argument(
    "--pc_output", type=str, default=None,
    help="Output point-cloud path (default: scene_capture_<cam>_idx<idx>_pc_<frame>.npy)",
)
parser.add_argument(
    "--pc_frame", type=str, default="base", choices=["base", "world"],
    help="Frame to express the saved cloud in: 'base'=robot root frame (default, natural for "
    "extrinsics matching), 'world'=sim world frame (matches overlay_orbbec_pc.py synthetic_world.npy).",
)
parser.add_argument(
    "--pc_points", type=int, default=4096,
    help="Per-class point budget for the dense sampler; the saved cloud has ~pc_points*pc_oversample points.",
)
parser.add_argument("--pc_oversample", type=int, default=3, help="Dense-cloud oversample factor.")
parser.add_argument(
    "--pc_include_objects", action="store_true",
    help="Also sample the insertive/receptive objects into the cloud (default: robot-only).",
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

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg  # noqa: E402

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.camera_align_cfg import (  # noqa: E402
    CameraAlignEnvCfg,
)

# ---- RGB key lookup (matches CameraAlignObservationsCfg) ----
CAMERA_TO_RGB = {
    "front_camera": "front_rgb",
    "front_orbbec": "front_orbbec_rgb",
    "side_camera": "side_rgb",
    "wrist_camera": "wrist_rgb",
}

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


def build_robot_pc_term(env, num_points, oversample, include_objects):
    """Instantiate an OccludedScenePointCloud term just to sample the dense robot cloud.

    We reuse the exact area-weighted mesh sampler the Sim2RealPC env uses (so the
    saved cloud matches the synthetic cloud overlay_orbbec_pc.py aligns against),
    but we only consume its ``dense_local`` / ``dense_body_idx`` -- the occlusion /
    augmentation pipeline is never run.
    """
    params = {
        "robot_cfg": SceneEntityCfg("robot"),
        "num_points": num_points,
        "oversample": oversample,
        "visualize": False,
    }
    if include_objects:
        params["insertive_cfg"] = SceneEntityCfg("insertive_object")
        params["receptive_cfg"] = SceneEntityCfg("receptive_object")
    term_cfg = ObservationTermCfg(func=task_mdp.OccludedScenePointCloud, params=params)
    return task_mdp.OccludedScenePointCloud(term_cfg, env)


def dense_cloud_to_world(term):
    """Transform the term's dense local cloud (env 0) into world coordinates.

    Robot points ride their body pose; object points (when sampled) ride the
    object's root pose -- mirrors OccludedScenePointCloud._transform_to_world_env.
    """
    robot = term.robot
    bp = robot.data.body_pos_w[0]  # (B, 3)
    bq = robot.data.body_quat_w[0]  # (B, 4)
    local = term.dense_local  # (M, 3)
    obj_idx = term.dense_obj_idx  # (M,)  -1 for robot points
    world = torch.empty_like(local)

    robot_mask = obj_idx < 0
    bidx = term.dense_body_idx[robot_mask]
    world[robot_mask] = math_utils.quat_apply(bq[bidx], local[robot_mask]) + bp[bidx]

    for oi, asset in enumerate(term.object_assets):
        m = obj_idx == oi
        if m.any():
            n = int(m.sum())
            pos = asset.data.root_pos_w[0]
            quat = asset.data.root_quat_w[0]
            world[m] = math_utils.quat_apply(quat.unsqueeze(0).expand(n, -1), local[m]) + pos
    return world


def world_to_base(points_world, robot):
    """Express world-frame points in the robot root (base) frame of env 0."""
    root_pos = robot.data.root_pos_w[0]
    root_quat = robot.data.root_quat_w[0]
    inv = math_utils.quat_inv(root_quat.unsqueeze(0)).expand(points_world.shape[0], -1)
    return math_utils.quat_apply(inv, points_world - root_pos)


def save_robot_pc(env):
    """Sample the dense robot cloud at the current pose and save it as .npy."""
    term = build_robot_pc_term(env, args_cli.pc_points, args_cli.pc_oversample, args_cli.pc_include_objects)
    points_world = dense_cloud_to_world(term)
    robot = env.scene["robot"]
    if args_cli.pc_frame == "base":
        points = world_to_base(points_world, robot)
    else:
        points = points_world
    pts_np = points.cpu().numpy().astype(np.float32)

    camera_key = CAMERA_ALIASES.get(args_cli.camera, args_cli.camera)
    out_path = args_cli.pc_output or f"scene_capture_{camera_key}_idx{args_cli.idx}_pc_{args_cli.pc_frame}.npy"
    np.save(out_path, pts_np)

    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    print(
        f"Saved robot point cloud ({pts_np.shape[0]} pts, frame={args_cli.pc_frame}, "
        f"objects={'on' if args_cli.pc_include_objects else 'off'}) to {out_path}"
    )
    print(f"  robot root (world): pos={np.round(root_pos, 4).tolist()}  quat(wxyz)={np.round(root_quat, 4).tolist()}")

    # ---- interactive HTML view next to the .npy ----
    try:
        from visualize_pc import save_pointcloud_html

        html_path = os.path.splitext(out_path)[0] + ".html"
        save_pointcloud_html(pts_np, html_path, title=os.path.basename(html_path))
        print(f"Saved interactive point-cloud view to {html_path}")
    except Exception as e:  # don't lose the .npy if plotly is missing / errors
        print(f"[warn] skipped interactive HTML ({type(e).__name__}: {e})")


def main():
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

    # Apply the state and re-render the cameras (no physics stepping, so floating
    # objects stay put instead of falling under gravity).
    obs, _ = env.reset_to(state, env_ids=torch.tensor([0], device=env.device), is_relative=True)

    # ---- save the rendered camera image ----
    camera_key = CAMERA_ALIASES.get(args_cli.camera, args_cli.camera)
    rgb_key = CAMERA_TO_RGB[camera_key]
    img = obs["policy"][rgb_key][0]
    if img.shape[0] in (3, 4):
        img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    if img.max() > 1.5:
        img = (img / 255.0).clip(0, 1)
    out_path = args_cli.output or f"scene_capture_{camera_key}_idx{args_cli.idx}.png"
    plt.imsave(out_path, img[..., :3])
    print(f"Saved sim view to {out_path}")

    # ---- save a dense robot point cloud (for sim2real extrinsics matching) ----
    if args_cli.save_pc:
        save_robot_pc(env)

    # ---- print the first 6 robot joint angles ----
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos[0]
    joint_names = list(robot.data.joint_names)
    print("\n" + "=" * 60)
    print(f"Reset: {args_cli.reset_type}  idx={args_cli.idx}  camera={args_cli.camera}")
    print(f"{'joint':<20}{'rad':>12}{'deg':>12}")
    for name in ARM_JOINT_NAMES:
        rad = joint_pos[joint_names.index(name)].item()
        print(f"{name:<20}{rad:>12.6f}{torch.rad2deg(torch.tensor(rad)).item():>12.4f}")
    degs = [torch.rad2deg(joint_pos[joint_names.index(n)]).item() for n in ARM_JOINT_NAMES]
    print("\njoint_angles (deg): " + " ".join(f"{d:.4f}" for d in degs))
    print("=" * 60 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
