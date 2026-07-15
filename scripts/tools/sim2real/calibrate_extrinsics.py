# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the sim front-camera extrinsics to a real capture via ICP.

We reset the sim arm to the recorded joint config (so sim and real robot are at
the SAME pose), then align the **real** point cloud onto the **sim** robot cloud
with point-to-point ICP. The recovered rigid correction is folded into the sim
camera world pose and reported back as ``camera_offset_pos`` / ``camera_offset_quat``
(the opengl-convention values the env cfg expects).

Why this works: with identical robot pose, any residual offset between the real
cloud (placed via the *current* sim camera pose) and the sim robot surface is the
error in that camera pose. ICP recovers it.

Usage::

    python scripts/tools/sim2real/calibrate_extrinsics.py \\
        --init_joint_pos_npz .../trajectory.npz \\
        --real_pc .../pointclouds/arm/cloud_arm_00000.ply \\
        --out_json exported/sim2real_pc_sanity/calibrated_extrinsics.json
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ICP calibration of the sim front-camera extrinsics to a real cloud.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-Sim2RealPC-v0")
parser.add_argument("--init_joint_pos_npz", type=str, required=True, help="trajectory.npz; arm reset to arm_joint_pos[0].")
parser.add_argument("--real_pc", type=str, required=True, help="Real cloud .ply in the front-camera optical frame.")
parser.add_argument("--out_json", type=str, default="exported/sim2real_pc_sanity/calibrated_extrinsics.json")
parser.add_argument("--icp_iters", type=int, default=80)
parser.add_argument("--icp_trim", type=float, default=0.7, help="Keep this fraction of nearest correspondences each iter.")
parser.add_argument("--icp_max_corr", type=float, default=0.15, help="Reject correspondences farther than this (m).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def icp(src: np.ndarray, tgt: np.ndarray, iters: int, trim: float, max_corr: float):
    """Point-to-point ICP. Returns (T 4x4 world correction, rms_before, rms_after, n_inliers)."""
    tree = cKDTree(tgt)
    T = np.eye(4)
    s = src.copy()
    d0, _ = tree.query(s)
    rms_before = float(np.sqrt((d0**2).mean()))
    n_in = len(s)
    for _ in range(iters):
        d, idx = tree.query(s)
        # trim: keep the closest `trim` fraction AND cap at max_corr (kills fliers)
        keep_thr = np.quantile(d, trim)
        m = (d <= keep_thr) & (d <= max_corr)
        if m.sum() < 10:
            m = d <= np.quantile(d, 0.9)
        n_in = int(m.sum())
        A, B = s[m], tgt[idx[m]]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cb - R @ ca
        s = (R @ s.T).T + t
        dT = np.eye(4)
        dT[:3, :3], dT[:3, 3] = R, t
        T = dT @ T
        if np.linalg.norm(t) < 1e-6 and np.allclose(R, np.eye(3), atol=1e-7):
            break
    d1, _ = tree.query(s)
    d1 = d1[d1 <= max_corr]
    rms_after = float(np.sqrt((d1**2).mean()))
    return T, rms_before, rms_after, n_in


def main():
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # robot-only target: drop the peg from scene + PC so we align against the arm.
    env_cfg.scene.insertive_object = None
    env_cfg.observations.policy.scene_pc.params.pop("insertive_cfg", None)
    env_cfg.observations.policy.scene_pc.params["visualize"] = False

    traj = np.load(args_cli.init_joint_pos_npz, allow_pickle=True)
    q0 = np.asarray(traj["arm_joint_pos"][0], dtype=float)
    for name, val in zip(ARM_JOINTS, q0):
        env_cfg.scene.robot.init_state.joint_pos[name] = float(val)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.reset()
    device = env.unwrapped.device

    # grab the obs-term instance (class-based term: term_cfg.func IS the instance)
    om = env.unwrapped.observation_manager
    term = None
    for tc in om._group_obs_class_term_cfgs["policy"]:
        if tc.func.__class__.__name__ == "OccludedScenePointCloud":
            term = tc.func
            break
    assert term is not None, "OccludedScenePointCloud term not found"

    # -- sim robot dense cloud in WORLD (full surface = good ICP target) --
    body_pos_w = term.robot.data.body_pos_w[0]   # (B, 3)
    body_quat_w = term.robot.data.body_quat_w[0]  # (B, 4)
    bidx = term.dense_body_idx
    local = term.dense_local
    sim_world = math_utils.quat_apply(body_quat_w[bidx], local) + body_pos_w[bidx]  # (M, 3)

    # -- real cloud placed in WORLD via the CURRENT sim camera pose --
    cam_pos_w, cam_quat_w = term._camera_pose_w()  # (1,3), (1,4) -- optical convention
    cam_pos_w, cam_quat_w = cam_pos_w[0], cam_quat_w[0]
    real_cam = torch.from_numpy(
        np.asarray(trimesh.load(args_cli.real_pc, process=False).vertices, dtype=np.float32)
    ).to(device)
    real_world0 = math_utils.quat_apply(cam_quat_w.unsqueeze(0).expand(real_cam.shape[0], -1), real_cam) + cam_pos_w

    src = real_world0.cpu().numpy().astype(np.float64)
    tgt = sim_world.cpu().numpy().astype(np.float64)

    T, rms_b, rms_a, n_in = icp(src, tgt, args_cli.icp_iters, args_cli.icp_trim, args_cli.icp_max_corr)
    R_d = torch.tensor(T[:3, :3], dtype=torch.float32, device=device)
    t_d = torch.tensor(T[:3, 3], dtype=torch.float32, device=device)

    # -- corrected camera world pose: apply ΔT after the original camera transform --
    cam_R = math_utils.matrix_from_quat(cam_quat_w.unsqueeze(0))[0]  # (3,3)
    cam_R2 = R_d @ cam_R
    cam_pos_w2 = R_d @ cam_pos_w + t_d
    cam_quat_w2 = math_utils.quat_from_matrix(cam_R2.unsqueeze(0))[0]

    # -- convert world pose -> offset relative to robot root (cfg convention) --
    root_pos = term.robot.data.root_pos_w[0]
    root_quat = term.robot.data.root_quat_w[0]
    root_quat_inv = math_utils.quat_inv(root_quat.unsqueeze(0))[0]
    offset_pos = math_utils.quat_apply(root_quat_inv.unsqueeze(0), (cam_pos_w2 - root_pos).unsqueeze(0))[0]
    offset_quat_optical = math_utils.quat_mul(root_quat_inv.unsqueeze(0), cam_quat_w2.unsqueeze(0))[0]
    # optical -> opengl: cfg builds optical = quat_mul(q_opengl, q_x180), q_x180=(0,1,0,0) (self-inverse).
    q_x180 = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    offset_quat_opengl = math_utils.quat_mul(offset_quat_optical.unsqueeze(0), q_x180.unsqueeze(0))[0]

    # correction magnitude (for choosing a DR range)
    corr_trans = float(torch.linalg.norm(t_d).item())
    corr_ang = float(2.0 * torch.acos(torch.clamp(math_utils.quat_from_matrix(R_d.unsqueeze(0))[0][0].abs(), max=1.0)).item())
    corr_ang_deg = corr_ang * 180.0 / np.pi

    cur_pos = term.cam_offset_pos.cpu().numpy().tolist()

    result = {
        "calibrated_camera_offset_pos": [round(float(v), 7) for v in offset_pos.cpu().numpy()],
        "calibrated_camera_offset_quat_opengl": [round(float(v), 7) for v in offset_quat_opengl.cpu().numpy()],
        "previous_camera_offset_pos": [round(float(v), 7) for v in cur_pos],
        "icp_rms_before_m": round(rms_b, 5),
        "icp_rms_after_m": round(rms_a, 5),
        "icp_inliers": n_in,
        "correction_translation_m": round(corr_trans, 5),
        "correction_rotation_deg": round(corr_ang_deg, 4),
        "real_pc": args_cli.real_pc,
        "init_joint_pos_npz": args_cli.init_joint_pos_npz,
    }
    os.makedirs(os.path.dirname(args_cli.out_json), exist_ok=True)
    with open(args_cli.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("[CALIB] " + json.dumps(result, indent=2), flush=True)
    print(f"[CALIB] wrote {args_cli.out_json}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
