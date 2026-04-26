# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Record gripper close trajectory — all unique joint positions from open to closed.

Spawns a minimal env with just the robot, commands the gripper to close via
the BinaryJointPositionAction, records unique gripper joint positions, and
saves as gripper_close_trajectory.pt.

Usage:
    python scripts_v2/tools/record_gripper_close.py \
        --task OmniReset-Ur5eRobotiq2f85-GripperClose-v0 \
        --num_envs 1

Output (gripper_close_trajectory.pt):
    joint_names: list[str]
    joint_positions: Tensor (N, J) — unique positions along open->closed trajectory
"""

from __future__ import annotations

"""Launch Isaac Sim first."""

import argparse
import os

import torch
from typing import cast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record gripper close trajectory.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-GripperClose-v0")
parser.add_argument("--num_steps", type=int, default=500, help="Env steps to run close action.")
parser.add_argument("--similarity_threshold", type=float, default=0.001, help="L2 threshold for uniqueness.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./Datasets/OmniReset/GripperClose/Ur5eRobotiq2f85",
    help="Output directory for gripper_close_trajectory.pt",
)
parser.add_argument(
    "--gripper_joints",
    nargs="+",
    default=[
        "finger_joint",
        "right_outer_knuckle_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_knuckle_joint",
        "right_inner_finger_knuckle_joint",
    ],
)

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after sim launch."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped

    robot = env.scene["robot"]
    device = env.device

    # Find joint indices
    gripper_indices = []
    gripper_names = []
    for name in args_cli.gripper_joints:
        if name in robot.joint_names:
            gripper_indices.append(list(robot.joint_names).index(name))
            gripper_names.append(name)
    print(f"Gripper joints: {gripper_names} (indices: {gripper_indices})")

    # Reset env
    env.reset()

    # Command gripper to close: BinaryJointAction uses negative = close, positive = open
    action = torch.zeros(env.num_envs, env.action_space.shape[0], device=device)
    action[:, -1] = -1.0

    # Record trajectory, deduplicating on the fly
    recorded_poses = None  # (M, J) tensor of unique poses so far
    total_steps = 0
    print(f"Recording {args_cli.num_steps} steps...")

    for step in range(args_cli.num_steps):
        obs, _, _, _, _ = env.step(action)
        pos = robot.data.joint_pos[0, gripper_indices].clone().cpu().unsqueeze(0)  # (1, J)

        # Check uniqueness against all recorded poses
        if recorded_poses is not None:
            dists = torch.cdist(pos, recorded_poses, p=2)  # (1, M)
            min_dist = dists.min().item()
            if min_dist > args_cli.similarity_threshold:
                recorded_poses = torch.cat([recorded_poses, pos], dim=0)
        else:
            recorded_poses = pos

        total_steps += 1
        if step % 50 == 0:
            finger_val = pos[0, 0].item()
            n_unique = recorded_poses.shape[0] if recorded_poses is not None else 0
            print(f"  step {step}: finger_joint={finger_val:.6f}, unique_states={n_unique}")

    print(f"\n{recorded_poses.shape[0]} unique states from {total_steps} steps")
    print(f"finger_joint range: [{recorded_poses[:, 0].min():.6f}, {recorded_poses[:, 0].max():.6f}]")
    print(f"All joint ranges:")
    for i, name in enumerate(gripper_names):
        print(f"  {name}: [{recorded_poses[:, i].min():.6f}, {recorded_poses[:, i].max():.6f}]")

    # Save
    os.makedirs(args_cli.output_dir, exist_ok=True)
    save_path = os.path.join(args_cli.output_dir, "gripper_close_trajectory.pt")

    data = {
        "joint_names": gripper_names,
        "joint_positions": recorded_poses,
    }
    torch.save(data, save_path)
    print(f"\nSaved to: {save_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
