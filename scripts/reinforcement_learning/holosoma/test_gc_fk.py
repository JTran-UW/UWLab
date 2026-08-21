# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the analytical FK used for GC goal EE poses against the simulator.

At every step, computes wrist_3_link's pose from the live joint angles via
``compute_ee_pose_analytical`` (composed with the robot root pose) and compares it to the pose
PhysX reports for the same state. This validates the exact code path used to precompute goal EE
poses in GoalConditionedMultiResetManager, including joint ordering and frame conventions.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify analytical FK EE poses against sim.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=200, help="Number of env steps to check.")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-OffPolicy-v0",
    help="Name of the task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg

import uwlab_tasks  # noqa: F401
from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import ARM_JOINT_NAMES, compute_ee_pose_analytical
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    arm_joint_indices = [robot.joint_names.index(name) for name in ARM_JOINT_NAMES]
    wrist_body_idx = robot.body_names.index("wrist_3_link")

    env.reset()
    actions = torch.zeros(
        (unwrapped.num_envs, unwrapped.action_manager.total_action_dim), device=unwrapped.device
    )

    max_pos_err = 0.0
    max_ang_err = 0.0
    for step in range(args_cli.steps):
        q = robot.data.joint_pos[:, arm_joint_indices]
        ee_pose_b = compute_ee_pose_analytical(q, device=str(unwrapped.device))
        fk_pos_w, fk_quat_w = math_utils.combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, ee_pose_b[:, :3], ee_pose_b[:, 3:7]
        )
        sim_pos_w = robot.data.body_link_pos_w[:, wrist_body_idx]
        sim_quat_w = robot.data.body_link_quat_w[:, wrist_body_idx]

        pos_err = torch.norm(fk_pos_w - sim_pos_w, dim=1)
        ang_err = math_utils.quat_error_magnitude(fk_quat_w, sim_quat_w)
        max_pos_err = max(max_pos_err, pos_err.max().item())
        max_ang_err = max(max_ang_err, ang_err.max().item())
        if step % 50 == 0:
            print(
                f"step {step:4d}: pos err mean {pos_err.mean().item():.2e} max {pos_err.max().item():.2e} m | "
                f"ang err mean {ang_err.mean().item():.2e} max {ang_err.max().item():.2e} rad"
            )
        env.step(actions)

    print(f"\nOverall max position error: {max_pos_err:.2e} m")
    print(f"Overall max orientation error: {max_ang_err:.2e} rad")
    tol_pos, tol_ang = 5e-3, 2e-2
    if max_pos_err < tol_pos and max_ang_err < tol_ang:
        print(f"FK VERIFICATION PASSED (tolerances: {tol_pos} m, {tol_ang} rad)")
    else:
        print(f"FK VERIFICATION FAILED (tolerances: {tol_pos} m, {tol_ang} rad)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
