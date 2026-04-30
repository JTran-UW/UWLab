# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the reaching task with a hand-coded OSC controller.

The controller does a per-step proportional move toward the target marker pose
expressed in the robot base frame. With the relative-OSC action's scale
(0.02 m, 0.2 rad), an action of 1.0 corresponds to one step of 2 cm / 0.2 rad,
so the policy effectively interpolates linearly toward the target.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Hand-coded OSC reaching controller.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Reaching-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument("--plot_rewards", action="store_true", help="Record and plot per-term rewards for one episode.")
parser.add_argument("--reward_plot_path", type=str, default="reaching_rewards.png")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


# Action scale must match RelCartesianOSCActionCfg.scale_xyz_axisangle in actions.py.
ARM_SCALE = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.2])


def aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (N, 3) to quaternion (N, 4) [w, x, y, z]."""
    angle = torch.norm(aa, dim=-1, keepdim=True)
    safe_angle = angle.clamp(min=1e-6)
    axis = aa / safe_angle
    half = angle / 2.0
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    device = env.unwrapped.device
    arm_scale = ARM_SCALE.to(device)
    action_dim = env.unwrapped.action_manager.total_action_dim

    # Find slice indices for ee and target poses in the critic obs (history_length=1).
    om = env.unwrapped.observation_manager
    offset = 0
    critic_term_start = {}
    for name, dims in zip(om.active_terms["critic"], om.group_obs_term_dim["critic"]):
        critic_term_start[name] = offset
        offset += dims[0]  # dims is a tuple of the term's shape; dims[0] is the flat size
    ee_start = critic_term_start["end_effector_pose"]
    tgt_start = critic_term_start["target_pose"]

    rm = env.unwrapped.reward_manager
    term_names = list(rm.active_terms)
    reward_history = {name: [] for name in term_names}
    ee_pos_history = []
    tgt_pos_history = []

    # One episode = episode_length_s / (decimation * sim.dt) policy steps.
    cfg = env.unwrapped.cfg
    steps_per_episode = int(round(cfg.episode_length_s / (cfg.decimation * cfg.sim.dt)))

    for step in range(args_cli.num_steps):
        # EE and target poses in robot base frame, read from critic obs [pos(3), aa(3)].
        critic_obs = obs["critic"]
        ee_obs = critic_obs[:, ee_start:ee_start + 6]
        tgt_obs = critic_obs[:, tgt_start:tgt_start + 6]

        # end_effector_pose: EE in robot base frame
        ee_pos_b = ee_obs[:, :3]
        ee_quat_b = aa_to_quat(ee_obs[:, 3:6])
        # target_pose: target in EE frame (root_asset_cfg = wrist_3_link)
        tgt_pos_ee = tgt_obs[:, :3]
        tgt_quat_ee = aa_to_quat(tgt_obs[:, 3:6])

        # Position delta: rotate EE-frame offset into base frame
        delta_pos = math_utils.quat_apply(ee_quat_b, tgt_pos_ee)

        # Orientation delta: tgt_quat_ee is right-multiply (EE frame); OSC needs left-multiply (base frame)
        # delta_left = ee_quat_b * tgt_quat_ee * inv(ee_quat_b)
        delta_quat = math_utils.quat_mul(
            math_utils.quat_mul(ee_quat_b, tgt_quat_ee),
            math_utils.quat_inv(ee_quat_b),
        )
        delta_quat = torch.where(delta_quat[:, :1] < 0, -delta_quat, delta_quat)
        delta_aa = math_utils.axis_angle_from_quat(delta_quat)

        # Target in base frame for plotting
        tgt_pos_b = ee_pos_b + delta_pos

        arm_action = torch.cat([delta_pos, delta_aa], dim=-1) / arm_scale
        arm_action = torch.clamp(arm_action, -1.0, 1.0)

        # Gripper: hold open (0)
        gripper_action = torch.zeros((args_cli.num_envs, action_dim - 6), device=device)
        action = torch.cat([arm_action, gripper_action], dim=-1)

        obs, _, _, _, _ = env.step(action)

        if args_cli.plot_rewards:
            ee_pos_history.append(ee_pos_b[0].cpu())
            tgt_pos_history.append(tgt_pos_b[0].cpu())
            for term_idx, name in enumerate(term_names):
                reward_history[name].append(rm._step_reward[0, term_idx].item())

        if step % 50 == 0:
            pos_err = torch.norm(delta_pos, dim=-1).mean().item()
            ang_err = torch.norm(delta_aa, dim=-1).mean().item()
            print(f"[step {step}] mean pos err: {pos_err:.4f} m, mean ang err: {ang_err:.4f} rad")

        if args_cli.plot_rewards and step + 1 >= steps_per_episode:
            break

    if args_cli.plot_rewards:
        import matplotlib.pyplot as plt

        ee_pos = torch.stack(ee_pos_history).numpy()    # (T, 3)
        tgt_pos = torch.stack(tgt_pos_history).numpy()  # (T, 3)
        steps = range(len(ee_pos))
        labels = ["x", "y", "z"]

        n = len(term_names)
        fig, axes = plt.subplots(n + 3, 1, figsize=(8, 2.0 * (n + 3)), sharex=True)

        for i, label in enumerate(labels):
            axes[i].plot(steps, ee_pos[:, i], label=f"ee_{label}")
            axes[i].plot(steps, tgt_pos[:, i], label=f"tgt_{label}", linestyle="--")
            axes[i].set_ylabel(f"pos {label} (m)")
            axes[i].legend(fontsize=7)
            axes[i].grid(True, alpha=0.3)

        for ax, name in zip(axes[3:], term_names):
            ax.plot(reward_history[name])
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("policy step")
        fig.suptitle(f"EE/target pos + rewards (env 0, {len(ee_pos)} steps)")
        fig.tight_layout()
        fig.savefig(args_cli.reward_plot_path, dpi=120)
        print(f"[INFO] Saved reward plot to {args_cli.reward_plot_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
