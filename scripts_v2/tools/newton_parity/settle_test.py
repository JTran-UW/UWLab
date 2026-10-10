# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settle test: do dataset reset states stay put on this backend under a hold action?

Every env is reset to a fixed (reset type, dataset index) entry. The arm action is zero (hold pose) and the gripper
closes on grasped states (as ``visualize_reset_states.py`` does). Terminations are disabled; ``abnormal_robot`` is
evaluated at each env-step boundary instead. Per env, the peg's drift in the world and in the gripper frame, arm
drift and drops are recorded.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/settle_test.py --visualizer none --backend newton --out newton_settle.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Settle test of dataset reset states.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--states_per_type", type=int, default=256)
parser.add_argument("--env_steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.backend == "newton" and not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import events as omnireset_events
from uwlab_tasks.utils.hydra import hydra_task_compose

_FORCED: dict = {}


def _forced_call(self, env, env_ids, dataset_dir, reset_types, probs, success=None):
    if env_ids is None:
        env_ids = torch.arange(self.num_envs, device=self._env.device)
    task_ids = _FORCED["task_ids"].to(self.device)[env_ids]
    state_ids = _FORCED["state_ids"].to(self.device)[env_ids]
    self.task_id[env_ids] = task_ids
    for dataset_idx in range(self.num_tasks):
        mask = task_ids == dataset_idx
        if mask.any():
            idx = state_ids[mask] % self.num_states[dataset_idx]
            states = omnireset_events.sample_from_nested_dict(self.datasets[dataset_idx], idx)
            self._reset_to(states["initial_state"], env_ids=env_ids[mask], is_relative=True)
    robot = self._env.scene["robot"]
    robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)


def _angle(qa, qb):
    """Rotation angle of qa^-1 qb (xyzw) [deg]."""
    rel = quat_mul(quat_inv(qa), qb)
    return torch.rad2deg(2.0 * torch.atan2(rel[:, :3].norm(dim=-1), rel[:, 3].abs()))


def _snapshot(env, hand):
    robot, peg = env.scene["robot"], env.scene["insertive_object"]
    h = robot.data.body_link_pose_w.torch[:, hand]
    p = peg.data.root_link_pose_w.torch
    return {
        "joint_pos": robot.data.joint_pos.torch.clone(),
        "peg_pos": (p[:, :3] - env.scene.env_origins).clone(),
        "peg_quat": p[:, 3:7].clone(),
        "peg_in_hand_pos": quat_apply_inverse(h[:, 3:7], p[:, :3] - h[:, :3]).clone(),
        "peg_in_hand_quat": quat_mul(quat_inv(h[:, 3:7]), p[:, 3:7]).clone(),
    }


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    reset_types = env_cfg.events.reset_from_reset_states.params["reset_types"]
    num_envs = args_cli.states_per_type * len(reset_types)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.terminations.abnormal_robot = None
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.enable_corruption = False
    gen = torch.Generator().manual_seed(args_cli.seed)
    _FORCED["task_ids"] = torch.arange(num_envs) % len(reset_types)
    _FORCED["state_ids"] = torch.randint(0, 1 << 30, (num_envs,), generator=gen)
    omnireset_events.MultiResetManager.__call__ = _forced_call

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    hand = robot.body_names.index("robotiq_base_link")
    finger = robot.joint_names.index("finger_joint")
    torch.manual_seed(args_cli.seed)
    env.reset(seed=args_cli.seed)
    start = _snapshot(env, hand)
    close_q = env_cfg.actions.gripper.close_command_expr["finger_joint"]
    grasped = (start["joint_pos"][:, finger].abs() / close_q) > 0.1
    action = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)
    gripper_slice = sum(env.action_manager.action_term_dim[: env.action_manager.active_terms.index("gripper")])
    action[:, gripper_slice:] = torch.where(grasped, -1.0, 1.0)[:, None]
    abnormal = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    worst_ratio = torch.zeros(num_envs, device=env.device)
    worst_joint = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    for _ in range(args_cli.env_steps):
        env.step(action)
        ratio = robot.data.joint_vel.torch.abs() / (2.0 * robot.data.joint_vel_limits.torch)
        value, joint = ratio.max(dim=1)
        update = value > worst_ratio
        worst_ratio = torch.where(update, value, worst_ratio)
        worst_joint = torch.where(update, joint, worst_joint)
        abnormal |= value > 1.0
    end = _snapshot(env, hand)
    out = {
        "meta": {
            "task": args_cli.task,
            "backend": args_cli.backend,
            "physics_cfg": repr(env_cfg.sim.physics),
            "env_steps": args_cli.env_steps,
            "step_dt": env.step_dt,
            "reset_types": reset_types,
            "joint_names": list(robot.joint_names),
        },
        "task_ids": _FORCED["task_ids"],
        "state_ids": _FORCED["state_ids"],
        "grasped": grasped.cpu(),
        "peg_drift_mm": ((end["peg_pos"] - start["peg_pos"]).norm(dim=-1) * 1000).cpu(),
        "peg_dz_mm": ((end["peg_pos"][:, 2] - start["peg_pos"][:, 2]) * 1000).cpu(),
        "peg_rot_deg": _angle(start["peg_quat"], end["peg_quat"]).cpu(),
        "peg_in_hand_slip_mm": ((end["peg_in_hand_pos"] - start["peg_in_hand_pos"]).norm(dim=-1) * 1000).cpu(),
        "peg_in_hand_rot_deg": _angle(start["peg_in_hand_quat"], end["peg_in_hand_quat"]).cpu(),
        "arm_drift_deg": torch.rad2deg(
            (end["joint_pos"][:, :6] - start["joint_pos"][:, :6]).abs().max(dim=1).values
        ).cpu(),
        "finger_start": start["joint_pos"][:, finger].cpu(),
        "finger_end": end["joint_pos"][:, finger].cpu(),
        "abnormal": abnormal.cpu(),
        "worst_vel_ratio": worst_ratio.cpu(),
        "worst_vel_joint": worst_joint.cpu(),
    }
    torch.save(out, args_cli.out)
    print(
        f"[settle_test] wrote {args_cli.out}: {num_envs} envs, {int(grasped.sum())} grasped,"
        f" {int(abnormal.sum())} abnormal"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
