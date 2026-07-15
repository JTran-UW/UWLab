# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step behavioural heuristics for offline policy comparison.

Each function in this module returns ``(num_envs, 1)`` and is wrapped by an
``ObsTerm`` in :class:`HeuristicsCfg`. The values are meant to be aggregated
(typically via mean) over an episode to produce a low-dim "behavioural
signature" of the policy — useful for UMAP'ing many trajectories and seeing
whether different policies actually visit different parts of the state space
in a *human-interpretable* way (unlike raw state-action UMAP, where the PC
dominates).

These heuristics are NEVER fed to a policy at inference; they're purely
diagnostic. Add :class:`HeuristicsCfg` to an env's observations cfg as an
extra group (alongside ``policy``, ``teacher`` etc.) — the obs manager will
compute them each step, the runner will ignore them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ── thresholds (kept here so they're easy to tune in one place) ─────────────
# Gripper-tip-to-peg distance considered "grasping". The Robotiq 2F-85 fingertips
# sit ~15 cm forward of robotiq_base_link, so when the gripper holds the peg
# the peg root is ~0.10–0.15 m from robotiq_base_link. 0.15 captures the grasp
# window without false positives during free-space approach (where peg is
# typically > 0.25 m from the gripper base).
_GRASP_DIST_M       = 0.15
# Peg-to-receptive-object distance for "task essentially solved" — the
# success_termination's threshold is around 1 cm at convergence; 0.05 here
# is a looser "in the immediate vicinity" check that fires for several steps
# leading up to success, giving a non-trivial mean over the trajectory.
_NEAR_RECEPTIVE_M   = 0.05
_GRIPPER_CLOSED_RAD = 0.10    # finger_joint > this rad → gripper closed (Robotiq 2F85)
_FINGER_JOINT_NAME  = "finger_joint"
# wrist_3_link for kinematic-arm features (height, speed, force) — it's the
# kinematic arm endpoint. robotiq_base_link for grasp proximity — it's at the
# gripper mount, so its distance to the held peg is a meaningful "grasping"
# signal (whereas wrist_3 → peg always reads ~0.15 m offset by gripper length).
_ARM_TIP_BODY_NAME    = "wrist_3_link"
_GRIPPER_BASE_BODY_NAME = "robotiq_base_link"


def _arm_tip_idx(robot: Articulation) -> int:
    return robot.body_names.index(_ARM_TIP_BODY_NAME)


def _gripper_base_idx(robot: Articulation) -> int:
    return robot.body_names.index(_GRIPPER_BASE_BODY_NAME)


def _finger_idx(robot: Articulation) -> int:
    return robot.joint_names.index(_FINGER_JOINT_NAME)


def _ee_pos_w(robot: Articulation) -> torch.Tensor:
    """wrist_3_link world position — used for arm-kinematic features."""
    return robot.data.body_link_pos_w[:, _arm_tip_idx(robot), :]


def _gripper_base_pos_w(robot: Articulation) -> torch.Tensor:
    """robotiq_base_link world position — used for grasp-proximity checks."""
    return robot.data.body_link_pos_w[:, _gripper_base_idx(robot), :]


def ee_to_insertive_distance(
    env: "ManagerBasedEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    insertive_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
) -> torch.Tensor:
    """Euclidean distance from EE (wrist_3_link origin) to insertive object root."""
    robot: Articulation = env.scene[robot_cfg.name]
    insertive: RigidObject = env.scene[insertive_cfg.name]
    return torch.norm(_ee_pos_w(robot) - insertive.data.root_link_pos_w, dim=-1, keepdim=True)


def insertive_to_receptive_distance(
    env: "ManagerBasedEnv",
    insertive_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
    receptive_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    """3D Euclidean distance peg→hole — overall task progress signal."""
    insertive: RigidObject = env.scene[insertive_cfg.name]
    receptive: RigidObject = env.scene[receptive_cfg.name]
    return torch.norm(
        insertive.data.root_link_pos_w - receptive.data.root_link_pos_w,
        dim=-1, keepdim=True,
    )


def is_grasping(
    env: "ManagerBasedEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    insertive_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
) -> torch.Tensor:
    """Binary: 1 if peg is within ``_GRASP_DIST_M`` of robotiq_base_link AND gripper is closed.

    Distance is measured to ``robotiq_base_link`` (the gripper mount), not
    ``wrist_3_link``, because the Robotiq 2F-85 fingertips are ~15 cm forward
    of wrist_3 and the held-peg-to-wrist_3 distance never falls below ~0.15 m
    even mid-grasp, which silently zeroed out the previous threshold.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    insertive: RigidObject = env.scene[insertive_cfg.name]
    dist = torch.norm(_gripper_base_pos_w(robot) - insertive.data.root_link_pos_w, dim=-1)
    finger_pos = robot.data.joint_pos[:, _finger_idx(robot)]
    grasping = (dist < _GRASP_DIST_M) & (finger_pos > _GRIPPER_CLOSED_RAD)
    return grasping.float().unsqueeze(-1)


def is_near_receptive(
    env: "ManagerBasedEnv",
    insertive_cfg: SceneEntityCfg = SceneEntityCfg("insertive_object"),
    receptive_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    """Binary: 1 if PEG is within ``_NEAR_RECEPTIVE_M`` of the receptive object.

    Switched from EE-to-receptive to peg-to-receptive: the task is "peg in
    hole", not "wrist in hole". Peg-to-receptive < 5 cm is a clean "almost
    solved" signal that fires for several steps before the success threshold
    (~1 cm) and gives a non-trivial trajectory mean.
    """
    insertive: RigidObject = env.scene[insertive_cfg.name]
    receptive: RigidObject = env.scene[receptive_cfg.name]
    dist = torch.norm(
        insertive.data.root_link_pos_w - receptive.data.root_link_pos_w,
        dim=-1,
    )
    return (dist < _NEAR_RECEPTIVE_M).float().unsqueeze(-1)


def wrist_force_magnitude(
    env: "ManagerBasedEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Magnitude of the joint reaction force at wrist_3_link (proxy for EE contact force)."""
    robot: Articulation = env.scene[robot_cfg.name]
    wrench = robot.data.body_incoming_joint_wrench_b[:, _arm_tip_idx(robot), :3]  # (N, 3)
    return torch.norm(wrench, dim=-1, keepdim=True)


# Filter order for the peg-mounted ContactSensor (see CONTACT_SENSOR_CFG_HELP in
# sim2real_pc_cfg). force_matrix_w columns are [left finger, right finger, hole].
_CONTACT_FILTER_GRIPPER_COLS = (0, 1)
_CONTACT_FILTER_RECEPTIVE_COL = 2


def object_contact_flags(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("peg_contacts"),
    threshold: float = 0.0,
) -> torch.Tensor:
    """Ground-truth (PhysX) binary contact flags, two per step.

    Reads ``force_matrix_w`` (shape ``(N, 1, M, 3)``) from a ContactSensor mounted
    on the insertive object (peg) and filtered, in this column order, to
    ``[left_inner_finger, right_inner_finger, receptive_object]``. Unlike the
    ``wrist_force_magnitude`` reaction-force proxy (noisy, can't separate which
    body is touched), this is the solver's own pairwise contact force, so the two
    flags cleanly segment the trajectory into reach / grasp-transport / insertion.

    Returns ``(N, 2)``: ``[gripper_touching_peg, peg_touching_hole]`` (1.0/0.0),
    where ``gripper`` fires if *either* finger is in contact above ``threshold``.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # (N, 1, M, 3) -> per-filter force magnitude (N, M).
    mag = torch.norm(sensor.data.force_matrix_w, dim=-1).view(env.num_envs, -1)
    gripper = (mag[:, _CONTACT_FILTER_GRIPPER_COLS].amax(dim=1) > threshold).float()
    insertion = (mag[:, _CONTACT_FILTER_RECEPTIVE_COL] > threshold).float()
    return torch.stack([gripper, insertion], dim=-1)


def joint_speed_magnitude(
    env: "ManagerBasedEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"]),
) -> torch.Tensor:
    """L2 norm of arm joint velocities (gripper joints excluded by the joint_names filter)."""
    robot: Articulation = env.scene[robot_cfg.name]
    qdot = robot.data.joint_vel[:, robot_cfg.joint_ids]
    return torch.norm(qdot, dim=-1, keepdim=True)


def action_magnitude(env: "ManagerBasedEnv") -> torch.Tensor:
    """L2 norm of the most recent action — proxy for control aggressiveness."""
    return torch.norm(env.action_manager.action, dim=-1, keepdim=True)


def ee_z_height(
    env: "ManagerBasedEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World Z of the EE — separates "high-carrier" vs "low-pusher" policies."""
    robot: Articulation = env.scene[robot_cfg.name]
    return _ee_pos_w(robot)[:, 2:3]


def ee_world_speed(
    env: "ManagerBasedEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear speed of the EE in world frame (m/s)."""
    robot: Articulation = env.scene[robot_cfg.name]
    v = robot.data.body_link_lin_vel_w[:, _arm_tip_idx(robot), :]
    return torch.norm(v, dim=-1, keepdim=True)


def gripper_closed_indicator(
    env: "ManagerBasedEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary: 1 if finger_joint angle exceeds the closed threshold."""
    robot: Articulation = env.scene[robot_cfg.name]
    finger_pos = robot.data.joint_pos[:, _finger_idx(robot)]
    return (finger_pos > _GRIPPER_CLOSED_RAD).float().unsqueeze(-1)


# Feature ordering matches HeuristicsCfg declaration order. Live OUTSIDE the
# configclass — IsaacLab's ObsManager scans class attrs and rejects anything
# that isn't an ObservationTermCfg, so non-ObsTerm constants must be module-level.
HEURISTIC_FEATURE_NAMES = (
    "ee_to_insertive_dist",
    "insertive_to_receptive_dist",
    "is_grasping",
    "is_near_receptive",
    "wrist_force_magnitude",
    "joint_speed_magnitude",
    "action_magnitude",
    "ee_z_height",
    "ee_world_speed",
    "gripper_closed",
)


@configclass
class HeuristicsCfg(ObsGroup):
    """10-dim per-step behavioural signature; mean over an episode → policy fingerprint.

    Term order is fixed (configclass iteration order matches declaration). The
    canonical ordered name list is :data:`HEURISTIC_FEATURE_NAMES`.
    """

    ee_to_insertive_dist        = ObsTerm(func=ee_to_insertive_distance)
    insertive_to_receptive_dist = ObsTerm(func=insertive_to_receptive_distance)
    is_grasping                 = ObsTerm(func=is_grasping)
    is_near_receptive           = ObsTerm(func=is_near_receptive)
    wrist_force_magnitude       = ObsTerm(func=wrist_force_magnitude)
    joint_speed_magnitude       = ObsTerm(func=joint_speed_magnitude)
    action_magnitude            = ObsTerm(func=action_magnitude)
    ee_z_height                 = ObsTerm(func=ee_z_height)
    ee_world_speed              = ObsTerm(func=ee_world_speed)
    gripper_closed              = ObsTerm(func=gripper_closed_indicator)

    def __post_init__(self):
        # No noise on diagnostic obs; concatenate to a single (N, 10) tensor.
        self.enable_corruption = False
        self.concatenate_terms = True
