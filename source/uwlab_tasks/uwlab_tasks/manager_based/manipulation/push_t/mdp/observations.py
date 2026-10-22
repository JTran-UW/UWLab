# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push-T observations and the shared T / goal / EE-tip state helpers."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv

T_CFG = SceneEntityCfg("t_block")
ROBOT_CFG = SceneEntityCfg("robot")


def quat_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw of an (x, y, z, w) quaternion."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def goal_pose(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env goal (pos (N,3) in the env frame, yaw (N,)); lazily seeded from ``env.cfg.goal_pose``.

    Kept as a buffer on the env so a later variant can resample it per episode. z = resting T center height.
    """
    if not hasattr(env, "_push_t_goal_pos"):
        x, y, yaw = env.cfg.goal_pose
        z = env.cfg.table_top_z + 0.5 * env.scene[T_CFG.name].cfg.spawn.height
        env._push_t_goal_pos = torch.tensor([x, y, z], device=env.device).repeat(env.num_envs, 1)
        env._push_t_goal_yaw = torch.full((env.num_envs,), yaw, device=env.device)
    return env._push_t_goal_pos, env._push_t_goal_yaw


def t_pose(env: ManagerBasedEnv, t_cfg: SceneEntityCfg = T_CFG) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """T (pos (N,3) in the env frame, yaw (N,), quat_w (N,4))."""
    t: RigidObject = env.scene[t_cfg.name]
    pos = t.data.root_link_pos_w.torch - env.scene.env_origins
    quat = t.data.root_link_quat_w.torch
    return pos, quat_yaw(quat), quat


def robot_root_pose_w(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = ROBOT_CFG):
    # robot base body pose
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.body_link_pos_w.torch[:, 0], robot.data.body_link_quat_w.torch[:, 0]


def ee_tip_pose_w(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = ROBOT_CFG) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_push_t_ee_body_idx"):
        env._push_t_ee_body_idx = robot.find_bodies(env.cfg.ee_body_name)[0][0]
    idx = env._push_t_ee_body_idx
    pos, quat = robot.data.body_link_pos_w.torch[:, idx], robot.data.body_link_quat_w.torch[:, idx]
    offset = env.cfg.ee_body_offset
    if offset is not None:
        pos = pos + math_utils.quat_apply(quat, torch.tensor(offset, device=pos.device).expand_as(pos))
    return pos, quat


def _to_root(env: ManagerBasedEnv, pos_w: torch.Tensor, quat_w: torch.Tensor | None = None):
    root_pos, root_quat = robot_root_pose_w(env)
    return math_utils.subtract_frame_transforms(root_pos, root_quat, pos_w, quat_w)


def _root_yaw(env: ManagerBasedEnv) -> torch.Tensor:
    return quat_yaw(robot_root_pose_w(env)[1])


def _sincos(a: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sin(a), torch.cos(a)], dim=-1)


def ee_tip_pose_root(env: ManagerBasedEnv) -> torch.Tensor:
    """EE tip position (3) + 6D rotation (first two rotation-matrix columns) in the robot root frame."""
    pos_w, quat_w = ee_tip_pose_w(env)
    pos_b, quat_b = _to_root(env, pos_w, quat_w)
    rot = math_utils.matrix_from_quat(quat_b)
    return torch.cat([pos_b, rot[..., 0], rot[..., 1]], dim=-1)


def t_pose_root(env: ManagerBasedEnv) -> torch.Tensor:
    """T position (3) + sin/cos yaw (2) in the robot root frame."""
    pos, yaw, _ = t_pose(env)
    pos_b, _ = _to_root(env, pos + env.scene.env_origins)
    return torch.cat([pos_b, _sincos(yaw - _root_yaw(env))], dim=-1)


def goal_pose_root(env: ManagerBasedEnv) -> torch.Tensor:
    """Goal position (3) + sin/cos yaw (2) in the robot root frame."""
    pos, yaw = goal_pose(env)
    pos_b, _ = _to_root(env, pos + env.scene.env_origins)
    return torch.cat([pos_b, _sincos(yaw - _root_yaw(env))], dim=-1)


def t_to_goal_root(env: ManagerBasedEnv) -> torch.Tensor:
    """Goal - T planar offset (dx, dy) in the robot root frame + sin/cos of the wrapped yaw error."""
    t_pos, t_yaw, _ = t_pose(env)
    g_pos, g_yaw = goal_pose(env)
    delta_w = torch.zeros_like(t_pos)
    delta_w[:, :2] = (g_pos - t_pos)[:, :2]
    delta_b = math_utils.quat_apply_inverse(robot_root_pose_w(env)[1], delta_w)
    return torch.cat([delta_b[:, :2], _sincos(wrap_angle(g_yaw - t_yaw))], dim=-1)


def ee_to_t_root(env: ManagerBasedEnv) -> torch.Tensor:
    """T position - EE tip position in the robot root frame."""
    t_pos, _, _ = t_pose(env)
    tip_w, _ = ee_tip_pose_w(env)
    delta_w = t_pos + env.scene.env_origins - tip_w
    return math_utils.quat_apply_inverse(robot_root_pose_w(env)[1], delta_w)


def t_velocity_root(env: ManagerBasedEnv, t_cfg: SceneEntityCfg = T_CFG) -> torch.Tensor:
    """T linear + angular velocity in the robot root frame."""
    t: RigidObject = env.scene[t_cfg.name]
    root_quat = robot_root_pose_w(env)[1]
    lin = math_utils.quat_apply_inverse(root_quat, t.data.root_com_lin_vel_w.torch)
    ang = math_utils.quat_apply_inverse(root_quat, t.data.root_com_ang_vel_w.torch)
    return torch.cat([lin, ang], dim=-1)
