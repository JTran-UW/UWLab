# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
MDP terminations.
"""


def invalid_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return true if the RigidBody position reads nan"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.isnan(asset.data.body_pos_w.torch).any(dim=-1).any(dim=-1)


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.torch.abs() > (robot.data.joint_vel_limits.torch * 2)).any(dim=1)


def abnormal_robot_state_or_blowup(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_joint_pos: float = 100.0
) -> torch.Tensor:
    """:func:`abnormal_robot_state`, or a non-finite joint state, or a joint position beyond ``max_joint_pos``.

    The extra checks catch solver blow-ups that never trip the velocity check (Newton never exceeds 2x the limit, and
    a NaN velocity compares False).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_vel = robot.data.joint_pos.torch, robot.data.joint_vel.torch
    non_finite = ~(torch.isfinite(joint_pos).all(dim=1) & torch.isfinite(joint_vel).all(dim=1))
    blowup = non_finite | (joint_pos.abs() > max_joint_pos).any(dim=1)
    return abnormal_robot_state(env, asset_cfg) | blowup
