# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared point-cloud observation configs.

Lifted out of ``gravity_cfg.py`` so both the ZeroG gravity-curriculum envs and the
normal-gravity ``rl_state_cfg.py`` pre-training envs can use the *same* PC obs
parameterization (true encoder/policy parity). ``gravity_cfg`` already imports from
``rl_state_cfg``, so this neutral module avoids a circular import.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp


# ===========================================================================
# Observations
# ===========================================================================
@configclass
class ScenePCObsCfg:
    """512-pt scene PC (robot+insertive+receptive) + proprio. Shared encoder compresses PC to 32d."""

    @configclass
    class GroupCfg(ObsGroup):
        prev_actions = ObsTerm(func=task_mdp.last_action)
        joint_pos = ObsTerm(func=task_mdp.joint_pos)
        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PointcloudCfg(ObsGroup):
        scene_pc = ObsTerm(
            func=task_mdp.ScenePointCloud,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "insertive_cfg": SceneEntityCfg("insertive_object"),
                "receptive_cfg": SceneEntityCfg("receptive_object"),
                "visualize": False,
                "num_points": 512,
                # Re-sample mesh points for ins/rec subsets at episode reset.
                # Kills the "fixed-init point arrangement → absolute yaw"
                # memorization channel. Hydra-overridable.
                "resample_on_reset": False,
                # Same idea but for robot points: per-env random subset of each
                # body's oversampled mesh pool, re-rolled each reset. Forces the
                # encoder to be agnostic to specific robot point selections so
                # the trained teacher generalizes across env init RNG state at
                # deploy time. Hydra-overridable.
                "resample_on_reset_robot": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TimeLeftCfg(ObsGroup):
        time_left = ObsTerm(func=task_mdp.time_left)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SuccessClassifierCfg(ObsGroup):
        """Obs for the auxiliary V_success head: kinematic state + time_left.

        ee_pose is intentionally omitted — no offline FK utility exists, and
        joint_pos is a bijection to ee_pose, so V_success can learn it.
        """

        prev_actions = ObsTerm(func=task_mdp.last_action)
        joint_pos = ObsTerm(func=task_mdp.joint_pos)
        insertive_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "quat",
            },
        )
        receptive_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "quat",
            },
        )
        time_left = ObsTerm(func=task_mdp.time_left)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: GroupCfg = GroupCfg()
    pointcloud: PointcloudCfg = PointcloudCfg()
    time_left: TimeLeftCfg = TimeLeftCfg()
    success_classifier: SuccessClassifierCfg = SuccessClassifierCfg()


@configclass
class ScenePCEECentricObsCfg(ScenePCObsCfg):
    """``ScenePCObsCfg`` with the scene pointcloud expressed in the end-effector frame.

    Identical to ``ScenePCObsCfg`` except the 512-pt scene PC is output in the
    ``wrist_3_link`` (end-effector) frame rather than the robot base frame. The task
    is wrist-centric, so EE-frame points are nearly invariant to arm pose and tend to
    learn more effectively. Obs dimensionality is unchanged (512x3), so the shared
    encoder / runner config is identical. ``ScenePCObsCfg`` defines no ``__post_init__``
    of its own, so there is nothing to chain to super.
    """

    def __post_init__(self):
        self.pointcloud.scene_pc.params["ref_cfg"] = SceneEntityCfg("robot", body_names="wrist_3_link")
