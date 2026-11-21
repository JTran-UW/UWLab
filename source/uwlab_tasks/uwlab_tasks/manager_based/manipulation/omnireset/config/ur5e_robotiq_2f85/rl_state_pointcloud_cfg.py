# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Point-cloud actor (proprio + object point clouds) with the privileged state critic of ``...-State-v0``."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp

from .rl_state_cfg import POSE_OBS_CLIP, ObservationsCfg, Ur5eRobotiq2f85RelCartesianOSCTrainCfg
from .rl_state_sgs_cfg import Ur5eRobotiq2f85RelCartesianOSCSGSTrainCfg


@configclass
class ProprioObsCfg(ObsGroup):
    prev_actions = ObsTerm(func=task_mdp.last_action)
    joint_pos = ObsTerm(func=task_mdp.joint_pos)
    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "rotation_repr": "axis_angle",
        },
        clip=POSE_OBS_CLIP,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PointCloudObsCfg(ObsGroup):
    objects = ObsTerm(
        func=task_mdp.ObjectsPointCloud,
        params={
            "object_cfgs": [SceneEntityCfg("insertive_object"), SceneEntityCfg("receptive_object")],
            "labels": [-1.0, 1.0],
            "num_points": 64,
            "pool_factor": 8,
            "resample_on_reset": True,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsPointCloudCfg:
    policy: ProprioObsCfg = ProprioObsCfg()
    pointcloud: PointCloudObsCfg = PointCloudObsCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPointCloudTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    observations: ObservationsPointCloudCfg = ObservationsPointCloudCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCPointCloudSGSTrainCfg(Ur5eRobotiq2f85RelCartesianOSCSGSTrainCfg):
    observations: ObservationsPointCloudCfg = ObservationsPointCloudCfg()
