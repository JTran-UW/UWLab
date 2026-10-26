# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State PPO with symmetry-invariant peg/hole observations (the fast-off-policy ``ObservationsSymmetryCfg``).

Same task as ``...-State-v0``; only the observations change. The critic observes exactly what the policy does.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp

from .physics_presets import joint_pos_obs
from .rl_state_cfg import (
    POSE_OBS_CLIP,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
)

# Rectangular peg: 4 identical yaws about its long axis, and the same end over end. The hole is 4-fold, no flip.
PEG_SYMMETRY = {"axis": [0.0, 0.0, 1.0], "order": 4, "flip": True}
HOLE_SYMMETRY = {"axis": [0.0, 0.0, 1.0], "order": 4, "flip": False}


@configclass
class SymmetryObsCfg(ObsGroup):
    """67-D per frame x 5 history = 335-D."""

    prev_actions = ObsTerm(func=task_mdp.last_action)
    joint_pos = joint_pos_obs()
    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "rotation_repr": "axis_angle",
        },
        clip=POSE_OBS_CLIP,
    )
    insertive_asset_pose = ObsTerm(
        func=task_mdp.SymmetricPoseObs,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "symmetry": PEG_SYMMETRY,
        },
        clip=POSE_OBS_CLIP,
    )
    receptive_asset_pose = ObsTerm(
        func=task_mdp.SymmetricPoseObs,
        params={
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "symmetry": HOLE_SYMMETRY,
        },
        clip=POSE_OBS_CLIP,
    )
    insertive_minus_receptive_position = ObsTerm(
        func=task_mdp.position_difference_in_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "origin_asset_cfg": SceneEntityCfg("receptive_object"),
            "frame_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
        },
        clip=POSE_OBS_CLIP,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 5


@configclass
class ObservationsSymmetryCfg:
    policy: SymmetryObsCfg = SymmetryObsCfg()
    critic: SymmetryObsCfg = SymmetryObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCSymmetryTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """``...-State-v0`` (uniform over the 4 reset types) with symmetry-invariant observations and success."""

    observations: ObservationsSymmetryCfg = ObservationsSymmetryCfg()

    def __post_init__(self):
        super().__post_init__()
        set_symmetric_success(self)


def set_symmetric_success(cfg) -> None:
    """Success at any of the 8 identical-looking peg poses, matching the flip-invariant observations."""
    cfg.commands.task_command.symmetric_success = True
    cfg.commands.task_command.insertive_symmetry = PEG_SYMMETRY
    cfg.commands.task_command.receptive_symmetry = HOLE_SYMMETRY


@configclass
class Ur5eRobotiq2f85RelCartesianOSCSymmetryFinetuneCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg):
    """``...-State-Finetune-v0`` (Stage 2: explicit actuator, ADR sysid + OSC gains, action-scale curriculum) with
    symmetry-invariant observations and success."""

    observations: ObservationsSymmetryCfg = ObservationsSymmetryCfg()

    def __post_init__(self):
        super().__post_init__()
        set_symmetric_success(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCSymmetryFinetuneEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """``...-State-Finetune-Play-v0`` (fixed sysid + OSC gains at full range, eval action term) with symmetry obs."""

    observations: ObservationsSymmetryCfg = ObservationsSymmetryCfg()

    def __post_init__(self):
        super().__post_init__()
        set_symmetric_success(self)
