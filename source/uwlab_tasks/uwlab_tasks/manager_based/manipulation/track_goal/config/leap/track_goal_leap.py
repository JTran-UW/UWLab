# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import uwlab_assets.robots.leap as leap

from isaaclab.utils import configclass

from ... import track_goal_env

episode_length = 50.0


@configclass
class TrackGoalLeap(track_goal_env.TrackGoalEnv):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = leap.IMPLICIT_LEAP6D.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.ee_pose.body_name = "palm_lower"
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = "palm_lower"
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = "palm_lower"
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = "palm_lower"
        self.rewards.end_effector_orientation_tracking_fine_grained.params["asset_cfg"].body_names = "palm_lower"


@configclass
class TrackGoalLeapJointPosition(TrackGoalLeap):
    actions = leap.LeapJointPositionAction()  # type: ignore


@configclass
class TrackGoalLeapMcIkAbs(TrackGoalLeap):
    actions = leap.LeapMcIkAbsoluteAction()  # type: ignore


@configclass
class TrackGoalLeapMcIkDel(TrackGoalLeap):
    actions = leap.LeapMcIkDeltaAction()  # type: ignore
