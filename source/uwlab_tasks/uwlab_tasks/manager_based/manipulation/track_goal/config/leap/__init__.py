# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

"""
Leap
"""
gym.register(
    id="UW-TrackGoal-Leap-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_goal_leap:TrackGoalLeapJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="UW-TrackGoal-Leap-IkAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_goal_leap:TrackGoalLeapMcIkAbs",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
        "teleop_cfg_entry_point": "uwlab_assets.robots.leap.teleop:LeapTeleopCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="UW-TrackGoal-Leap-IkDel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_goal_leap:TrackGoalLeapMcIkDel",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)
