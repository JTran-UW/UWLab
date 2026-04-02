# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OmniReset config with gravity trick: procedural resets + gravity curriculum.

50-50 procedural reset distribution:
  - ObjectAnywhereEEAnywhere: object + gripper randomly spawned (fully procedural)
  - ObjectPartiallyAssembledEENear: object from partial assembly dataset, gripper
    spawned near object via IK (no grasp dataset — avoids collision issues)

Gravity ramps from 0 to -9.81 via ADR. In zero-g, the floating object lets the robot
learn to approach and grasp without the object falling.
"""

from __future__ import annotations

import numpy as np

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp
from ...mdp.gravity_curriculum import GravityScheduler, gravity_interpolate_fn

from .rl_state_cfg import (
    BaseEventCfg,
    TerminationsCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
)


# ---------------------------------------------------------------------------
# Events: procedural anywhere + partial assembly near + gravity curriculum
# ---------------------------------------------------------------------------
@configclass
class GravityTrickEventCfg(BaseEventCfg):
    """Procedural 50-50 resets + gravity trick.

    50% ObjectAnywhereEEAnywhere: object random position/orientation, EE random IK pose.
        In zero-g objects float — robot learns approach and manipulation freely.
    50% ObjectPartiallyAssembledEENear: object from partial assembly dataset (near goal),
        EE spawned near object via IK (NOT from grasp dataset — avoids collision issues).
        Near-goal states ensure non-zero success rate (Octi's condition #1).

    Gravity ramps 0 → -9.81 via GravityScheduler ADR.
    """

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.3),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 12, np.pi / 12),
            },
            "velocity_range": {},
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )

    reset_procedural = EventTerm(
        func=task_mdp.GravityTrickResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "probs": [0.5, 0.5],
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            "obj_anywhere_range": {
                "x": (0.3, 0.55),
                "y": (-0.1, 0.5),
                "z": (0.0, 0.3),
                "roll": (-np.pi, np.pi),
                "pitch": (-np.pi, np.pi),
                "yaw": (-np.pi, np.pi),
            },
            "ee_anywhere_range": {
                "x": (0.3, 0.7),
                "y": (-0.4, 0.4),
                "z": (0.0, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            "ee_near_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.05),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 4, 3 * np.pi / 4),
                "yaw": (np.pi / 2, 3 * np.pi / 2),
            },
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

    # Gravity trick: start at zero-g, ramp to full gravity via curriculum
    variable_gravity = EventTerm(
        func=task_mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )


# ---------------------------------------------------------------------------
# Terminations: add out-of-bounds for zero-g object drift
# ---------------------------------------------------------------------------
@configclass
class GravityTrickTerminationsCfg(TerminationsCfg):
    """Standard terminations + object out-of-bounds (for zero-g drift)."""

    object_off_table = DoneTerm(
        func=task_mdp.object_off_table,
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "min_height": -0.1,
        },
    )


# ---------------------------------------------------------------------------
# Curriculum: gravity ramp via GravityScheduler
# ---------------------------------------------------------------------------
@configclass
class GravityTrickCurriculumsCfg:
    """Gravity curriculum: ramps from 0 to -9.81 based on success rate."""

    gravity_scheduler = CurrTerm(
        func=GravityScheduler,
        params={
            "success_str": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "max_difficulty": 10,
        },
    )

    gravity_ramp = CurrTerm(
        func=task_mdp.modify_term_cfg,
        params={
            "address": "events.variable_gravity.params.gravity_distribution_params",
            "modify_fn": gravity_interpolate_fn,
            "modify_params": {
                "initial_gravity": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "final_gravity": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                "scheduler_term_str": "gravity_scheduler",
            },
        },
    )


# ---------------------------------------------------------------------------
# Full env config
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Gravity trick training: procedural 50-50 resets + gravity curriculum. Single-task peg by default."""

    events: GravityTrickEventCfg = GravityTrickEventCfg()
    terminations: GravityTrickTerminationsCfg = GravityTrickTerminationsCfg()
    curriculum: GravityTrickCurriculumsCfg = GravityTrickCurriculumsCfg()
