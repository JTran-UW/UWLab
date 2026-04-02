# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified OmniReset config with gravity trick: 2 reset distributions + gravity curriculum.

Inspired by Octi's approach: ObjectAnywhereEEAnywhere + ObjectPartiallyAssembledEEGrasped,
with gravity ramping from 0 to -9.81 via ADR. No grasp sampling datasets needed for
ObjectAnywhereEEAnywhere.
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
# Events: 2-reset + gravity curriculum
# ---------------------------------------------------------------------------
@configclass
class GravityTrickEventCfg(BaseEventCfg):
    """2 reset distributions + variable gravity.

    Resets:
      - ObjectAnywhereEEAnywhere (50%): object spawned randomly on table, EE random IK pose
      - ObjectPartiallyAssembledEEGrasped (50%): near-goal, ensures non-zero success rate

    Gravity starts at 0 and ramps to -9.81 via GravityScheduler ADR.
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.5, 0.5],
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
    """Gravity trick training: 2 resets + gravity curriculum. Single-task peg by default."""

    events: GravityTrickEventCfg = GravityTrickEventCfg()
    terminations: GravityTrickTerminationsCfg = GravityTrickTerminationsCfg()
    curriculum: GravityTrickCurriculumsCfg = GravityTrickCurriculumsCfg()
