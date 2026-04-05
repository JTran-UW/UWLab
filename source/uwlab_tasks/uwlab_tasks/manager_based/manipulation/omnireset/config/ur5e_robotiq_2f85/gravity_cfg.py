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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp

from .rl_state_cfg import (
    BaseEventCfg,
    RewardsCfg,
    RlStateSceneCfg,
    TerminationsCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
)


# ---------------------------------------------------------------------------
# Rewards: sparse success + mechanical work + fail penalty
# ---------------------------------------------------------------------------
@configclass
class GravityTrickRewardsCfg(RewardsCfg):
    """Sparse reward: success +1, fail -1, mechanical work penalty."""

    # -- Disable inherited dense rewards --
    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )
    abnormal_robot = None
    dense_success_reward = None
    ee_asset_distance = None

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-5)
    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-4)

    # Must be non-zero or IsaacLab skips calling it entirely
    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    # Sparse success
    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)

    # Fail penalty: fires on abnormal_robot or object_out_of_bound
    fail = RewTerm(
        func=task_mdp.is_terminated_term,
        weight=-1.0,
        params={"term_keys": ["abnormal_robot", "object_out_of_bound"]},
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
        func=task_mdp.IKCurriculumResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
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
            "max_resample_attempts": 10,
        },
    )

    # Gravity is now controlled by GravityCurriculum (see GravityTrickCurriculumsCfg)


# ---------------------------------------------------------------------------
# Terminations: add out-of-bounds for zero-g object drift
# ---------------------------------------------------------------------------
@configclass
class GravityTrickTerminationsCfg(TerminationsCfg):
    """Standard terminations + object out-of-bounds (for zero-g drift)."""

    object_out_of_bound = DoneTerm(
        func=task_mdp.object_out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "in_bound_range": {"x": (-0.5, 1.0), "y": (-0.5, 1.0), "z": (-0.1, 1.0)},
        },
    )

    success = DoneTerm(func=task_mdp.success_termination)


# ---------------------------------------------------------------------------
# Curriculum: gravity ramp via GravityScheduler
# ---------------------------------------------------------------------------
@configclass
class GravityTrickCurriculumsCfg:
    """Gravity curriculum: ramps from zero-g to full gravity based on success rate."""

    gravity_curriculum = CurrTerm(
        func=task_mdp.GravityCurriculum,
        params={
            "success_str": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "max_difficulty": 10,
            "full_gravity": -9.81,
        },
    )


# ---------------------------------------------------------------------------
# Full env config
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Gravity trick training: procedural 50-50 resets + gravity curriculum. Single-task peg by default."""

    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    rewards: GravityTrickRewardsCfg = GravityTrickRewardsCfg()
    events: GravityTrickEventCfg = GravityTrickEventCfg()
    terminations: GravityTrickTerminationsCfg = GravityTrickTerminationsCfg()
    curriculum: GravityTrickCurriculumsCfg = GravityTrickCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrick10to1TrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """Gravity trick with success=10.0, fail=-1.0 (10:1 ratio)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.success_reward.weight = 10.0


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickGPSTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """Gravity trick with success=1.0, fail=-1.0 + GPS curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_procedural.params["curriculum_target"] = 0.33
        self.events.reset_procedural.params["curriculum_kappa"] = 2.0
        self.events.reset_procedural.params["curriculum_temperature"] = 2.0
