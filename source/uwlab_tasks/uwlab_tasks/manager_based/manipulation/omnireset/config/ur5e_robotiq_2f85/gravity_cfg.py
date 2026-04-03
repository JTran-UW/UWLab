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
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
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
# Scene: add gripper contact sensors
# ---------------------------------------------------------------------------
@configclass
class GravityTrickSceneCfg(RlStateSceneCfg):
    """Scene with contact sensors on gripper fingers for contact-gated rewards."""

    right_inner_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_inner_finger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_friction_forces=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"],
    )

    left_inner_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_inner_finger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        track_friction_forces=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"],
    )


# ---------------------------------------------------------------------------
# Rewards: pat/hand-style weights with contact-gated rewards
# ---------------------------------------------------------------------------
@configclass
class GravityTrickRewardsCfg(RewardsCfg):
    """Rewards mirroring pat/hand structure: reach → contact → contact-gated progress → success."""

    # -- Override weights to match pat/hand --
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-0.005)
    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-0.005)
    joint_vel = None

    # Reaching: 1.0
    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=1.0,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
        },
    )

    # Gripper contact: 0.5 (analogous to good_finger_contact)
    gripper_contact = RewTerm(
        func=task_mdp.gripper_contact,
        weight=0.5,
        params={"threshold": 1.0},
    )

    # Contact-gated dense success: 2.0 (analogous to position_command_error_tanh)
    contact_gated_dense_success = RewTerm(
        func=task_mdp.contact_gated_dense_success,
        weight=2.0,
        params={"std": 0.2, "threshold": 1.0},
    )

    # Termination penalty: -1.0 (replaces abnormal_robot -100)
    abnormal_robot = None

    early_termination = RewTerm(
        func=task_mdp.is_terminated_term,
        weight=-1.0,
        params={"term_keys": ["abnormal_robot", "object_out_of_bound"]},
    )

    # Must be non-zero or IsaacLab skips calling it entirely
    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    # Dense success: drop (replaced by contact_gated_dense_success)
    dense_success_reward = None

    # Success: 10.0 (was 1.0)
    success_reward = RewTerm(func=task_mdp.success_reward, weight=10.0)


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
            "ee_above_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.10, 0.20),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 2 - np.pi / 12, np.pi / 2 + np.pi / 12),
                "yaw": (-np.pi, np.pi),
            },
            "collision_analyzer_cfg": task_mdp.CollisionAnalyzerCfg(
                num_points=512,
                max_dist=0.5,
                min_dist=0.005,
                asset_cfg=SceneEntityCfg("insertive_object"),
                obstacle_cfgs=[SceneEntityCfg("robot")],
            ),
            "max_resample_attempts": 10,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

    # Uniform gravity randomization: sample z-gravity from [0, -9.81] each reset
    variable_gravity = EventTerm(
        func=task_mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )


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


# ---------------------------------------------------------------------------
# Curriculum: gravity ramp via GravityScheduler
# ---------------------------------------------------------------------------
@configclass
class GravityTrickCurriculumsCfg:
    """No curriculum — gravity is uniformly randomized each reset."""

    pass


# ---------------------------------------------------------------------------
# Full env config
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Gravity trick training: procedural 50-50 resets + gravity curriculum. Single-task peg by default."""

    scene: GravityTrickSceneCfg = GravityTrickSceneCfg(num_envs=32, env_spacing=1.5)
    rewards: GravityTrickRewardsCfg = GravityTrickRewardsCfg()
    events: GravityTrickEventCfg = GravityTrickEventCfg()
    terminations: GravityTrickTerminationsCfg = GravityTrickTerminationsCfg()
    curriculum: GravityTrickCurriculumsCfg = GravityTrickCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.insertive_object.spawn.activate_contact_sensors = True


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickSuccessTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """Gravity trick with 50% of partial assemblies sampled near-success."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_procedural.params["partial_assembly_success_fraction"] = 0.5
