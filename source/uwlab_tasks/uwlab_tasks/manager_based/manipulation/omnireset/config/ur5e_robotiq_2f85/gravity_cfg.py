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
# Rewards: sparse success + regularizers + fail penalty
# ---------------------------------------------------------------------------
@configclass
class GravityTrickRewardsCfg(RewardsCfg):
    """Sparse reward with regularizers. Inherits from RewardsCfg, disables dense rewards."""

    # -- Disable inherited dense rewards --
    abnormal_robot = None
    dense_success_reward = None
    ee_asset_distance = None

    # Regularizers (10x smaller than baseline)
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-5)
    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-4)
    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
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

    # Sparse success
    success_reward = RewTerm(func=task_mdp.success_reward, weight=10.0)

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
    """Procedural 50-50 resets + gravity trick."""

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
# Scene pointcloud observations: single combined PC from robot+insertive+receptive
# ---------------------------------------------------------------------------
@configclass
class ScenePCObservationsCfg:
    """512-pt scene pointcloud (robot+insertive+receptive) in robot base frame."""

    @configclass
    class ProprioCfg(ObsGroup):
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
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    pointcloud: PointcloudCfg = PointcloudCfg()


# ---------------------------------------------------------------------------
# Base env config
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """Gravity trick baseline: sparse reward + gravity curriculum + ScenePC."""

    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ScenePCObservationsCfg = ScenePCObservationsCfg()
    rewards: GravityTrickRewardsCfg = GravityTrickRewardsCfg()
    events: GravityTrickEventCfg = GravityTrickEventCfg()
    terminations: GravityTrickTerminationsCfg = GravityTrickTerminationsCfg()
    curriculum: GravityTrickCurriculumsCfg = GravityTrickCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()


# ---------------------------------------------------------------------------
# Ablation: PA EE lerp range (0.9, 1.0) — EE spawns very close to object for partial assemblies
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickPALerpTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """Gravity trick + partial assembly EE lerp (0.9, 1.0)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_procedural.params["pa_ee_lerp_range"] = (0.9, 1.0)


# ---------------------------------------------------------------------------
# Ablation: success_reward weight = 50
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickSuccess50TrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """Gravity trick + success_reward weight = 50."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.success_reward.weight = 50.0


# ---------------------------------------------------------------------------
# Ablation: uniform random gravity (no curriculum)
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickRandomGravTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """Gravity trick + uniform random gravity [0, -9.81] each reset (no curriculum)."""

    def __post_init__(self):
        super().__post_init__()
        # Disable gravity curriculum
        self.curriculum = None
        # Randomize gravity every 16s (interval mode = global, not per-env)
        self.events.variable_gravity = EventTerm(
            func=task_mdp.randomize_physics_scene_gravity,
            mode="interval",
            interval_range_s=(16.0, 16.0),
            params={
                "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.0, 0.0, 0.0]),
                "operation": "abs",
            },
        )


# ---------------------------------------------------------------------------
# Ablation: contact rewards (sanity check — reach + contact + contact-gated dense success)
# ---------------------------------------------------------------------------
@configclass
class GravityTrickContactSceneCfg(RlStateSceneCfg):
    """Scene with contact sensors on gripper fingers."""

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


@configclass
class GravityTrickContactRewardsCfg(GravityTrickRewardsCfg):
    """Sparse rewards + reach + contact + contact-gated dense success."""

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

    gripper_contact = RewTerm(
        func=task_mdp.gripper_contact,
        weight=0.5,
        params={"threshold": 1.0},
    )

    contact_gated_dense_success = RewTerm(
        func=task_mdp.contact_gated_dense_success,
        weight=2.0,
        params={"std": 0.2, "threshold": 1.0},
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGravityTrickContactTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """Gravity trick + contact rewards (sanity check that env works)."""

    scene: GravityTrickContactSceneCfg = GravityTrickContactSceneCfg(num_envs=32, env_spacing=1.5)
    rewards: GravityTrickContactRewardsCfg = GravityTrickContactRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success = None
        self.scene.robot.spawn.activate_contact_sensors = True


# ===========================================================================
# Zero-G State-Based Ablations
# ===========================================================================
# All three load pre-recorded ZeroG states via MultiResetManager + gravity curriculum.


# ---------------------------------------------------------------------------
# Events: load ZeroG states 50-50 (no GPS)
# ---------------------------------------------------------------------------
@configclass
class ZeroGStatesEventCfg(BaseEventCfg):
    """Load pre-recorded ZeroG states with 50-50 anywhere/partial-assembly mix."""

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

    reset_from_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ZeroGAnywhere", "ZeroGPartialAssembly"],
            "probs": [0.5, 0.5],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


# ---------------------------------------------------------------------------
# Events: load ZeroG states with GPS curriculum over all 20K states
# ---------------------------------------------------------------------------
@configclass
class ZeroGStatesGPSEventCfg(BaseEventCfg):
    """Load pre-recorded ZeroG states with GPS curriculum."""

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

    reset_from_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ZeroGAnywhere", "ZeroGPartialAssembly"],
            "probs": [0.5, 0.5],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "curriculum_target": 0.33,
            "curriculum_kappa": 2.0,
            "curriculum_temperature": 2.0,
        },
    )


# ---------------------------------------------------------------------------
# Ablation A: ZeroG states baseline (50-50 sampling, sparse rewards)
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCZeroGBaselineTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """ZeroG states + 50-50 sampling + sparse rewards + gravity curriculum."""

    events: ZeroGStatesEventCfg = ZeroGStatesEventCfg()

    def __post_init__(self):
        super().__post_init__()


# ---------------------------------------------------------------------------
# Ablation B: ZeroG states + GPS over all 20K states
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCZeroGGPSTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """ZeroG states + GPS curriculum over all states + sparse rewards + gravity curriculum."""

    events: ZeroGStatesGPSEventCfg = ZeroGStatesGPSEventCfg()

    def __post_init__(self):
        super().__post_init__()


# ---------------------------------------------------------------------------
# Ablation C: ZeroG states + 50-50 + dense rewards (curricuumed out)
# ---------------------------------------------------------------------------
@configclass
class ZeroGDenseRewardsCfg(GravityTrickRewardsCfg):
    """Sparse + dense rewards (contact + dense success) that get curricuumed out."""

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

    gripper_contact = RewTerm(
        func=task_mdp.gripper_contact,
        weight=0.5,
        params={"threshold": 1.0},
    )

    contact_gated_dense_success = RewTerm(
        func=task_mdp.contact_gated_dense_success,
        weight=2.0,
        params={"std": 0.2, "threshold": 1.0},
    )


@configclass
class ZeroGDenseCurriculumsCfg(GravityTrickCurriculumsCfg):
    """Gravity curriculum + reward weight decay for dense rewards."""

    reward_weight_decay = CurrTerm(
        func=task_mdp.RewardWeightCurriculum,
        params={
            "reward_term_names": ["ee_asset_distance", "gripper_contact", "contact_gated_dense_success"],
            "gravity_curriculum_name": "gravity_curriculum",
        },
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCZeroGDenseTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg):
    """ZeroG states + 50-50 + dense rewards curricuumed out as gravity increases."""

    scene: GravityTrickContactSceneCfg = GravityTrickContactSceneCfg(num_envs=32, env_spacing=1.5)
    events: ZeroGStatesEventCfg = ZeroGStatesEventCfg()
    rewards: ZeroGDenseRewardsCfg = ZeroGDenseRewardsCfg()
    curriculum: ZeroGDenseCurriculumsCfg = ZeroGDenseCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.activate_contact_sensors = True
