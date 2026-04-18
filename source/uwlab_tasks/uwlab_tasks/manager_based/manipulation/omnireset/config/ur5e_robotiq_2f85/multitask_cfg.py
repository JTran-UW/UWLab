# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-task configuration: peg + leg (or N tasks) in a single env via MultiAssetSpawnerCfg.

Phase 1 (debug): history_length=1, no dynamics randomization, policy = critic obs,
mesh-sampled pointcloud for implicit task conditioning.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions import (
    Ur5eRobotiq2f85RelativeOSCAction,
)

from ... import mdp as task_mdp

# ---------------------------------------------------------------------------
# Shared rigid-body property templates
# ---------------------------------------------------------------------------
_INSERTIVE_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
    disable_gravity=False,
    kinematic_enabled=False,
)

_RECEPTIVE_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
    disable_gravity=False,
    kinematic_enabled=True,
)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class MultiTaskSceneCfg(InteractiveSceneCfg):
    """Scene with MultiAssetSpawnerCfg for per-env object assignment.

    replicate_physics=False is required for heterogeneous meshes.
    """

    replicate_physics = False

    robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UsdFileCfg(
                    usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
                    scale=(1, 1, 1),
                ),
                sim_utils.UsdFileCfg(
                    usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd",
                    scale=(1, 1, 1),
                ),
            ],
            random_choice=False,
            rigid_props=_INSERTIVE_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UsdFileCfg(
                    usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
                    scale=(1, 1, 1),
                ),
                sim_utils.UsdFileCfg(
                    usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd",
                    scale=(1, 1, 1),
                ),
            ],
            random_choice=False,
            rigid_props=_RECEPTIVE_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, -0.013), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# Events (Phase 1: no dynamics randomization)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskEventCfg:
    """Only reset_everything + multi-pair resets. No material/mass/actuator DR."""

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


# ---------------------------------------------------------------------------
# Commands (same as single-task; TaskCommand auto-detects multi-asset)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskCommandsCfg:
    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


# ---------------------------------------------------------------------------
# Observations (Phase 1: history_length=1, policy = critic, pointcloud)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

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

        insertive_pc_in_base = ObsTerm(
            func=task_mdp.MeshPointCloud,
            params={
                "ref_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("insertive_object"),
                "num_points": 64,
            },
        )

        receptive_pc_in_base = ObsTerm(
            func=task_mdp.MeshPointCloud,
            params={
                "ref_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("receptive_object"),
                "num_points": 64,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards (same structure as single-task)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskRewardsCfg:

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-100.0)

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
        },
    )

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.1, params={"std": 1.0})

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)


# ---------------------------------------------------------------------------
# Rewards (quadruped-style: sparse success/fail + dense explore analog)
#
# Quadruped-inspired reward budgets:
#   success: +50 per step × ~10 consecutive steps (dominant signal)
#   explore: +0.3 × ~160 steps × ~0.5 avg ≈ 24
#   fail:    -25 × 1 step
# ---------------------------------------------------------------------------
@configclass
class MultiTaskSimplifiedRewardsCfg:

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    success_reward = RewTerm(func=task_mdp.success_reward, weight=50.0)

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.3, params={"std": 1.0})

    fail = RewTerm(
        func=task_mdp.is_terminated_term,
        params={"term_keys": ["abnormal_robot", "object_off_table"]},
        weight=-25.0,
    )

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )


# ---------------------------------------------------------------------------
# Terminations (task-agnostic)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)


# ---------------------------------------------------------------------------
# Terminations (simplified: + object_off_table + success_terminate)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskSimplifiedTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)
    object_off_table = DoneTerm(
        func=task_mdp.object_off_table,
        params={"asset_cfg": SceneEntityCfg("insertive_object"), "min_height": -0.05},
    )
    success_terminate = DoneTerm(
        func=task_mdp.consecutive_success_state,
        params={"num_consecutive_successes": 10},
    )


# ---------------------------------------------------------------------------
# No curriculum for Phase 1
# ---------------------------------------------------------------------------
@configclass
class MultiTaskNoCurriculumsCfg:
    pass


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------
@configclass
class Ur5eRobotiq2f85RelCartesianOSCMultiTaskTrainCfg(ManagerBasedRLEnvCfg):
    """Multi-task training: peg + leg, shared policy, pointcloud obs, no DR."""

    scene: MultiTaskSceneCfg = MultiTaskSceneCfg(num_envs=32, env_spacing=1.5)
    observations: MultiTaskObservationsCfg = MultiTaskObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: MultiTaskRewardsCfg = MultiTaskRewardsCfg()
    terminations: MultiTaskTerminationsCfg = MultiTaskTerminationsCfg()
    curriculum: MultiTaskNoCurriculumsCfg = MultiTaskNoCurriculumsCfg()
    events: MultiTaskEventCfg = MultiTaskEventCfg()
    commands: MultiTaskCommandsCfg = MultiTaskCommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 16.0
        self.sim.dt = 1 / 120.0

        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


@configclass
class Ur5eRobotiq2f85RelCartesianOSCMultiTaskSimplifiedTrainCfg(Ur5eRobotiq2f85RelCartesianOSCMultiTaskTrainCfg):
    """Multi-task training with quadruped-style rewards: sparse success/fail + dense explore analog."""

    rewards: MultiTaskSimplifiedRewardsCfg = MultiTaskSimplifiedRewardsCfg()
    terminations: MultiTaskSimplifiedTerminationsCfg = MultiTaskSimplifiedTerminationsCfg()


# ---------------------------------------------------------------------------
# Events with GPS curriculum (adaptive reset-type sampling)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskCurriculumEventCfg:
    """Reset events with GPS-style curriculum: Beta-shaped sampling targeting ~33% success per partition."""

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "curriculum_target": 0.5,
            "curriculum_kappa": 2.0,
            "curriculum_temperature": 2.0,
        },
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCMultiTaskCurriculumTrainCfg(Ur5eRobotiq2f85RelCartesianOSCMultiTaskTrainCfg):
    """Multi-task training with original rewards + GPS curriculum over reset types."""

    events: MultiTaskCurriculumEventCfg = MultiTaskCurriculumEventCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCMultiTaskSimplifiedCurriculumTrainCfg(
    Ur5eRobotiq2f85RelCartesianOSCMultiTaskSimplifiedTrainCfg,
):
    """Multi-task training with simplified rewards + GPS curriculum over reset types."""

    events: MultiTaskCurriculumEventCfg = MultiTaskCurriculumEventCfg()


# ---------------------------------------------------------------------------
# Events with per-state GPS curriculum (fine-grained per-reset-state sampling)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskPerStateCurriculumEventCfg:
    """Reset events with per-state GPS curriculum: tracks success per individual reset state (~80K partitions)."""

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "curriculum_target": 0.5,
            "curriculum_kappa": 2.0,
            "curriculum_temperature": 2.0,
        },
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCMultiTaskPerStateCurriculumTrainCfg(
    Ur5eRobotiq2f85RelCartesianOSCMultiTaskTrainCfg,
):
    """Multi-task training with original rewards + per-state GPS curriculum."""

    events: MultiTaskPerStateCurriculumEventCfg = MultiTaskPerStateCurriculumEventCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCMultiTaskSimplifiedPerStateCurriculumTrainCfg(
    Ur5eRobotiq2f85RelCartesianOSCMultiTaskSimplifiedTrainCfg,
):
    """Multi-task training with simplified rewards + per-state GPS curriculum."""

    events: MultiTaskPerStateCurriculumEventCfg = MultiTaskPerStateCurriculumEventCfg()


# ---------------------------------------------------------------------------
# 128-pt wrist-frame PCs with shared MLP encoder (same as single-task ablation)
# ---------------------------------------------------------------------------
@configclass
class MultiTaskSharedEncoder128PCObservationsCfg:
    """128-pt wrist-frame PCs with 2 obs groups: proprio (pass-through) + pointcloud (shared encoder).

    Groups: proprio (~21d), pointcloud (768d = 128×3×2 objects).
    Shared encoder compresses concatenated PCs to 32d; main MLP sees 21+32=53d.
    """

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
        insertive_pc_in_wrist = ObsTerm(
            func=task_mdp.MeshPointCloud,
            params={
                "ref_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "object_cfg": SceneEntityCfg("insertive_object"),
                "num_points": 128,
            },
        )
        receptive_pc_in_wrist = ObsTerm(
            func=task_mdp.MeshPointCloud,
            params={
                "ref_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "object_cfg": SceneEntityCfg("receptive_object"),
                "num_points": 128,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    pointcloud: PointcloudCfg = PointcloudCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCMultiTaskPC128SharedEncTrainCfg(
    Ur5eRobotiq2f85RelCartesianOSCMultiTaskTrainCfg,
):
    """Multi-task training with 128-pt wrist-frame PCs + shared MLP encoder."""

    observations: MultiTaskSharedEncoder128PCObservationsCfg = MultiTaskSharedEncoder128PCObservationsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCMultiTaskPC128SharedEncPerStateCurriculumTrainCfg(
    Ur5eRobotiq2f85RelCartesianOSCMultiTaskPC128SharedEncTrainCfg,
):
    """Multi-task 128-pt wrist-frame PC + shared encoder + per-state GPS curriculum."""

    events: MultiTaskPerStateCurriculumEventCfg = MultiTaskPerStateCurriculumEventCfg()
