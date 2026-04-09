# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""ZeroG gravity trick configs for OmniReset peg insertion.

Self-contained — no inheritance from rl_state_cfg. No dynamics randomization.
Policy and critic share the same observations.

Ablations:
  A) baseline       — ScenePC obs, 50-50 reset sampling
  B) baseline-state — state obs (pose, no history), 50-50 reset sampling
  C) gps            — ScenePC obs, GPS curriculum
  D) gps-state      — state obs, GPS curriculum
  E) gps-state-nosuccessterm — state obs, GPS, no success termination
  F) gps-state-consecutiveterm — state obs, GPS, consecutive success termination (T=10)
  G) gps-state-highreward — state obs, GPS, success termination + high success reward (10x)
  H) gps-state-truncatesuccess — state obs, GPS, success as truncation (bootstraps V)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85

from ... import mdp as task_mdp


# ===========================================================================
# Scene
# ===========================================================================
@configclass
class ZeroGSceneCfg(InteractiveSceneCfg):

    robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
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


# ===========================================================================
# Observations — two variants: ScenePC and State (pose)
# ===========================================================================
@configclass
class ScenePCObsCfg:
    """512-pt scene PC (robot+insertive+receptive). Same obs for policy and critic."""

    @configclass
    class GroupCfg(ObsGroup):
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

    proprio: GroupCfg = GroupCfg()
    pointcloud: PointcloudCfg = PointcloudCfg()


@configclass
class StateObsCfg:
    """State obs (poses, no history). Same obs for policy and critic."""

    @configclass
    class GroupCfg(ObsGroup):
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
        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )
        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )
        insertive_in_receptive_frame = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: GroupCfg = GroupCfg()
    critic: GroupCfg = GroupCfg()


# ===========================================================================
# Actions
# ===========================================================================
from .actions import Ur5eRobotiq2f85RelativeOSCAction  # noqa: E402


# ===========================================================================
# Commands
# ===========================================================================
@configclass
class CommandsCfg:
    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


# ===========================================================================
# Rewards — sparse success + regularizers
# ===========================================================================
@configclass
class ZeroGRewardsCfg:
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

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)

    fail = RewTerm(
        func=task_mdp.is_terminated_term,
        weight=-1.0,
        params={"term_keys": ["abnormal_robot", "object_out_of_bound"]},
    )


@configclass
class ZeroGConsecutiveRewardsCfg(ZeroGRewardsCfg):
    """Reward only on consecutive success (T=10). No per-frame success reward."""

    success_reward = None
    consecutive_success_reward = RewTerm(
        func=task_mdp.consecutive_success_reward,
        weight=1.0,
        params={"num_consecutive": 10},
    )


@configclass
class ZeroGHighRewardsCfg(ZeroGRewardsCfg):
    """Same as base but success reward weight 50x higher."""

    success_reward = RewTerm(func=task_mdp.success_reward, weight=50.0)


# ===========================================================================
# Terminations
# ===========================================================================
@configclass
class ZeroGTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)
    object_out_of_bound = DoneTerm(
        func=task_mdp.object_out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "in_bound_range": {"x": (-0.5, 1.0), "y": (-0.5, 1.0), "z": (-0.1, 1.0)},
        },
    )
    success = DoneTerm(func=task_mdp.success_termination)


@configclass
class ZeroGNoSuccessTerminationsCfg(ZeroGTerminationsCfg):
    success = None


@configclass
class ZeroGTruncateSuccessTerminationsCfg(ZeroGTerminationsCfg):
    """Success is a truncation (time_out=True) so rsl_rl bootstraps V(s) at success."""

    success = DoneTerm(func=task_mdp.success_termination, time_out=True)


@configclass
class ZeroGConsecutiveSuccessTerminationsCfg(ZeroGTerminationsCfg):
    success = DoneTerm(
        func=task_mdp.consecutive_success_state,
        params={"num_consecutive_successes": 10},
    )


# ===========================================================================
# Curriculum — gravity ramp
# ===========================================================================
@configclass
class ZeroGCurriculumCfg:
    gravity_curriculum = CurrTerm(
        func=task_mdp.GravityCurriculum,
        params={
            "success_str": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "max_difficulty": 10,
            "full_gravity": -9.81,
        },
    )


# ===========================================================================
# Events — two variants: 50-50 uniform and GPS
# ===========================================================================
@configclass
class ZeroGUniformEventCfg:
    """Reset from pre-recorded ZeroG states, 50-50 anywhere/partial-assembly."""

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

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


@configclass
class ZeroGGPSEventCfg:
    """Reset from pre-recorded ZeroG states with GPS curriculum."""

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

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


# ===========================================================================
# Hydra variants (single-task object swaps)
# ===========================================================================
def _make_rigid_obj(usd_path: str, kinematic: bool = False, mass: float = 0.02) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject" if not kinematic else "{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=kinematic,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "peg": _make_rigid_obj(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "fbleg": _make_rigid_obj(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": _make_rigid_obj(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
    },
    "scene.receptive_object": {
        "peghole": _make_rigid_obj(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd", kinematic=True, mass=0.5),
        "fbtabletop": _make_rigid_obj(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd", kinematic=True, mass=0.5
        ),
        "fbdrawerbox": _make_rigid_obj(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd", kinematic=True, mass=0.5
        ),
    },
}


# ===========================================================================
# Base env config
# ===========================================================================
@configclass
class ZeroGBaseCfg(ManagerBasedRLEnvCfg):
    """Base config for all ZeroG ablations. Subclasses set observations and events."""

    scene: ZeroGSceneCfg = ZeroGSceneCfg(num_envs=32, env_spacing=1.5)
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: ZeroGRewardsCfg = ZeroGRewardsCfg()
    terminations: ZeroGTerminationsCfg = ZeroGTerminationsCfg()
    curriculum: ZeroGCurriculumCfg = ZeroGCurriculumCfg()
    commands: CommandsCfg = CommandsCfg()
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
        self.variants = variants


# ===========================================================================
# Ablation A: baseline (ScenePC, 50-50)
# ===========================================================================
@configclass
class ZeroGBaselineTrainCfg(ZeroGBaseCfg):
    observations: ScenePCObsCfg = ScenePCObsCfg()
    events: ZeroGUniformEventCfg = ZeroGUniformEventCfg()


# ===========================================================================
# Ablation B: baseline-state (state obs, 50-50)
# ===========================================================================
@configclass
class ZeroGBaselineStateTrainCfg(ZeroGBaseCfg):
    observations: StateObsCfg = StateObsCfg()
    events: ZeroGUniformEventCfg = ZeroGUniformEventCfg()


# ===========================================================================
# Ablation C: gps (ScenePC, GPS)
# ===========================================================================
@configclass
class ZeroGGPSTrainCfg(ZeroGBaseCfg):
    observations: ScenePCObsCfg = ScenePCObsCfg()
    events: ZeroGGPSEventCfg = ZeroGGPSEventCfg()


# ===========================================================================
# Ablation D: gps-state (state obs, GPS)
# ===========================================================================
@configclass
class ZeroGGPSStateTrainCfg(ZeroGBaseCfg):
    observations: StateObsCfg = StateObsCfg()
    events: ZeroGGPSEventCfg = ZeroGGPSEventCfg()


# ===========================================================================
# Ablation E: gps-state-nosuccessterm (state obs, GPS, no success termination)
# ===========================================================================
@configclass
class ZeroGGPSStateNoTermTrainCfg(ZeroGBaseCfg):
    observations: StateObsCfg = StateObsCfg()
    events: ZeroGGPSEventCfg = ZeroGGPSEventCfg()
    terminations: ZeroGNoSuccessTerminationsCfg = ZeroGNoSuccessTerminationsCfg()


# ===========================================================================
# Ablation F: gps-state-consecutiveterm (state obs, GPS, consecutive success T=10)
# ===========================================================================
@configclass
class ZeroGGPSStateConsecutiveTermTrainCfg(ZeroGBaseCfg):
    observations: StateObsCfg = StateObsCfg()
    events: ZeroGGPSEventCfg = ZeroGGPSEventCfg()
    rewards: ZeroGConsecutiveRewardsCfg = ZeroGConsecutiveRewardsCfg()
    terminations: ZeroGConsecutiveSuccessTerminationsCfg = ZeroGConsecutiveSuccessTerminationsCfg()


# ===========================================================================
# Ablation G: gps-state-highreward (state obs, GPS, success term + 10x reward)
# ===========================================================================
@configclass
class ZeroGGPSStateHighRewardTrainCfg(ZeroGBaseCfg):
    observations: StateObsCfg = StateObsCfg()
    events: ZeroGGPSEventCfg = ZeroGGPSEventCfg()
    rewards: ZeroGHighRewardsCfg = ZeroGHighRewardsCfg()


# ===========================================================================
# Ablation H: gps-state-truncatesuccess (state obs, GPS, success as truncation)
# ===========================================================================
@configclass
class ZeroGGPSStateTruncateSuccessTrainCfg(ZeroGBaseCfg):
    observations: StateObsCfg = StateObsCfg()
    events: ZeroGGPSEventCfg = ZeroGGPSEventCfg()
    terminations: ZeroGTruncateSuccessTerminationsCfg = ZeroGTruncateSuccessTerminationsCfg()
