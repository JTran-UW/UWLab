# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""ZeroG gravity trick configs for OmniReset.

Self-contained — no inheritance from rl_state_cfg. No dynamics randomization.
Policy and critic share the same observations. ScenePC 512pt obs only.

GPS resets + terminate on success + gravity curriculum.
Hydra variants swap insertive/receptive objects for peg, leg, drawer.

Configs:
  ZeroGGPSScenePCTrainCfg           — single-task ScenePC 512pt + GPS
  ZeroGMultiTaskGPSScenePCTrainCfg  — multi-task peg+leg ScenePC 512pt + GPS
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
from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85, IMPLICIT_UR5E_ROBOTIQ_2F85

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg, Ur5eRobotiq2f85RelativeOSCEvalAction

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
# Observations
# ===========================================================================
@configclass
class ScenePCObsCfg:
    """512-pt scene PC (robot+insertive+receptive) + proprio. Shared encoder compresses PC to 32d."""

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
                # Re-sample mesh points for ins/rec subsets at episode reset.
                # Kills the "fixed-init point arrangement → absolute yaw"
                # memorization channel. Hydra-overridable.
                "resample_on_reset": False,
                # Same idea but for robot points: per-env random subset of each
                # body's oversampled mesh pool, re-rolled each reset. Forces the
                # encoder to be agnostic to specific robot point selections so
                # the trained teacher generalizes across env init RNG state at
                # deploy time. Hydra-overridable.
                "resample_on_reset_robot": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TimeLeftCfg(ObsGroup):
        time_left = ObsTerm(func=task_mdp.time_left)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SuccessClassifierCfg(ObsGroup):
        """Obs for the auxiliary V_success head: kinematic state + time_left.

        ee_pose is intentionally omitted — no offline FK utility exists, and
        joint_pos is a bijection to ee_pose, so V_success can learn it.
        """

        prev_actions = ObsTerm(func=task_mdp.last_action)
        joint_pos = ObsTerm(func=task_mdp.joint_pos)
        insertive_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "quat",
            },
        )
        receptive_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "quat",
            },
        )
        time_left = ObsTerm(func=task_mdp.time_left)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: GroupCfg = GroupCfg()
    pointcloud: PointcloudCfg = PointcloudCfg()
    time_left: TimeLeftCfg = TimeLeftCfg()
    success_classifier: SuccessClassifierCfg = SuccessClassifierCfg()


@configclass
class StateObsCfg:
    """State-only obs (no PC): proprio + object poses. Single ``policy`` group for both actor and critic."""

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
        insertive_asset_in_receptive_asset_frame = ObsTerm(
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

    policy: PolicyCfg = PolicyCfg()


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

    success_reward = RewTerm(func=task_mdp.success_reward, weight=100.0)

    fail = RewTerm(
        func=task_mdp.is_terminated_term,
        weight=-1.0,
        params={"term_keys": ["abnormal_robot", "object_out_of_bound"]},
    )


# ===========================================================================
# Terminations — terminate on success (success is a hard reset, not a truncation)
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
    success = DoneTerm(func=task_mdp.success_termination, time_out=False)


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
            "reduction": "mean",  # "mean" (legacy, per-env counter) or "min_per_bucket" (decouples from GPS sampling)
            "floor": 0.0,  # minimum difficulty_frac; never demote below this
        },
    )


# ===========================================================================
# Events — GPS curriculum resets
# ===========================================================================
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
            "curriculum_target": 0.5,
            "curriculum_kappa": 2.0,
            "curriculum_temperature": 2.0,
            "use_classifier": True,
            "classifier_hidden_dim": 64,
            "classifier_lr": 1e-3,
            "use_success_critic": False,
            "curriculum_monitor_history_len": 100,
            "success_monitor_history_len": 100,
            # Real grouped GRPO: K envs in each group share a reset state
            # (replicated by MultiResetManager). Set via Hydra at launch:
            #   env.events.reset_from_states.params.group_size=K
            # 1 disables grouping (per-env independent resets, default behavior).
            "group_size": 1,
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
    """Base config — GPS resets, terminate on success, gravity curriculum."""

    scene: ZeroGSceneCfg = ZeroGSceneCfg(num_envs=32, env_spacing=1.5)
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: ZeroGRewardsCfg = ZeroGRewardsCfg()
    terminations: ZeroGTerminationsCfg = ZeroGTerminationsCfg()
    curriculum: ZeroGCurriculumCfg = ZeroGCurriculumCfg()
    events: ZeroGGPSEventCfg = ZeroGGPSEventCfg()
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
# ScenePC 512pt + GPS + terminate on success (single-task)
# ===========================================================================
@configclass
class ZeroGGPSScenePCTrainCfg(ZeroGBaseCfg):
    observations: ScenePCObsCfg = ScenePCObsCfg()


# ===========================================================================
# State-only (no PC) + uniform routing (no GPS). Use for gravity-reduction ablations.
# ===========================================================================
@configclass
class ZeroGStateTrainCfg(ZeroGBaseCfg):
    """State obs + uniform reset-type sampling. Gravity reduction via Hydra override.

    Default ``reduction="mean"`` (per-env ±1 counter). Override
    ``env.curriculum.gravity_curriculum.params.reduction=monitor_mean`` for the
    threshold-gated variant.
    """

    observations: StateObsCfg = StateObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Disable GPS routing → uniform multinomial over reset types.
        self.events.reset_from_states.params["curriculum_target"] = None
        self.events.reset_from_states.params["use_classifier"] = False
        self.events.reset_from_states.params["use_success_critic"] = False


@configclass
class ZeroGStateGPSTrainCfg(ZeroGBaseCfg):
    """State obs + empirical GPS routing (no classifier / critic).

    Matches ``ZeroGStateTrainCfg`` for obs/reward/scene, but keeps GPS enabled
    (target=0.5 by default, empirical per-state monitor). Intended for
    ablations that combine monitor-based gravity with GPS routing.
    """

    observations: StateObsCfg = StateObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_states.params["use_classifier"] = False
        self.events.reset_from_states.params["use_success_critic"] = False


# ===========================================================================
# ScenePC + uniform routing — winning gravity recipe with PC obs
# Mirrors ZeroGStateTrainCfg's reset-event overrides but feeds the actor/critic
# the 512-pt scene point cloud instead of object pose terms. Used to retrain
# the peg RL teacher with PC obs + the existing multi-canonical sparse reward,
# so the teacher can't lock onto a specific yaw via explicit pose input
# (cylindrical peg PC has approximate SO(2) symmetry → policy is forced
# toward a yaw-invariant action under the 4-canonical reward).
# ===========================================================================
@configclass
class ZeroGScenePCUniformTrainCfg(ZeroGGPSScenePCTrainCfg):
    """ScenePC obs + uniform reset-type sampling (no GPS).

    Pair with Hydra overrides for the winning gravity recipe::

        env.curriculum.gravity_curriculum.params.reduction=monitor_mean
        env.curriculum.gravity_curriculum.params.floor=0.1

    Multi-canonical reward is inherited from ``ZeroGRewardsCfg`` via the
    ``assembled_offsets`` migration (commit ``4b06dc5``); for cylindrical peg
    success fires for any of the 4 yaw-equivalent poses.
    """

    def __post_init__(self):
        super().__post_init__()
        # Disable GPS routing → uniform multinomial over reset types
        # (matches ZeroGStateTrainCfg, the winning recipe from `34722401`).
        self.events.reset_from_states.params["curriculum_target"] = None
        self.events.reset_from_states.params["use_classifier"] = False
        self.events.reset_from_states.params["use_success_critic"] = False


@configclass
class ZeroGScenePCUniformNoPrevActTrainCfg(ZeroGScenePCUniformTrainCfg):
    """Ablation: ``ZeroGScenePCUniformTrainCfg`` minus ``prev_actions`` from proprio.

    Tests whether last-action carryover is providing a yaw "anchor" that lets
    the teacher implicitly pick a canonical (and so produces multimodal teacher
    actions across rollouts). Without prev_actions the proprio still has
    ``joint_pos`` and ``end_effector_pose`` (both yaw-revealing), so this is
    not a strict yaw-invariance test — it isolates the prev_action contribution.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.proprio.prev_actions = None


# ===========================================================================
# State-only + DR for sim-to-real teacher distillation
# Retains pat/gravity's ZeroG resets + gravity curriculum + IMPLICIT actuator.
# Adds: arm sysid DR (armature/delays U(0.8,1.2), friction widened to U(0,1.5))
#       + OSC gain DR (U(0.8,1.2) around terminal_kp=(1000,1000,1000,50,50,50)).
# ===========================================================================
@configclass
class ZeroGGPSSysidEventCfg(ZeroGGPSEventCfg):
    """ZeroG GPS resets + arm sysid DR + OSC gain DR (widened arm friction).

    Uses the ``*_fixed`` randomizer variants so DR is full-strength from iter 0
    (no curriculum warmup — gravity curriculum already provides easier-to-harder
    progression on its own axis).
    """

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "friction_scale_range": (0.0, 1.5),
            "delay_range": (0, 1),
        },
    )

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (0.8, 1.2),
            "terminal_kp": (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
            "terminal_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        },
    )


@configclass
class ZeroGStateSysidTrainCfg(ZeroGStateTrainCfg):
    """ZeroGStateTrainCfg + arm sysid DR + OSC gain DR layered on the events.

    Identical to ``ZeroGStateTrainCfg`` (uniform sampling, IMPLICIT actuator,
    gravity curriculum, ZeroG resets) plus the dynamics randomization needed
    for the eventual depth student to be exposed to the same dynamics
    distribution at distillation time.

    Same Hydra knobs apply (e.g. ``env.curriculum.gravity_curriculum.params.\
reduction=monitor_mean env.curriculum.gravity_curriculum.params.floor=0.1``).
    """

    events: ZeroGGPSSysidEventCfg = ZeroGGPSSysidEventCfg()


# ===========================================================================
# Leg-DR-isolation events: each disables one suspect (arm friction rand or
# OSC gain rand) by setting its scale_range to (1.0, 1.0). The event still
# runs (so OSC kp stays at the spec value (1000,...,50)) but produces no
# variability. Used to identify which DR specifically prevents leg from
# training -- prior runs show legacy ZeroG-State-v0 (no DR) trains leg
# fine, but ZeroG-State-Sysid-v0 (sysid DR added) breaks it.
# ===========================================================================
@configclass
class ZeroGGPSSysidNoArmFricEventCfg(ZeroGGPSSysidEventCfg):
    """Sysid events but arm friction NOT randomized (held at sysid baseline 1.0x)."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "friction_scale_range": (1.0, 1.0),  # disabled
            "delay_range": (0, 1),
        },
    )


@configclass
class ZeroGGPSSysidNoOSCGainEventCfg(ZeroGGPSSysidEventCfg):
    """Sysid events but OSC gain randomization disabled (terminal_kp still set
    to the spec value via the event, but scale held at 1.0x)."""

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (1.0, 1.0),  # disabled
            "terminal_kp": (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
            "terminal_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        },
    )


@configclass
class ZeroGStateSysidNoArmFricTrainCfg(ZeroGStateTrainCfg):
    """ZeroGStateSysidTrainCfg minus arm friction randomization."""

    events: ZeroGGPSSysidNoArmFricEventCfg = ZeroGGPSSysidNoArmFricEventCfg()


@configclass
class ZeroGStateSysidNoOSCGainTrainCfg(ZeroGStateTrainCfg):
    """ZeroGStateSysidTrainCfg minus OSC gain randomization."""

    events: ZeroGGPSSysidNoOSCGainEventCfg = ZeroGGPSSysidNoOSCGainEventCfg()


# ===========================================================================
# Full-DR ZeroG training: ZeroGGPSSysidEventCfg + the 9 BaseEventCfg DR terms
# (mass/material/gripper actuator) + narrowed arm friction. Pairs with a
# training cfg that drops ZeroGPartialAssembly so reset distribution matches
# DAgger.
#
# Motivation: sim2sim debug (2026-04-29) showed teacher rate 0.985 -> 0.224
# when DAgger envs swap in BaseEventCfg's extra DRs that the legacy ZeroG
# training set never included. Long-term fix is to train future teachers on
# this Full-DR env so any DAgger config (which inherits the same DR set + adds
# scene cameras/curtains) is a strict SUPERSET only in scene, not DR.
# ===========================================================================
@configclass
class ZeroGGPSSysidFullDREventCfg(ZeroGGPSSysidEventCfg):
    """ZeroG events + arm sysid + OSC DR + 9 BaseEventCfg DRs + narrowed arm friction.

    Deltas vs ``ZeroGGPSSysidEventCfg``:
      + 4 material-randomization startup events (robot, insertive, receptive, table)
      + 4 mass-randomization startup events (robot, insertive, receptive, table)
      + 1 gripper-actuator-parameter reset event (finger_joint stiffness/damping
        log-uniform 0.5-2x)
      ~ randomize_arm_sysid friction range narrowed from (0.0, 1.5) to (0.8, 1.2)
        (matches the ``randomize_arm_from_sysid_fixed`` defaults that DAgger envs
         already use; the widened (0.0, 1.5) was a one-off for sim-to-real
         robustness experiments and overshot what DAgger applies.)

    Default reset_types from ZeroGGPSEventCfg = [ZeroGAnywhere, ZeroGPartialAssembly]
    50/50; the paired training cfg ``ZeroGStateSysidFullDRTrainCfg`` drops
    ZeroGPartialAssembly to match DAgger's reset distribution.
    """

    # ---- Material DR (startup) -------------------------------------------
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )
    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (1.0, 2.0),
            "dynamic_friction_range": (0.9, 1.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )
    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.2, 0.6),
            "dynamic_friction_range": (0.15, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )
    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.2, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    # ---- Mass DR (startup) -----------------------------------------------
    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            # 20-200g range matches BaseEventCfg.
            "mass_distribution_params": (0.02, 0.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_table_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # ---- Gripper actuator DR (reset) -------------------------------------
    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # ---- Override arm sysid: narrow friction from (0, 1.5) to (0.8, 1.2) -
    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "actuator_name": "arm",
            "scale_range": (0.8, 1.2),
            "friction_scale_range": (0.8, 1.2),
            "delay_range": (0, 1),
        },
    )


@configclass
class ZeroGStateSysidFullDRTrainCfg(ZeroGStateTrainCfg):
    """ZeroG state RL training with full DR + [ZeroGAnywhere, ZeroGPartialAssembly] resets.

    Pairs with ``ZeroGGPSSysidFullDREventCfg``. Future teachers trained here
    see the full DR distribution that depth-DAgger envs apply, so DAgger
    becomes a strict SUBSET of training on the DR axis (DAgger only differs
    by scene additions: cameras, curtains).

    Reset distribution: keeps the parent ``ZeroGGPSEventCfg`` default of
    ``[ZeroGAnywhere, ZeroGPartialAssembly]`` 50/50. The PartialAssembly
    near-goal states are essential for cold-start bootstrap — without them,
    100% of envs start far from goal and the policy never lucks into a
    success event under sparse reward (verified failure: peg/leg/drawer
    all sat at task_0=0 for 12h on `ZeroGAnywhere`-only FullDR).

    Note: this means RL training's reset distribution is broader than the
    deployment-time DAgger env (which uses ZeroGAnywhere only). That's fine
    for sim2sim — the teacher just needs to be robust to reset states it
    will see at deployment, and PartialAssembly states are strictly easier.
    """

    events: ZeroGGPSSysidFullDREventCfg = ZeroGGPSSysidFullDREventCfg()

## ^^^ this is bugged because wrong action and robot

@configclass
class ZeroGScenePCSysidSim2RealTrainCfg(ZeroGScenePCUniformTrainCfg):
    events: ZeroGGPSSysidFullDREventCfg = ZeroGGPSSysidFullDREventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

# Eval cfg's observations cfg: same as ScenePCObsCfg (proprio, pointcloud,
# time_left, success_classifier) plus a diagnostic ``heuristics`` group used
# by the policy-comparison tooling. The runner only reads the policy obs group,
# so adding ``heuristics`` is invisible to inference.
@configclass
class _ScenePCEvalObsCfgWithHeuristics(ScenePCObsCfg):
    heuristics: task_mdp.HeuristicsCfg = task_mdp.HeuristicsCfg()


@configclass
class ZeroGScenePCSysidSim2RealEvalCfg(ZeroGScenePCSysidSim2RealTrainCfg):
    observations: _ScenePCEvalObsCfgWithHeuristics = _ScenePCEvalObsCfgWithHeuristics()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 10})
        self.events.reset_from_states.params["reset_types"] = ["ZeroGAnywhere"]
        self.events.reset_from_states.params["probs"] = [1.0]

        self.curriculum = None

# ===========================================================================
# Sim2Real DR: UWLab-ICL-style unified arm+OSC+action-scale DR + narrow contact
# DR. Mirrors ``Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg`` from
# UWLab-ICL on the dynamics axis: a single ``randomize_env_cfg_unified`` event
# with one coupled progress scalar drives arm sysid + OSC gain interpolation +
# motor delay together (so we never sample "high friction + low OSC Kp", which
# is unsolvable). Action scale gets an independent progress (decoupled because
# scale doesn't gate task feasibility). On top of that we add narrow-range
# material / mass / gripper-actuator DR for sim2real contact robustness.
# Pairs with the ScenePC ZeroG training task for 0-shot transfer experiments.
# ===========================================================================
@configclass
class ZeroGGPSSysidSim2RealEventCfg(ZeroGGPSEventCfg):
    """Unified arm+OSC+scale DR (UWLab-ICL parity) + narrow contact DR + ZeroG resets.

    Inherits the two reset events from ``ZeroGGPSEventCfg``. Adds:

    * ``randomize_env_cfg_unified`` -- ported from UWLab-ICL's
      ``Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg``. Single coupled
      progress ~ U(0, 1.5) drives armature/friction (0 to 1.5x sysid),
      motor delay (0 to ceil(1.5*delay_hi) steps), and OSC Kp/Kd interpolation
      from ``_kp_default`` to ``terminal_kp * U(0.8, 1.2)``. Action scale uses
      an independent progress ~ U(0, 1.5) to slide between ``initial_scales``
      (= training Z=0.02) and ``target_scales`` (= eval Z=0.002).
    * 4 narrow material-friction startup events (robot / insertive / receptive / table).
    * 4 narrow mass-randomization startup events (~halved-width vs FullDR ranges).
    * 1 narrow gripper-actuator log-uniform reset event.
    """

    # ---- Unified arm + OSC + action-scale DR (reset) ---------------------
    randomize_env_cfg_unified = EventTerm(
        func=task_mdp.randomize_env_cfg_unified,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "actuator_name": "arm",
            "action_name": "arm",
            "arm_scale_range": (0.8, 1.2),
            "delay_range": (0, 1),
            "kp_scale_range": (0.8, 1.2),
            "terminal_kp": (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
            "terminal_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "initial_scales": (0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
            "target_scales": (0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
            "coupled_progress_range": (0.0, 1.5),
            "action_scale_progress_range": (0.0, 1.5),
        },
    )

    # ---- Material DR (startup) -- narrow ---------------------------------
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.6, 0.9),
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )
    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (1.2, 1.8),
            "dynamic_friction_range": (1.1, 1.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )
    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.5),
            "dynamic_friction_range": (0.25, 0.45),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )
    table_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.4, 0.55),
            "dynamic_friction_range": (0.3, 0.45),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    # ---- Mass DR (startup) -- narrow -------------------------------------
    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            # 50-120g: narrowed around a realistic ~80g insertive part
            # (vs FullDR's 20-200g sweep).
            "mass_distribution_params": (0.05, 0.12),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_table_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # ---- Gripper actuator DR (reset) -- narrow ---------------------------
    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "stiffness_distribution_params": (0.7, 1.4),
            "damping_distribution_params": (0.7, 1.4),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class ZeroGScenePCUniformSim2RealDRTrainCfg(ZeroGScenePCUniformTrainCfg):
    """ScenePC uniform-routing + wide arm DR (0-1.5x sysid) + narrow contact DR.

    Designed to mirror UWLab-ICL's ``Ur5eRobotiq2f85RelCartesianOSCPrivilegedTrainCfg``
    (unified-DR privileged training) on the dynamics axis while keeping this
    branch's ZeroG scene/obs/curriculum. Two deltas vs the parent
    ``ZeroGScenePCUniformTrainCfg``:

    * ``events``    -- swapped to ``ZeroGGPSSysidSim2RealEventCfg`` (wide arm
      sysid friction U(0, 1.5) inherited from ``ZeroGGPSSysidEventCfg`` plus
      narrow-range material/mass/gripper-actuator DR).
    * ``scene.robot`` -- swapped (in ``__post_init__``) to
      ``EXPLICIT_UR5E_ROBOTIQ_2F85`` (DelayedPDActuator). Required for the
      ``delay_range=(0, 1)`` term in ``randomize_arm_sysid`` to actually
      fire -- IMPLICIT actuators have no ``positions_delay_buffer`` and the
      delay DR is silently a no-op on them.

    The ``actions`` field stays at the parent's ``Ur5eRobotiq2f85RelativeOSCAction``
    (training scales, Z=0.02). The OSC terminal Kp is handled entirely by
    ``randomize_osc_gains`` (``_fixed`` variant, scale_progress=1) -- it
    interpolates from the action's ``_kp_default`` to ``terminal_kp * U(0.8, 1.2)``
    and lands at the terminal point regardless of the action cfg's initial Kp.
    Training scales (Z=0.02) also match UWLab-ICL's ``initial_scales`` in the
    unified-DR privileged cfg.

    The parent's ``__post_init__`` (called via ``super().__post_init__()``)
    still nulls GPS classifier/critic on the inherited ``reset_from_states``,
    preserving the uniform-routing behavior of the base task.
    """

    events: ZeroGGPSSysidSim2RealEventCfg = ZeroGGPSSysidSim2RealEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class ScenePCRealEvalTrainCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):

    scene: ZeroGSceneCfg = ZeroGSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ScenePCObsCfg = ScenePCObsCfg()
    
    def __post_init__(self):
        super().__post_init__()


# ===========================================================================
# DAgger-identical baseline: bit-for-bit ZeroGStateSysidTrainCfg, only the
# obs group is renamed `teacher` so State_DAggerFastRunnerCfg can route a JIT
# teacher into it. Curriculum floor pinned to 1.0 (full gravity) so the JIT
# teacher's converged training-time regime is reproduced from iter 0 — the
# teacher saw difficulty_frac~1 at the end of training, and student-driven
# eval would otherwise read the curriculum-warmup floor.
# ===========================================================================
@configclass
class _StateObsAsTeacherCfg:
    """Wraps StateObsCfg.PolicyCfg under the group name `teacher`.

    State_DAggerFastRunnerCfg expects ``obs_groups[policy/teacher] = ['teacher']``.
    """

    teacher: StateObsCfg.PolicyCfg = StateObsCfg.PolicyCfg()


@configclass
class ZeroGStateSysidDAggerIdenticalCfg(ZeroGStateSysidTrainCfg):
    """ZeroGStateSysidTrainCfg with the obs group renamed `teacher`.

    Used as the known-good baseline for the sim2sim debug ladder: if the JIT
    teacher hits ~95% here but degrades when DAgger features (cameras,
    curtains, FinetuneEvalEvent, EXPLICIT actuator, DAgger termination set,
    eval-action scales) are layered on, the offender is in the delta.
    """

    observations: _StateObsAsTeacherCfg = _StateObsAsTeacherCfg()

    def __post_init__(self):
        super().__post_init__()
        # Force full gravity from iter 0 — match the converged regime the
        # teacher trained in. Default floor=0.0 starts at zero gravity.
        self.curriculum.gravity_curriculum.params["floor"] = 1.0


# ===========================================================================
# Sim2sim debug ladder variants. Each layers ONE class of DAgger feature on
# top of ZeroGStateSysidDAggerIdenticalCfg to bisect which one breaks teacher
# rate. Pair with Hydra overrides where finer-grained changes suffice.
#
# Inheritance chains skip ZeroGStateTrainCfg.__post_init__ where it would
# touch a non-existent ``reset_from_states`` (the param name on ZeroGGPSEvent;
# FinetuneEvalEventCfg uses ``reset_from_reset_states`` instead). Inheriting
# from ZeroGBaseCfg directly avoids that crash.
# ===========================================================================
from .rl_state_cfg import FinetuneEvalEventCfg as _FinetuneEvalEventCfg  # noqa: E402


@configclass
class ZeroGStateSysidDAggerEvalEventsCfg(ZeroGBaseCfg):
    """V_events: DAggerIdentical scene + curriculum, but events from FinetuneEvalEventCfg.

    Tests whether replacing ZeroGGPSSysidEventCfg (sysid arm friction widening,
    GPS routing on ZeroGAnywhere+ZeroGPartialAssembly) with FinetuneEvalEventCfg
    (eval-grade arm sysid randomization, uniform-over-ZeroGAnywhere reset) drops
    the teacher rate. Mirrors 4Canonical's __post_init__ for terminal_kp +
    reset_types so the only delta from the existing 4Canonical run is the
    scene/cameras/curtains/terminations.
    """

    observations: _StateObsAsTeacherCfg = _StateObsAsTeacherCfg()
    events: _FinetuneEvalEventCfg = _FinetuneEvalEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Full gravity from iter 0 (matches DAggerIdentical).
        self.curriculum.gravity_curriculum.params["floor"] = 1.0
        # 4Canonical's reset distribution: ZeroGAnywhere only (matches teacher's training).
        self.events.reset_from_reset_states.params["reset_types"] = ["ZeroGAnywhere"]
        self.events.reset_from_reset_states.params["probs"] = [1.0]
        # 4Canonical's stiff terminal Kp; FinetuneEvalEventCfg leaves these unset.
        self.events.randomize_osc_gains.params["terminal_kp"] = (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0)
        self.events.randomize_osc_gains.params["terminal_damping_ratio"] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# V_scene: V0 + curtains + side/wrist cameras (matches 4Canonical scene-add
# delta). Keeps ZeroG events/curriculum/terminations. Tests the camera/
# curtain visual-entity contribution to the teacher rate gap.
# ---------------------------------------------------------------------------
import isaaclab.sim as _sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg as _RigidObjCfg2  # noqa: E402
from isaaclab.managers import EventTermCfg as _EventTermV  # noqa: E402
from isaaclab.sensors import TiledCameraCfg as _TiledCameraCfg  # noqa: E402

_DEPTH_CLIP_V = (0.01, 2.0)
_RENDER_HW_V = 224


@configclass
class _ZeroGDAggerSceneCfg(ZeroGSceneCfg):
    """ZeroGSceneCfg + curtains + side + wrist cameras (mirrors 4Canonical scene)."""

    curtain_left = _RigidObjCfg2(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=_RigidObjCfg2.InitialStateCfg(pos=(0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=_sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=_sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=_sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    curtain_back = _RigidObjCfg2(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=_RigidObjCfg2.InitialStateCfg(pos=(-0.15, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=_sim_utils.CuboidCfg(
            size=(0.01, 1.3, 1.125),
            rigid_props=_sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=_sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    curtain_right = _RigidObjCfg2(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=_RigidObjCfg2.InitialStateCfg(pos=(0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=_sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=_sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=_sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    side_camera = _TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_side_camera",
        update_period=0,
        height=_RENDER_HW_V,
        width=_RENDER_HW_V,
        offset=_TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=_sim_utils.PinholeCameraCfg(focal_length=20.10, clipping_range=_DEPTH_CLIP_V),
    )
    wrist_camera = _TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=_RENDER_HW_V,
        width=_RENDER_HW_V,
        offset=_TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=_sim_utils.PinholeCameraCfg(focal_length=24.55, clipping_range=_DEPTH_CLIP_V),
    )


@configclass
class _ZeroGGPSSysidWristResetEventCfg(ZeroGGPSSysidEventCfg):
    """ZeroGGPSSysidEventCfg + reset_wrist_camera_pose (required when wrist cam exists)."""

    reset_wrist_camera_pose = _EventTermV(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera",
            "base_position": (0.0182505, -0.00408447, -0.0689107),
            "base_rotation": (0.34254336, -0.61819255, -0.6160212, 0.347879),
            "position_deltas": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "euler_deltas": {"pitch": (0.0, 0.0), "yaw": (0.0, 0.0), "roll": (0.0, 0.0)},
        },
    )


@configclass
class ZeroGStateSysidDAggerWithSceneCfg(ZeroGStateSysidDAggerIdenticalCfg):
    """V_scene: V0 + DAgger curtains + side/wrist cameras + wrist-cam reset event.

    Tests whether the visual-entity additions (kinematic curtains + 2 TiledCameras)
    perturb teacher rate independent of the events change. If teacher rate matches
    V0 here, scene additions are physics-neutral and the gap is in events.
    """

    scene: _ZeroGDAggerSceneCfg = _ZeroGDAggerSceneCfg(num_envs=32, env_spacing=1.5)
    events: _ZeroGGPSSysidWristResetEventCfg = _ZeroGGPSSysidWristResetEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match 4Canonical's render-light settings (with cameras present, dlssg
        # is undesirable since denoiser/AO can mess depth output).
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render_interval = self.decimation


# ===========================================================================
# Multi-task ZeroG: peg + leg via MultiAssetSpawner
# ===========================================================================
@configclass
class ZeroGMultiTaskSceneCfg(ZeroGSceneCfg):
    """Peg + leg in one scene via MultiAssetSpawnerCfg. replicate_physics=False required."""

    replicate_physics = False

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


@configclass
class ZeroGMultiTaskBaseCfg(ZeroGBaseCfg):
    """Multi-task base: peg+leg scene, no Hydra object variants."""

    scene: ZeroGMultiTaskSceneCfg = ZeroGMultiTaskSceneCfg(num_envs=32, env_spacing=1.5)

    def __post_init__(self):
        super().__post_init__()
        self.variants = {}


@configclass
class ZeroGMultiTaskGPSScenePCTrainCfg(ZeroGMultiTaskBaseCfg):
    """Multi-task: peg+leg + ScenePC 512pt + GPS + terminate on success."""

    observations: ScenePCObsCfg = ScenePCObsCfg()
