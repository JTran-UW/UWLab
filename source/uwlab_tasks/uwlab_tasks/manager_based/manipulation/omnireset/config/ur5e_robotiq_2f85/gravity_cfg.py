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
