# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Depth DAgger env configs for distilling state RL experts into depth-image students.

Scene mirrors ``data_collection_rgb_cfg.py`` (curtains + 3 cameras at front/side/wrist
poses) but the cameras render depth (``distance_to_camera``) instead of RGB.

Observation layout:
    * ``proprio``   — student input, 1D concatenated (prev_action + joint_pos + ee_pose)
    * ``front_depth``, ``side_depth``, ``wrist_depth`` — 1-channel depth images per camera
    * ``teacher``   — state-expert input, 1D, history_length=5 (matches JIT teacher's training obs)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.gravity_cfg import ZeroGGPSSysidFullDREventCfg

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85RelativeOSCEvalAction
from .rl_state_cfg import FinetuneEvalEventCfg, RlStateSceneCfg, Ur5eRobotiq2f85RlStateCfg

DEPTH_CLIP = (0.01, 2.0)
IMG_H, IMG_W = 224, 224
RENDER_H, RENDER_W = 240, 320


@configclass
class DepthDAggerSceneCfg(RlStateSceneCfg):
    """Scene with black curtain backdrops and 3 depth TiledCameras."""

    curtain_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_back = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.3, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_side_camera",
        update_period=0,
        height=RENDER_H,
        width=RENDER_W,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.10, clipping_range=DEPTH_CLIP
        ),
    )

def _depth_obs_term(
    sensor_name: str,
    depth_noise_range: float = 0.0,
    depth_global_bias_range: float = 0.0,
    depth_global_scale_range: float = 0.0,
    depth_dropout_prob: float = 0.0,
) -> ObsTerm:
    return ObsTerm(
        func=task_mdp.process_image,
        params={
            "sensor_cfg": SceneEntityCfg(sensor_name),
            "data_type": "distance_to_camera",
            "process_image": True,
            "output_size": (IMG_H, IMG_W),
            "depth_clip": DEPTH_CLIP,
            "depth_noise_range": depth_noise_range,
            "depth_global_bias_range": depth_global_bias_range,
            "depth_global_scale_range": depth_global_scale_range,
            "depth_dropout_prob": depth_dropout_prob,
        },
    )


@configclass
class DepthDAggerObservationsCfg:
    """Three obs groups: ``proprio`` (student scalars), per-camera depth, ``teacher`` (state expert input)."""

    @configclass
    class ProprioCfg(ObsGroup):
        """Student proprioception — must match what a real robot can measure."""

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
            self.history_length = 5

    @configclass
    class SideDepthCfg(ObsGroup):
        image = _depth_obs_term("side_camera", depth_noise_range=0.01, depth_global_bias_range=0.05, depth_global_scale_range=0.05, depth_dropout_prob=0.05)

        def __post_init__(self):
            # Single-term group: concatenate_terms=True so the image tensor is hoisted to
            # the group level (obs["front_depth"]) instead of being nested under a term key.
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """State expert input — must match the JIT teacher's training-time observation layout.

        NOTE: term *ordering* here is structural, not cosmetic. ``@configclass`` is a
        dataclass wrapper, so annotated fields (``x: T = ...``) are promoted above
        plain class attributes (``x = ...``) in iteration order. The state expert's
        ``PolicyCfg`` in ``rl_state_cfg.py`` annotates only
        ``insertive_asset_in_receptive_asset_frame`` — which lands that term at
        index 0 of the concatenated obs tensor. Reproduce the annotation here or
        the teacher sees a scrambled 215d vector and outputs garbage.
        """

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )
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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class AuxTargetCfg(ObsGroup):
        """Ground-truth object poses for CNN aux loss.

        Three pose terms: peg↔wrist, hole↔wrist, peg↔hole. Single-frame,
        no history, no corruption — this is the target a CNN aux head
        regresses to, forcing image features to be pose-aware.
        """

        insertive_in_wrist = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )
        receptive_in_wrist = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )
        insertive_in_receptive = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    side_depth: SideDepthCfg = SideDepthCfg()
    teacher: TeacherCfg = TeacherCfg()
    aux_target: AuxTargetCfg = AuxTargetCfg()


@configclass
class DepthDAggerTerminationsCfg:
    """Termination criteria for DAgger envs.

    `success` matches the state-RL training env's `success_termination` (single
    trigger from `progress_context.success`), not the prior 5-consecutive
    requirement. The 5-consecutive criterion was capping teacher rate in DAgger
    (~27% peg, ~41% drawer) vs training (~98%) since teacher policies often
    reach the success state momentarily then drift back from gripper jitter /
    OSC oscillation. Matching training removes this artificial gap.

    `early_success` retained for the runner's reset-on-early-success behavior;
    its 5-consecutive setting is a different (no-grad) signal for the runner,
    not the training termination metric.
    """

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)
    early_success = DoneTerm(
        func=task_mdp.early_success_termination,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )
    success = DoneTerm(func=task_mdp.success_termination, time_out=False)


@configclass
class Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg(Ur5eRobotiq2f85RlStateCfg):
    """Depth DAgger env: 3 depth cameras + separate student/teacher obs groups.

    Uses the **eval** action scales/gains (``Ur5eRobotiq2f85RelativeOSCEvalAction``)
    because the downloaded state expert was exported from the Stage-2 finetune *eval*
    config (``Finetune-Play-v0``); training scales are 10× larger on Z and would make
    teacher-driven rollouts overshoot.
    """

    # replicate_physics=True matches how the state expert was trained/evaluated
    # (single-task peg/leg/drawer). Switch to False only if we later need
    # heterogeneous per-env assets (multi-task depth DAgger).
    scene: DepthDAggerSceneCfg = DepthDAggerSceneCfg(num_envs=256, env_spacing=1.5, replicate_physics=True)
    observations: DepthDAggerObservationsCfg = DepthDAggerObservationsCfg()
    terminations: DepthDAggerTerminationsCfg = DepthDAggerTerminationsCfg()
    events: FinetuneEvalEventCfg = FinetuneEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        # Match the Stage-2 finetune-eval config the teacher was exported from.
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 16.0
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 1


# ---------------------------------------------------------------------------
# 2-camera variant: side + wrist (wrist parented to robotiq_base_link).
# Wrist pose/focal copied from data_collection_rgb_cfg.py; we set
# ``update_latest_camera_pose=True`` because the link moves every step
# (otherwise the cached scene-creation pose is reused and render goes all-inf).
# ---------------------------------------------------------------------------


@configclass
class DAggerWristSideSceneCfg(RlStateSceneCfg):
    """Flat scene class (no deep inheritance) with curtains + side + wrist cameras.

    Mirrors ``DataCollectionRGBObjectSceneCfg`` structure — declaring all scene
    entities at the same class level avoids a configclass MRO/field-ordering
    gotcha where nesting the wrist camera under a child scene produced a
    degenerate sky view.
    """

    # Background curtains (copied from DepthDAggerSceneCfg for black-backdrop consistency).
    curtain_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_back = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.3, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_side_camera",
        update_period=0,
        height=RENDER_H,
        width=RENDER_W,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["rgb"],
        # clipping_range added so depth variants work without redefining the cam.
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.10, clipping_range=DEPTH_CLIP),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=RENDER_H,
        width=RENDER_W,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55, clipping_range=DEPTH_CLIP),
    )


@configclass
class WristSideEventCfg(FinetuneEvalEventCfg):
    """FinetuneEval events + a reset-mode call to ``randomize_tiled_cameras`` with
    zero deltas for the wrist camera.

    Why: USD's ``reset_scene_to_default`` (part of ``BaseEventCfg``) resets the
    wrist camera's local XformOps on every reset, which undoes the
    ``TiledCameraCfg.OffsetCfg`` rotation and leaves the camera pointing at the
    sky. DataCollection RGB works because it has a ``randomize_wrist_camera``
    reset event that re-writes the correct base pose every reset. We copy that
    pattern with zero delta ranges (no randomization, just pose restoration).
    """

    reset_wrist_camera_pose = EventTerm(
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
class DepthDAggerWristSideObservationsCfg:
    """2-camera depth (side + wrist) obs layout."""

    @configclass
    class WristDepthCfg(ObsGroup):
        image = _depth_obs_term("wrist_camera", depth_noise_range=0.01, depth_global_bias_range=0.05, depth_global_scale_range=0.05, depth_dropout_prob=0.05)

        def __post_init__(self):
            self.concatenate_terms = True

    proprio: DepthDAggerObservationsCfg.ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg()
    side_depth: DepthDAggerObservationsCfg.SideDepthCfg = DepthDAggerObservationsCfg.SideDepthCfg()
    wrist_depth: WristDepthCfg = WristDepthCfg()
    teacher: DepthDAggerObservationsCfg.TeacherCfg = DepthDAggerObservationsCfg.TeacherCfg()
    aux_target: DepthDAggerObservationsCfg.AuxTargetCfg = DepthDAggerObservationsCfg.AuxTargetCfg()


@configclass
class Ur5eRobotiq2f85DepthDAggerWristSideCfg(Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg):
    """2-camera depth DAgger with wrist (parented to gripper base link) + side cam."""

    scene: DAggerWristSideSceneCfg = DAggerWristSideSceneCfg(
        num_envs=256, env_spacing=1.5, replicate_physics=True
    )
    observations: DepthDAggerWristSideObservationsCfg = DepthDAggerWristSideObservationsCfg()
    events: WristSideEventCfg = WristSideEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Scene cfg defaults to rgb for the shared DAggerWristSideSceneCfg — flip to depth here.
        self.scene.side_camera.data_types = ["distance_to_camera"]
        self.scene.wrist_camera.data_types = ["distance_to_camera"]


# ---------------------------------------------------------------------------
# 4-canonical peg DAgger: matches a state teacher trained on
# ``OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-v0`` (pat/gravity).
# Three deltas vs the existing wristside env:
#   1. Actuator: IMPLICIT_UR5E_ROBOTIQ_2F85 (the ZeroG-State-v0 default), not EXPLICIT.
#   2. Action:   Ur5eRobotiq2f85RelativeOSCAction (training scales), not Eval.
#   3. Teacher:  43d single-frame obs (history=1), term order = source-code order
#                (NO first-annotated insertive-in-receptive trick).
# Runtime success math is auto-aware of the 4 receptive offsets via the merged
# ``ProgressContext`` (rewards.py): ``orientation_aligned`` is True if rel_quat
# matches ANY of the 4 yaw offsets in the peghole metadata.
# ---------------------------------------------------------------------------


from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85  # noqa: E402

from .actions import Ur5eRobotiq2f85RelativeOSCAction  # noqa: E402


@configclass
class Teacher4CanonicalCfg(ObsGroup):
    """43d single-frame obs matching ZeroGStateTrainCfg.StateObsCfg.PolicyCfg.

    Term order = source-code declaration order (all bare attributes, no
    annotated overrides). The new state teacher (model_3700.pt from gravity
    SLURM 34840064) was trained on this exact layout with history_length=1.
    """

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
        self.history_length = 1


@configclass
class DepthDAggerWristSide4CanonicalObservationsCfg(DepthDAggerWristSideObservationsCfg):
    """Wrist+side depth obs with the 4-canonical TeacherCfg (43d single-frame)."""

    teacher: Teacher4CanonicalCfg = Teacher4CanonicalCfg()


@configclass
class Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalCfg(Ur5eRobotiq2f85DepthDAggerWristSideCfg):
    """Wrist+side depth DAgger paired with a 4-canonical-trained state teacher."""

    observations: DepthDAggerWristSide4CanonicalObservationsCfg = (
        DepthDAggerWristSide4CanonicalObservationsCfg()
    )
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        # Match ZeroGStateTrainCfg's actuator (the env that trained model_3700.pt).
        self.scene.robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Match the teacher's training-time stiff terminal OSC Kp. Without this,
        # FinetuneEvalEventCfg.randomize_osc_gains falls back to the action's
        # soft pre-train _kp_default=(200, 200, 200, 3, 3, 3), underdriving the
        # controller 5x and producing zero successes (verified via 50/50 split).
        self.events.randomize_osc_gains.params["terminal_kp"] = (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0)
        self.events.randomize_osc_gains.params["terminal_damping_ratio"] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        # Match the teacher's training-time reset distribution: ZeroG-recorded
        # anywhere placements (loaded from dataset), not procedural ObjectAnywhere.
        self.events.reset_from_reset_states.params["reset_types"] = ["ZeroGAnywhere"]
        self.events.reset_from_reset_states.params["probs"] = [1.0]


@configclass
class _DepthDAggerWristSide4CanonicalNoCamObsCfg:
    """For camera-presence A/B test: just `teacher` (43d). State_DAggerFastRunnerCfg
    sets `obs_groups[policy] = teacher`, so no `proprio` group is needed."""

    teacher: Teacher4CanonicalCfg = Teacher4CanonicalCfg()


@configclass
class Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalNoCamCfg(
    Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalCfg
):
    """4-canonical with cameras + curtains REMOVED — A/B test for sim2sim gap.

    Same env as 4-canonical except the scene drops side_camera, wrist_camera,
    and the 3 curtains. Used to measure teacher rate without camera-induced
    physics perturbation. If teacher rate jumps from ~0.25 to ~0.95 here,
    the cameras are the sim2sim gap source.
    """

    observations: _DepthDAggerWristSide4CanonicalNoCamObsCfg = (
        _DepthDAggerWristSide4CanonicalNoCamObsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        # Drop scene cameras + curtains. Setting attribute to None removes from
        # the scene cfg (configclass tolerates None for these RigidObject/Sensor slots).
        self.scene.side_camera = None
        self.scene.wrist_camera = None
        self.scene.curtain_left = None
        self.scene.curtain_back = None
        self.scene.curtain_right = None
        # Drop the wrist-camera reset event since the camera no longer exists.
        if hasattr(self.events, "reset_wrist_camera_pose"):
            self.events.reset_wrist_camera_pose = None


# ---------------------------------------------------------------------------
# Drawer DAgger control (single-canonical, non-symmetric task).
# Symmetry-hypothesis test: if drawer easily breaks 31% under the same recipe,
# then the peg ceiling is symmetry-specific (not modality / arch / scaffolding).
#
# Drawer assets have a single ``assembled_offset`` in metadata, so the merged
# ProgressContext.multi-offset code falls through to the single-canonical path
# (``shape[0] > 1`` check is False). No additional cfg changes needed beyond
# swapping the scene's insertive/receptive USDs.
# ---------------------------------------------------------------------------


from isaaclab.assets import RigidObjectCfg as _RigidObjectCfg  # noqa: E402


def _make_drawer_insertive() -> _RigidObjectCfg:
    """Drawer bottom (rigid, has gravity) — to be inserted into the drawer box."""
    return _RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=_RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _make_drawer_receptive() -> _RigidObjectCfg:
    """Drawer box (kinematic, the receptacle)."""
    return _RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=_RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR  # noqa: E402


@configclass
class Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalDrawerCfg(
    Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalCfg
):
    """Drawer variant: same recipe as 4-canonical peg, but with drawer USDs swapped in."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.insertive_object = _make_drawer_insertive()
        self.scene.receptive_object = _make_drawer_receptive()


# ---------------------------------------------------------------------------
# LEAN DAgger envs (peg + drawer): training-events only, no BaseEventCfg DRs.
# Built after sim2sim ladder bisection (2026-04-29) showed the existing 4Canonical
# inherits BaseEventCfg's 9 extra DR terms (mass/material/gripper) which the
# state teachers never saw at training time -> abnormal_robot trip 0.006 -> 0.585,
# teacher rate 0.985 -> 0.224. These lean envs swap WristSideEventCfg for a ZeroG
# events sibling so the existing peg_sysid_mean.pt / drawer_sysid_full.pt
# teachers experience the same DR distribution they trained under.
#
# Inheritance NOTE: cannot extend Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalCfg
# directly — its __post_init__ touches `self.events.reset_from_reset_states`
# (FinetuneEvalEventCfg's param name), which does not exist on ZeroG events
# (param is `reset_from_states`). Inheriting from the 4Canonical's parent
# (DepthDAggerWristSideCfg) and re-doing the 4Canonical-specific overrides
# (IMPLICIT actuator, training-action scales, 4-canonical 43d teacher obs)
# avoids that crash.
# ---------------------------------------------------------------------------

from .gravity_cfg import ZeroGGPSSysidEventCfg as _ZeroGGPSSysidEventCfg  # noqa: E402


@configclass
class _ZeroGSysidWristSideEventCfg(_ZeroGGPSSysidEventCfg):
    """ZeroGGPSSysidEventCfg + reset_wrist_camera_pose (rewrites wrist cam offset
    after USD reset_scene_to_default zeroes its local XformOps)."""

    reset_wrist_camera_pose = EventTerm(
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
class Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalLeanCfg(
    Ur5eRobotiq2f85DepthDAggerWristSideCfg
):
    """Lean 4-canonical DAgger: training events only (no BaseEventCfg DR superset).

    Inherits scene/cameras/curtains from the WristSide parent, overrides events
    to the ZeroG training event set so the existing `peg_sysid_mean.pt` teacher
    sees its training dynamics distribution exactly. 4Canonical's IMPLICIT
    actuator + training-scale OSC action + 43d single-frame teacher obs are
    re-applied here (since we can't inherit from 4Canonical — see file note).
    """

    observations: DepthDAggerWristSide4CanonicalObservationsCfg = (
        DepthDAggerWristSide4CanonicalObservationsCfg()
    )
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    events: _ZeroGSysidWristSideEventCfg = _ZeroGSysidWristSideEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # 4Canonical actuator (IMPLICIT, not EXPLICIT from grandparent post_init).
        self.scene.robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Match teacher's training-time reset distribution: ZeroGAnywhere only
        # (test B confirmed dropping ZeroGPartialAssembly is innocent), uniform
        # routing (no GPS classifier, since teacher trained without it).
        self.events.reset_from_states.params["reset_types"] = ["ZeroGAnywhere"]
        self.events.reset_from_states.params["probs"] = [1.0]
        self.events.reset_from_states.params["curriculum_target"] = None
        self.events.reset_from_states.params["use_classifier"] = False
        self.events.reset_from_states.params["use_success_critic"] = False


@configclass
class Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalLeanDrawerCfg(
    Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalLeanCfg
):
    """Lean 4-canonical DAgger, drawer task variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.insertive_object = _make_drawer_insertive()
        self.scene.receptive_object = _make_drawer_receptive()


# ---------------------------------------------------------------------------
# BCPPO env: mirrors teacher's RL training cfg (ZeroGStateSysidTrainCfg) +
# cameras + 4 obs groups (proprio, wrist_depth, side_depth, teacher).
#
# Why a separate env (not the existing Lean DAgger env): Lean inherits
# RewardsCfg from RlStateCfg with dense shaping rewards (ee_asset_distance,
# dense_success_reward, weighted progress_context). The teacher trained on
# ZeroGRewardsCfg (sparse: action regularizers + progress_context=0.1 only
# for the ProgressContext class to be called by IsaacLab + success_reward=100
# + fail=-1 on terminations). PPO under shaped rewards on a deployed-by-BC
# student can reward-hack the shaping signal without ever inserting (the
# 4-job DAgger sweep showed this -- mean_reward 1.7 with ZERO success
# events).
#
# This env keeps gravity_frac=1.0 from iter 0 (teacher's deployment regime).
# ---------------------------------------------------------------------------

from .gravity_cfg import (  # noqa: E402
    ZeroGStateSysidTrainCfg as _ZeroGStateSysidTrainCfg,
    _ZeroGDAggerSceneCfg,
    _ZeroGGPSSysidWristResetEventCfg,
)


@configclass
class _BCPPOObsCfg:
    """4 obs groups for BCPPO depth student.

    - proprio: 1d student input (prev_action + joint_pos + ee_pose, history=5)
    - wrist_depth, side_depth: 1-channel 224x224 depth images
    - teacher: 43d state input matching ZeroG teacher's training obs layout
      (single-frame, term-order from StateObsCfg.PolicyCfg)
    """

    proprio: DepthDAggerObservationsCfg.ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg()
    wrist_depth: DepthDAggerWristSideObservationsCfg.WristDepthCfg = (
        DepthDAggerWristSideObservationsCfg.WristDepthCfg()
    )
    side_depth: DepthDAggerObservationsCfg.SideDepthCfg = DepthDAggerObservationsCfg.SideDepthCfg()
    teacher: Teacher4CanonicalCfg = Teacher4CanonicalCfg()


@configclass
class Ur5eRobotiq2f85BCPPOSysidCfg(_ZeroGStateSysidTrainCfg):
    """BCPPO env: ZeroGStateSysidTrainCfg (teacher's RL training cfg) + cameras
    + 4 obs groups (proprio, wrist_depth, side_depth, teacher).

    Inherits sparse ZeroG rewards/terminations/curriculum/sysid events from
    parent. Adds DAgger scene (curtains + side/wrist cameras) and the
    reset_wrist_camera_pose event. Forces full gravity (floor=1.0) since we
    deploy the trained student at full gravity.
    """

    scene: _ZeroGDAggerSceneCfg = _ZeroGDAggerSceneCfg(num_envs=32, env_spacing=1.5)
    observations: _BCPPOObsCfg = _BCPPOObsCfg()
    events: _ZeroGGPSSysidWristResetEventCfg = _ZeroGGPSSysidWristResetEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Full gravity from iter 0 (deployment regime; teacher trained to convergence at this).
        self.curriculum.gravity_curriculum.params["floor"] = 1.0
        # Match teacher training reset distribution: drop GPS routing.
        self.events.reset_from_states.params["curriculum_target"] = None
        self.events.reset_from_states.params["use_classifier"] = False
        self.events.reset_from_states.params["use_success_critic"] = False
        # Camera-friendly render settings (matches Lean env).
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render_interval = self.decimation


@configclass
class Ur5eRobotiq2f85DepthGRPOGroupedDAggerNativeCfg(Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalLeanCfg):
    """GRPO env subclassing the *exact* env DAgger 34955212 trained in (Lean
    4-canonical). The plain WristSide parent uses different events / actuator /
    obs and the loaded DAgger weights produce 0% success there.

    Reset event is named ``reset_from_states`` (4-canonical Lean uses ZeroG
    sysid event set, not FinetuneEvalEventCfg); set group_size via Hydra at:
       env.events.reset_from_states.params.group_size=K
    """


@configclass
class Ur5eRobotiq2f85BCPPOSysidCurriculumCfg(Ur5eRobotiq2f85BCPPOSysidCfg):
    """Same as BCPPOSysidCfg but with GPS curriculum re-enabled (target=0.5).

    Used by grouped GRPO: routes resets to states with ~50% empirical success
    rate, ensuring within-group return variance so the per-group baseline gives
    a meaningful advantage signal.
    """

    def __post_init__(self):
        super().__post_init__()
        # Re-enable GPS curriculum (parent disabled it).
        self.events.reset_from_states.params["curriculum_target"] = 0.5
        self.events.reset_from_states.params["curriculum_kappa"] = 2.0
        self.events.reset_from_states.params["curriculum_temperature"] = 2.0


# ---------------------------------------------------------------------------
# ScenePC peg teacher DAgger (pat/dagger-symmetry)
#
# Replaces the 215d state teacher (pose-based MLP) with the new ScenePC teacher
# (proprio + 512-pt scene point cloud through a shared MLP encoder, then actor
# MLP). The hypothesis: cylindrical peg PC ~ SO(2)-symmetric → teacher action
# becomes yaw-invariant → DAgger MSE no longer collapses to averaged-across-
# canonicals actions, lifting peg DAgger above the ~50% ceiling.
#
# Teacher checkpoints from Tillicum 107905 (with prev_actions) and 107906
# (NoPrevAct) — both converged ~99% on task_0/task_1 at full gravity. Synthetic
# yaw probe showed scene_std/yaw_std ratio of 126x (107905) / 197x (107906),
# consistent with yaw-invariant action map.
#
# Two env variants below:
#   * Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherLeanCfg          (prev_act)
#   * Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherLeanNoPrevActCfg (no prev_act)
# In the NoPrevAct variant the *student* proprio also drops prev_actions, so
# what the student sees mirrors what the teacher saw at training time.
# ---------------------------------------------------------------------------

from .gravity_cfg import (  # noqa: E402
    ZeroGGPSEventCfg as _ZeroGGPSEventCfg,
    ScenePCObsCfg as _ScenePCObsCfg,
)


@configclass
class _ZeroGNoSysidWristSideEventCfg(_ZeroGGPSEventCfg):
    """Plain ZeroG GPS events (no sysid) + reset_wrist_camera_pose.

    Matches the ScenePC teacher's training events (it trained on
    ``ZeroGGPSEventCfg`` via ``ZeroGScenePCUniformTrainCfg`` → ``ZeroGBaseCfg``,
    which never layered in sysid DR).
    """

    reset_wrist_camera_pose = EventTerm(
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
class FullSysidDRWristSideEventCfg(ZeroGGPSSysidFullDREventCfg):
    reset_wrist_camera_pose = EventTerm(
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

# Term order is structural — the JIT teacher splits its input as
# ``[proprio_first | pointcloud_second]``. proprio terms must come first and in
# the same order as ``ScenePCObsCfg.GroupCfg`` (prev_actions, joint_pos, ee_pose).
# Then scene_pc.
@configclass
class TeacherProprioWithPCCfg(ObsGroup):
    """ScenePC teacher input: proprio (25d) + scene_pc (1536d) = 1561d, single-frame.

    Mirrors ``ScenePCObsCfg.GroupCfg`` (proprio) + ``ScenePCObsCfg.PointcloudCfg``
    (pointcloud) concatenated into one obs group. The JIT teacher splits this
    1561d tensor into proprio (first 25) and pc (last 1536) internally.
    """

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
    scene_pc = ObsTerm(
        func=task_mdp.ScenePointCloud,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_cfg": SceneEntityCfg("insertive_object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            "visualize": False,
            "num_points": 512,
            # Match teacher's training: resample_on_reset=False (default).
            "resample_on_reset": False,
        },
    )

    def __post_init__(self):
        # Match training-time corruption flag from ScenePCObsCfg.GroupCfg
        # (PointcloudCfg also had enable_corruption=True; both were True).
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class TeacherProprioWithPCNoPrevActCfg(TeacherProprioWithPCCfg):
    """Same as TeacherProprioWithPCCfg minus prev_actions (18d proprio + 1536d pc = 1554d).

    Pair with the NoPrevAct teacher checkpoint (Tillicum 107906).
    """

    def __post_init__(self):
        super().__post_init__()
        self.prev_actions = None


@configclass
class _DepthProprioNoPrevActCfg(DepthDAggerObservationsCfg.ProprioCfg):
    """Student proprio with prev_actions removed (paired with NoPrevAct teacher)."""

    def __post_init__(self):
        super().__post_init__()
        self.prev_actions = None


@configclass
class DepthDAggerWristSidePCTeacherObsCfg(DepthDAggerWristSideObservationsCfg):
    """WristSide depth obs with the ScenePC teacher group (proprio+pc=1561d)."""

    teacher: TeacherProprioWithPCCfg = TeacherProprioWithPCCfg()


@configclass
class DepthDAggerWristSidePCTeacherNoPrevActObsCfg(DepthDAggerWristSidePCTeacherObsCfg):
    """NoPrevAct variant: drops prev_actions from BOTH student proprio and teacher proprio."""

    proprio: _DepthProprioNoPrevActCfg = _DepthProprioNoPrevActCfg()
    teacher: TeacherProprioWithPCNoPrevActCfg = TeacherProprioWithPCNoPrevActCfg()


@configclass
class Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherLeanCfg(
    Ur5eRobotiq2f85DepthDAggerWristSideCfg
):
    """Lean DAgger paired with ScenePC peg teacher (with prev_actions).

    Mirrors ``Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalLeanCfg`` but:
      * teacher obs group = proprio + scene_pc (1561d) instead of 215d state
      * events are no-sysid (matches ScenePC teacher's training) + wrist cam reset
      * IMPLICIT actuator + training-scale OSC action (same as the 4-canonical Lean)

    Teacher JIT: ``teachers/peg_pc_prevact_jit.pt``.
    """

    observations: DepthDAggerWristSidePCTeacherObsCfg = DepthDAggerWristSidePCTeacherObsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    events: _ZeroGNoSysidWristSideEventCfg = _ZeroGNoSysidWristSideEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # IMPLICIT actuator (same as ScenePC teacher's training env, ZeroGSceneCfg).
        self.scene.robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Match the OLD 4-canonical Lean DAgger reset distribution: ZeroGAnywhere
        # only. The teacher was trained on 50/50 (Anywhere/PA) but at convergence
        # solves both task types >99%, so deploying on Anywhere alone is fine and
        # gives apples-to-apples comparison with the historical 4-canonical Lean
        # run. Without this, half the reset states are PA which fire
        # success_termination immediately, inflating early student_eval to ~98%
        # even at random init.
        self.events.reset_from_states.params["reset_types"] = ["ZeroGAnywhere"]
        self.events.reset_from_states.params["probs"] = [1.0]
        self.events.reset_from_states.params["curriculum_target"] = None
        self.events.reset_from_states.params["use_classifier"] = False
        self.events.reset_from_states.params["use_success_critic"] = False

@configclass
class Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherSysidTrainCfg(
    Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherLeanCfg
):
    events: FullSysidDRWristSideEventCfg = FullSysidDRWristSideEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.curriculum = None


@configclass
class Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherLeanNoPrevActCfg(
    Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherLeanCfg
):
    """NoPrevAct variant — student proprio AND teacher proprio drop prev_actions.

    Pair with ``teachers/peg_pc_noprevact_jit.pt``.
    """

    observations: DepthDAggerWristSidePCTeacherNoPrevActObsCfg = (
        DepthDAggerWristSidePCTeacherNoPrevActObsCfg()
    )
