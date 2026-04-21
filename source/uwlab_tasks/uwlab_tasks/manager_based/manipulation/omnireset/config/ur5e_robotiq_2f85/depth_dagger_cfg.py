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

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85RelativeOSCEvalAction
from .rl_state_cfg import FinetuneEvalEventCfg, RlStateSceneCfg, Ur5eRobotiq2f85RlStateCfg

DEPTH_CLIP = (0.01, 2.0)
IMG_H, IMG_W = 224, 224
RENDER_H, RENDER_W = 224, 224


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

def _depth_obs_term(sensor_name: str) -> ObsTerm:
    return ObsTerm(
        func=task_mdp.process_image,
        params={
            "sensor_cfg": SceneEntityCfg(sensor_name),
            "data_type": "distance_to_camera",
            "process_image": True,
            "output_size": (IMG_H, IMG_W),
            "depth_clip": DEPTH_CLIP,
        },
    )


def _rgb_obs_term(sensor_name: str) -> ObsTerm:
    return ObsTerm(
        func=task_mdp.process_image,
        params={
            "sensor_cfg": SceneEntityCfg(sensor_name),
            "data_type": "rgb",
            "process_image": True,
            "output_size": (IMG_H, IMG_W),
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
        image = _depth_obs_term("side_camera")

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

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)
    early_success = DoneTerm(
        func=task_mdp.early_success_termination,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )
    success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )


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


@configclass
class RgbDAggerObservationsCfg:
    """RGB variant: swap ``side_depth`` for ``side_rgb`` (3×H×W). Proprio + teacher unchanged."""

    ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg
    TeacherCfg = DepthDAggerObservationsCfg.TeacherCfg

    @configclass
    class SideRgbCfg(ObsGroup):
        image = _rgb_obs_term("side_camera")

        def __post_init__(self):
            self.concatenate_terms = True

    proprio: DepthDAggerObservationsCfg.ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg()
    side_rgb: SideRgbCfg = SideRgbCfg()
    teacher: DepthDAggerObservationsCfg.TeacherCfg = DepthDAggerObservationsCfg.TeacherCfg()
    aux_target: DepthDAggerObservationsCfg.AuxTargetCfg = DepthDAggerObservationsCfg.AuxTargetCfg()


@configclass
class Ur5eRobotiq2f85RgbDAggerRelCartesianOSCCfg(Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg):
    """RGB DAgger env: side camera renders RGB instead of depth. Everything else identical."""

    observations: RgbDAggerObservationsCfg = RgbDAggerObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Swap the side camera from depth to rgb.
        self.scene.side_camera.data_types = ["rgb"]


# ---------------------------------------------------------------------------
# 2-camera variants: side + front (front offset from data_collection_rgb_cfg.py)
# ---------------------------------------------------------------------------


@configclass
class DepthDAgger2CamSceneCfg(DepthDAggerSceneCfg):
    """Base DAgger scene + a front TiledCamera (pose copied from data_collection_rgb_cfg)."""

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_front_camera",
        update_period=0,
        height=RENDER_H,
        width=RENDER_W,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20, clipping_range=DEPTH_CLIP),
    )


@configclass
class DepthDAgger2CamObservationsCfg:
    """2-camera depth: side_depth + front_depth + proprio + teacher."""

    @configclass
    class FrontDepthCfg(ObsGroup):
        image = _depth_obs_term("front_camera")

        def __post_init__(self):
            self.concatenate_terms = True

    proprio: DepthDAggerObservationsCfg.ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg()
    side_depth: DepthDAggerObservationsCfg.SideDepthCfg = DepthDAggerObservationsCfg.SideDepthCfg()
    front_depth: FrontDepthCfg = FrontDepthCfg()
    teacher: DepthDAggerObservationsCfg.TeacherCfg = DepthDAggerObservationsCfg.TeacherCfg()
    aux_target: DepthDAggerObservationsCfg.AuxTargetCfg = DepthDAggerObservationsCfg.AuxTargetCfg()


@configclass
class RgbDAgger2CamObservationsCfg:
    """2-camera rgb: side_rgb + front_rgb + proprio + teacher."""

    @configclass
    class FrontRgbCfg(ObsGroup):
        image = _rgb_obs_term("front_camera")

        def __post_init__(self):
            self.concatenate_terms = True

    proprio: DepthDAggerObservationsCfg.ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg()
    side_rgb: RgbDAggerObservationsCfg.SideRgbCfg = RgbDAggerObservationsCfg.SideRgbCfg()
    front_rgb: FrontRgbCfg = FrontRgbCfg()
    teacher: DepthDAggerObservationsCfg.TeacherCfg = DepthDAggerObservationsCfg.TeacherCfg()
    aux_target: DepthDAggerObservationsCfg.AuxTargetCfg = DepthDAggerObservationsCfg.AuxTargetCfg()


@configclass
class Ur5eRobotiq2f85DepthDAgger2CamCfg(Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg):
    """2-camera depth DAgger: adds a front depth cam alongside the side depth cam."""

    scene: DepthDAgger2CamSceneCfg = DepthDAgger2CamSceneCfg(
        num_envs=256, env_spacing=1.5, replicate_physics=True
    )
    observations: DepthDAgger2CamObservationsCfg = DepthDAgger2CamObservationsCfg()


@configclass
class Ur5eRobotiq2f85RgbDAgger2CamCfg(Ur5eRobotiq2f85DepthDAgger2CamCfg):
    """2-camera RGB DAgger: both side + front cameras render RGB."""

    observations: RgbDAgger2CamObservationsCfg = RgbDAgger2CamObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.side_camera.data_types = ["rgb"]
        self.scene.front_camera.data_types = ["rgb"]


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
class RgbDAggerWristSideObservationsCfg:
    """2-camera RGB (side + wrist) obs layout. Other groups identical to 1cam RGB."""

    @configclass
    class WristRgbCfg(ObsGroup):
        image = _rgb_obs_term("wrist_camera")

        def __post_init__(self):
            self.concatenate_terms = True

    proprio: DepthDAggerObservationsCfg.ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg()
    side_rgb: RgbDAggerObservationsCfg.SideRgbCfg = RgbDAggerObservationsCfg.SideRgbCfg()
    wrist_rgb: WristRgbCfg = WristRgbCfg()
    teacher: DepthDAggerObservationsCfg.TeacherCfg = DepthDAggerObservationsCfg.TeacherCfg()
    aux_target: DepthDAggerObservationsCfg.AuxTargetCfg = DepthDAggerObservationsCfg.AuxTargetCfg()


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
class Ur5eRobotiq2f85RgbDAggerWristSideCfg(Ur5eRobotiq2f85RgbDAggerRelCartesianOSCCfg):
    """2-camera RGB DAgger with a wrist camera (parented to gripper base link) + side cam."""

    scene: DAggerWristSideSceneCfg = DAggerWristSideSceneCfg(
        num_envs=256, env_spacing=1.5, replicate_physics=True
    )
    observations: RgbDAggerWristSideObservationsCfg = RgbDAggerWristSideObservationsCfg()
    events: WristSideEventCfg = WristSideEventCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class DepthDAggerWristSideObservationsCfg:
    """2-camera depth (side + wrist) obs layout."""

    @configclass
    class WristDepthCfg(ObsGroup):
        image = _depth_obs_term("wrist_camera")

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
# Recurrent-student variant: proprio single-frame (LSTM supplies temporal context).
# Everything else identical to ``Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg``.
# ---------------------------------------------------------------------------


@configclass
class DepthDAggerRecurrentObservationsCfg(DepthDAggerObservationsCfg):
    """Same obs groups as depth DAgger, but proprio ``history_length=1``."""

    @configclass
    class ProprioSingleFrameCfg(DepthDAggerObservationsCfg.ProprioCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    proprio: ProprioSingleFrameCfg = ProprioSingleFrameCfg()


@configclass
class Ur5eRobotiq2f85DepthDAggerRecurrentCfg(Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg):
    """Depth DAgger env with single-frame proprio (for LSTM student)."""

    observations: DepthDAggerRecurrentObservationsCfg = DepthDAggerRecurrentObservationsCfg()
