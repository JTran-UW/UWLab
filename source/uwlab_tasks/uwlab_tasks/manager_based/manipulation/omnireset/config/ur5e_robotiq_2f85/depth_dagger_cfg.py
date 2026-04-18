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
IMG_H, IMG_W = 120, 160
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

    proprio: ProprioCfg = ProprioCfg()
    side_depth: SideDepthCfg = SideDepthCfg()
    teacher: TeacherCfg = TeacherCfg()


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
