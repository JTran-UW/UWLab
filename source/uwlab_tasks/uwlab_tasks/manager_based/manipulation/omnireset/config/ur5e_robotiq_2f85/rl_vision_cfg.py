# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision RL task for the FlashSAC-RGB port: two 64x64 grayscale cameras plus a non-privileged state.

The ``policy`` group keeps the privileged state so the PPO state expert can drive expert-buffer collection;
the vision agent reads only ``camera`` and ``vision_state``. Terminations are removed: the vision wrapper owns
the fixed horizon so every env's episode stays synchronized (required by the episodic image replay buffer).
"""

from __future__ import annotations

import torch

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from ... import mdp as task_mdp
from .physics_presets import joint_pos_obs
from .rl_state_symmetry_cfg import HOLE_SYMMETRY, PEG_SYMMETRY, SymmetryObsCfg
from .rl_camera_bench_cfg import _RGB_SCENE, _camera, _static_curtain, apply_isaac_rtx_settings_if_kit, grayscale_image
from .rl_state_cfg import (
    POSE_OBS_CLIP,
    ObservationsCfg,
    RlStateSceneCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
    variants,
)

VISION_RES = 64
VISION_HORIZON = 160


@configclass
class VisionSceneCfg(RlStateSceneCfg):
    curtain_left = _static_curtain(_RGB_SCENE.curtain_left)
    curtain_back = _static_curtain(_RGB_SCENE.curtain_back)
    curtain_right = _static_curtain(_RGB_SCENE.curtain_right)
    front_camera = _camera(
        "{ENV_REGEX_NS}/Robot/rgb_front_camera",
        (1.0770121, -0.1679045, 0.4486344),
        (0.46613815, 0.25072644, 0.47107948, 0.70564552),
        13.20,
        res=VISION_RES,
    )
    wrist_camera = _camera(
        "{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        (0.0182505, -0.00408447, -0.0689107),
        (-0.61819255, -0.6160212, 0.347879, 0.34254336),
        24.55,
        moving=True,
        res=VISION_RES,
    )


@configclass
class VisionImageObsCfg(ObsGroup):
    front = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("front_camera")})
    wrist = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("wrist_camera")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class VisionStateObsCfg(ObsGroup):
    """Non-privileged state (no object poses): 7 + 12 + 6 = 25-D."""

    prev_actions = ObsTerm(func=task_mdp.last_action)
    joint_pos = joint_pos_obs()
    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "rotation_repr": "axis_angle",
        },
        clip=POSE_OBS_CLIP,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionObservationsCfg(ObservationsCfg):
    camera: VisionImageObsCfg = VisionImageObsCfg()
    vision_state: VisionStateObsCfg = VisionStateObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    scene: VisionSceneCfg = VisionSceneCfg(num_envs=1024, env_spacing=1.5)
    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.scene.insertive_object = variants["scene.insertive_object"]["peg"].replace()
        self.scene.receptive_object = variants["scene.receptive_object"]["peghole"].replace()
        for name in ("front_camera", "wrist_camera"):
            cam = getattr(self.scene, name)
            cam.renderer_cfg = MultiBackendRendererCfg(isaacsim_rtx=cam.renderer_cfg)
        self.events.render_settings.func = apply_isaac_rtx_settings_if_kit
        # the vision wrapper resets all envs together every VISION_HORIZON steps
        self.terminations.time_out = None
        self.episode_length_s = 1.0e6
        self.events.reset_from_reset_states.params["max_states"] = None
        self.events.reset_from_reset_states.params["max_joint_vel"] = -1.0
        # symmetric success is opt-in (env.commands.task_command.symmetric_success=true)
        self.commands.task_command.insertive_symmetry = PEG_SYMMETRY
        self.commands.task_command.receptive_symmetry = HOLE_SYMMETRY


@configclass
class Vision3CamSceneCfg(VisionSceneCfg):
    side_camera = _camera(
        "{ENV_REGEX_NS}/Robot/rgb_side_camera",
        (0.8323904, 0.5877843, 0.2805111),
        (0.22122445, 0.51336143, 0.77676798, 0.29008842),
        20.10,
        res=VISION_RES,
    )


@configclass
class Vision3CamImageObsCfg(VisionImageObsCfg):
    side = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("side_camera")})


@configclass
class Vision3CamObservationsCfg(VisionObservationsCfg):
    camera: Vision3CamImageObsCfg = Vision3CamImageObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVision3CamCfg(Ur5eRobotiq2f85RelCartesianOSCVisionCfg):
    """Front + side + wrist cameras (the RGB data-collection set) at 64x64 grayscale."""

    scene: Vision3CamSceneCfg = Vision3CamSceneCfg(num_envs=1024, env_spacing=1.5)
    observations: Vision3CamObservationsCfg = Vision3CamObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        cam = self.scene.side_camera
        cam.renderer_cfg = MultiBackendRendererCfg(isaacsim_rtx=cam.renderer_cfg)


# Symmetry declarations of the jtran-uw main-branch assets the symmetry-invariant PPO expert was trained with.


# the symmetric state expert's policy obs, exactly as it was trained (State-Symmetry-v0)
SymExpertPolicyObsCfg = SymmetryObsCfg


@configclass
class VisionSymObservationsCfg(VisionObservationsCfg):
    policy: SymExpertPolicyObsCfg = SymExpertPolicyObsCfg()


@configclass
class Vision3CamSymObservationsCfg(Vision3CamObservationsCfg):
    policy: SymExpertPolicyObsCfg = SymExpertPolicyObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionSymCfg(Ur5eRobotiq2f85RelCartesianOSCVisionCfg):
    """Vision task whose ``policy`` group is the symmetry-invariant expert observation (expert collection only)."""

    observations: VisionSymObservationsCfg = VisionSymObservationsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVision3CamSymCfg(Ur5eRobotiq2f85RelCartesianOSCVision3CamCfg):
    observations: Vision3CamSymObservationsCfg = Vision3CamSymObservationsCfg()
