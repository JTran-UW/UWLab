# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL training task with three 128x128 grayscale cameras, for renderer benchmarks (renderer picked by preset)."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import os
import tempfile

from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from ... import mdp as task_mdp
from .data_collection_rgb_cfg import DataCollectionRGBObjectSceneCfg
from .rl_state_cfg import ObservationsCfg, RlStateSceneCfg, Ur5eRobotiq2f85RelCartesianOSCTrainCfg, variants

CAMERA_RES = 128


def _camera(prim_path: str, pos, rot, focal_length: float, moving: bool = False, res: int = CAMERA_RES) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0,
        update_latest_camera_pose=moving,
        height=res,
        width=res,
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal_length),
        renderer_cfg=MultiBackendRendererCfg(),
    )


def grayscale_image(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    rgb = env.scene.sensors[sensor_cfg.name].data.output["rgb"][..., :3].float()
    luma = torch.tensor([0.299, 0.587, 0.114], device=rgb.device)
    return (rgb @ luma).unsqueeze(-1) / 255.0


def apply_isaac_rtx_settings_if_kit(env, env_ids, settings):
    from isaaclab.utils import has_kit

    if has_kit():
        task_mdp.apply_isaac_rtx_settings(env, env_ids, settings)


_RGB_SCENE = DataCollectionRGBObjectSceneCfg(num_envs=1)


def _static_curtain(curtain) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=curtain.prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=curtain.init_state.pos, rot=curtain.init_state.rot),
        spawn=sim_utils.CuboidCfg(size=curtain.spawn.size, visual_material=curtain.spawn.visual_material),
    )


@configclass
class CameraBenchSceneCfg(RlStateSceneCfg):
    # RGB data-collection curtains (as static visuals: Newton makes no body for collision-free kinematic boxes)
    curtain_left = _static_curtain(_RGB_SCENE.curtain_left)
    curtain_back = _static_curtain(_RGB_SCENE.curtain_back)
    curtain_right = _static_curtain(_RGB_SCENE.curtain_right)

    front_camera = _camera(
        "{ENV_REGEX_NS}/Robot/rgb_front_camera",
        (1.0770121, -0.1679045, 0.4486344),
        (0.46613815, 0.25072644, 0.47107948, 0.70564552),
        13.20,
    )
    side_camera = _camera(
        "{ENV_REGEX_NS}/Robot/rgb_side_camera",
        (0.8323904, 0.5877843, 0.2805111),
        (0.22122445, 0.51336143, 0.77676798, 0.29008842),
        20.10,
    )
    wrist_camera = _camera(
        "{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        (0.0182505, -0.00408447, -0.0689107),
        (-0.61819255, -0.6160212, 0.347879, 0.34254336),
        24.55,
        moving=True,
    )


@configclass
class CameraObsCfg(ObsGroup):
    front = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("front_camera")})
    side = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("side_camera")})
    wrist = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("wrist_camera")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CameraBenchObservationsCfg(ObservationsCfg):
    camera: CameraObsCfg = CameraObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCCameraBenchCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    scene: CameraBenchSceneCfg = CameraBenchSceneCfg(num_envs=1024, env_spacing=1.5)
    observations: CameraBenchObservationsCfg = CameraBenchObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        # peg/peghole variants, as selected by the RL training commands
        self.scene.insertive_object = variants["scene.insertive_object"]["peg"].replace()
        self.scene.receptive_object = variants["scene.receptive_object"]["peghole"].replace()
        # the base cfg pins every camera to Isaac RTX; keep its settings as the isaacsim_rtx choice
        for name in ("front_camera", "side_camera", "wrist_camera"):
            cam = getattr(self.scene, name)
            # distinct OVRTX cfgs: equal renderer cfgs share one instance, and an OVRTX instance renders one camera
            if os.environ.get("UWLAB_CAM_RTX_PARTITION") == "1":
                cam.renderer_cfg.enable_scene_partitioning = True
            ovrtx = OVRTXRendererCfg(log_file_path=os.path.join(tempfile.gettempdir(), f"ovrtx_renderer_{name}.log"))
            cam.renderer_cfg = MultiBackendRendererCfg(isaacsim_rtx=cam.renderer_cfg, ovrtx=ovrtx)
        self.events.render_settings.func = apply_isaac_rtx_settings_if_kit
