# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB DAgger env configs: same structure as the sysid depth-DAgger envs but with RGB cameras and visual DR.

Mirrors ``Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherSysidTrainCfg`` from
``depth_dagger_cfg.py`` with three changes:
  * cameras render RGB instead of depth (no clipping / noising in obs terms)
  * camera pose + focal length randomization on every reset
  * full visual DR (texture + HDRI) from ``data_collection_rgb_cfg.py``
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .depth_dagger_cfg import (
    DepthDAggerObservationsCfg,
    FullSysidDRWristSideEventCfg,
    TeacherProprioWithPCCfg,
    Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherSysidTrainCfg,
)

IMG_H, IMG_W = 224, 224
# IMG_H, IMG_W = 168, 168

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Event config: sysid full-DR + camera pose/focal/visual randomization
# ---------------------------------------------------------------------------


@configclass
class RGBDAggerSysidEventCfg(FullSysidDRWristSideEventCfg):
    """Sysid full-DR events with RGB camera pose / focal / visual appearance randomization.

    Inherits all ZeroG GPS sysid DR terms from ``FullSysidDRWristSideEventCfg``.
    Overrides ``reset_wrist_camera_pose`` (zero-delta restore → small jitter),
    adds side-camera pose and focal randomization, and adds all visual appearance
    and HDRI DR terms from ``RGBEventCfg`` in ``data_collection_rgb_cfg.py``.

    NOTE: the side camera prim is named ``depth_side_camera`` (inherited from
    ``DAggerWristSideSceneCfg``) even when rendering RGB — the name is structural.
    """

    # Override: apply small jitter instead of zero-delta pose restore.
    reset_wrist_camera_pose = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera",
            "base_position": (0.0182505, -0.00408447, -0.0689107),
            "base_rotation": (0.34254336, -0.61819255, -0.6160212, 0.347879),
            "position_deltas": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            "euler_deltas": {"pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0), "roll": (-1.0, 1.0)},
        },
    )

    randomize_wrist_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera",
            "focal_length_range": (23.55, 25.55),
        },
    )

    randomize_side_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/depth_side_camera",
            "base_position": (0.8323904, 0.5877843, 0.2805111),
            "base_rotation": (0.29008842, 0.22122445, 0.51336143, 0.77676798),
            "position_deltas": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            "euler_deltas": {"pitch": (-2.0, 2.0), "yaw": (-2.0, 2.0), "roll": (-2.0, 2.0)},
        },
    )

    randomize_side_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/depth_side_camera",
            "focal_length_range": (18.1, 22.1),
        },
    )

    # --- Visual appearance DR (interval, copied from RGBEventCfg) ---

    randomize_wrist_mount_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "event_name": "randomize_wrist_mount_event",
            "mesh_names": ["robotiq_base_link/visuals/D415_to_Robotiq_Mount"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.2, 1.0),
            "metallic_range": (0.0, 0.8),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_inner_finger_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "event_name": "randomize_inner_finger_event",
            "mesh_names": ["left_inner_finger/visuals/mesh_1", "right_inner_finger/visuals/mesh_1"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.2, 1.0),
            "metallic_range": (0.0, 0.8),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_insertive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "event_name": "randomize_insertive_object_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_receptive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "event_name": "randomize_receptive_object_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_table_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "event_name": "randomize_table_event",
            "mesh_names": ["visuals/vention_mat"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.3, 0.9),
            "metallic_range": (0.0, 0.3),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_left_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_left"),
            "event_name": "randomize_curtain_left_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_back_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_back"),
            "event_name": "randomize_curtain_back_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_right_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_right"),
            "event_name": "randomize_curtain_right_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_sky_light = EventTerm(
        func=task_mdp.randomize_hdri,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "light_path": "/World/skyLight",
            "hdri_config_path": str(Path(__file__).parent / "resources" / "hdri_paths.yaml"),
            "intensity_range": (1000.0, 4000.0),
            "rotation_range": (0.0, 360.0),
        },
    )


# ---------------------------------------------------------------------------
# Observation groups
# ---------------------------------------------------------------------------


@configclass
class _WristRGBCfg(ObsGroup):
    image = _rgb_obs_term("wrist_camera")

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class _SideRGBCfg(ObsGroup):
    image = _rgb_obs_term("side_camera")

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class RGBDAggerWristSidePCTeacherObsCfg:
    """4-group obs layout: proprio + side_rgb + wrist_rgb + teacher (ScenePC)."""

    proprio: DepthDAggerObservationsCfg.ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg()
    side_rgb: _SideRGBCfg = _SideRGBCfg()
    wrist_rgb: _WristRGBCfg = _WristRGBCfg()
    teacher: TeacherProprioWithPCCfg = TeacherProprioWithPCCfg()
    aux_target: DepthDAggerObservationsCfg.AuxTargetCfg = DepthDAggerObservationsCfg.AuxTargetCfg()


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------


@configclass
class Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidTrainCfg(
    Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherSysidTrainCfg
):
    """RGB variant of the PC-teacher sysid DAgger env (wrist + side cameras).

    Identical to ``Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherSysidTrainCfg``
    except cameras render RGB. The parent chain flips ``data_types`` to depth
    in ``Ur5eRobotiq2f85DepthDAggerWristSideCfg.__post_init__``; this class
    restores them to RGB and enables higher-quality render settings.
    """

    observations: RGBDAggerWristSidePCTeacherObsCfg = RGBDAggerWristSidePCTeacherObsCfg()
    events: RGBDAggerSysidEventCfg = RGBDAggerSysidEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Parent chain flipped cameras to depth — restore to RGB.
        self.scene.side_camera.data_types = ["rgb"]
        self.scene.wrist_camera.data_types = ["rgb"]
        # randomize_visual_appearance_multiple_meshes requires replicate_physics=False.
        # Parent DAgger scene uses True; override here.
        self.scene.replicate_physics = False
        # Enable quality render settings needed for visual fidelity.
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.antialiasing_mode = "DLAA"
        self.num_rerenders_on_reset = 1