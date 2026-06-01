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
from .rl_state_cfg import (
    RlStateSceneCfg
)
from .depth_dagger_cfg import (
    DAggerWristSideSceneCfg,
    DepthDAggerObservationsCfg,
    FullSysidDRWristSideEventCfg,
    TeacherProprioWithPCCfg,
    Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherSysidTrainCfg,
    WristSideEventCfg,
    joint_pos_zero_gripper,
)
from .data_collection_rgb_cfg import Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg
from .rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg
from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85, EXPLICIT_UR5E_ROBOTIQ_2F85

# IMG_H, IMG_W = 168, 224  # 4:3 finetune / eval of finetuned policies
# IMG_H, IMG_W = 168, 168  # 1:1 (intermediate)
IMG_H, IMG_W = 224, 224  # 1:1 — changed back to 224x224 for full parity with working BC policy

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
        self.history_length = 1
        self.flatten_history_dim = False


@configclass
class _SideRGBCfg(ObsGroup):
    image = _rgb_obs_term("side_camera")

    def __post_init__(self):
        self.concatenate_terms = True
        self.history_length = 1
        self.flatten_history_dim = False


@configclass
class RGBDAggerWristSidePCTeacherObsCfg:
    """7-group obs layout: proprio + side_rgb + wrist_rgb + teacher (ScenePC) + 3 aux pose groups."""

    proprio: DepthDAggerObservationsCfg.ProprioCfg = DepthDAggerObservationsCfg.ProprioCfg()
    side_rgb: _SideRGBCfg = _SideRGBCfg()
    wrist_rgb: _WristRGBCfg = _WristRGBCfg()
    teacher: TeacherProprioWithPCCfg = TeacherProprioWithPCCfg()
    aux_insertive_in_wrist: DepthDAggerObservationsCfg.AuxInsertiveInWristCfg = DepthDAggerObservationsCfg.AuxInsertiveInWristCfg()
    aux_receptive_in_wrist: DepthDAggerObservationsCfg.AuxReceptiveInWristCfg = DepthDAggerObservationsCfg.AuxReceptiveInWristCfg()
    aux_insertive_in_receptive: DepthDAggerObservationsCfg.AuxInsertiveInReceptiveCfg = DepthDAggerObservationsCfg.AuxInsertiveInReceptiveCfg()


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
        self.sim.render.enable_dlssg = True
        self.sim.render.antialiasing_mode = "DLAA"
        self.num_rerenders_on_reset = 1


# ---------------------------------------------------------------------------
# Eval env config
# ---------------------------------------------------------------------------


@configclass
class RGBDAggerEvalEventCfg(WristSideEventCfg):
    """Color-only visual DR for eval: no textures, mild colors on all surfaces.

    No texture randomization (texture_prob=0). Objects, table, and curtains get
    mild near-neutral color variation. Curtains/back wall use near-white colors to
    match a typical lab background. HDRI varies for lighting diversity.
    ``replicate_physics`` must be ``False`` in the scene when this cfg is used.
    """

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

    randomize_sky_light = EventTerm(
        func=task_mdp.randomize_hdri,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "light_path": "/World/skyLight",
            "hdri_config_path": str(Path(__file__).parent / "resources" / "hdri_paths.yaml"),
            "intensity_range": (1500.0, 3500.0),
            "rotation_range": (0.0, 360.0),
        },
    )

@configclass
class DebugObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=joint_pos_zero_gripper)

@configclass
class _PCTeacherOnlyObsCfg:
    """Single obs group for direct JIT teacher evaluation: 25d proprio + 1536d PC = 1561d."""

    policy: TeacherProprioWithPCCfg = TeacherProprioWithPCCfg()
    debug: DebugObsCfg = DebugObsCfg()


@configclass
class Ur5eRobotiq2f85PCTeacherFinetuneEvalCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """FinetuneEval env for direct JIT ScenePC-teacher evaluation (no cameras).

    Identical dynamics to the RGB DAgger eval env (FinetuneEvalEventCfg: fixed sysid +
    OSC gains, ObjectAnywhereEEAnywhere resets, EXPLICIT actuator) but swaps out
    cameras and visual DR for a plain ScenePC obs group. Allows measuring teacher
    performance against the exact dynamics the student is evaluated on.
    """

    observations: _PCTeacherOnlyObsCfg = _PCTeacherOnlyObsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_from_reset_states.params["reset_types"] = ["ZeroGAnywhere"]

def nullify_matching(obj, keywords=("appearance", "wrist", "side")):
    for name in vars(obj):  # or dir(obj) if you need class attrs too
        if any(k in name.lower() for k in keywords):
            setattr(obj, name, None)

@configclass
class DebugEvalCfg(Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidTrainCfg):
    observations: _PCTeacherOnlyObsCfg = _PCTeacherOnlyObsCfg()
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    
    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
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

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True

        # Parent post_init touches scene.side_camera / scene.wrist_camera which
        # don't exist on RlStateSceneCfg. Apply only the non-camera parts manually.
        # IMPLICIT actuator (same as ScenePC teacher's training env, ZeroGSceneCfg).
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        self.curriculum = None

        # take out wrist, side, appearance
        nullify_matching(self.events)
        self.terminations.corrupted_camera = None


@configclass
class Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidEvalCfg(
    Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg
):
    """Eval cfg for the student trained on
    ``Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidTrainCfg``.

    Inherits the Stage-2 finetune-eval env (EXPLICIT actuator, ``RelativeOSCEvalAction``,
    ``FinetuneEvalEventCfg`` fixed sysid + OSC gains, 10-consecutive success termination)
    and layers on the visual machinery the student needs at inference:

      * scene swapped to ``DAggerWristSideSceneCfg`` (= ``RlStateSceneCfg`` + 3 curtains
        + wrist & side RGB ``TiledCameraCfg``s, defaults to ``rgb`` data type),
      * events swapped to ``RGBDAggerEvalEventCfg`` (FinetuneEval sysid gains + zero-delta
        wrist camera pose restore + tame visual DR),
      * observations swapped to ``RGBDAggerWristSidePCTeacherObsCfg`` (proprio +
        side_rgb + wrist_rgb + teacher + aux_target — identical group layout to the
        train cfg, so the student receives its training-time input dict).

    ``replicate_physics=False`` required by ``randomize_visual_appearance_multiple_meshes``.
    """

    scene: DAggerWristSideSceneCfg = DAggerWristSideSceneCfg(
        num_envs=256, env_spacing=1.5, replicate_physics=False
    )
    observations: RGBDAggerWristSidePCTeacherObsCfg = RGBDAggerWristSidePCTeacherObsCfg()
    events: RGBDAggerEvalEventCfg = RGBDAggerEvalEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Cameras default to rgb in DAggerWristSideSceneCfg — no flip needed.
        # Match the train cfg's render settings so the vision encoder sees inputs
        # with the same lighting / AO / reflections it was trained on.
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.antialiasing_mode = "DLAA"
        self.num_rerenders_on_reset = 1


# ---------------------------------------------------------------------------
# DataCollection-based DAgger env: state teacher + 224x224 RGB cameras
# ---------------------------------------------------------------------------


_UR5E_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@configclass
class _ProprioCfgArmOnly(DepthDAggerObservationsCfg.ProprioCfg):
    """ProprioCfg restricted to the 6 UR5e arm joints (gripper dims dropped).

    The real robot only reports the 6 arm joint encoders reliably; gripper
    joint state is rarely available from RGB and can introduce spurious
    correlations. Dropping (not zeroing) matches the real-robot obs dimension
    exactly and avoids a constant-zero input that the normalizer treats as signal.
    Output shape: (B, 6) instead of (B, 12).
    """

    joint_pos = ObsTerm(
        func=task_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_UR5E_ARM_JOINT_NAMES)},
    )


@configclass
class RGBDAggerDataCollectionStateObsCfg:
    """4-group obs layout for state-teacher DAgger from data-collection scene.

    Reuses the RGB image groups from the PC-teacher DAgger env (side_rgb,
    wrist_rgb at IMG_H x IMG_W) and the standard 215d state teacher obs
    (TeacherCfg, history_length=5). The data-collection scene names its
    cameras ``side_camera`` / ``wrist_camera``, matching these groups.
    """

    proprio: _ProprioCfgArmOnly = _ProprioCfgArmOnly()
    side_rgb: _SideRGBCfg = _SideRGBCfg()
    wrist_rgb: _WristRGBCfg = _WristRGBCfg()
    teacher: DepthDAggerObservationsCfg.TeacherCfg = DepthDAggerObservationsCfg.TeacherCfg()


@configclass
class Ur5eRobotiq2f85RGBDAggerDataCollectionStateCfg(Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg):
    """RGB DAgger env based on the data-collection scene with a state teacher.

    Inherits ``Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg``:
      * scene: DataCollectionRGBObjectSceneCfg (curtains + side/wrist 240x320 RGB;
        front_camera removed — not used by student obs groups)
      * events: DataCollectionRGBEventCfg (FinetuneEval sysid base + camera DR +
        full visual DR + 4-reset-type sampling: Anywhere/Grasped/PA/Resting)
      * 224x224 policy input (crop/resize via process_image)

    Swaps in DAgger-compatible obs groups (proprio + side_rgb + wrist_rgb +
    teacher) in place of the DataCollection policy/data_collection groups.

    Pair with RGB_DAggerWristSidePretrainedWeightedRunnerCfg and a state
    expert teacher (e.g. peg_state_rl_expert_finetuned_seed42.pt).

    NOTE: the runner sets teacher_returns_std=True — confirm the teacher JIT
    was exported with --std before launching, otherwise evaluate_with_std fails.
    """

    observations: RGBDAggerDataCollectionStateObsCfg = RGBDAggerDataCollectionStateObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Remove front camera — not used by student obs (side_rgb + wrist_rgb only).
        self.scene.front_camera = None
        # Drop front-camera event terms so they don't fire on a None prim.
        self.events.randomize_front_camera = None
        self.events.randomize_front_camera_focal_length = None
        # Update corrupted_camera termination to only check the two active cameras.
        self.terminations.corrupted_camera.params["camera_names"] = ["side_camera", "wrist_camera"]


# DataCollection-based DAgger env: PC teacher + 224x224 RGB cameras
# ---------------------------------------------------------------------------


@configclass
class RGBDAggerDataCollectionPCTeacherObsCfg:
    """7-group obs layout for PC-teacher DAgger from data-collection scene.

    Same student groups as the state-teacher variant (proprio + side_rgb + wrist_rgb)
    but swaps in TeacherProprioWithPCCfg (25d proprio + 1536d ScenePC = 1561d) to
    match seed23_sysidenv.pt teacher input format. Three standalone aux pose groups
    (each a separate top-level key) are included for the CNN pose-regression aux loss
    — no concatenation, each key has its own independently-shaped tensor.
    """

    proprio: _ProprioCfgArmOnly = _ProprioCfgArmOnly()
    side_rgb: _SideRGBCfg = _SideRGBCfg()
    wrist_rgb: _WristRGBCfg = _WristRGBCfg()
    teacher: TeacherProprioWithPCCfg = TeacherProprioWithPCCfg()
    aux_insertive_in_wrist: DepthDAggerObservationsCfg.AuxInsertiveInWristCfg = DepthDAggerObservationsCfg.AuxInsertiveInWristCfg()
    aux_receptive_in_wrist: DepthDAggerObservationsCfg.AuxReceptiveInWristCfg = DepthDAggerObservationsCfg.AuxReceptiveInWristCfg()
    aux_insertive_in_receptive: DepthDAggerObservationsCfg.AuxInsertiveInReceptiveCfg = DepthDAggerObservationsCfg.AuxInsertiveInReceptiveCfg()


@configclass
class Ur5eRobotiq2f85RGBDAggerDataCollectionPCTeacherCfg(Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg):
    """RGB DAgger env based on the data-collection scene with the ScenePC teacher.

    Identical to Ur5eRobotiq2f85RGBDAggerDataCollectionStateCfg except the teacher
    obs group uses TeacherProprioWithPCCfg (1561d) instead of the 215d state teacher.
    Pair with seed23_sysidenv.pt and RGB_DAggerWristSidePretrainedWeightedRunnerCfg.
    """

    observations: RGBDAggerDataCollectionPCTeacherObsCfg = RGBDAggerDataCollectionPCTeacherObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.front_camera = None
        self.events.randomize_front_camera = None
        self.events.randomize_front_camera_focal_length = None
        self.terminations.corrupted_camera.params["camera_names"] = ["side_camera", "wrist_camera"]