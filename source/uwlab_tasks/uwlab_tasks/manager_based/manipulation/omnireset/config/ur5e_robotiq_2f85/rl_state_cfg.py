# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

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
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
 
from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85, IMPLICIT_UR5E_ROBOTIQ_2F85

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions import (
    Ur5eRobotiq2f85RelativeOSCAction,
    Ur5eRobotiq2f85RelativeOSCEvalAction,
)

from ... import mdp as task_mdp


def _look_at_quat(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for an OpenGL-convention camera (forward=-Z, up=+Y) at `pos`
    looking at `target`."""
    zx, zy, zz = pos[0] - target[0], pos[1] - target[1], pos[2] - target[2]
    zlen = math.sqrt(zx * zx + zy * zy + zz * zz)
    zx, zy, zz = zx / zlen, zy / zlen, zz / zlen

    ux, uy, uz = world_up
    xx, xy, xz = uy * zz - uz * zy, uz * zx - ux * zz, ux * zy - uy * zx
    xlen = math.sqrt(xx * xx + xy * xy + xz * xz)
    xx, xy, xz = xx / xlen, xy / xlen, xz / xlen

    yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx

    # rotation matrix columns are the camera's local x/y/z axes expressed in world coords
    m00, m01, m02 = xx, yx, zx
    m10, m11, m12 = xy, yy, zy
    m20, m21, m22 = xz, yz, zz

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s

    return (w, x, y, z)


@configclass
class RlStateSceneCfg(InteractiveSceneCfg):
    """Scene configuration for RL state environment."""

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

    # Environment
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

@configclass
class RlStateReachingSceneCfg(InteractiveSceneCfg):
    """Scene configuration for RL state environment."""

    robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    target_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
        ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.3, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Environment
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

    # front_camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/depth_front_camera",
    #     update_period=0,
    #     height=240,
    #     width=320,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.0770121, -0.1679045, 0.4486344),
    #         rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
    #         convention="opengl",
    #     ),
    #     data_types=["distance_to_camera"],
    #     spawn=sim_utils.PinholeCameraCfg(focal_length=13.20, clipping_range=(0.1, 1.25)),
    # )

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_front_camera",
        update_period=0,
        height=240,
        width=320,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.5, 0, 0.75),
            rot=_look_at_quat(pos=(1.5, 0, 0.75), target=(0, 0, 0)),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20, clipping_range=(0.1, 2.0)),
    )


@configclass
class RlStateReachingGrayscaleSceneCfg(RlStateReachingSceneCfg):
    """Reaching scene with the peg-insertion three-camera rig (front / side / wrist) in rgb.

    Camera poses, focal lengths, and prim paths are taken verbatim from
    ``data_collection_rgb_cfg.DataCollectionRGBObjectSceneCfg`` (the DAgger rig) so a policy trained
    here sees the same viewpoints as the peg-insertion task. Images are converted to grayscale at the
    observation level (see ``ObservationsReachingGrayscaleCfg``), not by the sensor.

    Rendered at 168x126 rather than the DAgger rig's 320x240: the observation is downsampled to
    84x84 anyway, so 320x240 renders ~3.6x more pixels than are ever used. The 4:3 aspect is kept
    deliberately -- ``vertical_aperture`` defaults to ``horizontal_aperture * height/width``, so
    changing the aspect (e.g. rendering a square 84x84) would silently widen the vertical FOV and
    change what the cameras see. 168x126 still supersamples ~1.5x into the 84x84 output, which keeps
    thin structures like the peg from aliasing away.
    """

    # Black backdrop panels around the workspace, matching the DAgger rig
    # (data_collection_rgb_cfg.DataCollectionRGBObjectSceneCfg). Without them the front and side
    # cameras see straight past the workspace into neighbouring environments, so the background
    # becomes a function of env index and grid position -- spurious structure a vision policy can
    # latch onto, and a domain gap against the real cell. Kinematic, collision disabled: purely
    # visual, no effect on physics.
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

    # Declared explicitly because this `init_state` IS the center that `randomize_target_marker`
    # samples around -- `reset_root_state_uniform` offsets from `default_root_state`. Keeping it here
    # (rather than inheriting) pins the randomization center next to the task that depends on it.
    # The reset dataset's fixed (0.3, 0.0, 0.3) write is overwritten by that event, so it is not the
    # pose the marker ends up at.
    target_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.3, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.10),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
    )


@configclass
class RlStateGrayscaleSceneCfg(RlStateSceneCfg):
    """Peg-insertion scene plus the three-camera rig (front / side / wrist), for grayscale collection.

    Camera poses, focal lengths and prim paths are taken verbatim from
    ``data_collection_rgb_cfg.DataCollectionRGBObjectSceneCfg`` (the DAgger rig), and the black
    curtains come with them -- without a backdrop the peg scene renders the skybox behind the table
    and the images would not match the RGB collection distribution. Declared inline rather than
    inherited because ``data_collection_rgb_cfg`` imports *from* this module; the reaching grayscale
    scene below does the same.

    Rendered at 168x126 rather than the DAgger rig's 320x240: the observation is downsampled to
    84x84 anyway, so 320x240 renders ~3.6x more pixels than are ever used. The 4:3 aspect is kept
    deliberately -- ``vertical_aperture`` defaults to ``horizontal_aperture * height/width``, so
    changing the aspect would silently widen the vertical FOV and change what the cameras see.
    """

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

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.10),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
    )


@configclass
class RlStateDepthSceneCfg(RlStateGrayscaleSceneCfg):
    """Same three-camera rig as ``RlStateGrayscaleSceneCfg`` but rendering depth instead of rgb.

    Subclassed rather than re-declared so the poses, focal lengths, resolution and the black
    curtains stay in exact correspondence with the grayscale rig -- only ``data_types`` and the prim
    names change. The curtains are kept deliberately: they give the background a finite depth
    (~1.3 m) instead of leaving it unhit, which ``process_image`` would map to 0.0 and make
    indistinguishable from a surface at the near plane.

    ``clipping_range`` is set because depth is consumed as raw metric distance, so the far plane
    directly bounds the observation's value range. (0.1, 2.0) follows the existing depth camera in
    this module and comfortably covers the workspace -- the furthest curtain is ~1.3 m from the
    front camera. Anything past 2.0 m returns inf and is mapped to 0.0.
    """

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_front_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20, clipping_range=(0.1, 2.0)),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/depth_side_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.10, clipping_range=(0.1, 2.0)),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/depth_wrist_camera",
        update_period=0,
        height=126,
        width=168,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55, clipping_range=(0.1, 2.0)),
    )


@configclass
class RlStateGrayscale2CamSceneCfg(RlStateGrayscaleSceneCfg):
    """Grayscale rig with the FRONT camera removed -- side + wrist only.

    ``front_camera = None`` drops the sensor from the scene entirely, so it is never rendered.
    Removing the camera only from the observation term would keep paying its render cost, which is
    the whole point of this ablation: rendering is the dominant per-env cost (measured 0.70 ms/env
    for the 3-camera grayscale rig), so a third fewer cameras should cut roughly a third of it.

    Side and wrist are kept rather than front because front and side are near-duplicate external
    views (both ~85% dark backdrop in the collected buffer), while wrist is the only close-range
    view of the grasp.
    """

    front_camera = None


@configclass
class RlStateGrayscale2CamLowResSceneCfg(RlStateGrayscale2CamSceneCfg):
    """Side + wrist rig rendered at 112x84 instead of 168x126 -- 2.25x fewer rendered pixels.

    Every observation is downsampled to 84x84 regardless, so 168x126 (21,168 px) renders 3x more
    pixels than are ever consumed. 112x84 (9,408 px) is the smallest 4:3 size that still covers
    84x84 without upsampling in either axis.

    The 4:3 aspect is preserved deliberately: ``vertical_aperture`` defaults to None and is derived
    from ``horizontal_aperture * height/width``, so changing the aspect would silently alter the
    vertical FOV and change what the cameras see. Pose and focal lengths are untouched, so this is
    purely a rasterization-cost change -- the observation dim is unchanged at 14,112.
    """

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=84,
        width=112,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.10),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=84,
        width=112,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
    )


@configclass
class BaseEventWithDynamicsGapCfg:
    """Shared events: material/mass randomization, gripper gains, scene reset.

    Does NOT include arm sysid or OSC gain randomization -- those differ
    between finetune (curriculum-ramped) and eval (fixed) stages.  See
    ``FinetuneEventCfg`` and ``FinetuneEvalEventCfg``.
    """

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (0.8, 0.8),
        },
    )

    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})


@configclass
class BaseEventCfg:
    """Shared events: material/mass randomization, gripper gains, scene reset.

    Does NOT include arm sysid or OSC gain randomization -- those differ
    between finetune (curriculum-ramped) and eval (fixed) stages.  See
    ``FinetuneEventCfg`` and ``FinetuneEvalEventCfg``.
    """

    # mode: startup (randomize dynamics)
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="reset",
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
        mode="reset",
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
        mode="reset",
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
        mode="reset",
        params={
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.2, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="reset",
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
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            # we assume insertive object is somewhere between 20g and 200g
            "mass_distribution_params": (0.02, 0.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_receptive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="reset",
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
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

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

    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

@configclass
class BaseEventNoDRCfg:
    """Shared events: material/mass randomization, gripper gains, scene reset.

    Does NOT include arm sysid or OSC gain randomization -- those differ
    between finetune (curriculum-ramped) and eval (fixed) stages.  See
    ``FinetuneEventCfg`` and ``FinetuneEvalEventCfg``.
    """

    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

    
@configclass
class BaseEventNoDR_6bdbe5e_Cfg:
    """Shared events: material/mass randomization, gripper gains, scene reset.

    Does NOT include arm sysid or OSC gain randomization -- those differ
    between finetune (curriculum-ramped) and eval (fixed) stages.  See
    ``FinetuneEventCfg`` and ``FinetuneEvalEventCfg``.
    """

    # mode: startup (randomize dynamics)
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.3), # (0.3, 1.2),
            "dynamic_friction_range": (0.5, 0.5), # (0.2, 1.0),
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
            "static_friction_range": (1.5, 1.5), # (1.0, 2.0),
            "dynamic_friction_range": (1.5, 1.5), #  (0.9, 1.9),
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
            "static_friction_range": (0.4, 0.4), # (0.2, 0.6),
            "dynamic_friction_range": (0.3, 0.3), # (0.15, 0.5),
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
            "static_friction_range": (0.4, 0.4), # (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.3), # (0.2, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 0.7), # (0.7, 1.3),
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
            # we assume insertive object is somewhere between 20g and 200g
            "mass_distribution_params": (0.1, 0.1), # (0.02, 0.2),
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
            "mass_distribution_params": (0.5, 0.5), # (0.5, 1.5),
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
            "mass_distribution_params": (1.0, 1.0), # (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "stiffness_distribution_params": (1.5, 1.5), # (0.5, 2.0),
            "damping_distribution_params": (1.5, 1.5), # (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

@configclass
class BaseReachingEventCfg:
    """Shared events: material/mass randomization, gripper gains, scene reset.

    Does NOT include arm sysid or OSC gain randomization -- those differ
    between finetune (curriculum-ramped) and eval (fixed) stages.  See
    ``FinetuneEventCfg`` and ``FinetuneEvalEventCfg``.
    """

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
    # mode: reset
    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})

@configclass
class TrainEventCfg(BaseEventCfg):
    """Training events: material/mass randomization + 4-path resets. No sysid or OSC gain randomization."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class TrainEventPegMassGapFullResetCfg(TrainEventCfg):
    """``TrainEventCfg`` with the peg pinned to 500 g, keeping the full 4-path reset mixture.

    ``randomize_insertive_object_mass`` REPLACES the inherited term, so both value and mode change:
    the base samples 0.02-0.2 kg fresh on every reset, this pins 0.5 kg once at startup. That is
    2.5x the heaviest peg seen in training and 25x the lightest, held fixed for the whole run.

    Everything else -- resets, materials, the other mass events -- is inherited untouched, so peg
    mass is the ONLY difference from the base task. Use this when the dynamics gap has to be
    isolated from any change of initial-state distribution.
    """

    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "mass_distribution_params": (0.5, 0.5),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )


@configclass
class TrainEventPegMassGapCfg(TrainEventPegMassGapFullResetCfg):
    """The peg-mass gap PLUS resets restricted to ``ObjectAnywhereEEAnywhere``.

    Inherits the mass override from its parent rather than restating it, so the two gap variants
    cannot drift apart if the pinned mass is retuned. Only the reset distribution is added here:
    every episode starts from the hardest path (object loose, gripper empty and anywhere) instead
    of a quarter of them starting already grasped or partly assembled.

    Matches ``TrainEvalEventAnywhereOnlyPegMassGapCfg`` term for term, so this finetune and its
    eval env see the same initial-state distribution.
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class TrainEventNoDRCfg(BaseEventNoDRCfg):
    """Training events: material/mass randomization + 4-path resets. No sysid or OSC gain randomization."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class TrainEventNoDR_6bdbe5e_Cfg(BaseEventNoDR_6bdbe5e_Cfg):
    """Training events: material/mass randomization + 4-path resets. No sysid or OSC gain randomization."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class TrainEventWithDynamicsGapCfg(BaseEventWithDynamicsGapCfg):
    """Training events: material/mass randomization + 4-path resets. No sysid or OSC gain randomization."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere"
            ],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class TrainEventWithSuboptimalCfg(BaseEventNoDRCfg):
    """Training events: material/mass randomization + 4-path resets. No sysid or OSC gain randomization."""

    # TEMPORARILY SETTING THIS TO FULL RESET DISTRIBUTION
    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

import numpy as np
@configclass
class TrainReachingEventCfg(BaseReachingEventCfg):
    """Reaching training events: scene reset to default only (fixed target, fixed arm init)."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.SingleResetManager,
        mode="reset",
        params={
            "dataset_dir": "Datasets/OmniReset/Resets/resets_reaching.pt",
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class TrainReachingGrayscaleEventCfg(TrainReachingEventCfg):
    """Reaching training events + target randomization.

    ``resets_reaching.pt`` pins ``target_marker`` to a single pose (0.3, 0.0, 0.3) across all 10675
    reset states, so with the inherited events alone the target never moves and a vision policy has
    nothing to localize. This term resamples it each reset, around the scene's ``init_state``.

    Ordering matters: ``EventManager`` iterates ``cfg.__dict__`` in declaration order and dataclass
    inheritance puts base-class fields first, so this term is applied *after* the inherited
    ``reset_from_reset_states`` and its write wins. Declaring it in the base class instead would let
    the dataset overwrite it.
    """

    randomize_target_marker = EventTerm(
        func=task_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target_marker"),
            # Offsets from init_state (0.5, 0.3, 0.1) -> x in [0.41, 0.59], y in [0.17, 0.43],
            # z in [0.05, 0.32]. Every pose in this box is kinematically reachable by the UR5e:
            # the whole box sits between 0.42 m and 0.73 m from the shoulder in the arm plane
            # (<= 90% of the 0.817 m full extension, so clear of the elbow singularity) and never
            # closer than 0.44 m to the base axis, so it avoids the base dead cylinder. x/y are kept
            # symmetric so the sampled distribution stays centered on init_state; z is clipped
            # asymmetrically because the center sits low (0.1 m) and dropping much below that would
            # put the target at the robot's base plane.
            # To pin the marker at the center for a visual check, set every range to (0.0, 0.0) --
            # but keep the term, or the dataset's fixed (0.3, 0.0, 0.3) wins.
            "pose_range": {"x": (-0.09, 0.09), "y": (-0.13, 0.13), "z": (-0.05, 0.22)},
            # Marker is kinematic with gravity disabled; keep it at rest.
            "velocity_range": {},
        },
    )


@configclass
class TrainEasyEventCfg(BaseEventCfg):
    """Training events: material/mass randomization + 4-path resets. No sysid or OSC gain randomization."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class TrainEasyEventNoDRCfg(BaseEventNoDRCfg):
    """Training events: material/mass randomization + 4-path resets. No sysid or OSC gain randomization."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class TrainEvalEventCfg(BaseEventCfg):
    """Eval after Stage 1: no sysid/OSC gain randomization, 1-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class TrainEvalEventAnywhereOnlyCfg(TrainEventCfg):
    """``TrainEventCfg`` with resets restricted to ``ObjectAnywhereEEAnywhere`` (prob 1.0).

    Material/mass events are inherited unchanged, so the initial-state distribution is the only
    difference from the training task.
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class TrainEvalEventAnywhereOnlyPegMassGapCfg(TrainEvalEventAnywhereOnlyCfg):
    """``TrainEvalEventAnywhereOnlyCfg`` with the peg pinned to 500 g.

    Same single-path resets as its parent, so this and ``TrainEvalEventAnywhereOnlyCfg`` differ only
    in peg mass -- the eval-side counterpart of ``TrainEventPegMassGapCfg``.
    """

    # randomize_osc_gains = EventTerm(
    #     func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
    #     mode="reset",
    #     params={
    #         "action_name": "arm",
    #         "scale_range": (0.6, 0.6),
    #     },
    # )


    # randomize_robot_mass = EventTerm(
    #     func=task_mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mass_distribution_params": (1.6, 1.6),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #         "recompute_inertia": True,
    #     },
    # )


    randomize_insertive_object_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            # we assume insertive object is somewhere between 20g and 200g
            "mass_distribution_params": (0.5, 0.5), # (0.02, 0.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )


@configclass
class TrainEvalEventNoDRCfg(BaseEventNoDRCfg):
    """Eval after Stage 1: no sysid/OSC gain randomization, 1-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class TrainEvalEventNoDRWristCamCfg(TrainEvalEventNoDRCfg):
    """``TrainEvalEventNoDRCfg`` plus a reset-time re-author of the wrist camera's transform.

    Required for the wrist camera to render at all. A TiledCamera parented to a moving articulation
    link renders the skybox in these peg-derived scenes unless something writes its ``xformOp`` on
    the prim; without this term the wrist view is a static image of the sky while front/side (which
    hang off the static ``/Robot`` root) are unaffected. The DAgger RGB rig only works because its
    ``randomize_wrist_camera`` term happens to perform that write.

    Deltas are zero on purpose: this is a deterministic re-placement at the pose the scene cfg
    already specifies, not domain randomization -- these tasks are explicitly no-DR. Verified by
    A/B: identical robot pose, wrist frame brightness 213.6 (skybox) without this term vs 20.6
    (gripper and peg in view) with it.

    Declared in a subclass so it runs AFTER the inherited reset terms -- EventManager iterates
    ``cfg.__dict__`` in declaration order, base fields first, so the camera is placed once the
    robot is already in its reset pose.
    """

    place_wrist_camera = EventTerm(
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
class TrainEvalEventNoDRWristCamDepthCfg(TrainEvalEventNoDRWristCamCfg):
    """As above, but targeting the depth rig's prim name (``depth_wrist_camera``).

    The path is matched by name, so the rgb-named term would silently no-op here --
    ``randomize_tiled_cameras`` skips prims it cannot resolve.
    """

    place_wrist_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/robotiq_base_link/depth_wrist_camera",
            "base_position": (0.0182505, -0.00408447, -0.0689107),
            "base_rotation": (0.34254336, -0.61819255, -0.6160212, 0.347879),
            "position_deltas": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "euler_deltas": {"pitch": (0.0, 0.0), "yaw": (0.0, 0.0), "roll": (0.0, 0.0)},
        },
    )


@configclass
class TrainEvalEventNoDR_6bdbe5e_Cfg(BaseEventNoDR_6bdbe5e_Cfg):
    """Eval after Stage 1: no sysid/OSC gain randomization, 1-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class TrainEvalEasyEventCfg(BaseEventCfg):
    """Eval after Stage 1: no sysid/OSC gain randomization, 1-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectPartiallyAssembledEEGrasped"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class TrainEvalEasyEventNoDRCfg(BaseEventNoDRCfg):
    """Eval after Stage 1: no sysid/OSC gain randomization, 1-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectPartiallyAssembledEEGrasped"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class TrainEvalEventWithDynamicsGapCfg(BaseEventWithDynamicsGapCfg):
    """Eval after Stage 1: no sysid/OSC gain randomization, 1-path resets."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FinetuneEvalEventCfg(BaseEventCfg):
    """Eval after Stage 2: fixed sysid + OSC gains (scale_progress=1) + 1-path resets."""

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
            "delay_range": (0, 1),
        },
    )

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (0.8, 1.2),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FinetuneFullResetEventCfg(FinetuneEvalEventCfg):
    """Stage-2 fixed sysid + OSC gains, but resetting from all four distributions.

    Sits between the two existing finetune event configs: it keeps ``FinetuneEvalEventCfg``'s
    ``*_fixed`` randomizers -- fully ramped (scale_progress=1) gains that never change over training,
    so no curriculum is required to drive them -- while restoring ``TrainEventCfg``'s uniform 4-way
    reset distribution instead of the eval config's ObjectAnywhereEEAnywhere-only resets.

    Use for training runs that want the hardest, final dynamics from step 0 across the full reset
    distribution. ``FinetuneEventCfg`` is the alternative when you do want the gains ramped, but that
    one requires ``FinetuneCurriculumsCfg`` to advance ``scale_progress``.
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FinetuneEventCfg(TrainEventCfg):
    """Finetune events: curriculum-ramped sysid + OSC gains + 4-path resets. Explicit actuator from start."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid,
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
            "delay_range": (0, 1),
            "initial_scale_progress": 0.0,
        },
    )

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (0.8, 1.2),
            "terminal_kp": (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
            "terminal_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "initial_scale_progress": 0.0,
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )

@configclass
class CommandsReachingCfg:
    """Reaching command term -- logs equivalent metrics to the peg-insertion TaskCommand."""

    task_command = task_mdp.TaskCommandReachingCfg(
        resampling_time_range=(1e6, 1e6),
        ee_asset_cfg=SceneEntityCfg("robot", body_names="wrist_3_link"),
        target_asset_cfg=SceneEntityCfg("target_marker"),
        success_position_threshold=0.15,
        success_orientation_threshold=0.2,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
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
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for policy group."""

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

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        # privileged observations
        time_left = ObsTerm(func=task_mdp.time_left)

        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        robot_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        insertive_object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )

        receptive_object_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )

        table_material_properties = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table")}
        )

        robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

        insertive_object_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )

        receptive_object_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )

        table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

        robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_stiffness = ObsTerm(
            func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObservationsNoPrivilegedObsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
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
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for policy group."""

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

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
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
            # 5 frames to match the policy group's history, so the critic sees the same temporal
            # window the actor does (the terms are already the same set).
            self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObservationsDataCollectionNoPrivilegedObsCfg(ObservationsCfg):
    """Data-collection observations: expert's groups intact, plus the target task's critic group.

    Inherits ``policy`` and the *privileged* ``critic`` unchanged, so a PPO expert trained on
    ``ObservationsCfg`` loads and acts normally (rsl_rl sizes its critic head from the env's
    ``critic`` group and loads strictly -- stripping it here would fail the checkpoint load).
    ``critic_no_priv`` carries exactly what ``ObservationsNoPrivilegedObsCfg``'s critic sees (same
    six terms as the policy, same 5-frame history), so ``play.py --record_critic_obs_keys
    critic_no_priv`` records a buffer for the No-Privileged-Obs task while the expert keeps acting
    from its own observations.
    """

    critic_no_priv: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsDataCollectionGrayscaleCfg(ObservationsCfg):
    """State groups for the expert, grayscale group for recording.

    Inherits ``policy`` (state) and the privileged ``critic`` unchanged, so the state-trained PPO
    expert acts and its checkpoint loads (rsl_rl sizes its critic head from the env's ``critic``
    group and loads strictly). ``grayscale`` holds the three-camera luma stack a vision student
    consumes -- record it with ``play.py --record_actor_obs_keys grayscale``. Same term, params and
    history as ``ObservationsReachingGrayscaleCfg`` so a policy sees the same layout either way.
    """

    @configclass
    class GrayscaleCfg(ObsGroup):
        """Three-camera grayscale stack -- (history_length, 3, 84, 84), one luma channel per camera."""

        grayscale_image = ObsTerm(
            func=task_mdp.process_multi_camera_image,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("front_camera"),
                    SceneEntityCfg("side_camera"),
                    SceneEntityCfg("wrist_camera"),
                ],
                "data_type": "rgb",
                "output_size": (84, 84),
                "grayscale": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = False

    grayscale: GrayscaleCfg = GrayscaleCfg()


@configclass
class ObservationsGrayscaleAsymmetricCfg:
    """Vision actor, state critic -- genuinely asymmetric, unlike the reaching grayscale task.

    - ``policy``  : three-camera grayscale stack, what the student actually sees.
    - ``proprio`` : joint state / end-effector / previous action, shared by actor and critic.
    - ``critic``  : the full state vector with NO privileged terms and NO ``time_left``
      (``ObservationsNoPrivilegedObsCfg.CriticCfg`` is exactly that -- the same six terms as the
      state policy; the privileged critic would additionally expose time_left, joint and ee
      velocities, masses, material properties and joint friction/armature/stiffness/damping).

    Note the actor and critic streams have different shapes here, which the reaching grayscale task
    avoided. Training code that applies one shared encoder to both (the MR.Q vision script asserts
    ``policy`` and ``critic`` match) needs a separate critic encoder to consume this.
    """

    @configclass
    class ProprioCfg(ObsGroup):
        """Proprioception, shared by actor and critic. Excludes any target/object pose."""

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
            self.history_length = 3

    policy: ObservationsDataCollectionGrayscaleCfg.GrayscaleCfg = (
        ObservationsDataCollectionGrayscaleCfg.GrayscaleCfg()
    )
    proprio: ProprioCfg = ProprioCfg()
    critic: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsDataCollectionGrayscaleAsymmetricCfg(ObservationsDataCollectionGrayscaleCfg):
    """Collection-side mirror of ``ObservationsGrayscaleAsymmetricCfg``.

    Keeps the expert's own ``policy`` (state) and privileged ``critic`` so a state-trained PPO
    checkpoint loads and acts unchanged, and adds every group the asymmetric vision task needs to
    record: ``grayscale`` (inherited), ``proprio``, and ``critic_no_priv``. Collect with
    ``--record_actor_obs_keys grayscale --record_critic_obs_keys critic_no_priv``.
    """

    proprio: ObservationsGrayscaleAsymmetricCfg.ProprioCfg = ObservationsGrayscaleAsymmetricCfg.ProprioCfg()
    critic_no_priv: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsDataCollectionDepthCfg(ObservationsCfg):
    """Depth counterpart of ``ObservationsDataCollectionGrayscaleCfg``.

    Inherits the expert's state ``policy`` and privileged ``critic`` unchanged so a state-trained
    PPO checkpoint still loads and acts, and adds a ``depth`` group holding the three-camera stack.

    The tensor layout matches the grayscale group exactly -- ``distance_to_camera`` yields one
    channel per camera, so three cameras concatenate to (history, 3, 84, 84) just as three luma
    channels do, and a buffer recorded here has the same 63504-element rows. The *values* differ:
    depth is raw metric distance in metres with inf mapped to 0.0, not scaled to [0, 1] like rgb.
    """

    @configclass
    class DepthCfg(ObsGroup):
        """Three-camera depth stack -- (history_length, 3, 84, 84), one distance channel per camera."""

        depth_image = ObsTerm(
            func=task_mdp.process_multi_camera_image,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("front_camera"),
                    SceneEntityCfg("side_camera"),
                    SceneEntityCfg("wrist_camera"),
                ],
                "data_type": "distance_to_camera",
                "output_size": (84, 84),
                # grayscale stays False -- process_image asserts it is rgb-only.
                "grayscale": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = False

    depth: DepthCfg = DepthCfg()


@configclass
class ObservationsDataCollectionDepthAsymmetricCfg(ObservationsDataCollectionDepthCfg):
    """Collection-side observation set for the asymmetric *depth* task.

    Exactly ``ObservationsDataCollectionGrayscaleAsymmetricCfg`` with the vision group swapped from
    grayscale to depth. Collect with
    ``--record_actor_obs_keys depth --record_critic_obs_keys critic_no_priv
    --record_proprio_obs_keys proprio``.
    """

    proprio: ObservationsGrayscaleAsymmetricCfg.ProprioCfg = ObservationsGrayscaleAsymmetricCfg.ProprioCfg()
    critic_no_priv: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsGrayscale2CamAsymmetricCfg(ObservationsGrayscaleAsymmetricCfg):
    """Asymmetric grayscale observations with the front camera dropped -- side + wrist only.

    Vision obs becomes (history 3, 2 cameras, 84, 84) = 42,336 elements, down from 63,504. That
    changes the actor's input width, so buffers recorded against the 3-camera task are NOT
    compatible -- an expert buffer for this task has to be re-collected.

    ``proprio`` and the non-privileged state ``critic`` are inherited unchanged, so the critic side
    of the asymmetry is identical to the 3-camera task and the comparison isolates the camera count.
    """

    @configclass
    class Grayscale2CamCfg(ObsGroup):
        """Two-camera grayscale stack -- (history_length, 2, 84, 84), one luma channel per camera."""

        grayscale_image = ObsTerm(
            func=task_mdp.process_multi_camera_image,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("side_camera"),
                    SceneEntityCfg("wrist_camera"),
                ],
                "data_type": "rgb",
                "output_size": (84, 84),
                "grayscale": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = False

    policy: Grayscale2CamCfg = Grayscale2CamCfg()


@configclass
class ObservationsGrayscale2CamNoHistAsymmetricCfg(ObservationsGrayscale2CamAsymmetricCfg):
    """Side + wrist cameras with NO image history -- a single frame instead of a 3-frame stack.

    Vision obs becomes (1, 2, 84, 84) = 14,112 elements, vs 42,336 for the 2-camera stack and
    63,504 for the original 3-camera stack.

    Note this targets a different bottleneck than the camera-count ablation. History costs nothing
    to render -- each frame is rendered once and retained -- so this saves no render time. What it
    saves is 3x on everything sized by the observation: replay-buffer memory and, notably, the
    CPU->GPU sample path, which profiling put at 39% of an iteration.

    ``proprio`` keeps its 3-step history: only the image stack is collapsed, so the policy retains
    joint/end-effector velocity information and the ablation isolates *visual* history.
    """

    @configclass
    class Grayscale2CamNoHistCfg(ObservationsGrayscale2CamAsymmetricCfg.Grayscale2CamCfg):
        """Single-frame two-camera grayscale -- same term and cameras, history collapsed to 1."""

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    policy: Grayscale2CamNoHistCfg = Grayscale2CamNoHistCfg()


@configclass
class ObservationsGrayscale2CamNoHistObs32AsymmetricCfg(ObservationsGrayscale2CamNoHistAsymmetricCfg):
    """Side + wrist, single frame, downsampled to 32x32 instead of 84x84.

    Vision obs becomes (1, 2, 32, 32) = 2,048 elements, a 6.9x cut from 14,112 and 31x from the
    original 63,504. ``sample_H2D`` measured as ~2.2 ms fixed + 0.41 us/element, so this should take
    that block from ~8.0 ms to ~3.0 ms per call.

    32x32 is a deliberate midpoint: SQuInT-style pipelines go to 16x16, which risks losing the peg
    and hole at the scale they occupy in these views, while 84x84 is more resolution than the
    encoder demonstrably needs. Unlike the resolution rung below it, this DOES discard information
    the policy sees, so it is the rung most likely to cost task performance.

    The term is redeclared in full rather than mutating the inherited ``params`` dict, so there is
    no chance of aliasing the parent task's params and silently changing its resolution too.
    """

    @configclass
    class Grayscale2CamNoHist32Cfg(ObsGroup):
        """Two-camera single-frame grayscale at 32x32 -- (1, 2, 32, 32)."""

        grayscale_image = ObsTerm(
            func=task_mdp.process_multi_camera_image,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("side_camera"),
                    SceneEntityCfg("wrist_camera"),
                ],
                "data_type": "rgb",
                "output_size": (32, 32),
                "grayscale": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = False

    policy: Grayscale2CamNoHist32Cfg = Grayscale2CamNoHist32Cfg()


@configclass
class ObservationsDataCollectionGrayscale2CamNoHistObs32AsymmetricCfg(ObservationsCfg):
    """Collection-side mirror of ``ObservationsGrayscale2CamNoHistObs32AsymmetricCfg``.

    Keeps the expert's own state ``policy`` and privileged ``critic`` so a state-trained PPO
    checkpoint loads and acts unchanged, and adds the three groups the ablated vision task needs to
    record: ``grayscale`` (2 cameras, single frame, 32x32), ``proprio`` and ``critic_no_priv``.

    The vision group is the *same class object* the training task uses, so the recorded rows cannot
    drift from what training consumes -- resolution, camera order and history are shared by
    construction rather than by two copies of the same literals.

        --record_actor_obs_keys grayscale --record_critic_obs_keys critic_no_priv
        --record_proprio_obs_keys proprio
    """

    grayscale: ObservationsGrayscale2CamNoHistObs32AsymmetricCfg.Grayscale2CamNoHist32Cfg = (
        ObservationsGrayscale2CamNoHistObs32AsymmetricCfg.Grayscale2CamNoHist32Cfg()
    )
    proprio: ObservationsGrayscaleAsymmetricCfg.ProprioCfg = ObservationsGrayscaleAsymmetricCfg.ProprioCfg()
    critic_no_priv: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsGrayscale2CamNoHistObs64AsymmetricCfg(ObservationsGrayscale2CamNoHistAsymmetricCfg):
    """Side + wrist, single frame, downsampled to 64x64 -- the rung between Obs32 and the 84x84 parent.

    Vision obs is (1, 2, 64, 64) = 8,192 elements: 4x Obs32's 2,048 and 0.58x the parent's 14,112.
    The render is unchanged at 112x84, so this differs from its neighbours in downsampling only.

    64 is also the largest size Squint's CNNEncoder accepts (it supports 64/32/16 square only), so
    this is the top of the ladder for that architecture; the 84x84 parent is MR.Q-only.

    The term is redeclared in full rather than mutating the inherited ``params`` dict, so there is
    no chance of aliasing the parent task's params and silently changing its resolution too.
    """

    @configclass
    class Grayscale2CamNoHist64Cfg(ObsGroup):
        """Two-camera single-frame grayscale at 64x64 -- (1, 2, 64, 64)."""

        grayscale_image = ObsTerm(
            func=task_mdp.process_multi_camera_image,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("side_camera"),
                    SceneEntityCfg("wrist_camera"),
                ],
                "data_type": "rgb",
                "output_size": (64, 64),
                "grayscale": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = False

    policy: Grayscale2CamNoHist64Cfg = Grayscale2CamNoHist64Cfg()


@configclass
class ObservationsDataCollectionGrayscale2CamNoHistObs64AsymmetricCfg(ObservationsCfg):
    """Collection-side mirror of ``ObservationsGrayscale2CamNoHistObs64AsymmetricCfg``.

    Same construction as the Obs32 collection cfg: the expert's state ``policy`` and privileged
    ``critic`` are kept so a state-trained PPO checkpoint loads and acts unchanged, and the vision
    group is the *same class object* the training task uses, so recorded rows cannot drift from
    what training consumes.

        --record_actor_obs_keys grayscale --record_critic_obs_keys critic_no_priv
        --record_proprio_obs_keys proprio
    """

    grayscale: ObservationsGrayscale2CamNoHistObs64AsymmetricCfg.Grayscale2CamNoHist64Cfg = (
        ObservationsGrayscale2CamNoHistObs64AsymmetricCfg.Grayscale2CamNoHist64Cfg()
    )
    proprio: ObservationsGrayscaleAsymmetricCfg.ProprioCfg = ObservationsGrayscaleAsymmetricCfg.ProprioCfg()
    critic_no_priv: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsDataCollectionGrayscale2CamNoHistAsymmetricCfg(ObservationsCfg):
    """Collection-side mirror of ``ObservationsGrayscale2CamNoHistAsymmetricCfg`` (84x84).

    The 84x84 training task already existed as the LowRes rung; only its collection counterpart was
    missing. Vision obs is (1, 2, 84, 84) = 14,112 elements, consuming nearly all of the 112x84
    render rather than throwing most of it away as the smaller rungs do.
    """

    grayscale: ObservationsGrayscale2CamNoHistAsymmetricCfg.Grayscale2CamNoHistCfg = (
        ObservationsGrayscale2CamNoHistAsymmetricCfg.Grayscale2CamNoHistCfg()
    )
    proprio: ObservationsGrayscaleAsymmetricCfg.ProprioCfg = ObservationsGrayscaleAsymmetricCfg.ProprioCfg()
    critic_no_priv: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsGrayscale2CamNoHistObs126AsymmetricCfg(ObservationsGrayscale2CamNoHistAsymmetricCfg):
    """Side + wrist, single frame, downsampled to 126x126 -- the top of the resolution ladder.

    Vision obs is (1, 2, 126, 126) = 31,752 elements: 2.25x the 84x84 rung's 14,112, 3.9x Obs64 and
    15.5x Obs32.

    Unlike the Obs32/Obs64 rungs this one must NOT sit on the 112x84 LowRes scene: 126 exceeds that
    render's height, so the resize would upsample vertically and invent rows. It pairs with the
    168x126 two-camera rig instead, where 168->126 downsamples width and height passes through 1:1,
    so no axis is upsampled.

    The term is redeclared in full rather than mutating the inherited ``params`` dict, so there is
    no chance of aliasing the parent task's params and silently changing its resolution too.
    """

    @configclass
    class Grayscale2CamNoHist126Cfg(ObsGroup):
        """Two-camera single-frame grayscale at 126x126 -- (1, 2, 126, 126)."""

        grayscale_image = ObsTerm(
            func=task_mdp.process_multi_camera_image,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("side_camera"),
                    SceneEntityCfg("wrist_camera"),
                ],
                "data_type": "rgb",
                "output_size": (126, 126),
                "grayscale": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = False

    policy: Grayscale2CamNoHist126Cfg = Grayscale2CamNoHist126Cfg()


@configclass
class ObservationsDataCollectionGrayscale2CamNoHistObs126AsymmetricCfg(ObservationsCfg):
    """Collection-side mirror of ``ObservationsGrayscale2CamNoHistObs126AsymmetricCfg``.

    Same construction as the Obs32/Obs64 collection cfgs: the expert's state ``policy`` and
    privileged ``critic`` are kept so a state-trained PPO checkpoint loads and acts unchanged, and
    the vision group is the *same class object* the training task uses, so recorded rows cannot
    drift from what training consumes.

        --record_actor_obs_keys grayscale --record_critic_obs_keys critic_no_priv
        --record_proprio_obs_keys proprio
    """

    grayscale: ObservationsGrayscale2CamNoHistObs126AsymmetricCfg.Grayscale2CamNoHist126Cfg = (
        ObservationsGrayscale2CamNoHistObs126AsymmetricCfg.Grayscale2CamNoHist126Cfg()
    )
    proprio: ObservationsGrayscaleAsymmetricCfg.ProprioCfg = ObservationsGrayscaleAsymmetricCfg.ProprioCfg()
    critic_no_priv: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsDepthAsymmetricCfg:
    """Training-side observation set for the asymmetric depth task.

    Exactly ``ObservationsGrayscaleAsymmetricCfg`` with the actor's vision group swapped from
    grayscale to depth; ``proprio`` is reused verbatim so the two modalities stay interchangeable
    from the training script's point of view.

    - ``policy``  : three-camera depth stack (history, 3, 84, 84), raw metres with inf -> 0.0.
    - ``proprio`` : joint state / end-effector / previous action.
    - ``critic``  : full state, no privileged terms and no ``time_left``.

    As with the grayscale task the actor and critic streams have different shapes, so training code
    must not assume one shared encoder spans both.
    """

    policy: ObservationsDataCollectionDepthCfg.DepthCfg = ObservationsDataCollectionDepthCfg.DepthCfg()
    proprio: ObservationsGrayscaleAsymmetricCfg.ProprioCfg = ObservationsGrayscaleAsymmetricCfg.ProprioCfg()
    critic: ObservationsNoPrivilegedObsCfg.CriticCfg = ObservationsNoPrivilegedObsCfg.CriticCfg()


@configclass
class ObservationsReachingCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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

        target_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("target_marker"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for policy group."""

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

        target_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("target_marker"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_stiffness = ObsTerm(
            func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})


        # privileged observations
        time_left = ObsTerm(func=task_mdp.time_left)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObservationsReachingDepthCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        depth_image = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "distance_to_camera",
                "output_size": (84, 84),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for policy group."""


        depth_image = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "distance_to_camera",
                "output_size": (84, 84),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObservationsReachingGrayscaleCfg:
    """Reaching observations from the three-camera rig (front / side / wrist), grayscale + proprioception.

    Each camera contributes one luma channel, concatenated on the channel dim, so the image groups are
    (history_length, 3, *output_size) -- the same (history, C, H, W) layout the depth reaching task
    produces, with C=3 cameras instead of C=1 depth image.

    Three groups, because an image (history, 3, 84, 84) and a proprio vector (history, D) cannot be
    concatenated into one group (rank mismatch):

    - ``policy`` / ``critic``: the grayscale image stack (identical; neither is privileged).
    - ``proprio``: joint state / end-effector / previous action, shared by both.

    Neither stream sees the target pose: the target is randomized every reset (see
    ``TrainReachingGrayscaleEventCfg``), so it must be localized from the cameras. IsaacLab builds
    ``single_observation_space`` from every active group and the CleanRL ``IsaacLabVectorEnv``
    forwards it verbatim, so the extra group needs no wrapper change and does not disturb tasks that
    only define policy/critic.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        grayscale_image = ObsTerm(
            func=task_mdp.process_multi_camera_image,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("front_camera"),
                    SceneEntityCfg("side_camera"),
                    SceneEntityCfg("wrist_camera"),
                ],
                "data_type": "rgb",
                "output_size": (84, 84),
                "grayscale": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = False

    @configclass
    class ProprioCfg(ObsGroup):
        """Proprioception, shared by the actor and the critic.

        Deliberately excludes the target pose -- that is what the cameras are for. Flattened over
        history (the ObsGroup default) to a plain (history_length * D,) vector.
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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3

    # observation groups. There is deliberately no `critic` group: it would hold exactly the same
    # image as `policy`, and ObservationManager computes every group independently -- so a critic
    # group costs a second full clone -> float32 -> grayscale -> resize over all three cameras every
    # step (~340 MiB of GPU traffic at 64 envs), plus a second history buffer, for a bitwise copy.
    # The cameras themselves render only once regardless, since they are scene sensors. Consumers
    # get `obs["critic"]` as a zero-copy alias of `obs["policy"]` from IsaacLabVectorEnv.
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()


@configclass
class RewardsCfg:

    # safety rewards

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-100.0)

    # task rewards

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
        },
    )

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.1, params={"std": 1.0})

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)

@configclass
class RewardsScaledCfg:

    # safety rewards

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-10.0)

    # task rewards

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "std": 1.0,
        },
    )

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.1, params={"std": 1.0})

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)

@configclass
class RewardsScaledSparseCfg:
    """``RewardsScaledCfg`` without the dense shaping terms (``ee_asset_distance``,
    ``dense_success_reward``). ``progress_context`` is kept -- the reset managers read
    ``func.success`` off it."""

    # safety rewards

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-10.0)

    # task rewards

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)

@configclass
class RewardsReachingCfg:

    # safety rewards

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-3)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-100.0)

    # task rewards

    progress_context = RewTerm(
        func=task_mdp.ProgressContextReaching,  # type: ignore
        weight=0.1,
        params={
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "target_asset_cfg": SceneEntityCfg("target_marker"),
            "success_position_threshold": 0.15,
            "success_orientation_threshold": 0.2,
        },
    )

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward_no_angle, weight=0.1, params={"std": 1.0})

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    # Conservative failure: world Z only (cf. grasp_sampling check_grasp_success pos_above_ground on root_pos_w[:, 2])
    # insertive_fell_too_low = DoneTerm(
    #     func=task_mdp.object_root_w_z_below_threshold,
    #     params={
    #         "object_cfg": SceneEntityCfg("insertive_object"),
    #         "min_world_z": -0.2,
    #     },
    # )

    # Nullable so submissions can disable it from the CLI via Hydra:
    #   env.terminations.first_episode_termination=null
    # IsaacLab's TerminationManager skips None-valued term fields.
    first_episode_termination: DoneTerm | None = DoneTerm(func=task_mdp.terminate_first_episode)

    # success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 1})


@configclass
class TerminationsSuccessTerminationCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    # Conservative failure: world Z only (cf. grasp_sampling check_grasp_success pos_above_ground on root_pos_w[:, 2])
    # insertive_fell_too_low = DoneTerm(
    #     func=task_mdp.object_root_w_z_below_threshold,
    #     params={
    #         "object_cfg": SceneEntityCfg("insertive_object"),
    #         "min_world_z": -0.2,
    #     },
    # )

    # Nullable so submissions can disable it from the CLI via Hydra:
    #   env.terminations.first_episode_termination=null
    # IsaacLab's TerminationManager skips None-valued term fields.
    first_episode_termination: DoneTerm | None = DoneTerm(func=task_mdp.terminate_first_episode)

    success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 1})



@configclass
class TerminationsEvalCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    # Conservative failure: world Z only (cf. grasp_sampling check_grasp_success pos_above_ground on root_pos_w[:, 2])
    # insertive_fell_too_low = DoneTerm(
    #     func=task_mdp.object_root_w_z_below_threshold,
    #     params={
    #         "object_cfg": SceneEntityCfg("insertive_object"),
    #         "min_world_z": -0.2,
    #     },
    # )

    # first_episode_termination = DoneTerm(func=task_mdp.terminate_first_episode)

    success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 1})


@configclass
class TerminationsReachingCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=task_mdp.time_out) #, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    # Nullable so submissions can disable it from the CLI via Hydra:
    #   env.terminations.first_episode_termination=null
    first_episode_termination: DoneTerm | None = DoneTerm(func=task_mdp.terminate_first_episode, time_out=True)

    # success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 2})


@configclass
class FinetuneCurriculumsCfg:
    """Finetune curriculum: ADR sysid + action scale ramp. No actuator swap (explicit from start)."""

    adr_sysid = CurrTerm(
        func=task_mdp.adr_sysid_curriculum,
        params={
            "event_term_names": ["randomize_arm_sysid", "randomize_osc_gains"],
            "reset_event_name": "reset_from_reset_states",
            "success_threshold_up": 0.95,
            "success_threshold_down": 0.9,
            "delta": 0.01,
            "update_every_n_steps": 200,
            "initial_scale_progress": 0.0,
            "warmup_success_threshold": 0.95,
        },
    )

    action_scale = CurrTerm(
        func=task_mdp.action_scale_curriculum,
        params={
            "action_name": "arm",
            "reset_event_name": "reset_from_reset_states",
            "initial_scales": [0.02, 0.02, 0.02, 0.02, 0.02, 0.2],
            "target_scales": [0.01, 0.01, 0.002, 0.02, 0.02, 0.2],
            "success_threshold_up": 0.95,
            "success_threshold_down": 0.9,
            "delta": 0.01,
            "update_every_n_steps": 200,
            "initial_progress": 0.0,
        },
    )


@configclass
class NoCurriculumsCfg:
    """No curriculum (eval / data-collection with fixed 0.8--1.2 randomization)."""

    pass


def make_insertive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
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


variants = {
    "scene.insertive_object": {
        "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_insertive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
    },
    "scene.receptive_object": {
        "fbtabletop": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
        ),
        "fbdrawerbox": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"
        ),
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
        "plate": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
    },
}


@configclass
class Ur5eRobotiq2f85RlStateCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateRewardScalingCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledCfg = RewardsScaledCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledCfg = RewardsScaledCfg()
    terminations: TerminationsSuccessTerminationCfg = TerminationsSuccessTerminationCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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


@configclass
class Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledSparseCfg = RewardsScaledSparseCfg()
    terminations: TerminationsSuccessTerminationCfg = TerminationsSuccessTerminationCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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


@configclass
class Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg(ManagerBasedRLEnvCfg):
    """``Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseCfg`` with a non-privileged critic.

    Consumes buffers recorded by
    ``OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DataCollection-Reward-Scaling-Sparse-No-Privileged-Obs-v0``:
    both sides use ``ObservationsNoPrivilegedObsCfg``'s critic (six terms, 5-frame history), so the
    recorded ``n_critic_obs`` matches what ``load_expert_replay_buffer`` checks against.
    """

    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsNoPrivilegedObsCfg = ObservationsNoPrivilegedObsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledSparseCfg = RewardsScaledSparseCfg()
    terminations: TerminationsSuccessTerminationCfg = TerminationsSuccessTerminationCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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


@configclass
class Ur5eRobotiq2f85RlStateNoPrivilegedObsCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsNoPrivilegedObsCfg = ObservationsNoPrivilegedObsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateSuccessTerminationCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsSuccessTerminationCfg = TerminationsSuccessTerminationCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateEasyCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 4.0
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

@configclass
class Ur5eRobotiq2f85RlStateEvalCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsEvalCfg = TerminationsEvalCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateEvalRewardScalingCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledCfg = RewardsScaledCfg()
    terminations: TerminationsEvalCfg = TerminationsEvalCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateEvalRewardScalingSparseCfg(ManagerBasedRLEnvCfg):
    """Eval counterpart of ``Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseCfg``.

    ``TerminationsEvalCfg`` already carries the ``success`` termination (and drops
    ``first_episode_termination``), so the name does not repeat "SuccessTermination".
    """

    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledSparseCfg = RewardsScaledSparseCfg()
    terminations: TerminationsEvalCfg = TerminationsEvalCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateEvalRewardScalingNoPrivilegedObsCfg(ManagerBasedRLEnvCfg):
    """``Ur5eRobotiq2f85RlStateEvalRewardScalingCfg`` with a non-privileged critic.

    ``ObservationsNoPrivilegedObsCfg`` keeps the policy group identical and drops every privileged
    critic term -- ``time_left``, ``joint_vel``, ee velocity, material properties, masses, and joint
    friction/armature/stiffness/damping -- so the critic sees only what the policy sees.
    """

    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsNoPrivilegedObsCfg = ObservationsNoPrivilegedObsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledCfg = RewardsScaledCfg()
    terminations: TerminationsEvalCfg = TerminationsEvalCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateDataCollectionGrayscaleCfg(ManagerBasedRLEnvCfg):
    """Eval env for recording expert transitions with grayscale observations.

    Dynamics, action term, rewards and terminations match
    ``Ur5eRobotiq2f85RlStateDataCollectionRewardScalingSparseNoPrivilegedObsCfg`` so the expert
    behaves the same and recorded rewards are the sparse ones; only the scene gains cameras and the
    observations gain the ``grayscale`` group.
    """

    scene: RlStateGrayscaleSceneCfg = RlStateGrayscaleSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )
    observations: ObservationsDataCollectionGrayscaleCfg = ObservationsDataCollectionGrayscaleCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledSparseCfg = RewardsScaledSparseCfg()
    terminations: TerminationsEvalCfg = TerminationsEvalCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

        # Render settings (mirrors the RGB data-collection rig)
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.antialiasing_mode = "DLAA"

        # render once per env step rather than per physics substep
        self.sim.render_interval = self.decimation

        # rerender on reset so an episode's first image is not the pre-reset one
        self.num_rerenders_on_reset = 1


@configclass
class Ur5eRobotiq2f85RlStateDataCollectionRewardScalingSparseNoPrivilegedObsCfg(ManagerBasedRLEnvCfg):
    """Eval env for recording expert transitions destined for the No-Privileged-Obs task.

    Identical to ``Ur5eRobotiq2f85RlStateEvalRewardScalingSparseCfg`` except for the observations,
    which add a ``critic_no_priv`` group alongside the privileged ``critic``.
    """

    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsDataCollectionNoPrivilegedObsCfg = ObservationsDataCollectionNoPrivilegedObsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsScaledSparseCfg = RewardsScaledSparseCfg()
    terminations: TerminationsEvalCfg = TerminationsEvalCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

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

@configclass
class Ur5eRobotiq2f85RlStateEvalEasyCfg(ManagerBasedRLEnvCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsEvalCfg = TerminationsEvalCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseEventCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 4.0
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

@configclass
class Ur5eRobotiq2f85RlStateReachingCfg(ManagerBasedRLEnvCfg):
    scene: RlStateReachingSceneCfg = RlStateReachingSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsReachingCfg = ObservationsReachingCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsReachingCfg = RewardsReachingCfg()
    terminations: TerminationsReachingCfg = TerminationsReachingCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseReachingEventCfg = MISSING
    commands: CommandsReachingCfg = CommandsReachingCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 4.0
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

@configclass
class Ur5eRobotiq2f85RlStateReachingDepthCfg(ManagerBasedRLEnvCfg):
    scene: RlStateReachingSceneCfg = RlStateReachingSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsReachingDepthCfg = ObservationsReachingDepthCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    rewards: RewardsReachingCfg = RewardsReachingCfg()
    terminations: TerminationsReachingCfg = TerminationsReachingCfg()
    curriculum: NoCurriculumsCfg = NoCurriculumsCfg()
    events: BaseReachingEventCfg = MISSING
    commands: CommandsReachingCfg = CommandsReachingCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 4.0
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

# Training configuration (Stage 1: no curriculum, implicit actuator, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainCfg(Ur5eRobotiq2f85RlStateCfg):

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingCfg(Ur5eRobotiq2f85RlStateRewardScalingCfg):

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationCfg(Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationCfg):

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseCfg(
    Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseCfg
):

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg(
    Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSparseNoPrivilegedObsCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """Its parent minus the ``success`` termination: episodes end only on time-out, abnormal
    robot state, or ``first_episode_termination``.

    Observations, rewards, actions and the reset distribution are inherited unchanged, so a
    policy transfers between this and the parent and any difference is attributable to the
    termination alone.
    """

    terminations: TerminationsCfg = TerminationsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsPegMassGapCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """Dynamics-gap twin of its parent: identical in every respect except a fixed 0.8 OSC gain scale.

    Observations, rewards, terminations, actions and the reset distribution are all inherited, so a
    policy trained on the parent can be evaluated here (or finetuned from a parent checkpoint) and
    any difference is attributable to the controller gap alone.
    """

    events: TrainEventPegMassGapCfg = TrainEventPegMassGapCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsPegMassGapFullResetCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """Dynamics-gap twin of the base task with the 4-path reset mixture left intact.

    Differs from the base task in peg mass alone, so a comparison against it isolates the gap.
    The sibling ``...PegMassGapCfg`` additionally narrows resets to one path, which confounds the
    gap with a change of initial-state distribution -- use this one when that matters.
    """

    events: TrainEventPegMassGapFullResetCfg = TrainEventPegMassGapFullResetCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSparseNoPrivilegedObsPegMassGapFullResetCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsPegMassGapFullResetCfg
):
    """Its parent minus the ``success`` termination: episodes end only on time-out, abnormal
    robot state, or ``first_episode_termination``, so a solved episode runs to the horizon.

    Observations, rewards, actions and the 4-path reset mixture are inherited unchanged, as is the
    peg-mass gap, so any difference against the parent is attributable to the termination alone.
    """

    terminations: TerminationsCfg = TerminationsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """Eval twin of its parent: identical except resets are restricted to ``ObjectAnywhereEEAnywhere``.

    Observations, rewards, terminations and actions are inherited, so a checkpoint trained on the
    parent loads here unchanged.
    """

    events: TrainEvalEventAnywhereOnlyCfg = TrainEvalEventAnywhereOnlyCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingSuccessTerminationSparseNoPrivilegedObsPegMassGapCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """Eval twin with BOTH the single-path resets and the fixed 0.8 OSC gain scale.

    Differs from ``...EvalRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg`` in the gains
    alone, so the pair isolates the dynamics gap under a fixed initial-state distribution.
    """

    events: TrainEvalEventAnywhereOnlyPegMassGapCfg = TrainEvalEventAnywhereOnlyPegMassGapCfg()


# Training configuration (Stage 1: no curriculum, implicit actuator, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainNoPrivilegedObsCfg(Ur5eRobotiq2f85RlStateNoPrivilegedObsCfg):

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()



# Training configuration (Stage 1: no curriculum, implicit actuator, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainSuccessTerminationCfg(Ur5eRobotiq2f85RlStateSuccessTerminationCfg):

    events: TrainEventCfg = TrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainNoDRCfg(Ur5eRobotiq2f85RlStateCfg):

    events: TrainEventNoDRCfg = TrainEventNoDRCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainNoDR_6bdbe5e_Cfg(Ur5eRobotiq2f85RlStateCfg):

    events: TrainEventNoDR_6bdbe5e_Cfg = TrainEventNoDR_6bdbe5e_Cfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainFinetuneDynamicsCfg(Ur5eRobotiq2f85RlStateCfg):

    events: TrainEventWithDynamicsGapCfg = TrainEventWithDynamicsGapCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainFinetuneSuboptimalCfg(Ur5eRobotiq2f85RlStateCfg):

    events: TrainEventWithSuboptimalCfg = TrainEventWithSuboptimalCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainEasyCfg(Ur5eRobotiq2f85RlStateEasyCfg):

    events: TrainEasyEventCfg = TrainEasyEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainEasyNoDRCfg(Ur5eRobotiq2f85RlStateEasyCfg):

    events: TrainEasyEventNoDRCfg = TrainEasyEventNoDRCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainReachingCfg(Ur5eRobotiq2f85RlStateReachingCfg):

    events: TrainReachingEventCfg = TrainReachingEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainReachingDepthCfg(Ur5eRobotiq2f85RlStateReachingDepthCfg):

    events: TrainReachingEventCfg = TrainReachingEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


@configclass
class Ur5eRobotiq2f85RlStateReachingGrayscaleCfg(Ur5eRobotiq2f85RlStateReachingDepthCfg):
    """Reaching from the three-camera (front / side / wrist) grayscale rig.

    Inherits the depth reaching task's sim/physx/render settings, rewards, and terminations; swaps in
    the rgb three-camera scene and the grayscale multi-camera observations.
    """

    scene: RlStateReachingGrayscaleSceneCfg = RlStateReachingGrayscaleSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsReachingGrayscaleCfg = ObservationsReachingGrayscaleCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainReachingGrayscaleCfg(Ur5eRobotiq2f85RlStateReachingGrayscaleCfg):

    events: TrainReachingGrayscaleEventCfg = TrainReachingGrayscaleEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainReachingGrayscaleFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainReachingGrayscaleCfg
):
    """Grayscale reaching with photorealism disabled.

    Identical to the parent in every respect except rendering. The inherited config enables ambient
    occlusion, reflections, the DL denoiser and DLSSG -- features whose entire output is discarded
    by the downsample to an 84x84 single-channel observation. Profiling put rollout/env_step at
    95.8% of a training step on an L40S, so this is the only remaining lever of any size.

    Kept as a separate task rather than an edit to the parent so the two can be A/B'd, and so
    in-flight jobs are unaffected.
    """

    def __post_init__(self):
        super().__post_init__()
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.enable_translucency = False
        # The 168x126 -> 84x84 resize already supersamples ~1.5x, which is the antialiasing that
        # actually matters for this observation; a GPU AA pass on top of it is redundant.
        self.sim.render.antialiasing_mode = "Off"
        self.sim.render.rendering_mode = "performance"

# Finetune configuration (Stage 2: explicit actuator, curriculum ramps sysid + gains + scales)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg(Ur5eRobotiq2f85RlStateCfg):
    """Finetune config: loads converged Stage 1 policy, explicit actuator from start, curriculum ramps DR."""

    events: FinetuneEventCfg = FinetuneEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    curriculum: FinetuneCurriculumsCfg = FinetuneCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Finetune configuration with scaled sparse rewards, success termination and a non-privileged critic
@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg(
    Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """Off-policy finetune: Stage 2 events/curriculum on top of the scaled-sparse, no-privileged-obs base."""

    events: FinetuneEventCfg = FinetuneEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    curriculum: FinetuneCurriculumsCfg = FinetuneCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Finetune TRAINING configuration: no curriculum, fixed/maximal gains, full reset distribution
@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsFullResetCfg(
    Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """Off-policy finetune, trained on the final dynamics from step 0 over all four reset paths.

    Differs from the sibling
    ``Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg``
    in that nothing ramps: that one pairs ``FinetuneEventCfg`` with ``FinetuneCurriculumsCfg`` to
    walk sysid / OSC gains up over training, whereas this one takes the ``*_fixed`` randomizers at
    scale_progress=1 and inherits the base's ``NoCurriculumsCfg``.

    Differs from the ...EvalCfg variant only in the reset distribution -- same events family, action
    and explicit actuator, but resets are drawn uniformly from all four paths
    (ObjectAnywhereEEAnywhere / ObjectRestingEEGrasped / ObjectAnywhereEEGrasped /
    ObjectPartiallyAssembledEEGrasped) rather than ObjectAnywhereEEAnywhere alone.

    Terminations come from the base's ``TerminationsSuccessTerminationCfg``, which keeps
    ``first_episode_termination`` -- correct for training, unlike the eval/data-collection variants.
    """

    events: FinetuneFullResetEventCfg = FinetuneFullResetEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalCfg(Ur5eRobotiq2f85RlStateEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventCfg = TrainEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingCfg(Ur5eRobotiq2f85RlStateEvalRewardScalingCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventCfg = TrainEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingSparseCfg(Ur5eRobotiq2f85RlStateEvalRewardScalingSparseCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventCfg = TrainEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingNoPrivilegedObsCfg(
    Ur5eRobotiq2f85RlStateEvalRewardScalingNoPrivilegedObsCfg
):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventCfg = TrainEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Data-collection configuration: PPO expert acts from state obs, records `grayscale`
@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleCfg(Ur5eRobotiq2f85RlStateDataCollectionGrayscaleCfg):
    """Grayscale collection with domain randomization off.

    ``TrainEvalEventNoDRCfg`` keeps the same 4-path reset distribution as ``TrainEvalEventCfg`` but
    drops the material / mass / gripper-gain randomization, so the recorded images and dynamics come
    from the nominal system.
    """

    # WristCam variant: without the reset-time xformOp write the wrist camera renders only skybox.
    events: TrainEvalEventNoDRWristCamCfg = TrainEvalEventNoDRWristCamCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleCfg
):
    """Collection env for the asymmetric vision task: vision actor, non-privileged state critic.

    Identical dynamics, rewards (sparse + scaled), success termination and no-DR events as the
    parent -- only the observation set widens, adding ``proprio`` and ``critic_no_priv`` alongside
    the inherited ``grayscale``. The expert still acts from its own state ``policy`` group.

        --record_actor_obs_keys grayscale --record_critic_obs_keys critic_no_priv
    """

    observations: ObservationsDataCollectionGrayscaleAsymmetricCfg = (
        ObservationsDataCollectionGrayscaleAsymmetricCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionDepthAsymmetricCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricCfg
):
    """Depth version of the asymmetric collection env.

    Inherits the grayscale asymmetric env wholesale -- same dynamics, action term, sparse scaled
    rewards, terminations and no-DR events -- and swaps exactly two things: the camera rig now
    renders ``distance_to_camera`` and the vision observation group is ``depth`` instead of
    ``grayscale``. A buffer recorded here therefore has the same row layout (63504-element vision,
    proprio, non-privileged state critic) as the grayscale one, differing only in pixel semantics.

        --record_actor_obs_keys depth --record_critic_obs_keys critic_no_priv
        --record_proprio_obs_keys proprio
    """

    scene: RlStateDepthSceneCfg = RlStateDepthSceneCfg(num_envs=32, env_spacing=1.5, replicate_physics=False)
    observations: ObservationsDataCollectionDepthAsymmetricCfg = ObservationsDataCollectionDepthAsymmetricCfg()
    # Depth rig prims are named depth_wrist_camera, so the inherited rgb-named term would no-op.
    events: TrainEvalEventNoDRWristCamDepthCfg = TrainEvalEventNoDRWristCamDepthCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainDepthAsymmetricCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionDepthAsymmetricCfg
):
    """Training env: depth + proprio actor, non-privileged full-state critic.

    Depth counterpart of ``Ur5eRobotiq2f85RelCartesianOSCTrainGrayscaleAsymmetricCfg``, and stands
    in the same relation to the depth collection env: it shares that env's scene, dynamics, sparse
    scaled rewards and disabled domain randomization, so a buffer recorded there is on-distribution
    here. Two things change: the state ``policy`` / privileged ``critic`` groups the expert needed
    are dropped (leaving ``policy`` = depth, ``proprio``, ``critic`` = non-privileged state, which
    also spares ObservationManager the cost of groups nothing consumes), and terminations switch to
    the training set, which adds ``terminate_first_episode`` to stagger initial episodes across
    envs. Both sets already terminate on success.
    """

    observations: ObservationsDepthAsymmetricCfg = ObservationsDepthAsymmetricCfg()
    terminations: TerminationsSuccessTerminationCfg = TerminationsSuccessTerminationCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainGrayscaleAsymmetricCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricCfg
):
    """Training env: grayscale + proprio actor, non-privileged full-state critic.

    Shares the collection env's scene, dynamics, sparse scaled rewards and disabled domain
    randomization, so a buffer recorded there is on-distribution here. Two things change: the state
    ``policy``/privileged ``critic`` groups the expert needed are dropped (leaving ``policy`` =
    grayscale, ``proprio``, ``critic`` = non-privileged state, and sparing ObservationManager the
    cost of groups nothing consumes), and terminations switch to the training set, which adds
    ``terminate_first_episode`` to stagger initial episodes across envs. Both sets already terminate
    on success.
    """

    observations: ObservationsGrayscaleAsymmetricCfg = ObservationsGrayscaleAsymmetricCfg()
    terminations: TerminationsSuccessTerminationCfg = TerminationsSuccessTerminationCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainGrayscaleAsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscaleAsymmetricCfg
):
    """Asymmetric training env with photorealism disabled (~1.7x faster rendering)."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.enable_translucency = False
        self.sim.render.antialiasing_mode = "Off"
        self.sim.render.rendering_mode = "performance"


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamAsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscaleAsymmetricFastRenderCfg
):
    """Speed ablation: the FastRender grayscale asymmetric task with side + wrist cameras only.

    Identical to the parent in every other respect (dynamics, rewards, terminations, no-DR events,
    FastRender settings), so a throughput comparison against it isolates the camera count. The front
    camera is dropped from the scene, not merely from the observation, so its render cost is
    actually saved.

    Actor obs is (3, 2, 84, 84) = 42,336 vs 63,504; expert buffers are therefore not interchangeable
    with the 3-camera task.
    """

    scene: RlStateGrayscale2CamSceneCfg = RlStateGrayscale2CamSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )
    observations: ObservationsGrayscale2CamAsymmetricCfg = ObservationsGrayscale2CamAsymmetricCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistAsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamAsymmetricFastRenderCfg
):
    """Speed ablation stacked on the 2-camera task: single-frame images, no visual history.

    Shares the 2-camera scene, so render cost is unchanged from its parent -- the difference is
    purely observation size (14,112 vs 42,336), which drives replay-buffer memory and the
    host->device sample transfer. Comparing this against its parent isolates the cost of image
    history; comparing the parent against the 3-camera task isolates camera count.
    """

    observations: ObservationsGrayscale2CamNoHistAsymmetricCfg = ObservationsGrayscale2CamNoHistAsymmetricCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResAsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistAsymmetricFastRenderCfg
):
    """Third rung of the speed ablation: 2 cameras, no history, rendered at 112x84.

    Only the render resolution changes from the parent -- same cameras, same poses, same FOV, same
    84x84 observation. So this isolates rasterization cost, and unlike the earlier rungs it discards
    no information the policy sees (the extra pixels were being thrown away by the downsample).

    Observation dim stays 14,112, so buffers are dimensionally interchangeable with the parent task,
    though the images differ slightly (less pre-downsample detail).
    """

    scene: RlStateGrayscale2CamLowResSceneCfg = RlStateGrayscale2CamLowResSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResObs32AsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResAsymmetricFastRenderCfg
):
    """Fourth rung: 2 cameras, no history, 112x84 render, downsampled to 32x32.

    Only ``output_size`` changes from the parent, so render cost is identical and the delta is
    entirely in observation size -- 2,048 vs 14,112 elements. Expect the win in ``buffer/sample_H2D``
    and buffer memory, not in ``rollout/env_step``.

    Note the render is still 112x84 for a 32x32 observation, which is now 9x more pixels than are
    consumed. Dropping the render to ~44x33 would be the natural companion change, but is kept
    separate so this rung measures downsampling alone.
    """

    observations: ObservationsGrayscale2CamNoHistObs32AsymmetricCfg = (
        ObservationsGrayscale2CamNoHistObs32AsymmetricCfg()
    )



@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricCfg
):
    """Asymmetric collection env with photorealism disabled -- pair with the FastRender training env
    so recorded images and training images come from the same renderer."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.enable_translucency = False
        self.sim.render.antialiasing_mode = "Off"
        self.sim.render.rendering_mode = "performance"


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscale2CamNoHistLowResObs32AsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricFastRenderCfg
):
    """Collection env matching the fully-ablated vision task: 2 cameras, no history, 112x84 -> 32x32.

    Pairs with ``...TrainGrayscale2CamNoHistLowResObs32AsymmetricFastRenderCfg``. Inherits that
    task's dynamics, sparse scaled rewards, no-DR events and FastRender settings from the grayscale
    asymmetric collection env, and swaps in the ablated scene and observation set, so a buffer
    recorded here is on-distribution for training there.

    Renderer parity matters as much as observation parity: both sides are FastRender and both render
    at 112x84, so the student never sees a domain gap between the recorded images and the ones it
    trains against.
    """

    scene: RlStateGrayscale2CamLowResSceneCfg = RlStateGrayscale2CamLowResSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )
    observations: ObservationsDataCollectionGrayscale2CamNoHistObs32AsymmetricCfg = (
        ObservationsDataCollectionGrayscale2CamNoHistObs32AsymmetricCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResObs64AsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResAsymmetricFastRenderCfg
):
    """2 cameras, no history, 112x84 render, downsampled to 64x64.

    Only ``output_size`` changes from the parent, so render cost is identical and the delta is
    entirely in observation size -- 8,192 elements against the parent's 14,112 and Obs32's 2,048.
    """

    observations: ObservationsGrayscale2CamNoHistObs64AsymmetricCfg = (
        ObservationsGrayscale2CamNoHistObs64AsymmetricCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscale2CamNoHistLowResObs64AsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricFastRenderCfg
):
    """Collection env matching the 64x64 vision task: 2 cameras, no history, 112x84 -> 64x64.

    Both sides are FastRender and both render at 112x84, so the student never sees a domain gap
    between recorded images and the ones it trains against.
    """

    scene: RlStateGrayscale2CamLowResSceneCfg = RlStateGrayscale2CamLowResSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )
    observations: ObservationsDataCollectionGrayscale2CamNoHistObs64AsymmetricCfg = (
        ObservationsDataCollectionGrayscale2CamNoHistObs64AsymmetricCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscale2CamNoHistLowResAsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricFastRenderCfg
):
    """Collection env matching the 84x84 vision task (the LowRes rung, no Obs suffix).

    Pairs with ``...TrainGrayscale2CamNoHistLowResAsymmetricFastRenderCfg``, which already existed;
    this fills in its missing collection counterpart.
    """

    scene: RlStateGrayscale2CamLowResSceneCfg = RlStateGrayscale2CamLowResSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )
    observations: ObservationsDataCollectionGrayscale2CamNoHistAsymmetricCfg = (
        ObservationsDataCollectionGrayscale2CamNoHistAsymmetricCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistObs126AsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistAsymmetricFastRenderCfg
):
    """2 cameras, no history, 168x126 render downsampled to 126x126.

    Inherits the 168x126 two-camera scene unchanged and overrides only ``output_size``, so render
    cost matches the 84x84 non-LowRes rung and the delta is entirely observation size: 31,752
    elements against that rung's 14,112.
    """

    observations: ObservationsGrayscale2CamNoHistObs126AsymmetricCfg = (
        ObservationsGrayscale2CamNoHistObs126AsymmetricCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscale2CamNoHistObs126AsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricFastRenderCfg
):
    """Collection env matching the 126x126 vision task: 2 cameras, no history, 168x126 -> 126x126.

    Scene is pinned to the two-camera 168x126 rig rather than the LowRes 112x84 one, matching the
    training task; both sides are FastRender, so the student sees no domain gap between recorded
    images and the ones it trains against.
    """

    scene: RlStateGrayscale2CamSceneCfg = RlStateGrayscale2CamSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )
    observations: ObservationsDataCollectionGrayscale2CamNoHistObs126AsymmetricCfg = (
        ObservationsDataCollectionGrayscale2CamNoHistObs126AsymmetricCfg()
    )


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalGrayscaleAsymmetricCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscaleAsymmetricCfg
):
    """Eval env for the asymmetric grayscale task -- the training env minus the stagger hack.

    Observation groups are inherited unchanged, which is the point: a checkpoint's encoder, actor
    and critic only load against the exact ``policy``/``proprio``/``critic`` shapes they were
    trained on. The only change is terminations, which drop ``first_episode_termination``; that term
    cuts every env's first episode short to decorrelate rollouts during training, and would
    otherwise poison the first reported success rate here.
    """

    terminations: TerminationsEvalCfg = TerminationsEvalCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalGrayscaleAsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCEvalGrayscaleAsymmetricCfg
):
    """Asymmetric grayscale eval env with photorealism disabled.

    Pair this with checkpoints trained on the FastRender env -- the renderer is part of the
    observation distribution, so evaluating a FastRender policy under full photorealism (or vice
    versa) measures a domain gap rather than the policy.
    """

    def __post_init__(self):
        super().__post_init__()
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.enable_translucency = False
        self.sim.render.antialiasing_mode = "Off"
        self.sim.render.rendering_mode = "performance"


@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalGrayscale2CamNoHistLowResObs32AsymmetricFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResObs32AsymmetricFastRenderCfg
):
    """Eval env for the fully-ablated vision task: 2 cameras, no history, 112x84 render -> 32x32.

    Inherits the training env so scene, observation groups and renderer are byte-identical -- a
    checkpoint's encoder/actor/critic only load against the exact shapes they trained on, and the
    renderer is part of the observation distribution, so evaluating FastRender weights under full
    photorealism would measure a domain gap rather than the policy.

    The only change is terminations: ``TerminationsEvalCfg`` drops ``first_episode_termination``,
    which cuts every env's first episode short to decorrelate rollouts during training and would
    otherwise depress the first reported success rate here.
    """

    terminations: TerminationsEvalCfg = TerminationsEvalCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleFastRenderCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleCfg
):
    """Grayscale collection with photorealism disabled.

    Same rationale as ``Ur5eRobotiq2f85RelCartesianOSCTrainReachingGrayscaleFastRenderCfg``: ambient
    occlusion, reflections, the DL denoiser, DLSSG and translucency are all discarded by the
    downsample to an 84x84 single-channel observation.

    Note this deliberately breaks visual parity with the RGB DAgger rig, which the parent config
    keeps. Buffers recorded here should be trained against FastRender-rendered environments, not
    mixed with RGB-rig data, or the student sees a domain gap.
    """

    def __post_init__(self):
        super().__post_init__()
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.enable_translucency = False
        # The 168x126 -> 84x84 resize already supersamples ~1.5x, which is the antialiasing that
        # actually matters for this observation; a GPU AA pass on top of it is redundant.
        self.sim.render.antialiasing_mode = "Off"
        self.sim.render.rendering_mode = "performance"


# Data-collection configuration: PPO expert acts from its own obs, records `critic_no_priv`
@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionRewardScalingSparseNoPrivilegedObsCfg(
    Ur5eRobotiq2f85RlStateDataCollectionRewardScalingSparseNoPrivilegedObsCfg
):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventCfg = TrainEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Data-collection configuration for the Stage-2 FINETUNE task: PPO expert acts from its own obs and
# records `critic_no_priv`, but under finetune dynamics rather than Stage-1 ones.
@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg(
    Ur5eRobotiq2f85RlStateDataCollectionRewardScalingSparseNoPrivilegedObsCfg
):
    """Records expert transitions for
    ``Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsEvalCfg``.

    Same Stage-2 setup as that config -- explicit actuator, ``FinetuneEvalEventCfg`` and the Eval
    action -- so the recorded transitions come from the dynamics the finetune task actually trains
    under. ``FinetuneEvalEventCfg`` uses the ``*_fixed`` sysid / OSC-gain randomizers, i.e. the fully
    ramped (scale_progress=1) gains, and the base pins ``curriculum = NoCurriculumsCfg``, so nothing
    here ramps over time -- every recorded episode sees the same, maximally sys-IDed dynamics.

    Terminations come from the base's ``TerminationsEvalCfg`` rather than the consumer's
    ``TerminationsSuccessTerminationCfg``. The two are identical except that the latter also carries
    ``first_episode_termination``, which truncates the first episode after a reset -- wanted while
    training, not while recording, since it would throw away expert transitions.
    """

    events: FinetuneFullResetEventCfg = FinetuneFullResetEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalNoDRCfg(Ur5eRobotiq2f85RlStateEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventNoDRCfg = TrainEvalEventNoDRCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()

# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalNoDR_6bdbe5e_Cfg(Ur5eRobotiq2f85RlStateEvalCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventNoDR_6bdbe5e_Cfg = TrainEvalEventNoDR_6bdbe5e_Cfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalEasyCfg(Ur5eRobotiq2f85RlStateEvalEasyCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEasyEventCfg = TrainEvalEasyEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalEasyNoDRCfg(Ur5eRobotiq2f85RlStateEvalEasyCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEasyEventNoDRCfg = TrainEvalEasyEventNoDRCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCEvalFinetuneDynamicsCfg(Ur5eRobotiq2f85RlStateCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: TrainEvalEventWithDynamicsGapCfg = TrainEvalEventWithDynamicsGapCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Evaluation configuration (after Stage 2: explicit actuator, stiff gains, fixed sysid)
@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg(Ur5eRobotiq2f85RlStateCfg):
    """Eval after Stage 2: explicit actuator, stiff gains, small action scale, fixed sysid + OSC gains."""

    events: FinetuneEvalEventCfg = FinetuneEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Finetune configuration with scaled sparse rewards, success termination and a non-privileged critic
@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsEvalCfg(
    Ur5eRobotiq2f85RlStateRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """Off-policy finetune: Stage 2 events/curriculum on top of the scaled-sparse, no-privileged-obs base."""

    events: FinetuneEvalEventCfg = FinetuneEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
