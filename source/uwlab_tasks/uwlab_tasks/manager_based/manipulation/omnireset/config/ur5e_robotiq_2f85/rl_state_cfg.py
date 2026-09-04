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

    # Declared last so it runs after every DR term within the reset mode; negative = disabled
    # (float sentinel, not None -- update_class_from_dict rejects a float override of None).
    # Override scalars from the CLI to turn any task into a dynamics-gap benchmark, e.g.
    #   env.events.dynamics_gap.params.peg_mass=0.5
    dynamics_gap = EventTerm(
        func=task_mdp.apply_dynamics_gap,
        mode="reset",
        params={
            "peg_mass": -1.0,
            "socket_mass": -1.0,
            "table_mass_scale": -1.0,
            "robot_mass_scale": -1.0,
            "peg_friction": -1.0,
            "socket_friction": -1.0,
            "table_friction": -1.0,
            "robot_friction": -1.0,
            "gripper_stiffness_scale": -1.0,
            "gripper_damping_scale": -1.0,
            "osc_kp_xyz_scale": -1.0,
            "osc_kp_rpy_scale": -1.0,
            "osc_damping_ratio_xyz_scale": -1.0,
            "osc_damping_ratio_rpy_scale": -1.0,
        },
    )

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

        # Peg-pose terms use the gap-capable wrapper; all knobs None = identical to the plain
        # function. Override scalars from the CLI for an observation-gap benchmark, e.g.
        #   env.observations.policy.insertive_asset_pose.params.pos_noise_std=0.005
        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_gap_and_hold,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
                "pos_noise_std": 0.0,
                "rot_noise_std": 0.0,
                "pos_bias": [0.0, 0.0, 0.0],
                "rot_bias": [0.0, 0.0, 0.0],
                "world_pos_bias": [0.0, 0.0, 0.0],
                "hold_prob": 0.0,
                "hold_steps": 0,
            },
        )

        # Peghole pose: same gap-capable wrapper (own hold group), so a peghole miscalibration can
        # be injected via world_pos_bias here + root_world_pos_bias on the peg-in-peghole term.
        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_gap_and_hold,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
                "pos_noise_std": 0.0,
                "rot_noise_std": 0.0,
                "pos_bias": [0.0, 0.0, 0.0],
                "rot_bias": [0.0, 0.0, 0.0],
                "world_pos_bias": [0.0, 0.0, 0.0],
                "hold_prob": 0.0,
                "hold_steps": 0,
                "hold_group": "peghole_pose_hold",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_gap_and_hold,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
                "pos_noise_std": 0.0,
                "rot_noise_std": 0.0,
                "pos_bias": [0.0, 0.0, 0.0],
                "rot_bias": [0.0, 0.0, 0.0],
                "world_pos_bias": [0.0, 0.0, 0.0],
                "root_world_pos_bias": [0.0, 0.0, 0.0],
                "hold_prob": 0.0,
                "hold_steps": 0,
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

        # Peg-pose terms use the gap-capable wrapper; all knobs None = identical to the plain
        # function. Override scalars from the CLI for an observation-gap benchmark, e.g.
        #   env.observations.policy.insertive_asset_pose.params.pos_noise_std=0.005
        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_gap_and_hold,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
                "pos_noise_std": 0.0,
                "rot_noise_std": 0.0,
                "pos_bias": [0.0, 0.0, 0.0],
                "rot_bias": [0.0, 0.0, 0.0],
                "world_pos_bias": [0.0, 0.0, 0.0],
                "hold_prob": 0.0,
                "hold_steps": 0,
            },
        )

        # Peghole pose: same gap-capable wrapper (own hold group), so a peghole miscalibration can
        # be injected via world_pos_bias here + root_world_pos_bias on the peg-in-peghole term.
        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_gap_and_hold,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
                "pos_noise_std": 0.0,
                "rot_noise_std": 0.0,
                "pos_bias": [0.0, 0.0, 0.0],
                "rot_bias": [0.0, 0.0, 0.0],
                "world_pos_bias": [0.0, 0.0, 0.0],
                "hold_prob": 0.0,
                "hold_steps": 0,
                "hold_group": "peghole_pose_hold",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_gap_and_hold,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
                "pos_noise_std": 0.0,
                "rot_noise_std": 0.0,
                "pos_bias": [0.0, 0.0, 0.0],
                "rot_bias": [0.0, 0.0, 0.0],
                "world_pos_bias": [0.0, 0.0, 0.0],
                "root_world_pos_bias": [0.0, 0.0, 0.0],
                "hold_prob": 0.0,
                "hold_steps": 0,
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
class TerminationsSuccessAsTruncationCfg(TerminationsCfg):
    """``TerminationsCfg`` plus a ``success`` term flagged ``time_out=True``: episodes end at
    insertion, but IsaacLab reports the end through ``extras["time_outs"]``, so agents with
    ``handle_truncations`` bootstrap through it instead of treating success as a zero-value
    terminal. Online counterpart of play.py's --success_to_truncation recording option.
    """

    success = DoneTerm(
        func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 1}, time_out=True
    )


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
class TerminationsEvalNoSuccessCfg:
    """``TerminationsEvalCfg`` minus ``success``: episodes end only on time-out or abnormal robot state."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)


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
class TerminationsGCPlayCfg(TerminationsReachingCfg):
    """TerminationsReachingCfg plus success termination, for eval/play.

    Training deliberately has no success termination (the policy learns to HOLD the goal for the
    whole episode); for watching/eval it is nicer for the episode to end once the goal has been
    held for ~1 s. Reads GCProgressContext's continuous_success_counter.
    """

    success = DoneTerm(
        func=task_mdp.consecutive_success_state,
        params={"num_consecutive_successes": 10},
    )


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
class GCObservationsCfg(ObservationsCfg):
    """ObservationsCfg with goal-conditioned relative-pose terms.

    Replaces ``insertive_asset_in_receptive_asset_frame`` with the insertive object's pose in its
    goal-pose frame, and adds the EE pose in its goal-EE-pose frame. Applied to both groups.
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        insertive_asset_in_receptive_asset_frame: ObsTerm | None = None

        insertive_asset_in_goal_frame = ObsTerm(
            func=task_mdp.asset_pose_in_gc_goal_frame,
            params={"target": "insertive_object", "rotation_repr": "axis_angle"},
        )

        end_effector_in_goal_ee_frame = ObsTerm(
            func=task_mdp.asset_pose_in_gc_goal_frame,
            params={
                "target": "ee",
                "ee_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        insertive_asset_in_receptive_asset_frame: ObsTerm | None = None

        insertive_asset_in_goal_frame = ObsTerm(
            func=task_mdp.asset_pose_in_gc_goal_frame,
            params={"target": "insertive_object", "rotation_repr": "axis_angle"},
        )

        end_effector_in_goal_ee_frame = ObsTerm(
            func=task_mdp.asset_pose_in_gc_goal_frame,
            params={
                "target": "ee",
                "ee_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GCRewardsCfg:
    """Goal-conditioned reward weights, simplified for learnability.

    Success and the dense shaping term depend on the INSERTIVE OBJECT's pose only; the EE may end
    anywhere. Requiring the EE to hit its own goal too made success a conjunction of two 6-DoF
    constraints, a much sparser target. Both terms take ``include_ee`` (see GCProgressContext),
    so the stricter criterion is one config flip away.

    There are deliberately no per-asset intermediate rewards here -- see GCRewardIntermediateCfg.
    """

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
        func=task_mdp.GCProgressContext,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "include_ee": False,
            # Difficulty levers, all Hydra-overridable. require_orientation=False drops the
            # orientation conjunct; the *_threshold sentinels (-1.0 = use the object's USD
            # metadata value, any value > 0 overrides it) loosen the success ball.
            "require_orientation": True,
            "position_threshold": -1.0,
            "orientation_threshold": -1.0,
            # Keypoint objective: success when ALL axis keypoints are within keypoint_threshold
            # metres of their goal counterparts. Spin about the peg axis is free by construction,
            # matching the base task's euler-XY criterion ("yaw could be different").
            "success_mode": "pose",
            "num_keypoints": 4,
            "keypoint_extent": -1.0,
            "keypoint_threshold": 0.01,
            # Tolerance curriculum (second difficulty axis, alongside the reset manager's
            # neighbour-rank curriculum). keypoint_threshold_start <= 0 disables it and pins the
            # tolerance at keypoint_threshold. Both axes read the same per-episode success rate,
            # so a step on either depresses the shared rate and gates the other -- the pair is
            # self-regulating. The cooldown is deliberately 2x the rank curriculum's so tolerance
            # steps at half the rank cadence instead of compounding with it every window.
            "keypoint_threshold_start": -1.0,
            "keypoint_threshold_min": 0.01,
            "threshold_promote_at": 0.7,
            "threshold_demote_at": 0.55,
            "threshold_factor": 0.9,
            "threshold_cooldown": 12800,
        },
    )

    dense_success_reward = RewTerm(
        func=task_mdp.gc_dense_success_reward,
        weight=0.1,
        # std sets the length-scale of the shaping. At std=1.0 the position term only spans
        # ~0.15 over the whole 0-0.2 m working range, i.e. almost no gradient; drop it toward
        # 0.1 to sharpen. include_orientation=False pairs with require_orientation=False.
        params={
            "std": 1.0,
            "include_ee": False,
            "include_orientation": True,
            "rot_std": -1.0,
            # use_keypoints=True switches shaping to mean keypoint distance (one unit, one std);
            # pair it with progress_context success_mode="keypoint".
            "use_keypoints": False,
        },
    )

    # Straight from the base omnireset reward set: keeps the gripper on the peg. Success ignores
    # the EE, so nothing else in this config penalises dropping the object mid-transport.
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

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)


@configclass
class GCRewardIntermediateCfg(GCRewardsCfg):
    """GCRewardsCfg plus intermediate per-asset success rewards: 0.1 when the insertive object
    alone is aligned with its goal, and 0.1 when the EE alone is aligned with its goal.

    NOTE: ``ee_success_reward`` rewards EE alignment that the parent's success criterion ignores
    (``include_ee=False``), so this variant re-introduces an objective the task no longer scores.
    Kept for comparison; the plain GCRewardsCfg is the simplified default.
    """

    insertive_success_reward = RewTerm(func=task_mdp.gc_insertive_success_reward, weight=0.1)

    ee_success_reward = RewTerm(func=task_mdp.gc_ee_success_reward, weight=0.1)


@configclass
class GCTrainEventCfg(BaseEventCfg):
    """Training events with goal-conditioned 4-path resets and a goal-distance curriculum.

    The goal is sampled near each env's own start state at first (``curriculum_pos_start`` metres,
    ``curriculum_rot_start`` radians) and the radii widen linearly over ``curriculum_steps`` env
    steps until every dataset state qualifies -- at which point the distribution is exactly the
    original uniform one. Early goals are therefore reachable in a few steps, so the sparse
    success term fires often enough to learn from before long-horizon goals appear.

    Progress is logged as ``curriculum/goal_pos_radius``, ``curriculum/goal_rot_radius`` and
    ``curriculum/goal_in_range_frac`` (the fraction of envs that found an in-radius goal;
    persistently < 1 means the start radius is tighter than the dataset can supply).
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.GoalConditionedMultiResetManager,
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
            "debug_vis": True,
            "goal_curriculum": True,
            "curriculum_pos_start": 0.05,
            "curriculum_pos_end": 100.0,
            "curriculum_rot_start": 0.3,
            "curriculum_rot_end": 100.0,
            "curriculum_steps": 100000,
            # Candidates drawn per env per reset. The radius is only honoured if some candidate
            # lands inside it; with too few, the term degrades to "nearest of K" and the
            # effective radius is set by candidate density, not by the schedule. Watch
            # curriculum/goal_in_range_frac and raise this if it sits well below 1.
            "curriculum_candidates": 256,
            # Schedule mode. "success" widens the radii only when the rolling success rate clears
            # `curriculum_promote_at`, and shrinks below `curriculum_demote_at`. "steps" is the old
            # open-loop linear ramp, kept for comparison -- note it has to guess the data's scale
            # in advance, and a too-large `*_end` makes it a no-op within a few hundred steps.
            "curriculum_mode": "success",
            "curriculum_promote_at": 0.7,
            "curriculum_demote_at": 0.55,
            "curriculum_expand_factor": 1.15,
            # Reset events between adjustments. The success window is ~100 episodes, so adjusting
            # every reset would move the radius faster than the signal driving it can respond.
            "curriculum_cooldown": 6400,
            # k-NN rank curriculum ("knn" mode): difficulty is the neighbour RANK limit rather
            # than a metric radius, so it is independent of dataset density. Neighbours are ranked
            # by mean KEYPOINT distance -- the same quantity success is scored on. rank_limit >=
            # knn_k falls back to a uniform draw, i.e. the unrestricted distribution.
            "curriculum_rank_start": 8,
            # Per-task start ranks for resumes ("1800,114"); empty = rank_start everywhere.
            "curriculum_rank_starts": "",
            # Demote floor; 0 = fall back to rank_start.
            "curriculum_rank_floor": 0,
            "curriculum_rank_max": 0,
            "knn_k": 256,
            "keypoint_extent": -1.0,
            "num_keypoints": 4,
            # Fraction of envs whose goal IS their own start state (see GCMRM.__call__).
            "identity_goal_prob": 0.0,
        },
    )


@configclass
class GCGraspedTrainEventCfg(GCTrainEventCfg):
    """GCTrainEventCfg with resets restricted to ``ObjectAnywhereEEGrasped`` (prob 1.0).

    The object starts already grasped, so the policy never has to solve approach-and-grasp --
    the task collapses to "carry the held object to a goal pose", i.e. reaching. This is the
    simplest rung of the ladder: if success does not move here, the problem is not the reset
    distribution.
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.GoalConditionedMultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEGrasped"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "debug_vis": True,
            "goal_curriculum": True,
            "curriculum_pos_start": 0.05,
            "curriculum_pos_end": 100.0,
            "curriculum_rot_start": 0.3,
            "curriculum_rot_end": 100.0,
            "curriculum_steps": 100000,
            "curriculum_candidates": 256,
            # Schedule mode. "success" widens the radii only when the rolling success rate clears
            # `curriculum_promote_at`, and shrinks below `curriculum_demote_at`. "steps" is the old
            # open-loop linear ramp, kept for comparison -- note it has to guess the data's scale
            # in advance, and a too-large `*_end` makes it a no-op within a few hundred steps.
            "curriculum_mode": "success",
            "curriculum_promote_at": 0.7,
            "curriculum_demote_at": 0.55,
            "curriculum_expand_factor": 1.15,
            # Reset events between adjustments. The success window is ~100 episodes, so adjusting
            # every reset would move the radius faster than the signal driving it can respond.
            "curriculum_cooldown": 6400,
            # k-NN rank curriculum ("knn" mode): difficulty is the neighbour RANK limit rather
            # than a metric radius, so it is independent of dataset density. Neighbours are ranked
            # by mean KEYPOINT distance -- the same quantity success is scored on. rank_limit >=
            # knn_k falls back to a uniform draw, i.e. the unrestricted distribution.
            "curriculum_rank_start": 8,
            # Per-task start ranks for resumes ("1800,114"); empty = rank_start everywhere.
            "curriculum_rank_starts": "",
            # Demote floor; 0 = fall back to rank_start.
            "curriculum_rank_floor": 0,
            "curriculum_rank_max": 0,
            "knn_k": 256,
            "keypoint_extent": -1.0,
            "num_keypoints": 4,
            # Fraction of envs whose goal IS their own start state (see GCMRM.__call__).
            "identity_goal_prob": 0.0,
        },
    )


@configclass
class GCGraspedRestingTrainEventCfg(GCGraspedTrainEventCfg):
    """Two grasped reset paths: ``ObjectAnywhereEEGrasped`` + ``ObjectRestingEEGrasped``, 50/50.

    A step up from grasped-only without yet requiring approach-and-grasp: the object still starts
    held, but half the episodes begin with it resting on a surface rather than free in the air, so
    the policy must also lift/extract before transporting. Per-path success is logged separately as
    ``metrics/task_0_success_rate`` (anywhere) and ``metrics/task_1_success_rate`` (resting).
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.GoalConditionedMultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEGrasped", "ObjectRestingEEGrasped"],
            "probs": [0.5, 0.5],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "debug_vis": True,
            "goal_curriculum": True,
            "curriculum_pos_start": 0.05,
            "curriculum_pos_end": 1.0,
            "curriculum_rot_start": 0.3,
            "curriculum_rot_end": 3.2,
            "curriculum_steps": 100000,
            "curriculum_candidates": 512,
            "curriculum_mode": "success",
            "curriculum_promote_at": 0.7,
            "curriculum_demote_at": 0.55,
            "curriculum_expand_factor": 1.15,
            "curriculum_cooldown": 6400,
            # k-NN rank curriculum ("knn" mode): difficulty is the neighbour RANK limit rather
            # than a metric radius, so it is independent of dataset density. Neighbours are ranked
            # by mean KEYPOINT distance -- the same quantity success is scored on. rank_limit >=
            # knn_k falls back to a uniform draw, i.e. the unrestricted distribution.
            "curriculum_rank_start": 8,
            # Per-task start ranks for resumes ("1800,114"); empty = rank_start everywhere.
            "curriculum_rank_starts": "",
            # Demote floor; 0 = fall back to rank_start.
            "curriculum_rank_floor": 0,
            "curriculum_rank_max": 0,
            "knn_k": 256,
            "keypoint_extent": -1.0,
            "num_keypoints": 4,
            # Fraction of envs whose goal IS their own start state (see GCMRM.__call__).
            "identity_goal_prob": 0.0,
        },
    )


@configclass
class GCGraspedPlayEventCfg(GCGraspedTrainEventCfg):
    """``GCGraspedTrainEventCfg`` with the goal curriculum OFF.

    Goals are drawn uniformly from the reset dataset rather than near each env's start pose, so an
    evaluation measures the policy against the FULL goal distribution. With the curriculum on, an
    early-training policy is scored against goals deliberately sampled close to its start state,
    which flatters the number; this variant removes that.
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.GoalConditionedMultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEGrasped"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "debug_vis": True,
            "goal_curriculum": False,
            # Fraction of envs whose goal IS their own start state (see GCMRM.__call__).
            "identity_goal_prob": 0.0,
        },
    )


@configclass
class GCMRMVisualizationEventCfg(BaseEventNoDRCfg):
    """Goal-conditioned resets only, no domain randomization: for visual inspection."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.GoalConditionedMultiResetManager,
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
        },
    )


@configclass
class VisualizationTerminationsCfg:
    """Time-out only, so the env resets on a fixed cadence."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class Ur5eRobotiq2f85GCMRMVisualizationCfg(Ur5eRobotiq2f85RlStateCfg):
    """Peg-insertion scene with GoalConditionedMultiResetManager resets and 2 s episodes.

    No success termination and no DR: every episode runs exactly 2 s, then all envs reset to a
    fresh (initial state, goal state) pair. Intended for visual inspection with no policy.
    """

    events: GCMRMVisualizationEventCfg = GCMRMVisualizationEventCfg()
    terminations: VisualizationTerminationsCfg = VisualizationTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 2.0


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGCTrainCfg(Ur5eRobotiq2f85RlStateCfg):
    """Goal-conditioned training env: GCMRM resets, goal-frame observations, goal-reaching rewards.

    Terminations are the inherited TerminationsCfg (time_out, abnormal_robot,
    first_episode_termination) -- no success termination.
    """

    observations: GCObservationsCfg = GCObservationsCfg()
    rewards: GCRewardsCfg = GCRewardsCfg()
    events: GCTrainEventCfg = GCTrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGCGraspedRestingTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGCTrainCfg):
    """Grasped GC task widened to two reset paths (anywhere-grasped + resting-grasped).

    Observations, rewards, terminations and actions are inherited from the grasped task, so a
    checkpoint trained there loads here unchanged; only the reset distribution differs.
    """

    events: GCGraspedRestingTrainEventCfg = GCGraspedRestingTrainEventCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGCGraspedPlayCfg(Ur5eRobotiq2f85RelCartesianOSCGCTrainCfg):
    """Play/eval twin of the grasped GC task: uniform goal sampling, no curriculum.

    Observations, rewards, terminations and actions are inherited, so a checkpoint trained on
    ``...-GC-Grasped-v0`` loads here unchanged and is scored on the unrestricted goal distribution.
    """

    events: GCGraspedPlayEventCfg = GCGraspedPlayEventCfg()
    terminations: TerminationsGCPlayCfg = TerminationsGCPlayCfg()

    def __post_init__(self):
        super().__post_init__()
        # keypoint markers (orange = current peg, blue = goal) -- the quantity success scores on
        self.rewards.progress_context.params["debug_vis"] = True


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGCGraspedTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGCTrainCfg):
    """Its parent with resets restricted to ``ObjectAnywhereEEGrasped``: a reaching-style task.

    Observations, rewards and terminations are inherited unchanged, so the reset distribution is
    the only difference and a policy transfers between the two.
    """

    events: GCGraspedTrainEventCfg = GCGraspedTrainEventCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGCIntermediateTrainCfg(Ur5eRobotiq2f85RelCartesianOSCGCTrainCfg):
    """GC training env with intermediate per-asset success rewards."""

    rewards: GCRewardIntermediateCfg = GCRewardIntermediateCfg()

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
class GCAutoResetEventCfg(TrainEventPegMassGapFullResetCfg):
    """Its parent with the reset term upgraded to ``GoalConditionedMultiResetManager``.

    GCMRM SUBCLASSES MultiResetManager, so the 4-path reset mixture and per-path success
    accounting are inherited unchanged -- it only additionally samples and exposes ``goal_state``,
    which the ``gc`` observation group reads. ``goal_curriculum=False`` draws goals uniformly, so
    a sampled goal is exactly a draw from the task's own reset distribution.
    """

    reset_from_reset_states = EventTerm(
        func=task_mdp.GoalConditionedMultiResetManager,
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
            "goal_curriculum": False,
            # Draws the disassembly policy's target keypoints (blue) and the peg's current ones
            # (orange). The training loop toggles visibility so they show only while that policy
            # is driving.
            "debug_vis": True,
            "identity_goal_prob": 0.0,
        },
    )


@configclass
class GCAutoResetNoGapEventCfg(GCAutoResetEventCfg):
    """``GCAutoResetEventCfg`` minus the peg-mass gap.

    The gap is exactly one term -- ``randomize_insertive_object_mass``, pinned to 500 g at startup
    instead of resampled per reset. Restoring it by REFERENCE to ``BaseEventCfg`` rather than
    restating the distribution means the two variants cannot drift apart if the base range is
    retuned, which is the same reasoning the PegMassGap configs give for not restating the mass.
    """

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.randomize_insertive_object_mass = BaseEventCfg().randomize_insertive_object_mass


@configclass
class TerminationsGCAutoResetCfg(TerminationsCfg):
    """``TerminationsCfg`` with ``first_episode_termination`` disabled.

    That term staggers env start times by killing envs still on their first episode, and its guard
    (``common_step_counter >= max_episode_length``) never trips here because the horizon is huge --
    so it would keep teleporting roughly one env every ``max_episode_length / num_envs`` steps,
    forever. Its purpose, desynchronizing episode boundaries, is also the opposite of what this
    task wants: phases are deliberately synchronized. ``abnormal_robot`` stays the ONLY teleport.
    """

    first_episode_termination: DoneTerm | None = None


@configclass
class ObservationsGCAutoResetCfg(ObservationsNoPrivilegedObsCfg):
    """The finetune task's observations plus a ``gc`` group carrying the GC policy's own inputs.

    The group is ``GCObservationsCfg.PolicyCfg`` verbatim, so the PPO goal-conditioned policy sees
    byte-for-byte the observation it was trained on. FastSAC never reads it: the vec-env wrapper
    concatenates explicitly named actor/critic groups, so an extra group is inert for it.
    """

    gc: GCObservationsCfg.PolicyCfg = GCObservationsCfg.PolicyCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGCAutoResetFinetuneCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsPegMassGapFullResetCfg
):
    """Finetune task wired for AUTONOMOUS RESETS driven by the goal-conditioned policy.

    Differences from the plain dynamics-gap finetune task:

    * ``time_out`` is effectively disabled (huge ``episode_length_s``). The 160-step collect and
      160-step GC-reset phases are driven by the training script, because an env auto-reset at the
      phase boundary would TELEPORT the robot -- exactly the scripted reset this setup exists to
      avoid. ``abnormal_robot`` still terminates and resets immediately, and is the only teleport.
    * The reset term is GCMRM so a sampled goal state is available to condition the GC policy on.
    * A ``gc`` observation group supplies that policy's inputs.

    No observation seen by FastSAC changes, so a checkpoint trained on the plain task loads here
    unchanged (there is no time-remaining term whose meaning the longer horizon would alter).
    """

    events: GCAutoResetEventCfg = GCAutoResetEventCfg()
    observations: ObservationsGCAutoResetCfg = ObservationsGCAutoResetCfg()
    # Plain TerminationsCfg == time_out + abnormal_robot, i.e. NO success termination. Inheriting
    # the parent's would end an episode the moment the peg seats and TELEPORT the env -- the
    # scripted reset this task exists to avoid. Matches the `...Sparse...FullResetCfg` sibling.
    terminations: TerminationsGCAutoResetCfg = TerminationsGCAutoResetCfg()

    def __post_init__(self):
        super().__post_init__()
        # 1e6 s at a 0.1 s control step: time_out never fires within any realistic run.
        self.episode_length_s = 1.0e6


@configclass
class Ur5eRobotiq2f85RelCartesianOSCGCAutoResetFinetuneNoGapCfg(
    Ur5eRobotiq2f85RelCartesianOSCGCAutoResetFinetuneCfg
):
    """``Ur5eRobotiq2f85RelCartesianOSCGCAutoResetFinetuneCfg`` without the peg-mass gap.

    Peg mass is the ONLY difference, so a comparison against the gap twin isolates the dynamics
    gap under identical autonomous-reset mechanics: same GC-driven resets, same disabled time_out,
    same 4-path goal distribution, same terminations.
    """

    events: GCAutoResetNoGapEventCfg = GCAutoResetNoGapEventCfg()


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
class Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSparseNoPrivilegedObsPegMassGapCfg(
    Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSparseNoPrivilegedObsPegMassGapFullResetCfg
):
    """Its parent with resets narrowed to ``ObjectAnywhereEEAnywhere`` (prob 1.0).

    Observations, rewards, terminations (no ``success``) and the peg-mass gap are inherited, so the
    initial-state distribution is the only difference from the parent. Matches
    ``TrainEvalEventAnywhereOnlyPegMassGapCfg`` term for term, so an eval env sees the same states.
    """

    events: TrainEventPegMassGapCfg = TrainEventPegMassGapCfg()


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


# Robot variant whose USD (and therefore the co-located metadata.yaml the sysid randomizers read)
# comes from yandabao/uwlab-assets instead of UW-Lab/uwlab-assets. Only the arm sysid nominals
# (armature / static_friction / dynamic_ratio / viscous_friction) differ between the two.
YANDA_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/yandabao/uwlab-assets/resolve/main"

EXPLICIT_UR5E_ROBOTIQ_2F85_YANDA_SYSID = EXPLICIT_UR5E_ROBOTIQ_2F85.copy()  # type: ignore
EXPLICIT_UR5E_ROBOTIQ_2F85_YANDA_SYSID.spawn.usd_path = (
    f"{YANDA_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/"
    "ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd"
)


def _use_yanda_peg_and_hole(cfg):
    """Point the peg/hole at yandabao/uwlab-assets. Same nominal geometry as UW-Lab's, but the
    hole's collision mesh differs (53 verts vs 2444) and the PegHole metadata.yaml carries 8
    symmetric assembled offsets at 0.1 rad tolerance instead of one at 0.025 -- both what Yanda's
    experts trained against.

    Also rebinds the ``peg``/``peghole`` CLI variants to these same assets: variant tokens resolve
    AFTER ``__post_init__``, so without this ``env.scene.insertive_object=peg`` would silently revert
    the cfg to UW-Lab assets (and their 0.025 rad thresholds). With it, the tokens are no-ops here.
    """
    cfg.scene.insertive_object.spawn.usd_path = f"{YANDA_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"
    cfg.scene.receptive_object.spawn.usd_path = f"{YANDA_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"
    yanda_variants = {k: dict(v) for k, v in variants.items()}
    yanda_variants["scene.insertive_object"]["peg"] = cfg.scene.insertive_object.copy()
    yanda_variants["scene.receptive_object"]["peghole"] = cfg.scene.receptive_object.copy()
    cfg.variants = yanda_variants


@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsFullResetYandaSysidCfg(
    Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsFullResetCfg
):
    """Its parent with the robot USD (and thus sysid metadata.yaml) and the peg/hole assets sourced
    from yandabao/uwlab-assets."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85_YANDA_SYSID.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _use_yanda_peg_and_hole(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationFullResetYandaSysidCfg(
    Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsFullResetYandaSysidCfg
):
    """Its parent minus the ``success`` termination: episodes run past insertion to time-out, matching
    the ...NoSuccessTerminationYandaSysid data-collection config. Keeps ``first_episode_termination``
    for training; curriculum stays ``NoCurriculumsCfg`` from the base (nothing ramps).
    """

    terminations: TerminationsCfg = TerminationsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationFullResetYandaSysidRealDynamicsGapCfg(
    Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationFullResetYandaSysidCfg
):
    """Real Dynamics Gap: the Yanda sys-id task with HARDCODED gaps replicating failures observed on
    the real robot (unlike the CLI-knob sweeps, the gap is part of the task definition).

    Current gap: stale-pose hold on the peg's observed position -- each step, with probability 0.05,
    the observed WORLD position freezes for 1 step (both policy peg-pose terms share the dropout;
    rotation and proprioception stay live). Critic observations stay clean.
    History: 2026-08-29 shipped with a +1 cm x/z world-frame bias (93.4% -> 0.41% on 498k);
    reverted 2026-08-30 in favor of the hold gap.
    """

    def __post_init__(self):
        super().__post_init__()
        for term in (
            self.observations.policy.insertive_asset_pose,
            self.observations.policy.insertive_asset_in_receptive_asset_frame,
        ):
            term.params["hold_prob"] = 0.05
            term.params["hold_steps"] = 1


@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsSuccessTruncationFullResetYandaSysidCfg(
    Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationFullResetYandaSysidCfg
):
    """Its parent plus a ``success`` termination flagged ``time_out=True``: online episodes end at
    insertion (no post-success dwell) but are bootstrapped as truncations, matching a buffer
    recorded with play.py --success_to_truncation.
    """

    terminations: TerminationsSuccessAsTruncationCfg = TerminationsSuccessAsTruncationCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationAnywhereOnlyYandaSysidCfg(
    Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationFullResetYandaSysidCfg
):
    """Its parent with resets drawn from ``ObjectAnywhereEEAnywhere`` only (``FinetuneEvalEventCfg``:
    same fixed sysid + OSC-gain randomizers, single reset path).
    """

    events: FinetuneEvalEventCfg = FinetuneEvalEventCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCFinetuneYandaSysidCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg):
    """Curriculum-ramped on-policy finetune (``FinetuneEventCfg`` + ``FinetuneCurriculumsCfg``:
    adr_sysid and action_scale ramp with success) with the robot USD and peg/hole assets sourced
    from yandabao/uwlab-assets. On-policy counterpart of the Yanda OffPolicy finetune tasks.

    The privileged critic is pinned to the UW-Lab-robot body set: the yandabao USD carries an extra
    massless helper frame (``robotiq_fingertip_centered``) that would grow ``robot_mass`` to 205
    dims and break strict loading of Stage-1 checkpoints (204). Excluding it keeps the checkpoint's
    exact critic layout so warm-starts load the value function verbatim.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85_YANDA_SYSID.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _use_yanda_peg_and_hole(self)
        self.observations.critic.robot_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="^(?!robotiq_fingertip_centered$).*"
        )


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


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionRewardScalingSparseNoPrivilegedObsNoSuccessTerminationCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionRewardScalingSparseNoPrivilegedObsCfg
):
    """Its parent minus the ``success`` termination: episodes end only on time-out or abnormal
    robot state, so recorded episodes continue past insertion until the time limit.
    """

    terminations: TerminationsEvalNoSuccessCfg = TerminationsEvalNoSuccessCfg()


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


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationYandaSysidCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """The finetune data-collection config minus the ``success`` termination, with the robot USD
    (and thus sysid metadata.yaml) sourced from yandabao/uwlab-assets.
    """

    terminations: TerminationsEvalNoSuccessCfg = TerminationsEvalNoSuccessCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85_YANDA_SYSID.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _use_yanda_peg_and_hole(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCDataCollectionFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsYandaSysidCfg(
    Ur5eRobotiq2f85RelCartesianOSCDataCollectionFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg
):
    """The finetune data-collection config (success termination kept, so episodes end at insertion
    instead of dwelling in the success state to time-out) with the robot USD and peg/hole assets
    sourced from yandabao/uwlab-assets. Pair with play.py --success_to_truncation to record the
    success-caused done as a truncation (bootstrapped) in the replay buffer.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85_YANDA_SYSID.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _use_yanda_peg_and_hole(self)


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
