# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base push-T env cfg: fixed T start pose and fixed goal pose on the table (poses in the env frame)."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from . import mdp


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for a rotation of ``yaw`` rad about +z."""
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


@configclass
class PushTSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING

    t_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TBlock",
        spawn=mdp.T_BLOCK_SPAWN,
        init_state=RigidObjectCfg.InitialStateCfg(),  # set from PushTEnvCfg.t_init_pose
    )

    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TGoal",
        spawn=mdp.T_GOAL_MARKER_SPAWN,
        init_state=AssetBaseCfg.InitialStateCfg(),  # set from PushTEnvCfg.goal_pose
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.0, 0.0, -0.707, 0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
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
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # joint terms get the robot's arm joints from the robot-specific cfg
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, clip=(-10.0, 10.0))
        # clips keep a PhysX blow-up out of the running obs normalizer
        ee_pose = ObsTerm(func=mdp.ee_tip_pose_root, clip=(-2.0, 2.0))
        t_pose = ObsTerm(func=mdp.t_pose_root, clip=(-2.0, 2.0))
        goal_pose = ObsTerm(func=mdp.goal_pose_root)
        t_to_goal = ObsTerm(func=mdp.t_to_goal_root, clip=(-2.0, 2.0))
        ee_to_t = ObsTerm(func=mdp.ee_to_t_root, clip=(-2.0, 2.0))
        last_action = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        t_velocity = ObsTerm(func=mdp.t_velocity_root, clip=(-5.0, 5.0))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class PushTEnvCfg(ManagerBasedRLEnvCfg):
    scene: PushTSceneCfg = PushTSceneCfg(num_envs=4096, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: object = MISSING
    rewards: object = MISSING
    terminations: object = MISSING
    events: object = MISSING
    viewer: ViewerCfg = ViewerCfg(eye=(1.6, 0.0, 0.8), lookat=(0.45, 0.0, 0.0), origin_type="env", env_index=0)

    # planar (x, y, yaw) in the env frame
    t_init_pose: tuple[float, float, float] = (0.50, 0.12, 0.0)
    goal_pose: tuple[float, float, float] = (0.45, -0.10, 0.5 * math.pi)
    # measured: the T settles with its center 2.0 cm (half height) above this
    table_top_z: float = -0.0135
    # body whose pose is the EE / TCP in obs and rewards
    ee_body_name: str = MISSING
    # EE point in the ee_body frame (for frames the importer does not keep as bodies)
    ee_body_offset: tuple[float, float, float] | None = None

    def __post_init__(self):
        x, y, yaw = self.t_init_pose
        # +0.5 mm so the block drops onto the table instead of spawning in penetration
        self.scene.t_block.init_state.pos = (x, y, self.table_top_z + 0.5 * self.scene.t_block.spawn.height + 5e-4)
        self.scene.t_block.init_state.rot = yaw_quat(yaw)
        gx, gy, gyaw = self.goal_pose
        self.scene.goal_marker.init_state.pos = (gx, gy, self.table_top_z + 0.5 * self.scene.goal_marker.spawn.height)
        self.scene.goal_marker.init_state.rot = yaw_quat(gyaw)
