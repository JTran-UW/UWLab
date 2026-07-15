# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal env config for recording gripper close trajectory.

Robot only, no objects — just need the articulation to close the gripper
and record all joint positions along the trajectory.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import uwlab_assets.robots.ur5e_robotiq_gripper as ur5e_robotiq_gripper

from ... import mdp as task_mdp


@configclass
class GripperCloseSceneCfg(InteractiveSceneCfg):
    """Minimal scene: robot + ground + light."""

    robot = ur5e_robotiq_gripper.IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
class GripperCloseTerminationCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class GripperCloseObservationsCfg:
    pass


@configclass
class GripperCloseRewardsCfg:
    pass


@configclass
class Ur5eRobotiq2f85GripperCloseCfg(ManagerBasedRLEnvCfg):
    """Minimal env for recording gripper close trajectory."""

    scene: GripperCloseSceneCfg = GripperCloseSceneCfg(num_envs=1, env_spacing=1.5)
    terminations: GripperCloseTerminationCfg = GripperCloseTerminationCfg()
    observations: GripperCloseObservationsCfg = GripperCloseObservationsCfg()
    actions: ur5e_robotiq_gripper.Robotiq2f85BinaryGripperAction = ur5e_robotiq_gripper.Robotiq2f85BinaryGripperAction()
    rewards: GripperCloseRewardsCfg = GripperCloseRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120.0

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
