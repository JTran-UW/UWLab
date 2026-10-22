# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManiSkill PushT-v1 parity: panda_stick, pd_joint_delta_pos at 20 Hz, 100-step episodes, ManiSkill reward/success."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

from uwlab_assets.robots.panda_stick import (
    PANDA_STICK_ARM_JOINTS,
    PANDA_STICK_CFG,
    PANDA_STICK_TCP_BODY,
    PANDA_STICK_TCP_OFFSET,
)

from ... import mdp
from ...push_t_env_cfg import PushTEnvCfg, PushTSceneCfg

PANDA_ARM_CFG = SceneEntityCfg("robot", joint_names=PANDA_STICK_ARM_JOINTS)

# ManiSkill PushT WhiteTableSceneBuilder keyframe for panda_stick
PUSH_T_PANDA_INIT_QPOS = [0.662, 0.212, 0.086, -2.685, -0.115, 2.898, 1.673]


@configclass
class PandaStickJointDeltaPosAction:
    arm = mdp.JointDeltaPositionActionCfg(asset_name="robot", joint_names=PANDA_STICK_ARM_JOINTS, scale=0.1)


@configclass
class PandaStickEventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    # ManiSkill robot_init_qpos_noise
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_gaussian_offset, mode="reset", params={"std": 0.02, "asset_cfg": PANDA_ARM_CFG}
    )


@configclass
class PandaStickRewardsCfg:
    # weight is set to 1 / step_dt in __post_init__ so the manager's "* dt" cancels
    maniskill_dense = RewTerm(func=mdp.maniskill_push_t_reward, weight=1.0, params={"thresh": mdp.INTERSECTION_THRESH})


@configclass
class PandaStickTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # time_out=True: ppo_fast bootstraps V(final obs) on every done, success included; rsl_rl only does it for time-outs
    success = DoneTerm(func=mdp.push_t_intersection_success, params={"thresh": mdp.INTERSECTION_THRESH}, time_out=True)
    nonfinite = DoneTerm(func=mdp.nonfinite_state)


@configclass
class PandaStickPushTEnvCfg(PushTEnvCfg):
    actions: PandaStickJointDeltaPosAction = PandaStickJointDeltaPosAction()
    events: PandaStickEventCfg = PandaStickEventCfg()
    rewards: PandaStickRewardsCfg = PandaStickRewardsCfg()
    terminations: PandaStickTerminationsCfg = PandaStickTerminationsCfg()
    ee_body_name: str = PANDA_STICK_TCP_BODY
    ee_body_offset: tuple[float, float, float] | None = PANDA_STICK_TCP_OFFSET

    def __post_init__(self):
        super().__post_init__()
        # base on the table surface as in ManiSkill (table top at z=0 there): stick tip 1.5 cm above it at the keyframe
        self.scene.robot = PANDA_STICK_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=PANDA_STICK_CFG.init_state.replace(
                pos=(0.0, 0.0, self.table_top_z),
                joint_pos={j: q for j, q in zip(PANDA_STICK_ARM_JOINTS, PUSH_T_PANDA_INIT_QPOS)},
            ),
        )
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params = {"asset_cfg": PANDA_ARM_CFG}
            group.joint_vel.params = {"asset_cfg": PANDA_ARM_CFG}

        # 100 Hz physics, 20 Hz control, 100 policy steps
        self.decimation = 5
        self.sim.dt = 0.01
        self.episode_length_s = 5.0
        # the reward manager multiplies by step_dt; cancel it so the per-step value matches ManiSkill
        self.rewards.maniskill_dense.weight = 1.0 / (self.sim.dt * self.decimation)

        self.sim.physics = PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.02,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.0005,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=2**23,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**31,
        )


@configclass
class PandaStickPushTPlayEnvCfg(PandaStickPushTEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16


@configclass
class PandaStickEEDeltaPoseAction:
    # ManiSkill panda_stick ee_link = panda_hand_tcp (panda_hand + 15 cm along z)
    arm = mdp.EEDeltaPoseActionCfg(
        asset_name="robot",
        joint_names=PANDA_STICK_ARM_JOINTS,
        body_name=PANDA_STICK_TCP_BODY,
        body_offset=PANDA_STICK_TCP_OFFSET,
    )


@configclass
class PandaStickEEPushTEnvCfg(PandaStickPushTEnvCfg):
    """ManiSkill PushT-v1 with ``pd_ee_delta_pose`` (the control mode of ManiSkill's PushT FastSAC / FlashSAC runs)."""

    actions: PandaStickEEDeltaPoseAction = PandaStickEEDeltaPoseAction()


@configclass
class PandaStickEEPushTPlayEnvCfg(PandaStickEEPushTEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16


# ManiSkill PushT-v1 create_tee: bar 0.2 x 0.05 centered 0.0375 below the actor origin (bar toward -y), stem
# 0.05 x 0.15 flush with it; 0.8 kg (the reference FastSAC runs' explicit mass), friction 3 / 3
MS_T_BLOCK_SPAWN = mdp.T_BLOCK_SPAWN.replace(
    bar_size=(0.2, 0.05),
    stem_size=(0.05, 0.15),
    part_centers=(-0.0375, 0.0625),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=3.0, restitution=0.0),
)
MS_T_GOAL_MARKER_SPAWN = mdp.T_GOAL_MARKER_SPAWN.replace(
    bar_size=(0.2, 0.05), stem_size=(0.05, 0.15), part_centers=(-0.0375, 0.0625)
)
# ManiSkill world = robot base at (-0.615, 0, 0); here the base is the env origin
MS_BASE_X = 0.615
MS_GOAL_POSE = (-0.156 + MS_BASE_X, -0.1, 5.0 / 3.0 * math.pi)
# tee_spawnbox: x in goal_x + [-0.1, 0.1], y in goal_y + [-0.1, 0.2], yaw ~ U[0, 2 pi)
MS_T_X_RANGE = (MS_GOAL_POSE[0] - 0.1, MS_GOAL_POSE[0] + 0.1)
MS_T_Y_RANGE = (MS_GOAL_POSE[1] - 0.1, MS_GOAL_POSE[1] + 0.2)


@configclass
class ManiSkillStateObservationsCfg:
    """ManiSkill PushT-v1 ``obs_mode="state"`` (qpos, qvel, tcp_pose, goal_pos, obj_pose; 31-D), root frame."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": PANDA_ARM_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": PANDA_ARM_CFG})
        tcp_pose = ObsTerm(func=mdp.ms_tcp_pose)
        goal_pos = ObsTerm(func=mdp.ms_goal_pos)
        obj_pose = ObsTerm(func=mdp.ms_obj_pose)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class ManiSkillEventCfg(PandaStickEventCfg):
    reset_t = EventTerm(
        func=mdp.reset_t_uniform_pose,
        mode="reset",
        params={"x_range": MS_T_X_RANGE, "y_range": MS_T_Y_RANGE, "yaw_range": (0.0, 2.0 * math.pi)},
    )


@configclass
class ManiSkillTerminationsCfg:
    # ManiSkill partial_reset=False -> ignore_terminations: success is logged, episodes always run 100 steps
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.push_t_intersection_success, params={"thresh": mdp.INTERSECTION_THRESH, "terminate": False}
    )
    nonfinite = DoneTerm(func=mdp.nonfinite_state)


@configclass
class PandaStickManiSkillPushTEnvCfg(PandaStickEEPushTEnvCfg):
    """ManiSkill PushT-v1 parity (as run by the reference FastSAC / FlashSAC runs): tee geometry, mass and friction,
    tee spawn box and goal, pd_ee_delta_pose, 31-D state obs, success that does not terminate."""

    observations: ManiSkillStateObservationsCfg = ManiSkillStateObservationsCfg()
    events: ManiSkillEventCfg = ManiSkillEventCfg()
    terminations: ManiSkillTerminationsCfg = ManiSkillTerminationsCfg()
    t_init_pose: tuple[float, float, float] = (MS_GOAL_POSE[0], MS_GOAL_POSE[1] + 0.05, 0.0)
    goal_pose: tuple[float, float, float] = MS_GOAL_POSE
    table_top_z: float = 0.0

    def __post_init__(self):
        self.scene.t_block.spawn = MS_T_BLOCK_SPAWN
        self.scene.goal_marker.spawn = MS_T_GOAL_MARKER_SPAWN
        # ManiSkill's table is a box at z <= 0; robot and table use the default material (0.3 / 0.3)
        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, self.table_top_z - 0.05)),
            spawn=sim_utils.CuboidCfg(
                size=(1.6, 1.6, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
            ),
        )
        self.scene.ground.init_state.pos = (0.0, 0.0, self.table_top_z - 0.8)
        super().__post_init__()
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.3, dynamic_friction=0.3, restitution=0.0
        )


@configclass
class PandaStickManiSkillPushTPlayEnvCfg(PandaStickManiSkillPushTEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16


# ManiSkill zoom camera of the reference FlashSAC-RGB run: eye (-0.05, -0.05, 0.75) -> target (-0.15, -0.05, 0) in ManiSkill
# world, fov 0.9 rad; here in the env frame (base at the origin), rotation = Isaac look-at (USD / opengl convention)
MS_ZOOM_CAMERA = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/ZoomCamera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.05 + MS_BASE_X, -0.05, 0.75), rot=(0.04683, 0.04683, 0.705554, 0.705554), convention="opengl"
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=20.955 / (2.0 * math.tan(0.45)), horizontal_aperture=20.955, clipping_range=(0.01, 10.0)
    ),
    width=64,
    height=64,
)


@configclass
class ManiSkillRGBSceneCfg(PushTSceneCfg):
    camera: TiledCameraCfg = MS_ZOOM_CAMERA


@configclass
class ManiSkillRGBObservationsCfg:
    """ManiSkill ``obs_mode="rgb"`` as the FlashSAC-RGB run used it: one gray 64x64 camera + qpos / qvel / tcp_pose (21-D).
    ``policy`` (31-D state) is only for a state expert collecting demos."""

    @configclass
    class ImageCfg(ObsGroup):
        gray = ObsTerm(func=mdp.ms_gray_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StateCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": PANDA_ARM_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": PANDA_ARM_CFG})
        tcp_pose = ObsTerm(func=mdp.ms_tcp_pose)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ManiSkillStateObservationsCfg.PolicyCfg = ManiSkillStateObservationsCfg.PolicyCfg()
    critic: ManiSkillStateObservationsCfg.PolicyCfg = ManiSkillStateObservationsCfg.PolicyCfg()
    image: ImageCfg = ImageCfg()
    state: StateCfg = StateCfg()


@configclass
class PandaStickManiSkillRGBPushTEnvCfg(PandaStickManiSkillPushTEnvCfg):
    """ManiSkill PushT-v1 parity + the zoomed gray 64x64 camera of the reference FlashSAC-RGB run."""

    scene: ManiSkillRGBSceneCfg = ManiSkillRGBSceneCfg(num_envs=1024, env_spacing=1.5)
    observations: ManiSkillRGBObservationsCfg = ManiSkillRGBObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # one render per policy step
        self.sim.render_interval = self.decimation
        # Newton Warp renderer: RTX needs Vulkan, which the Hyak container lacks (RTX cameras there return black frames)
        self.scene.camera.renderer_cfg = NewtonWarpRendererCfg()
        # otherwise the first frame of every episode shows the pre-reset scene
        self.num_rerenders_on_reset = 1
        # ManiSkill look: flat lighting (no HDR sun shadows); tee / goal colors tuned so the gray image matches ManiSkill's
        # (tee ~70, goal ~150, table ~230)
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
        self.scene.t_block.spawn = self.scene.t_block.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.03, 0.04))
        )
        self.scene.goal_marker.spawn = self.scene.goal_marker.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))
        )


@configclass
class PandaStickManiSkillRGBPushTPlayEnvCfg(PandaStickManiSkillRGBPushTEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
