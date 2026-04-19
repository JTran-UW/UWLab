# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-only DAgger env (sanity check).

Student reads the same 215d state observation as the JIT teacher. No cameras,
no curtains. Isolates the distillation loop / teacher JIT / action plumbing
from any vision-representation issue — if this converges to near-zero
behavior loss and matched student/teacher success, the depth pipeline is
specifically the bottleneck.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85RelativeOSCEvalAction
from .rl_state_cfg import FinetuneEvalEventCfg, RlStateSceneCfg, Ur5eRobotiq2f85RlStateCfg


@configclass
class StateDAggerObservationsCfg:
    """Single ``teacher`` obs group — student reuses the same tensor via obs_groups."""

    @configclass
    class TeacherCfg(ObsGroup):
        """State expert input — must match the JIT teacher's training-time obs layout.

        Term order matches ``rl_state_cfg.py:ObservationsCfg.PolicyCfg`` exactly:
        only ``insertive_asset_in_receptive_asset_frame`` is annotated, which
        promotes it to index 0 under configclass/dataclass iteration order.
        Scrambling the order feeds the teacher a wrong 215d vector.
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

    teacher: TeacherCfg = TeacherCfg()


@configclass
class StateDAggerTerminationsCfg:

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
class Ur5eRobotiq2f85StateDAggerRelCartesianOSCCfg(Ur5eRobotiq2f85RlStateCfg):
    """State DAgger env: only the ``teacher`` obs group, eval action scales, EXPLICIT actuator.

    Scaffolding matches the depth DAgger env (same event cfg, action, robot,
    episode_length_s) except we drop cameras and the proprio/depth obs groups.
    """

    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=512, env_spacing=1.5, replicate_physics=True)
    observations: StateDAggerObservationsCfg = StateDAggerObservationsCfg()
    terminations: StateDAggerTerminationsCfg = StateDAggerTerminationsCfg()
    events: FinetuneEvalEventCfg = FinetuneEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 16.0
