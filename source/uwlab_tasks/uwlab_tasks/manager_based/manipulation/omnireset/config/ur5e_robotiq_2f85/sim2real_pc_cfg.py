# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2real point-cloud sanity-check env.

Same scene as :class:`RlStateSceneCfg` but with **no insertive / receptive
objects** -- just the robot on the table. The single observation term is
:class:`OccludedScenePointCloud`, which samples a dense robot point cloud and
applies the FoundationStereo sim2real augmentation pipeline from
``PC_DESIGN.MD`` (HPR self-occlusion -> edge bleed -> surface bias -> dropout).

This env exists to *visually* sanity-check that augmentation: spawn it, gently
move the arm, and overlay the point cloud markers in a recorded video
(``scripts/tools/sim2real/record_sim2real_pc.py``).

Registered as ``OmniReset-Ur5eRobotiq2f85-Sim2RealPC-v0``.
"""

from __future__ import annotations

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85RelativeOSCEvalAction
from .depth_dagger_cfg import TeacherProprioWithPCCfg
from .rl_state_cfg import (
    ObservationsCfg,
    RlStateSceneCfg,
    Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg,
    Ur5eRobotiq2f85RlStateCfg,
)

# ICP-calibrated D455 front-camera extrinsics (opengl convention), shared by the
# sanity env and the data-collection env. Real cloud aligned to the sim arm at a
# recorded pose: RMS 49mm -> 7mm. See calibrate_extrinsics.py /
# exported/sim2real_pc_sanity/calibrated_extrinsics.json. Original guess was
# pos=(1.0770121, -0.1679045, 0.4486344), quat=(0.70564552, 0.46613815, 0.25072644, 0.47107948).
CALIBRATED_CAMERA_OFFSET_POS = (1.001469, -0.1966215, 0.4574127)
CALIBRATED_CAMERA_OFFSET_QUAT = (0.7301757, 0.4487978, 0.2351745, 0.4583852)


@configclass
class Sim2RealPCSceneCfg(RlStateSceneCfg):
    """Same scene as ``RlStateSceneCfg`` (robot + peg + peg-hole + table).

    We keep the insertive (peg) and receptive (peg-hole) objects so the point
    cloud is sampled from them too; their placement + kinematic flags are set in
    :meth:`Ur5eRobotiq2f85Sim2RealPCCfg.__post_init__` so they sit still for the
    static sanity snapshot (no falling under gravity).
    """


@configclass
class Sim2RealPCEventCfg:
    """Minimal events: reset the scene to default each episode. No DR, no objects."""

    reset_everything = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset", params={})


@configclass
class Sim2RealPCObservationsCfg:
    """Single policy group: the occluded/augmented robot point cloud."""

    @configclass
    class PolicyCfg(ObsGroup):
        scene_pc = ObsTerm(
            func=task_mdp.OccludedScenePointCloud,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                # Also sample point clouds from the scene objects. Adding them to
                # the same dense cloud means the HPR/frustum occlusion pass is
                # shared -- robot<->object and object<->object occlusion fall out
                # for free.
                "insertive_cfg": SceneEntityCfg("insertive_object"),
                "receptive_cfg": SceneEntityCfg("receptive_object"),
                # Real FoundationStereo arm clouds are dense (~18k pts); use a
                # higher sim count so the densities are comparable.
                "num_points": 4096,
                "oversample": 3,
                # D455 front-camera offset relative to the /Robot root prim
                # (opengl convention). Output points are in this camera's optical
                # frame. These are the ICP-CALIBRATED values (real cloud aligned to
                # the sim arm at a recorded pose: RMS 49mm -> 7mm); the original
                # camera_align_cfg guess was (1.0770121, -0.1679045, 0.4486344) /
                # (0.70564552, 0.46613815, 0.25072644, 0.47107948). See
                # calibrate_extrinsics.py and exported/.../calibrated_extrinsics.json.
                "camera_offset_pos": CALIBRATED_CAMERA_OFFSET_POS,
                "camera_offset_quat": CALIBRATED_CAMERA_OFFSET_QUAT,
                # Extrinsics domain randomization around the calibrated center
                # (per-env, resampled each reset). 0 -> fixed for the sanity view.
                # The data-collection env turns this on (see below).
                "camera_offset_pos_range": (0.0, 0.0, 0.0),
                "camera_offset_rot_range_deg": 0.0,
                "bg_plane_z": -0.02,
                "visualize": True,
                "visualize_env_ids": [0],
            },
        )

        def __post_init__(self):
            # The term applies its own augmentation; no extra obs-group corruption.
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()


@configclass
class Sim2RealPCTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class Sim2RealPCEmptyCfg:
    """Empty rewards / commands / curriculum (no objects, no task)."""

    pass


@configclass
class Ur5eRobotiq2f85Sim2RealPCCfg(Ur5eRobotiq2f85RlStateCfg):
    """Sanity-check env for the sim2real point-cloud augmentation."""

    scene: Sim2RealPCSceneCfg = Sim2RealPCSceneCfg(num_envs=2, env_spacing=2.0)
    observations: Sim2RealPCObservationsCfg = Sim2RealPCObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()
    events: Sim2RealPCEventCfg = Sim2RealPCEventCfg()
    terminations: Sim2RealPCTerminationsCfg = Sim2RealPCTerminationsCfg()
    rewards: Sim2RealPCEmptyCfg = Sim2RealPCEmptyCfg()
    commands: Sim2RealPCEmptyCfg = Sim2RealPCEmptyCfg()
    curriculum: Sim2RealPCEmptyCfg = Sim2RealPCEmptyCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 16.0

        # Place the peg + peg-hole in a plausible pre-insertion pose for the
        # static sanity snapshot, and make both kinematic so they hold position
        # (the peg would otherwise fall: it isn't actually grasped here). The peg
        # sits in the gripper, the hole rests on the table in front of it.
        self.scene.insertive_object.init_state.pos = (0.49, 0.13, 0.36)
        self.scene.insertive_object.spawn.rigid_props.kinematic_enabled = True
        self.scene.receptive_object.init_state.pos = (0.55, -0.05, 0.0)
        self.scene.receptive_object.spawn.rigid_props.kinematic_enabled = True
        # Pulled-back 3/4 view (env-relative). The HPR occlusion is computed from
        # the fixed D455 camera_center *inside the obs term*, independent of this
        # viewer -- so we place the viewer off-axis and at a distance where the
        # point markers are clearly visible (a head-on D455 view fills the frame
        # and the on-surface markers blend into the robot). From this angle the
        # sanity check reads clearly: green = full cloud (both sides), red =
        # augmented cloud occluded toward the D455 (camera-facing side).
        self.viewer = ViewerCfg(
            eye=(1.7, -1.1, 0.95),
            lookat=(0.2, 0.0, 0.3),
            origin_type="env",
            env_index=0,
        )


# ----------------------------------------------------------------------------
# Data-collection env: Train cfg + sim2real point-cloud obs group
# ----------------------------------------------------------------------------
@configclass
class DataCollectionPCObservationsCfg(ObservationsCfg):
    """Eval (policy + critic state groups) PLUS:

    * ``teacher`` -- the ScenePC expert input (proprio 25 + clean ScenePointCloud
      512 = 1561d), identical to the DAgger teacher group, so the JIT expert
      (``teachers/patrick_jit_expert.pt``) can be run for action generation.
    * ``pointcloud`` -- the calibrated, domain-randomized :class:`OccludedScenePointCloud`
      with the FoundationStereo sim2real augmentation (incl. fliers). This is the
      student observation we record for sim2real point-cloud data collection.
    """

    # Expert (teacher) input -- 1561d, matches teachers/patrick_jit_expert.pt.
    teacher: TeacherProprioWithPCCfg = TeacherProprioWithPCCfg()

    @configclass
    class DataCollectObsCfg(ObsGroup):
        scene_pc = ObsTerm(
            func=task_mdp.OccludedScenePointCloud,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                # sample the objects too -> shared HPR/frustum occlusion
                "insertive_cfg": SceneEntityCfg("insertive_object"),
                "receptive_cfg": SceneEntityCfg("receptive_object"),
                "num_points": 1024,
                "oversample": 3,
                # ICP-calibrated D455 extrinsics as the center...
                "camera_offset_pos": CALIBRATED_CAMERA_OFFSET_POS,
                "camera_offset_quat": CALIBRATED_CAMERA_OFFSET_QUAT,
                # ...with extrinsics domain randomization around it (per-env, on
                # reset) so the student is robust to real-rig miscalibration.
                "camera_offset_pos_range": (0.05, 0.05, 0.05),
                "camera_offset_rot_range_deg": 5.0,
                "bg_plane_z": -0.02,
                "visualize": False,
            },
        )

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        # (N,1) magnitude of the wrist_3_link joint reaction force -- a proxy for EE
        # contact/insertion force. Recorded per step so training can UP-SAMPLE forceful
        # timesteps (it is NOT a policy input: train_point_net.py excludes it from proprio
        # and uses it only as a sampling weight). Placed last so proprio = joint_pos +
        # end_effector_pose (18d) is unchanged.
        wrist_force = ObsTerm(
            func=task_mdp.wrist_force_magnitude,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            # The term applies its own augmentation; no extra obs-group corruption.
            self.enable_corruption = False
            self.concatenate_terms = False
            self.history_length = 1

    data_collect: DataCollectObsCfg = DataCollectObsCfg()


@configclass
class Ur5eRobotiq2f85DataCollectionPCRelCartesianOSCCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Data-collection env: the **eval-after-Stage-2** config
    (:class:`Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg` -- explicit actuator,
    stiff gains, training 0.02 action scales, fixed sysid + OSC gains, 1-path
    ObjectAnywhereEEAnywhere reset states) with two added observation groups:

    * ``teacher`` -- the ScenePC expert input, so ``teachers/patrick_jit_expert.pt``
      can be rolled out to generate actions (see collect_pc_demos.py).
    * ``pointcloud`` -- the calibrated + domain-randomized sim2real point cloud
      (batched torch augmentation: z-buffer occlusion, edge bleed, surface bias,
      dropout, fliers) -- the student observation we record.

    Use this to collect (sim2real point cloud -> expert action) demonstrations.
    """

    observations: DataCollectionPCObservationsCfg = DataCollectionPCObservationsCfg()
    # The expert was trained with the EVAL action term (stiff end-of-curriculum
    # gains + the 0.01/0.002 xyz scale), NOT the soft 0.02 RelativeOSCAction the
    # finetune-eval base ships with. Using the wrong action term mis-scales the
    # expert's deltas (esp. 10x in z) and softens tracking -> the expert fails.
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_from_reset_states.params["probs"] = [0.25, 0.25, 0.25, 0.25]
        self.events.reset_from_reset_states.params["reset_types"] = [
            "ObjectAnywhereEEAnywhere",
            "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped",
            "ObjectPartiallyAssembledEEGrasped",
        ]


# ----------------------------------------------------------------------------
# BC-PointNet eval env: roll out a trained PointNet on the sim2real PC obs
# ----------------------------------------------------------------------------
@configclass
class BCPointNetObservationsCfg(ObservationsCfg):
    """Eval (policy + critic state) groups PLUS the ``data_collect`` student group --
    the exact sim2real-augmented ``scene_pc`` + ``joint_pos`` + ``end_effector_pose``
    the BC PointNet was trained on. Unlike :class:`DataCollectionPCObservationsCfg`
    there is **no ``teacher`` group**: BC eval rolls out the student PointNet itself,
    so the JIT expert input is not needed.
    """

    data_collect: DataCollectionPCObservationsCfg.DataCollectObsCfg = (
        DataCollectionPCObservationsCfg.DataCollectObsCfg()
    )


@configclass
class Ur5eRobotiq2f85BCPointNetEvalCfg(Ur5eRobotiq2f85DataCollectionPCRelCartesianOSCCfg):
    """Eval a behavior-cloned :class:`PointNet` (trained by
    ``scripts/imitation_learning/point_cloud/train_point_net.py``).

    Identical scene / action term / sysid + OSC gains / 4-path reset distribution as
    the data-collection env -- i.e. the same on-policy distribution the demos were
    drawn from -- but the ``teacher`` group is dropped (no JIT expert). Run with
    ``play.py --bc_checkpoint <lightning.ckpt>``: that path feeds the ``data_collect``
    group's ``scene_pc`` (reshaped to ``(N, num_points, 3)``) and concatenated proprio
    (``joint_pos`` + ``end_effector_pose`` = 18d) to the PointNet, denormalizes the
    predicted action with the checkpoint's saved stats, and steps the env. Success is
    tracked exactly as for any other policy (reward-triggered termination).
    """

    observations: BCPointNetObservationsCfg = BCPointNetObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_from_reset_states.params["probs"] = [1.0]
        self.events.reset_from_reset_states.params["reset_types"] = [
            "ObjectAnywhereEEAnywhere",
        ]