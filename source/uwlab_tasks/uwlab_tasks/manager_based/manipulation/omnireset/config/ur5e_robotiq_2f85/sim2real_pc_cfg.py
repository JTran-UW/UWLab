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
from isaaclab.sensors import ContactSensorCfg
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
                # Enforce a fixed robot/insertive/receptive point split (both the dense
                # pre-occlusion sample AND the final stratified resample). Area-weighting
                # alone gives ~93% robot; the small peg/hole were starved (~3%/5%).
                "class_ratios": (0.5, 0.25, 0.25),
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

        # self.events.reset_from_reset_states.params["probs"] = [0.25, 0.25, 0.25, 0.25]
        self.events.reset_from_reset_states.params["probs"] = [0.34, 0.33, 0.33]
        self.events.reset_from_reset_states.params["reset_types"] = [
            "ObjectAnywhereEEAnywhere",
            "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped",
            # "ObjectPartiallyAssembledEEGrasped",
        ]
        self.terminations.success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 2})


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

        # TEMPORARY: 4-path reset distribution (all reset types, equal weight) to match the
        # distribution the earlier BC sweep was evaluated under, for a fair comparison.
        # The harder single-path (probs=[1.0], ObjectAnywhereEEAnywhere only) is commented below.
        # self.events.reset_from_reset_states.params["probs"] = [0.25, 0.25, 0.25, 0.25]
        # self.events.reset_from_reset_states.params["reset_types"] = [
        #     "ObjectAnywhereEEAnywhere",
        #     "ObjectRestingEEGrasped",
        #     "ObjectAnywhereEEGrasped",
        #     "ObjectPartiallyAssembledEEGrasped",
        # ]
        self.events.reset_from_reset_states.params["probs"] = [1.0]
        self.events.reset_from_reset_states.params["reset_types"] = ["ObjectAnywhereEEAnywhere"]


@configclass
class Ur5eRobotiq2f85BCPointNetSegEvalCfg(Ur5eRobotiq2f85BCPointNetEvalCfg):
    """BC PointNet eval for models trained on SEGMENTED clouds (point_dim=4: xyz + a per-point
    class label). Identical to :class:`Ur5eRobotiq2f85BCPointNetEvalCfg` (4-path reset, no
    teacher group) but turns on the ``scene_pc`` segmentation channel so the live cloud is
    ``(num_points, 4)`` -- matching the contact_seg dataset the model was trained on. The labels
    MUST match training exactly (robot=0 / insertive=-1 / receptive=+1); ``bc_utils.bc_actions``
    reshapes the flat cloud with the model's ``point_dim`` so the 4th channel is fed to the encoder.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.data_collect.scene_pc.params["include_segmentation"] = True
        self.observations.data_collect.scene_pc.params["segmentation_labels"] = {
            "robot": 0.0,
            "insertive": -1.0,
            "receptive": 1.0,
        }
        # Match the contact-seg COLLECTION frame: end-effector (wrist_3_link), not
        # camera. Models trained on the EE-frame contact dataset must be evaluated on
        # an EE-frame cloud or the frame won't match (same gotcha as the clean side).
        self.observations.data_collect.scene_pc.params["ref_cfg"] = SceneEntityCfg(
            "robot", body_names="wrist_3_link"
        )


# ----------------------------------------------------------------------------
# Contact-sensor data-collection env: ground-truth phase flags
# ----------------------------------------------------------------------------
# A peg-mounted ContactSensor reports the PhysX solver's pairwise contact force
# between the insertive object (peg) and, in this fixed column order,
#   [left_inner_finger, right_inner_finger, receptive_object].
# mdp.heuristics.object_contact_flags turns that into two binary flags --
#   gripper_touching_peg (either finger) and peg_touching_hole -- which segment a
# demo into reach / grasp-transport / insertion phases far more cleanly than the
# noisy wrist reaction force, so training can sample differently per phase.
_PEG_CONTACT_SENSOR_NAME = "peg_contacts"


def _peg_contact_sensor_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",  # peg = sensor body (dynamic)
        # force_matrix_w columns, in EXACTLY this order (heuristics indexes them):
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/left_inner_finger",
            "{ENV_REGEX_NS}/Robot/right_inner_finger",
            "{ENV_REGEX_NS}/ReceptiveObject",
        ],
        history_length=0,  # current step only
        update_period=0.0,  # report every sim step
        track_pose=False,
        debug_vis=False,
    )


@configclass
class ContactDataCollectObsCfg(DataCollectionPCObservationsCfg.DataCollectObsCfg):
    """``data_collect`` student group + privileged aux targets + the ground-truth
    contact flags, all appended AFTER joint_pos + end_effector_pose so the 18d
    proprio is unchanged. Like ``wrist_force``, these are recorded metadata, NOT a
    policy input (train_point_net.py / bc_utils.py exclude them from proprio via the
    {joint_pos, end_effector_pose} allowlist)."""

    # --- Aux targets recorded ON THE SIDE (metadata, NOT a policy input) ---
    # Privileged scene state for the trainer's optional aux loss (_AUX_TERMS =
    # [insertive_asset_pose, receptive_asset_pose, insertive_in_receptive]). Mirrors
    # the CLEAN collect config so occluded datasets have the same schema and can be
    # used for aux-loss training. Object poses are in the EE (wrist_3_link) frame to
    # match the EE-frame cloud; insertive_in_receptive is peg-in-hole frame.
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

    insertive_in_receptive = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("receptive_object"),
            "rotation_repr": "axis_angle",
        },
    )

    contact_flags = ObsTerm(
        func=task_mdp.object_contact_flags,
        params={"sensor_cfg": SceneEntityCfg(_PEG_CONTACT_SENSOR_NAME), "threshold": 0.0},
    )


@configclass
class DataCollectionPCContactObservationsCfg(DataCollectionPCObservationsCfg):
    data_collect: ContactDataCollectObsCfg = ContactDataCollectObsCfg()


@configclass
class Ur5eRobotiq2f85DataCollectionPCContactCfg(Ur5eRobotiq2f85DataCollectionPCRelCartesianOSCCfg):
    """Data-collection env + a peg-mounted ContactSensor recording two
    ground-truth contact flags/step (gripper<->peg, peg<->hole). Otherwise
    identical scene / action / sysid / reset distribution to
    :class:`Ur5eRobotiq2f85DataCollectionPCRelCartesianOSCCfg` -- so demos are
    drawn from the same distribution, just with the extra phase signal recorded.
    """

    observations: DataCollectionPCContactObservationsCfg = DataCollectionPCContactObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Contact reporting must be active on every body the sensor pairs with:
        # the peg (sensor body), both fingers (robot articulation), and the hole.
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.insertive_object.spawn.activate_contact_sensors = True
        self.scene.receptive_object.spawn.activate_contact_sensors = True
        # Register the sensor on the scene (InteractiveScene picks it up at build).
        setattr(self.scene, _PEG_CONTACT_SENSOR_NAME, _peg_contact_sensor_cfg())

        # Per-point segmentation channel: scene_pc becomes (num_points, 4) = xyz + a
        # class label (robot=0 / insertive=-1 / receptive=+1). Configurable -- flip to
        # False (or override segmentation_labels) here. NOTE: with this on, scene_pc is
        # P*4, so the downstream PointNet must use point_dim=4 (see train_point_net.py).
        self.observations.data_collect.scene_pc.params["include_segmentation"] = True
        self.observations.data_collect.scene_pc.params["segmentation_labels"] = {
            "robot": 0.0, "insertive": -1.0, "receptive": 1.0,
        }
        # Express the occluded cloud in the END-EFFECTOR (wrist_3_link) frame instead
        # of the camera frame -- nearly arm-pose-invariant, matching the clean
        # ScenePointCloud. Occlusion is still computed from the camera viewpoint; only
        # the output coordinates are wrist-relative. At real-robot deploy, apply the
        # same camera->wrist transform (extrinsics + FK) to the incoming D455 cloud.
        self.observations.data_collect.scene_pc.params["ref_cfg"] = SceneEntityCfg(
            "robot", body_names="wrist_3_link"
        )


# ----------------------------------------------------------------------------
# Clean (vanilla) ScenePC data-collection env: NO sim2real occlusion / noise
# ----------------------------------------------------------------------------
# Records the ORIGINAL clean ScenePointCloud (full FPS-sampled robot+ins+rec
# cloud) instead of the OccludedScenePointCloud sim2real augmentation -- i.e. no
# z-buffer self-occlusion, no edge bleed, no surface bias, no dropout, no fliers,
# no extrinsics domain randomization. The only sampling stochasticity is
# resample_on_reset=True, which redraws the FPS subset each episode. 3-channel
# xyz only (no segmentation label). This is the same term + config the teacher
# group uses, just at the collection point count (1024) and with resampling on.
@configclass
class CleanDataCollectObsCfg(DataCollectionPCObservationsCfg.DataCollectObsCfg):
    """``data_collect`` student group with the CLEAN ScenePointCloud swapped in for
    the sim2real-augmented OccludedScenePointCloud. proprio (joint_pos +
    end_effector_pose = 18d) is unchanged; only the ``scene_pc`` term differs."""

    scene_pc = ObsTerm(
        func=task_mdp.ScenePointCloud,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_cfg": SceneEntityCfg("insertive_object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            # Express the cloud in the END-EFFECTOR (wrist_3_link) frame. The task is
            # wrist-centric, so EE-frame points are nearly invariant to arm pose and
            # learn far better than base-frame points (see ScenePointCloud docstring).
            # NOTE: the teacher/expert clean cloud stays in base frame (unchanged).
            "ref_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            # Match the teacher group's clean cloud exactly.
            "num_points": 512,
            # Enforce a fixed robot/insertive/receptive split (512 -> 256/128/128).
            # No occlusion here, so these are 256/128/128 DISTINCT points.
            "class_ratios": (0.5, 0.25, 0.25),
            # Redraw the FPS subset each episode (the requested stochasticity).
            "resample_on_reset": True,
            "visualize": False,
        },
    )

    # --- Aux terms recorded ON THE SIDE (metadata, NOT a policy input) ---
    # The trainer's proprio allowlist ({joint_pos, end_effector_pose}) excludes these,
    # so they never enter the BC input -- they are stored only for future aux losses
    # (e.g. predicting object poses / phase from the point cloud). Declared AFTER
    # joint_pos + end_effector_pose so the 18d proprio order is unchanged.
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

    insertive_in_receptive = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_cfg": SceneEntityCfg("receptive_object"),
            "rotation_repr": "axis_angle",
        },
    )

    contact_flags = ObsTerm(
        func=task_mdp.object_contact_flags,
        params={"sensor_cfg": SceneEntityCfg(_PEG_CONTACT_SENSOR_NAME), "threshold": 0.0},
    )


@configclass
class DataCollectionPCCleanObservationsCfg(DataCollectionPCObservationsCfg):
    data_collect: CleanDataCollectObsCfg = CleanDataCollectObsCfg()


@configclass
class Ur5eRobotiq2f85DataCollectionPCCleanCfg(Ur5eRobotiq2f85DataCollectionPCRelCartesianOSCCfg):
    """Data-collection env recording the CLEAN vanilla ScenePC (no sim2real
    augmentation), with resample_on_reset=True. Otherwise identical scene / action
    term / sysid + OSC gains / reset distribution to
    :class:`Ur5eRobotiq2f85DataCollectionPCRelCartesianOSCCfg` -- demos are drawn
    from the same on-policy distribution; only the recorded cloud's realism differs
    (clean full cloud vs occluded + noisy). The ``teacher`` group (and thus the JIT
    expert input) is unchanged, so the same expert generates actions."""

    observations: DataCollectionPCCleanObservationsCfg = DataCollectionPCCleanObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The ``data_collect`` group records a ground-truth ``contact_flags`` aux term
        # (gripper<->peg, peg<->hole), which needs the peg-mounted ContactSensor active
        # on every body it pairs with. Same wiring as the Contact env.
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.insertive_object.spawn.activate_contact_sensors = True
        self.scene.receptive_object.spawn.activate_contact_sensors = True
        setattr(self.scene, _PEG_CONTACT_SENSOR_NAME, _peg_contact_sensor_cfg())


# ----------------------------------------------------------------------------
# BC-PointNet eval env: CLEAN ScenePointCloud (point_dim=3, 512 pts)
# ----------------------------------------------------------------------------
@configclass
class BCPointNetCleanObservationsCfg(ObservationsCfg):
    """BC eval obs: policy + critic state groups PLUS the CLEAN ScenePointCloud
    ``data_collect`` group (512-pt, xyz only, no sim2real occlusion/noise) -- matching the
    ``clean_scenepc`` dataset. No ``teacher`` group (BC rolls out the student itself)."""

    data_collect: CleanDataCollectObsCfg = CleanDataCollectObsCfg()


@configclass
class Ur5eRobotiq2f85BCPointNetCleanEvalCfg(Ur5eRobotiq2f85BCPointNetEvalCfg):
    """BC eval for models trained on the CLEAN ScenePointCloud (point_dim=3, 512 pts).
    Identical reset / action / success setup to :class:`Ur5eRobotiq2f85BCPointNetEvalCfg`
    but swaps the occluded sim2real ``scene_pc`` for the clean 512-pt FPS cloud. The live
    cloud is a fixed 512 points (FlattenMLP requires exactly the trained ``num_points``)."""

    observations: BCPointNetCleanObservationsCfg = BCPointNetCleanObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The shared clean ``data_collect`` group declares the ``contact_flags`` aux term,
        # whose obs func reads the peg-mounted ContactSensor every step (the BC policy
        # ignores it -- allowlist proprio -- but the term still executes). Register the
        # sensor here too so eval matches the collection env and does not KeyError.
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.insertive_object.spawn.activate_contact_sensors = True
        self.scene.receptive_object.spawn.activate_contact_sensors = True
        setattr(self.scene, _PEG_CONTACT_SENSOR_NAME, _peg_contact_sensor_cfg())

@configclass
class Ur5eRobotiq2f85BCPointNetCleanNotEECentricEvalCfg(Ur5eRobotiq2f85BCPointNetCleanEvalCfg):
    def __post_init__(self):
        super().__post_init__()

        # we unset it, so it becomes the robot cfg. This is what some older runs were trained on.
        self.observations.data_collect.scene_pc.params["ref_cfg"] = None


# ----------------------------------------------------------------------------
# PER-PRIM occluded data-collection env: cloud partitioned by source prim
# ----------------------------------------------------------------------------
# The occluded scene_pc is the SAME ratio-enforced occluded cloud, but each output point
# additionally carries the SOURCE PRIM it belongs to (each robot body link + insertive +
# receptive) as a trailing channel. The collector peels that channel into a separate
# `scene_pc_prim_id` dataset, so a BC run can choose -- via its `pc_parts` config -- which
# prims to feed the model (drop individual links, all robot links, or an object) and
# zero-pad to a fixed size AT TRAINING TIME. No per-prim cap / padding at collection.
# The seg channel is OFF (the prim id is the finer signal). EE (wrist_3_link) frame, and
# the contact + aux side terms are inherited from the Contact cfg for reuse.
@configclass
class Ur5eRobotiq2f85DataCollectionPCPerPrimCfg(Ur5eRobotiq2f85DataCollectionPCContactCfg):
    """Occluded data-collection env that records a PER-PRIM-LABELED point cloud (ratio-enforced
    occluded cloud + a per-point prim id). See :class:`OccludedScenePointCloud` ``per_prim``
    and ``collect_pc_demos.py`` (per-prim prim-id split)."""

    def __post_init__(self):
        super().__post_init__()
        p = self.observations.data_collect.scene_pc.params
        p["per_prim"] = True
        # Prim id carries the source signal -> drop the redundant seg channel.
        p["include_segmentation"] = False


@configclass
class Ur5eRobotiq2f85BCPointNetPerPrimEvalCfg(Ur5eRobotiq2f85BCPointNetEvalCfg):
    """BC eval for models trained on a PER-PRIM cloud. The live env emits scene_pc with a
    trailing per-point prim-id channel; ``bc_utils.bc_actions`` peels it, keeps only the
    model's ``pc_parts``, and zero-pads to the trained size. The per_prim config MUST match
    collection (same robot) so the live prim ids match the trained ones."""

    def __post_init__(self):
        super().__post_init__()
        p = self.observations.data_collect.scene_pc.params
        p["per_prim"] = True
        p["include_segmentation"] = False
        # Match the per-prim COLLECTION frame: end-effector (wrist_3_link).
        p["ref_cfg"] = SceneEntityCfg("robot", body_names="wrist_3_link")