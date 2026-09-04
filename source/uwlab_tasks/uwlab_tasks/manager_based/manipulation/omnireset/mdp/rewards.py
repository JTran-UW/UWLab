# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from ..assembly_keypoints import Offset
from . import utils
from .collision_analyzer_cfg import CollisionAnalyzerCfg
from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import TaskCommand


class ee_asset_distance_tanh(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.root_asset_cfg = cfg.params.get("root_asset_cfg")
        self.target_asset_cfg = cfg.params.get("target_asset_cfg")
        self.std = cfg.params.get("std")

        root_asset_offset_metadata_key: str = cfg.params.get("root_asset_offset_metadata_key")
        target_asset_offset_metadata_key: str = cfg.params.get("target_asset_offset_metadata_key")

        self.root_asset = env.scene[self.root_asset_cfg.name]
        root_usd_path = self.root_asset.cfg.spawn.usd_path
        root_metadata = utils.read_metadata_from_usd_directory(root_usd_path)
        root_offset_data = root_metadata.get(root_asset_offset_metadata_key)
        self.root_asset_offset = Offset(pos=root_offset_data.get("pos"), quat=root_offset_data.get("quat"))

        self.target_asset = env.scene[self.target_asset_cfg.name]
        if target_asset_offset_metadata_key is not None:
            target_usd_path = self.target_asset.cfg.spawn.usd_path
            target_metadata = utils.read_metadata_from_usd_directory(target_usd_path)
            target_offset_data = target_metadata.get(target_asset_offset_metadata_key)
            self.target_asset_offset = Offset(pos=target_offset_data.get("pos"), quat=target_offset_data.get("quat"))
        else:
            self.target_asset_offset = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        root_asset_cfg: SceneEntityCfg,
        target_asset_cfg: SceneEntityCfg,
        root_asset_offset_metadata_key: str,
        target_asset_offset_metadata_key: str | None = None,
        std: float = 0.1,
    ) -> torch.Tensor:
        root_asset_alignment_pos_w, root_asset_alignment_quat_w = self.root_asset_offset.combine(
            self.root_asset.data.body_link_pos_w[:, root_asset_cfg.body_ids].view(-1, 3),
            self.root_asset.data.body_link_quat_w[:, root_asset_cfg.body_ids].view(-1, 4),
        )
        if self.target_asset_offset is None:
            target_asset_alignment_pos_w = self.target_asset.data.root_pos_w.view(-1, 3)
            target_asset_alignment_quat_w = self.target_asset.data.root_quat_w.view(-1, 4)
        else:
            target_asset_alignment_pos_w, target_asset_alignment_quat_w = self.target_asset_offset.apply(
                self.target_asset
            )
        target_asset_in_root_asset_frame_pos, target_asset_in_root_asset_frame_angle_axis = (
            math_utils.compute_pose_error(
                root_asset_alignment_pos_w,
                root_asset_alignment_quat_w,
                target_asset_alignment_pos_w,
                target_asset_alignment_quat_w,
            )
        )

        pos_distance = torch.norm(target_asset_in_root_asset_frame_pos, dim=1)

        return 1 - torch.tanh(pos_distance / std)



class ee_pos_distance_tanh(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.root_asset_cfg = cfg.params.get("root_asset_cfg")
        self.target_asset_cfg = cfg.params.get("target_cfg")
        self.std = cfg.params.get("std")

        root_asset_offset_metadata_key: str = cfg.params.get("root_asset_offset_metadata_key")
        target_asset_offset_metadata_key: str = cfg.params.get("target_asset_offset_metadata_key")

        self.root_asset = env.scene[self.root_asset_cfg.name]
        root_usd_path = self.root_asset.cfg.spawn.usd_path
        root_metadata = utils.read_metadata_from_usd_directory(root_usd_path)
        root_offset_data = root_metadata.get(root_asset_offset_metadata_key)
        self.root_asset_offset = Offset(pos=root_offset_data.get("pos"), quat=root_offset_data.get("quat"))

        self.target_asset = env.scene[self.target_asset_cfg.name]
        if target_asset_offset_metadata_key is not None:
            target_usd_path = self.target_asset.cfg.spawn.usd_path
            target_metadata = utils.read_metadata_from_usd_directory(target_usd_path)
            target_offset_data = target_metadata.get(target_asset_offset_metadata_key)
            self.target_asset_offset = Offset(pos=target_offset_data.get("pos"), quat=target_offset_data.get("quat"))
        else:
            self.target_asset_offset = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        root_asset_cfg: SceneEntityCfg,
        target_asset_cfg: SceneEntityCfg,
        root_asset_offset_metadata_key: str,
        target_asset_offset_metadata_key: str | None = None,
        std: float = 0.1,
    ) -> torch.Tensor:
        root_asset_alignment_pos_w, root_asset_alignment_quat_w = self.root_asset_offset.combine(
            self.root_asset.data.body_link_pos_w[:, root_asset_cfg.body_ids].view(-1, 3),
            self.root_asset.data.body_link_quat_w[:, root_asset_cfg.body_ids].view(-1, 4),
        )
        if self.target_asset_offset is None:
            target_asset_alignment_pos_w = self.target_asset.data.root_pos_w.view(-1, 3)
            target_asset_alignment_quat_w = self.target_asset.data.root_quat_w.view(-1, 4)
        else:
            target_asset_alignment_pos_w, target_asset_alignment_quat_w = self.target_asset_offset.apply(
                self.target_asset
            )
        target_asset_in_root_asset_frame_pos, target_asset_in_root_asset_frame_angle_axis = (
            math_utils.compute_pose_error(
                root_asset_alignment_pos_w,
                root_asset_alignment_quat_w,
                target_asset_alignment_pos_w,
                target_asset_alignment_quat_w,
            )
        )

        pos_distance = torch.norm(target_asset_in_root_asset_frame_pos, dim=1)

        return 1 - torch.tanh(pos_distance / std)


class ProgressContext(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.params.get("receptive_asset_cfg").name]  # type: ignore

        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        receptive_meta = utils.read_metadata_from_usd_directory(self.receptive_asset.cfg.spawn.usd_path)
        insertive_offset = utils.get_assembled_offset(insertive_meta)
        receptive_offset = utils.get_assembled_offset(receptive_meta)
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_offset.get("pos")),
            quat=tuple(insertive_offset.get("quat")),
        )
        self.receptive_asset_offset = Offset(
            pos=tuple(receptive_offset.get("pos")),
            quat=tuple(receptive_offset.get("quat")),
        )
        _offs = utils.get_assembled_offsets(receptive_meta)
        self.receptive_offsets_pos = torch.tensor([o["pos"] for o in _offs], dtype=torch.float32, device=env.device)
        self.receptive_offsets_quat = torch.tensor([o["quat"] for o in _offs], dtype=torch.float32, device=env.device)

        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.position_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.euler_xy_distance = torch.zeros((env.num_envs), device=env.device)
        self.xyz_distance = torch.zeros((env.num_envs), device=env.device)
        self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        # Only the envs that actually reset: a blanket [:] wipes every env's progress whenever any
        # one env ends, which starves consecutive_success_state (terminations run before rewards).
        if env_ids is not None and os.environ.get("UWLAB_LEGACY_SUCCESS_COUNTER_RESET") == "1" and not getattr(self, "_legacy_wipe_logged", False):
            self._legacy_wipe_logged = True
            print("[legacy] UWLAB_LEGACY_SUCCESS_COUNTER_RESET=1 -> blanket success-counter wipe ACTIVE (pre-8d15384 behavior)")
        if env_ids is None or os.environ.get("UWLAB_LEGACY_SUCCESS_COUNTER_RESET") == "1":
            # Pre-8d15384 behavior (blanket wipe; starves consecutive_success_state). Env-var gated,
            # for byte-reproducing pre-fix data collections ONLY -- never set this for training.
            # DO NOT remove as dead code: this gate is the regression fixture that byte-reproduces
            # expert_rb/full_resets_peghole_anywhere_w_dr_n_step_3_sparse_no_priv.pt (see
            # .claude/gap-finetune-handoff.md, "byte-certified").
            self.continuous_success_counter[:] = 0
        else:
            self.continuous_success_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        insertive_asset_cfg: SceneEntityCfg,
        receptive_asset_cfg: SceneEntityCfg,
        command_context: str = "task_command",
    ) -> torch.Tensor:
        task_command: TaskCommand = env.command_manager.get_term(command_context)
        success_position_threshold = task_command.success_position_threshold
        success_orientation_threshold = task_command.success_orientation_threshold
        # symmetry-aware: closest of the receptive asset's assembled offsets; yaw ignored
        ins_pos_w, ins_quat_w = self.insertive_asset_offset.apply(self.insertive_asset)
        xyz, exy = utils.assembled_alignment_error(
            ins_pos_w, ins_quat_w, self.receptive_asset.data.root_pos_w, self.receptive_asset.data.root_quat_w,
            self.receptive_offsets_pos, self.receptive_offsets_quat,
        )
        self.euler_xy_distance[:] = exy
        self.xyz_distance[:] = xyz
        self.position_aligned[:] = self.xyz_distance < success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < success_orientation_threshold
        self.success[:] = self.orientation_aligned & self.position_aligned

        # Update continuous success counter
        self.continuous_success_counter[:] = torch.where(
            self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )

        # Update success monitor
        self.success_monitor.success_update(
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), self.success
        )

        return torch.zeros(env.num_envs, device=env.device)


def axis_keypoints_local(extent: float, num: int, device) -> torch.Tensor:
    """``num`` points evenly spaced along the object's local +z axis over ``[-extent, +extent]``.

    Deliberately ON THE AXIS rather than at bounding-box corners. A peg is a solid of revolution:
    spin about its own long axis does not change whether it can be inserted. Corner keypoints would
    penalise that spin and make the learned task strictly harder than the real one. Axis keypoints
    are spin-invariant by construction while still capturing position AND tilt -- the two DoF that
    actually matter -- because a tilt moves the end points apart even when the centre coincides.
    """
    kp = torch.zeros(num, 3, device=device)
    kp[:, 2] = torch.linspace(-extent, extent, num, device=device)
    return kp


def transform_keypoints(pos: torch.Tensor, quat: torch.Tensor, kp_local: torch.Tensor) -> torch.Tensor:
    """Rigidly place ``kp_local`` [K,3] under each pose (``pos`` [N,3], ``quat`` [N,4]) -> [N,K,3]."""
    n, k = pos.shape[0], kp_local.shape[0]
    q = quat[:, None, :].expand(n, k, 4).reshape(-1, 4)
    p = kp_local[None, :, :].expand(n, k, 3).reshape(-1, 3)
    return math_utils.quat_apply(q, p).reshape(n, k, 3) + pos[:, None, :]


class GCProgressContext(ManagerTermBase):
    """Goal-conditioned analogue of ProgressContext.

    Tracks pose error of the insertive object and the EE (wrist_3_link) against the
    GCMRM-sampled goal state. Exposes the same success/aligned/counter attributes as
    ProgressContext so success_reward and the GCMRM success hook work unchanged.

    ``include_ee`` selects what counts as success. With ``include_ee=False`` (the default)
    ONLY the insertive object's pose matters -- the EE may be anywhere. Requiring the EE to
    also hit its goal makes success a conjunction of two independent 6-DoF constraints, which
    is a far sparser target to learn from. EE distances are still computed and exposed, so
    turning ``include_ee=True`` restores the stricter criterion without any other change.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.insertive_asset: RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore
        self.ee_asset_cfg: SceneEntityCfg = cfg.params.get("ee_asset_cfg")
        self.ee_asset: Articulation = env.scene[self.ee_asset_cfg.name]
        self.ee_body_idx = 0 if isinstance(self.ee_asset_cfg.body_ids, slice) else self.ee_asset_cfg.body_ids[0]

        self.insertive_xyz_distance = torch.zeros((env.num_envs), device=env.device)
        self.insertive_angle_distance = torch.zeros((env.num_envs), device=env.device)
        self.ee_xyz_distance = torch.zeros((env.num_envs), device=env.device)
        self.ee_angle_distance = torch.zeros((env.num_envs), device=env.device)
        self.insertive_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.ee_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.position_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.success = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.continuous_success_counter = torch.zeros((env.num_envs), dtype=torch.int32, device=env.device)

        # Keypoint objective. One quantity in METRES couples position and tilt, so there is a single
        # threshold and a single shaping length-scale -- no metres-vs-radians pairing to keep in sync.
        num_kp = int(cfg.params.get("num_keypoints", 4))
        extent = float(cfg.params.get("keypoint_extent", -1.0))
        if extent <= 0:
            # No bounding box in metadata; `bottom_offset` gives the origin->tip distance along z.
            try:
                meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
                extent = abs(float(meta["bottom_offset"]["pos"][2]))
            except Exception:
                extent = 0.03
        self.keypoint_extent = extent
        self.keypoints_local = axis_keypoints_local(extent, num_kp, env.device)
        self.keypoint_max_distance = torch.zeros((env.num_envs), device=env.device)
        self.keypoint_mean_distance = torch.zeros((env.num_envs), device=env.device)

        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        # keypoint debug markers, created lazily on first use (sim must be up)
        self._kp_markers = None

    def _get_kp_markers(self):
        if self._kp_markers is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            def sphere(path, color):
                return VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path=path,
                        markers={
                            "p": sim_utils.SphereCfg(
                                radius=0.006, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                            )
                        },
                    )
                )

            # ORANGE = current peg keypoints, BLUE = goal keypoints
            self._kp_markers = (
                sphere("/Visuals/GCKeypoints/current", (1.0, 0.45, 0.0)),
                sphere("/Visuals/GCKeypoints/goal", (0.0, 0.45, 1.0)),
            )
        return self._kp_markers

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self.continuous_success_counter[:] = 0
        else:
            self.continuous_success_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        insertive_asset_cfg: SceneEntityCfg,
        ee_asset_cfg: SceneEntityCfg,
        event_term_name: str = "reset_from_reset_states",
        command_context: str = "task_command",
        include_ee: bool = False,
        require_orientation: bool = True,
        position_threshold: float = -1.0,
        orientation_threshold: float = -1.0,
        success_mode: str = "pose",
        num_keypoints: int = 4,
        keypoint_extent: float = -1.0,
        keypoint_threshold: float = 0.01,
        keypoint_threshold_start: float = -1.0,
        keypoint_threshold_min: float = 0.01,
        debug_vis: bool = False,
        threshold_promote_at: float = 0.7,
        threshold_demote_at: float = 0.55,
        threshold_factor: float = 0.9,
        threshold_cooldown: int = 12800,
    ) -> torch.Tensor:
        task_command: TaskCommand = env.command_manager.get_term(command_context)
        # The command's thresholds come from the receptive object's USD metadata, so they are not
        # reachable by a Hydra override. These sentinels (any value > 0 wins) make them tunable
        # per-run for curriculum/difficulty sweeps. -1.0 rather than None deliberately: the config
        # type-checker compares against the DEFAULT's type and rejects float-over-None.
        success_position_threshold = (
            position_threshold if position_threshold > 0 else task_command.success_position_threshold
        )
        success_orientation_threshold = (
            orientation_threshold if orientation_threshold > 0 else task_command.success_orientation_threshold
        )

        goal_state = env.event_manager.get_term_cfg(event_term_name).func.goal_state
        env_origins = env.scene.env_origins

        insertive_goal_pose = goal_state["rigid_object"][insertive_asset_cfg.name]["root_pose"]
        self.insertive_xyz_distance[:] = torch.norm(
            self.insertive_asset.data.root_pos_w - (insertive_goal_pose[:, :3] + env_origins), dim=1
        )
        self.insertive_angle_distance[:] = math_utils.quat_error_magnitude(
            self.insertive_asset.data.root_quat_w, insertive_goal_pose[:, 3:7]
        )

        ee_goal_pose = goal_state["articulation"]["robot"]["ee_pose"]
        self.ee_xyz_distance[:] = torch.norm(
            self.ee_asset.data.body_link_pos_w[:, self.ee_body_idx] - (ee_goal_pose[:, :3] + env_origins), dim=1
        )
        self.ee_angle_distance[:] = math_utils.quat_error_magnitude(
            self.ee_asset.data.body_link_quat_w[:, self.ee_body_idx], ee_goal_pose[:, 3:7]
        )

        self.insertive_aligned[:] = (self.insertive_xyz_distance < success_position_threshold) & (
            self.insertive_angle_distance < success_orientation_threshold
        )
        self.ee_aligned[:] = (self.ee_xyz_distance < success_position_threshold) & (
            self.ee_angle_distance < success_orientation_threshold
        )
        # Success is insertive-object-only unless include_ee is set; see the class docstring.
        if include_ee:
            self.position_aligned[:] = (self.insertive_xyz_distance < success_position_threshold) & (
                self.ee_xyz_distance < success_position_threshold
            )
            self.orientation_aligned[:] = (self.insertive_angle_distance < success_orientation_threshold) & (
                self.ee_angle_distance < success_orientation_threshold
            )
        else:
            self.position_aligned[:] = self.insertive_xyz_distance < success_position_threshold
            self.orientation_aligned[:] = self.insertive_angle_distance < success_orientation_threshold

        # Position-only success: drop the orientation conjunct entirely. Hitting an arbitrary
        # dataset goal within BOTH 3cm and 0.2rad is a very small target; this is the next rung
        # down the simplification ladder when nothing is learning.
        if not require_orientation:
            self.orientation_aligned[:] = True

        # Keypoint distances are always computed (cheap, and useful to log even in pose mode).
        cur_kp = transform_keypoints(
            self.insertive_asset.data.root_pos_w, self.insertive_asset.data.root_quat_w, self.keypoints_local
        )
        goal_kp = transform_keypoints(
            insertive_goal_pose[:, :3] + env_origins, insertive_goal_pose[:, 3:7], self.keypoints_local
        )
        kp_d = torch.norm(cur_kp - goal_kp, dim=-1)
        self.keypoint_max_distance[:] = kp_d.max(dim=1).values
        self.keypoint_mean_distance[:] = kp_d.mean(dim=1)

        if debug_vis:
            cur_m, goal_m = self._get_kp_markers()
            cur_m.visualize(translations=cur_kp.reshape(-1, 3))
            goal_m.visualize(translations=goal_kp.reshape(-1, 3))

        if success_mode == "keypoint":
            active_threshold = self._advance_threshold_curriculum(
                env,
                event_term_name=event_term_name,
                static_threshold=keypoint_threshold,
                start=keypoint_threshold_start,
                minimum=keypoint_threshold_min,
                promote_at=threshold_promote_at,
                demote_at=threshold_demote_at,
                factor=threshold_factor,
                cooldown=threshold_cooldown,
            )
            # ALL keypoints must be within threshold -- the max is the honest criterion.
            aligned = self.keypoint_max_distance < active_threshold
            self.position_aligned[:] = aligned
            self.orientation_aligned[:] = True
        self.success[:] = self.orientation_aligned & self.position_aligned

        self.continuous_success_counter[:] = torch.where(
            self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )

        self.success_monitor.success_update(
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), self.success
        )

        return torch.zeros(env.num_envs, device=env.device)


    def _advance_threshold_curriculum(
        self,
        env: ManagerBasedRLEnv,
        event_term_name: str,
        static_threshold: float,
        start: float,
        minimum: float,
        promote_at: float,
        demote_at: float,
        factor: float,
        cooldown: int,
    ) -> float:
        """Tighten the keypoint success tolerance as the policy earns it; loosen when it stalls.

        A second difficulty axis alongside the reset manager's neighbour-rank curriculum. Both read
        the SAME per-episode success rate, which makes the pair self-regulating: whichever axis
        steps first depresses the shared rate below ``promote_at``, so they cannot both run away at
        once. ``cooldown`` defaults to twice the rank curriculum's so tolerance steps at half the
        rank cadence rather than compounding with it every window -- the 2026-08-20 collapse came
        from difficulty stacking faster than the 100-episode success window could report on it.

        ``start <= 0`` disables the curriculum and pins the tolerance at ``static_threshold``.
        """
        if start <= 0:
            return static_threshold

        if not hasattr(self, "_kp_threshold"):
            self._kp_threshold = float(start)
            self._kp_threshold_cooldown = -(10**9)  # allow the first adjustment immediately

        # Drive off the reset manager's monitor, not this term's own: this one is refreshed every
        # STEP with the whole env batch, so its 100-slot ring holds an instantaneous dwell fraction
        # rather than a per-episode success rate. The event term's monitor is written once per
        # episode on reset, which is the quantity the rank curriculum already gates on.
        rate = float("nan")
        reset_term = env.event_manager.get_term_cfg(event_term_name).func
        monitor = getattr(reset_term, "success_monitor", None)
        if monitor is not None:
            rates = monitor.get_success_rate()
            probs = getattr(reset_term, "probs", None)
            # Probability-weighted so a rarely-drawn reset path cannot dominate the tolerance.
            rate = float((rates * probs).sum().item()) if probs is not None else float(rates.mean().item())

        step = int(getattr(env, "common_step_counter", 0))
        if step - self._kp_threshold_cooldown >= max(1, cooldown) and rate == rate:  # rate==rate filters NaN
            self._kp_threshold_cooldown = step
            if rate >= promote_at:
                self._kp_threshold = max(minimum, self._kp_threshold * factor)
            elif rate <= demote_at:
                self._kp_threshold = min(start, self._kp_threshold / factor)

        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"]["curriculum/keypoint_threshold"] = self._kp_threshold
        env.extras["log"]["curriculum/threshold_driving_rate"] = rate
        return self._kp_threshold


class ProgressContextReaching(ManagerTermBase):
    """Reaching-task analogue of ProgressContext. Tracks EE-link → target-marker pose error.

    No USD metadata reads, no command lookup -- thresholds and asset cfgs are passed as params.
    Exposes the same attributes (xyz_distance, euler_xy_distance, position_aligned,
    orientation_aligned, success, continuous_success_counter) so dense_success_reward and
    success_reward can read it via context lookup.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ee_asset_cfg: SceneEntityCfg = cfg.params.get("ee_asset_cfg")
        self.target_asset_cfg: SceneEntityCfg = cfg.params.get("target_asset_cfg")
        self.success_position_threshold: float = cfg.params.get("success_position_threshold", 0.03)
        self.success_orientation_threshold: float = cfg.params.get("success_orientation_threshold", 0.2)

        self.ee_asset: Articulation = env.scene[self.ee_asset_cfg.name]
        self.target_asset: RigidObject = env.scene[self.target_asset_cfg.name]
        self.ee_body_idx = (
            0 if isinstance(self.ee_asset_cfg.body_ids, slice) else self.ee_asset_cfg.body_ids[0]
        )

        self.orientation_aligned = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self.position_aligned = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self.euler_xy_distance = torch.zeros((env.num_envs,), device=env.device)
        self.xyz_distance = torch.zeros((env.num_envs,), device=env.device)
        self.success = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self.continuous_success_counter = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is not None and os.environ.get("UWLAB_LEGACY_SUCCESS_COUNTER_RESET") == "1" and not getattr(self, "_legacy_wipe_logged", False):
            self._legacy_wipe_logged = True
            print("[legacy] UWLAB_LEGACY_SUCCESS_COUNTER_RESET=1 -> blanket success-counter wipe ACTIVE (pre-8d15384 behavior)")
        if env_ids is None or os.environ.get("UWLAB_LEGACY_SUCCESS_COUNTER_RESET") == "1":
            # Pre-8d15384 behavior (blanket wipe; starves consecutive_success_state). Env-var gated,
            # for byte-reproducing pre-fix data collections ONLY -- never set this for training.
            # DO NOT remove as dead code: this gate is the regression fixture that byte-reproduces
            # expert_rb/full_resets_peghole_anywhere_w_dr_n_step_3_sparse_no_priv.pt (see
            # .claude/gap-finetune-handoff.md, "byte-certified").
            self.continuous_success_counter[:] = 0
        else:
            self.continuous_success_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ee_asset_cfg: SceneEntityCfg,
        target_asset_cfg: SceneEntityCfg,
        success_position_threshold: float = 0.03,
        success_orientation_threshold: float = 0.2,
    ) -> torch.Tensor:
        ee_pos_w = self.ee_asset.data.body_link_pos_w[:, self.ee_body_idx]
        ee_quat_w = self.ee_asset.data.body_link_quat_w[:, self.ee_body_idx]
        target_pos_w = self.target_asset.data.root_pos_w
        target_quat_w = self.target_asset.data.root_quat_w

        ee_in_target_pos, ee_in_target_quat = math_utils.subtract_frame_transforms(
            target_pos_w, target_quat_w, ee_pos_w, ee_quat_w
        )

        e_x, e_y, _ = math_utils.euler_xyz_from_quat(ee_in_target_quat)
        self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xyz_distance[:] = torch.norm(ee_in_target_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        # Reaching task: success is position-only (orientation Kp is too weak to track reliably).
        self.orientation_aligned[:] = True
        self.success[:] = self.position_aligned

        self.continuous_success_counter[:] = torch.where(
            self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )

        return torch.zeros(env.num_envs, device=env.device)


def dense_success_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:

    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    angle_diff: torch.Tensor = getattr(context_term, "euler_xy_distance")
    xyz_distance: torch.Tensor = getattr(context_term, "xyz_distance")

    # Normalize the distances by std
    angle_diff = torch.exp(-angle_diff / std)
    xyz_distance = torch.exp(-xyz_distance / std)
    stacked = torch.stack([angle_diff, xyz_distance], dim=0)
    return torch.mean(stacked, dim=0)


def gc_insertive_success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    """1.0 when the insertive object is within the goal position and orientation thresholds."""
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    return context_term.insertive_aligned.float()


def gc_ee_success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    """1.0 when the EE is within the goal position and orientation thresholds."""
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    return context_term.ee_aligned.float()


def gc_keypoint_dense_reward(
    env: ManagerBasedRLEnv, std: float, context: str = "progress_context"
) -> torch.Tensor:
    """Dense shaping from mean keypoint distance: ``exp(-mean_kp_dist / std)``, in [0, 1].

    The counterpart to ``success_mode="keypoint"``. Because keypoint distance is a single quantity
    in metres, this needs ONE ``std`` -- there is no position/orientation pair to keep consistent,
    which is the failure mode `gc_dense_success_reward` had to grow a `rot_std` to work around.
    Mean (not max) is used for shaping so every keypoint contributes gradient; success still uses
    the max, so shaping is smoother than the criterion but points the same way.
    """
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    return torch.exp(-context_term.keypoint_mean_distance / std)


def gc_dense_success_reward(
    env: ManagerBasedRLEnv,
    std: float,
    context: str = "progress_context",
    include_ee: bool = False,
    include_orientation: bool = True,
    rot_std: float = -1.0,
    use_keypoints: bool = False,
) -> torch.Tensor:
    """Dense goal-reaching reward from angle and xyz distance to the goal, in [0, 1].

    With ``include_ee=False`` (the default) only the insertive object contributes, matching the
    insertive-only success criterion -- a dense term that rewards EE proximity while success
    ignores the EE would pull the policy toward states that never score. Set ``include_ee=True``
    to average the EE terms back in.
    """
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    if use_keypoints:
        # Single quantity in metres: position and tilt are already coupled, so one `std` suffices
        # and `rot_std`/`include_orientation` are irrelevant here. Pairs with success_mode="keypoint".
        return torch.exp(-context_term.keypoint_mean_distance / std)
    # Position is in METRES and orientation in RADIANS, whose useful ranges differ by ~10x. A single
    # shared `std` therefore cannot serve both: sharpening position to 0.1 makes the angle term
    # exp(-1.2/0.1) ~ 0 (dead), while a std large enough for angle leaves position nearly flat.
    # rot_std defaults to <=0 meaning "use std", so existing configs are unchanged.
    pos_std = std
    ang_std = rot_std if rot_std > 0 else std
    # When success is position-only (`require_orientation=False` on the context), keeping the
    # angle term here rewards alignment that success does not score -- the same incoherence as
    # shaping toward the EE while ignoring it. include_orientation=False drops it.
    if not include_orientation:
        insertive_pos = torch.exp(-context_term.insertive_xyz_distance / pos_std)
        if not include_ee:
            return insertive_pos
        return (insertive_pos + torch.exp(-context_term.ee_xyz_distance / pos_std)) / 2.0
    insertive = torch.exp(-context_term.insertive_angle_distance / ang_std) + torch.exp(
        -context_term.insertive_xyz_distance / pos_std
    )
    if not include_ee:
        return insertive / 2.0
    ee = torch.exp(-context_term.ee_angle_distance / ang_std) + torch.exp(
        -context_term.ee_xyz_distance / pos_std
    )
    return (insertive + ee) / 4.0


def dense_success_reward_no_angle(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:

    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    xyz_distance: torch.Tensor = getattr(context_term, "xyz_distance")

    # Normalize the distances by std
    xyz_distance = torch.exp(-xyz_distance / std)
    return torch.mean(xyz_distance, dim=0)


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_aligned: torch.Tensor = getattr(context_term, "position_aligned")
    return torch.where(orientation_aligned & position_aligned, 1.0, 0.0)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.clamp(torch.sum(torch.square(env.action_manager.action), dim=1), 0, 1e4)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.clamp(
        torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1), 0, 1e4
    )


def joint_vel_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1), 0, 1e4)


class collision_free(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._env = env

        self.collision_analyzer_cfg = cfg.params.get("collision_analyzer_cfg")
        self.collision_analyzer = self.collision_analyzer_cfg.class_type(self.collision_analyzer_cfg, self._env)

    def __call__(self, env: ManagerBasedRLEnv, collision_analyzer_cfg: CollisionAnalyzerCfg) -> torch.Tensor:
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_free = self.collision_analyzer(env, all_env_ids)

        return collision_free
