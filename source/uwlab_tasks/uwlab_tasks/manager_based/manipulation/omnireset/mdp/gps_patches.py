# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-computed reset patches with per-patch GPS curriculum.

Generates K valid reset states at init using the full IK + collision pipeline,
then at runtime teleports to stored patches sampled via Beta-shaped weights
on per-patch success rates. Same pattern as OctiLab's terrain patch GPS.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils

from .events import IKCurriculumResetManager
from .success_monitor import SuccessMonitor
from .success_monitor_cfg import SuccessMonitorCfg


class PatchGPSManager(IKCurriculumResetManager):
    """Pre-computed reset patches with per-patch success tracking and Beta-shaped sampling.

    At init: generates ``gps_num_patches`` valid reset states using the parent's
    full reset pipeline (IK solve, gripper close, collision check).

    At reset: records episode success, samples a patch via Beta-weighted GPS,
    and teleports the environment to the stored state. No IK or collision
    checking at runtime.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # GPS parameters
        self.gps_num_patches: int = cfg.params.get("gps_num_patches", 10000)
        self.gps_target: float = cfg.params.get("gps_target", 0.33)
        self.gps_kappa: float = cfg.params.get("gps_kappa", 2.0)
        self.gps_temperature: float = cfg.params.get("gps_temperature", 2.0)
        self.gps_anywhere_fraction: float = cfg.params.get("gps_anywhere_fraction", 0.5)

        # Receptive object randomization range (matches GravityTrickEventCfg)
        rec_range = cfg.params.get("rec_pose_range", {
            "x": (0.3, 0.55), "y": (-0.1, 0.3), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-np.pi / 12, np.pi / 12),
        })
        self.rec_range = torch.tensor(
            [rec_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=env.device,
        )

        # Receptive object bottom offset
        receptive_usd_path = self.receptive_object.cfg.spawn.usd_path
        from . import utils
        rec_metadata = utils.read_metadata_from_usd_directory(receptive_usd_path)
        rec_bottom = rec_metadata.get("bottom_offset")
        if rec_bottom is not None:
            self.rec_bottom_offset = torch.tensor(rec_bottom.get("pos"), device=env.device).unsqueeze(0)
        else:
            self.rec_bottom_offset = torch.zeros(1, 3, device=env.device)

        # Support offset (table surface)
        self.support_object = env.scene["ur5_metal_support"]

        # Generate patches
        K = self.gps_num_patches
        self.patch_arm_joints = torch.zeros(K, self.n_joints, device=env.device)
        self.patch_gripper_joints = torch.zeros(K, len(self.gripper_joint_ids), device=env.device)
        self.patch_ins_pose = torch.zeros(K, 7, device=env.device)  # relative to env origin
        self.patch_rec_pose = torch.zeros(K, 7, device=env.device)  # relative to env origin
        self._generate_patches(env)

        # Per-patch success monitor (reuse existing SuccessMonitor)
        patch_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100, num_monitored_data=K, device=str(env.device)
        )
        self.patch_success_monitor = patch_monitor_cfg.class_type(patch_monitor_cfg)

        # Track which patch each env is currently running
        self.env_patch_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        print(f"[PatchGPSManager] Generated {K} patches, GPS target={self.gps_target}")

    def _generate_patches(self, env: ManagerBasedEnv) -> None:
        """Generate all patches using the parent's full reset pipeline."""
        K = self.gps_num_patches
        batch_size = min(env.num_envs, K)
        generated = 0

        while generated < K:
            n = min(batch_size, K - generated)
            env_ids = torch.arange(n, device=env.device)

            # Assign groups: 50-50 anywhere vs partial assembly
            n_any = int(n * (1.0 - self.gps_anywhere_fraction))
            # Flip: fraction is "partial assembly fraction" in parent, so anywhere = 1 - fraction
            n_any = int(n * self.gps_anywhere_fraction)
            n_pa = n - n_any
            self.group_id[env_ids[:n_any]] = 0  # anywhere
            self.group_id[env_ids[n_any:n]] = 1  # partial assembly

            # Randomize receptive object pose (same range as GravityTrickEventCfg)
            self._randomize_receptive(env, env_ids)

            # Run full parent reset pipeline
            ids_anywhere = env_ids[:n_any]
            ids_partial = env_ids[n_any:n]
            if ids_anywhere.numel() > 0:
                self._reset_anywhere(env, ids_anywhere)
            if ids_partial.numel() > 0:
                self._reset_partial_assembly(env, ids_partial)
            self._reset_ee(env, env_ids)
            self._close_gripper(env, env_ids)

            # Collision resample loop (same as parent __call__)
            remaining_ids = env_ids.clone()
            for _ in range(self.max_resample_attempts):
                if remaining_ids.numel() == 0:
                    break
                self.robot.update(env.sim.get_physics_dt())
                collision_free = self._check_collision(env, remaining_ids)
                colliding = remaining_ids[~collision_free]
                if colliding.numel() == 0:
                    break
                col_anywhere = colliding[self.group_id[colliding] == 0]
                col_partial = colliding[self.group_id[colliding] == 1]
                if col_anywhere.numel() > 0:
                    self._reset_anywhere(env, col_anywhere)
                if col_partial.numel() > 0:
                    self._reset_partial_assembly(env, col_partial)
                self._reset_ee(env, colliding)
                self._close_gripper(env, colliding)
                remaining_ids = colliding

            # Update data buffers to get final poses
            self.robot.update(env.sim.get_physics_dt())
            self.insertive_object.update(env.sim.get_physics_dt())
            self.receptive_object.update(env.sim.get_physics_dt())

            # Extract final states (all poses relative to env origin)
            origins = env.scene.env_origins[env_ids]

            arm_joints = self.robot.data.joint_pos[env_ids][:, self.joint_ids]
            gripper_joints = self.robot.data.joint_pos[env_ids][:, self.gripper_joint_ids]

            ins_pos_rel = self.insertive_object.data.root_pos_w[env_ids] - origins
            ins_quat = self.insertive_object.data.root_quat_w[env_ids]
            ins_pose = torch.cat([ins_pos_rel, ins_quat], dim=-1)

            rec_pos_rel = self.receptive_object.data.root_pos_w[env_ids] - origins
            rec_quat = self.receptive_object.data.root_quat_w[env_ids]
            rec_pose = torch.cat([rec_pos_rel, rec_quat], dim=-1)

            # Store
            self.patch_arm_joints[generated:generated + n] = arm_joints
            self.patch_gripper_joints[generated:generated + n] = gripper_joints
            self.patch_ins_pose[generated:generated + n] = ins_pose
            self.patch_rec_pose[generated:generated + n] = rec_pose

            generated += n

        print(
            f"[PatchGPSManager] Patch generation complete: {K} patches stored. "
            f"Arm joints range: [{self.patch_arm_joints.min():.3f}, {self.patch_arm_joints.max():.3f}], "
            f"Ins pos range: [{self.patch_ins_pose[:, :3].min():.3f}, {self.patch_ins_pose[:, :3].max():.3f}]"
        )

    def _randomize_receptive(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        """Randomize receptive object pose (same logic as reset_receptive_object_pose event)."""
        n = env_ids.numel()
        samples = math_utils.sample_uniform(
            self.rec_range[:, 0], self.rec_range[:, 1], (n, 6), device=env.device
        )
        # Position relative to support surface + bottom offset
        support_pos = self.support_object.data.root_pos_w[env_ids]
        rec_pos = support_pos + samples[:, :3] - self.rec_bottom_offset.expand(n, -1)
        rec_quat = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])

        self.receptive_object.write_root_pose_to_sim(
            torch.cat([rec_pos, rec_quat], dim=-1), env_ids=env_ids
        )
        self.receptive_object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=env.device), env_ids=env_ids
        )
        self.receptive_object.update(env.sim.get_physics_dt())

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        # Parent params (used at init only, ignored at runtime)
        dataset_dir: str = "",
        robot_ik_cfg: SceneEntityCfg | None = None,
        obj_anywhere_range: dict | None = None,
        ee_anywhere_range: dict | None = None,
        max_resample_attempts: int = 10,
        ee_lerp_range: tuple[float, float] = (0.0, 1.0),
        pa_ee_lerp_range: tuple[float, float] = (0.0, 1.0),
        collision_num_points: int = 128,
        collision_max_dist: float = 0.3,
        collision_min_dist: float = 0.0,
        # GPS params (used at init only)
        gps_num_patches: int = 10000,
        gps_target: float = 0.33,
        gps_kappa: float = 2.0,
        gps_temperature: float = 2.0,
        gps_anywhere_fraction: float = 0.5,
        rec_pose_range: dict | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = env_ids.long()
        if env_ids.numel() == 0:
            return

        # -- 1. Record success for completed episodes --
        context_term = env.reward_manager.get_term_cfg("progress_context").func
        success_mask = getattr(context_term, "success")[env_ids].float()
        self.patch_success_monitor.success_update(self.env_patch_ids[env_ids], success_mask)

        # -- 2. Log metrics --
        rates = self.patch_success_monitor.get_success_rate()  # [K]
        if "log" not in env.extras:
            env.extras["log"] = {}
        visited_mask = self.patch_success_monitor.success_size > 0
        env.extras["log"]["Metrics/mean_patch_success_rate"] = (
            rates[visited_mask].mean().item() if visited_mask.any() else 0.0
        )
        env.extras["log"]["Metrics/frontier_patch_count"] = (
            ((rates > 0.1) & (rates < 0.6)).sum().item()
        )
        env.extras["log"]["Metrics/visited_patch_count"] = visited_mask.sum().item()

        # -- 3. Sample patches via Beta-shaped weights --
        chosen, probs = SuccessMonitor.sample_by_target_rate_from_rates(
            rates, env_ids.numel(),
            target=self.gps_target, kappa=self.gps_kappa, temperature=self.gps_temperature,
        )
        self.env_patch_ids[env_ids] = chosen

        # -- 4. Teleport to selected patches --
        self._teleport_to_patches(env, env_ids, chosen)

    def _teleport_to_patches(
        self, env: ManagerBasedEnv, env_ids: torch.Tensor, patch_ids: torch.Tensor
    ) -> None:
        """Write stored patch states directly to sim. No IK, no collision check."""
        n = env_ids.numel()
        device = env.device
        origins = env.scene.env_origins[env_ids]

        # Arm joints (position + zero velocity)
        self.robot.write_joint_state_to_sim(
            self.patch_arm_joints[patch_ids],
            torch.zeros(n, self.n_joints, device=device),
            joint_ids=self.joint_ids, env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            self.patch_arm_joints[patch_ids],
            joint_ids=self.joint_ids, env_ids=env_ids,
        )

        # Gripper joints
        self.robot.write_joint_state_to_sim(
            self.patch_gripper_joints[patch_ids],
            torch.zeros(n, len(self.gripper_joint_ids), device=device),
            joint_ids=self.gripper_joint_ids, env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            self.patch_gripper_joints[patch_ids],
            joint_ids=self.gripper_joint_ids, env_ids=env_ids,
        )

        # Insertive object (stored relative to env origin, add origin back)
        ins_pose = self.patch_ins_pose[patch_ids].clone()
        ins_pose[:, :3] += origins
        self.insertive_object.write_root_pose_to_sim(ins_pose, env_ids=env_ids)
        self.insertive_object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=device), env_ids=env_ids
        )

        # Receptive object
        rec_pose = self.patch_rec_pose[patch_ids].clone()
        rec_pose[:, :3] += origins
        self.receptive_object.write_root_pose_to_sim(rec_pose, env_ids=env_ids)
        self.receptive_object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=device), env_ids=env_ids
        )

        # Zero all velocities
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.robot.data.joint_vel[env_ids]), env_ids=env_ids
        )
