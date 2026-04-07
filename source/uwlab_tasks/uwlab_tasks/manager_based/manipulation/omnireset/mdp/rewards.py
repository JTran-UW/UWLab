# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

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
        penalize_distance: bool = False,
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

        if penalize_distance:
            # 0 when close, 1 when far — use with negative weight to penalize distance
            return torch.tanh(pos_distance / std)
        return 1 - torch.tanh(pos_distance / std)


class ProgressContext(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.params.get("receptive_asset_cfg").name]  # type: ignore

        insertive_usd_paths = utils.get_usd_paths_from_spawn_cfg(self.insertive_asset.cfg.spawn)
        receptive_usd_paths = utils.get_usd_paths_from_spawn_cfg(self.receptive_asset.cfg.spawn)
        self.num_task_types = len(insertive_usd_paths)
        self.is_multitask = self.num_task_types > 1

        if self.is_multitask:
            self.task_type_ids = torch.arange(env.num_envs, device=env.device) % self.num_task_types
            ins_offset_pos, ins_offset_quat = [], []
            rec_offset_pos, rec_offset_quat = [], []
            for ins_path, rec_path in zip(insertive_usd_paths, receptive_usd_paths):
                im = utils.read_metadata_from_usd_directory(ins_path)
                rm = utils.read_metadata_from_usd_directory(rec_path)
                ins_offset_pos.append(im["assembled_offset"]["pos"])
                ins_offset_quat.append(im["assembled_offset"]["quat"])
                rec_offset_pos.append(rm["assembled_offset"]["pos"])
                rec_offset_quat.append(rm["assembled_offset"]["quat"])
            self.ins_offset_pos = torch.tensor(ins_offset_pos, device=env.device, dtype=torch.float32)
            self.ins_offset_quat = torch.tensor(ins_offset_quat, device=env.device, dtype=torch.float32)
            self.rec_offset_pos = torch.tensor(rec_offset_pos, device=env.device, dtype=torch.float32)
            self.rec_offset_quat = torch.tensor(rec_offset_quat, device=env.device, dtype=torch.float32)
        else:
            insertive_meta = utils.read_metadata_from_usd_directory(insertive_usd_paths[0])
            receptive_meta = utils.read_metadata_from_usd_directory(receptive_usd_paths[0])
            self.insertive_asset_offset = Offset(
                pos=tuple(insertive_meta["assembled_offset"]["pos"]),
                quat=tuple(insertive_meta["assembled_offset"]["quat"]),
            )
            self.receptive_asset_offset = Offset(
                pos=tuple(receptive_meta["assembled_offset"]["pos"]),
                quat=tuple(receptive_meta["assembled_offset"]["quat"]),
            )

        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.position_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.euler_xy_distance = torch.zeros((env.num_envs), device=env.device)
        self.xyz_distance = torch.zeros((env.num_envs), device=env.device)
        self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

    def _apply_offset_multitask(self, asset, offset_pos_all, offset_quat_all):
        """Apply per-env offsets for multi-task mode."""
        env_offset_pos = offset_pos_all[self.task_type_ids]
        env_offset_quat = offset_quat_all[self.task_type_ids]
        return math_utils.combine_frame_transforms(
            asset.data.root_pos_w, asset.data.root_quat_w, env_offset_pos, env_offset_quat,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.continuous_success_counter[:] = 0

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

        if self.is_multitask:
            insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self._apply_offset_multitask(
                self.insertive_asset, self.ins_offset_pos, self.ins_offset_quat,
            )
            receptive_asset_alignment_pos_w, receptive_asset_alignment_quat_w = self._apply_offset_multitask(
                self.receptive_asset, self.rec_offset_pos, self.rec_offset_quat,
            )
        else:
            insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
                self.insertive_asset
            )
            receptive_asset_alignment_pos_w, receptive_asset_alignment_quat_w = self.receptive_asset_offset.apply(
                self.receptive_asset
            )

        insertive_asset_in_receptive_asset_frame_pos, insertive_asset_in_receptive_asset_frame_quat = (
            math_utils.subtract_frame_transforms(
                receptive_asset_alignment_pos_w,
                receptive_asset_alignment_quat_w,
                insertive_asset_alignment_pos_w,
                insertive_asset_alignment_quat_w,
            )
        )
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
        self.euler_xy_distance[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xyz_distance[:] = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < success_orientation_threshold
        self.success[:] = self.orientation_aligned & self.position_aligned

        self.continuous_success_counter[:] = torch.where(
            self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )

        self.success_monitor.success_update(
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), self.success
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


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_aligned: torch.Tensor = getattr(context_term, "position_aligned")
    return torch.where(orientation_aligned & position_aligned, 1.0, 0.0)


def gripper_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    penalize_no_contact: bool = False,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_inner_finger_contact_sensor"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_inner_finger_contact_sensor"),
) -> torch.Tensor:
    """Binary reward: both gripper fingers in contact with object above threshold.

    If penalize_no_contact is True, returns 1 when NOT in contact (use with negative weight).
    """
    left_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    left_force = torch.norm(left_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)[:, 0, :], dim=-1)
    right_force = torch.norm(right_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)[:, 0, :], dim=-1)
    in_contact = ((left_force > threshold) & (right_force > threshold)).float()
    if penalize_no_contact:
        return 1.0 - in_contact
    return in_contact


def contact_gated_dense_success(
    env: ManagerBasedRLEnv,
    std: float,
    threshold: float = 1.0,
    context: str = "progress_context",
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_inner_finger_contact_sensor"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_inner_finger_contact_sensor"),
) -> torch.Tensor:
    """Dense success reward gated by gripper contact (like pat/hand's position_command_error_tanh)."""
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    xyz_distance: torch.Tensor = getattr(context_term, "xyz_distance")
    in_contact = gripper_contact(env, threshold, left_sensor_cfg, right_sensor_cfg)
    return (1 - torch.tanh(xyz_distance / std)) * in_contact


def is_terminated_term(env: ManagerBasedRLEnv, term_keys: list[str]) -> torch.Tensor:
    """Returns 1.0 for envs where any of the named termination terms fired this step."""
    result = torch.zeros(env.num_envs, device=env.device)
    for key in term_keys:
        result = torch.max(result, env.termination_manager.get_term(key).float())
    return result


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.clamp(torch.sum(torch.square(env.action_manager.action), dim=1), 0, 1e4)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.clamp(
        torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1), 0, 1e4
    )


def mechanical_work(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"]),
) -> torch.Tensor:
    """Mechanical work: mean of abs(torque * joint_vel) * dt across arm joints only."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    work = (robot.data.applied_torque[:, joint_ids] * robot.data.joint_vel[:, joint_ids]).abs()
    return torch.clamp(work.mean(dim=1) * env.step_dt, max=1.0)


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
