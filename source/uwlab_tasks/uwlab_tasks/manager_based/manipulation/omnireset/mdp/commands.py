# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm

from ..assembly_keypoints import Offset
from . import utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TaskCommandCfg, TaskDependentCommandCfg


class TaskDependentCommand(CommandTerm):
    cfg: TaskDependentCommandCfg

    def __init__(self, cfg: TaskDependentCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.reset_terms_when_resample = cfg.reset_terms_when_resample
        self.interval_reset_terms = []
        self.reset_terms = []
        self.ALL_INDICES = torch.arange(self.num_envs, device=self.device)
        for name, term_cfg in self.reset_terms_when_resample.items():
            if not (term_cfg.mode == "reset" or term_cfg.mode == "interval"):
                raise ValueError(f"Term '{name}' in 'reset_terms_when_resample' must have mode 'reset' or 'interval'")
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
            if term_cfg.mode == "reset":
                self.reset_terms.append(term_cfg)
            elif term_cfg.mode == "interval":
                if term_cfg.interval_range_s != (0, 0):
                    raise ValueError(
                        "task dependent events term with interval mode current only supports range of (0, 0)"
                    )
                self.interval_reset_terms.append(term_cfg)

    def _resample_command(self, env_ids: Sequence[int]):
        for term in self.reset_terms:
            func = term.func
            func(self._env, env_ids, **term.params)
        for term in self.interval_reset_terms:
            func = term.func
            func.reset(env_ids)

    def _update_command(self):
        for term in self.interval_reset_terms:
            func = term.func
            func(self._env, self.ALL_INDICES, **term.params)

    def get_event(self, event_term_name: str):
        """Get the event term by name."""
        return self.reset_terms_when_resample.get(event_term_name).func


class TaskCommand(TaskDependentCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TaskCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TaskCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.receptive_asset_cfg.name]

        insertive_usd_paths = utils.get_usd_paths_from_spawn_cfg(self.insertive_asset.cfg.spawn)
        receptive_usd_paths = utils.get_usd_paths_from_spawn_cfg(self.receptive_asset.cfg.spawn)
        self.num_task_types = len(insertive_usd_paths)
        self.is_multitask = self.num_task_types > 1

        if self.is_multitask:
            # Multitask path uses canonical (first) receptive offset only. Multi-offset
            # receptacles (e.g. cylindrical peg) are still treated as single-canonical here;
            # extend to per-task variable-length offset lists if needed.
            self.task_type_ids = torch.arange(env.num_envs, device=self.device) % self.num_task_types
            ins_offset_pos, ins_offset_quat = [], []
            rec_offset_pos, rec_offset_quat = [], []
            pos_thresholds, ori_thresholds = [], []
            for ins_path, rec_path in zip(insertive_usd_paths, receptive_usd_paths):
                im = utils.read_metadata_from_usd_directory(ins_path)
                rm = utils.read_metadata_from_usd_directory(rec_path)
                rec_canonical_pos, rec_canonical_quat = utils.get_canonical_assembled_offset(rm)
                ins_offset_pos.append(im["assembled_offset"]["pos"])
                ins_offset_quat.append(im["assembled_offset"]["quat"])
                rec_offset_pos.append(rec_canonical_pos)
                rec_offset_quat.append(rec_canonical_quat)
                pos_thresholds.append(rm["success_thresholds"]["position"])
                ori_thresholds.append(rm["success_thresholds"]["orientation"])
            self.ins_offset_pos = torch.tensor(ins_offset_pos, device=self.device, dtype=torch.float32)
            self.ins_offset_quat = torch.tensor(ins_offset_quat, device=self.device, dtype=torch.float32)
            self.rec_offset_pos = torch.tensor(rec_offset_pos, device=self.device, dtype=torch.float32)
            self.rec_offset_quat = torch.tensor(rec_offset_quat, device=self.device, dtype=torch.float32)
            self._pos_thresholds = torch.tensor(pos_thresholds, device=self.device, dtype=torch.float32)
            self._ori_thresholds = torch.tensor(ori_thresholds, device=self.device, dtype=torch.float32)
            self.task_names = [utils.object_name_from_usd(p) for p in insertive_usd_paths]
            # Receptive offsets list (multitask = canonical-only for now: shape (T, 1, ...))
            self.rec_offsets_pos = self.rec_offset_pos.unsqueeze(1)
            self.rec_offsets_quat = self.rec_offset_quat.unsqueeze(1)
        else:
            insertive_meta = utils.read_metadata_from_usd_directory(insertive_usd_paths[0])
            receptive_meta = utils.read_metadata_from_usd_directory(receptive_usd_paths[0])
            self.insertive_asset_offset = Offset(
                pos=tuple(insertive_meta["assembled_offset"]["pos"]),
                quat=tuple(insertive_meta["assembled_offset"]["quat"]),
            )
            # Receptive may have multiple valid assembled-pose offsets. Store list for the
            # success check; canonical (first) is also stored as Offset for single-pose use.
            rec_offsets = utils.get_assembled_offsets(receptive_meta)
            self.receptive_asset_offset = Offset(pos=tuple(rec_offsets[0][0]), quat=tuple(rec_offsets[0][1]))
            self.receptive_asset_offsets_pos = torch.tensor(
                [o[0] for o in rec_offsets], dtype=torch.float32, device=self.device
            )
            self.receptive_asset_offsets_quat = torch.tensor(
                [o[1] for o in rec_offsets], dtype=torch.float32, device=self.device
            )
            self._success_position_threshold: float = receptive_meta["success_thresholds"]["position"]
            self._success_orientation_threshold: float = receptive_meta["success_thresholds"]["orientation"]

        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

    @property
    def success_position_threshold(self):
        if self.is_multitask:
            return self._pos_thresholds[self.task_type_ids]
        return self._success_position_threshold

    @property
    def success_orientation_threshold(self):
        if self.is_multitask:
            return self._ori_thresholds[self.task_type_ids]
        return self._success_orientation_threshold

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    def _apply_offset_multitask(self, asset, offset_pos_all, offset_quat_all):
        env_offset_pos = offset_pos_all[self.task_type_ids]
        env_offset_quat = offset_quat_all[self.task_type_ids]
        return math_utils.combine_frame_transforms(
            asset.data.root_pos_w, asset.data.root_quat_w, env_offset_pos, env_offset_quat,
        )

    def _update_metrics(self):
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_rot_align_error"][reset_env] = self.euler_distance[reset_env]
        self.metrics["end_of_episode_pos_align_error"][reset_env] = self.xyz_distance[reset_env]
        last_episode_success = (self.orientation_aligned & self.position_aligned)[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

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
        e_x, e_y, e_z = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
        self.euler_distance[:] = (
            math_utils.wrap_to_pi(e_x).abs()
            + math_utils.wrap_to_pi(e_y).abs()
            + math_utils.wrap_to_pi(e_z).abs()
        )
        self.xyz_distance[:] = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        # For multi-offset receptacles (e.g. cylindrical peg), orientation aligned counts when
        # rel_quat is within threshold of ANY of the receptive offsets. For single-offset
        # receptacles, this reduces to the canonical check above.
        if (not self.is_multitask) and self.receptive_asset_offsets_pos.shape[0] > 1:
            num_envs = self._env.num_envs
            ori_aligned = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            ins_pos_w = insertive_asset_alignment_pos_w
            ins_quat_w = insertive_asset_alignment_quat_w
            rec_root_pos = self.receptive_asset.data.root_pos_w
            rec_root_quat = self.receptive_asset.data.root_quat_w
            min_euler = torch.full((num_envs,), float("inf"), device=self.device)
            for k in range(self.receptive_asset_offsets_pos.shape[0]):
                off_pos = self.receptive_asset_offsets_pos[k].unsqueeze(0).expand(num_envs, -1)
                off_quat = self.receptive_asset_offsets_quat[k].unsqueeze(0).expand(num_envs, -1)
                rec_pos_w_k = rec_root_pos + math_utils.quat_apply(rec_root_quat, off_pos)
                rec_quat_w_k = math_utils.quat_mul(rec_root_quat, off_quat)
                _, rel_quat_k = math_utils.subtract_frame_transforms(
                    rec_pos_w_k, rec_quat_w_k, ins_pos_w, ins_quat_w
                )
                ex, ey, ez = math_utils.euler_xyz_from_quat(rel_quat_k)
                edist_k = (
                    math_utils.wrap_to_pi(ex).abs()
                    + math_utils.wrap_to_pi(ey).abs()
                    + math_utils.wrap_to_pi(ez).abs()
                )
                min_euler = torch.minimum(min_euler, edist_k)
                ori_aligned = ori_aligned | (edist_k < self.success_orientation_threshold)
            self.euler_distance[:] = min_euler  # report MIN error (closest matching offset)
            self.orientation_aligned[:] = ori_aligned
        else:
            self.orientation_aligned[:] = self.euler_distance < self.success_orientation_threshold
        self.metrics["average_rot_align_error"][:] = self.euler_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
