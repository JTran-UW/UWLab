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
from . import symmetry, utils

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
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.receptive_asset_cfg.name]
        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        receptive_meta = utils.read_metadata_from_usd_directory(self.receptive_asset.cfg.spawn.usd_path)
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
            quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
        )
        self.receptive_asset_offset = Offset(
            pos=tuple(receptive_meta.get("assembled_offset").get("pos")),
            quat=tuple(receptive_meta.get("assembled_offset").get("quat")),
        )
        self.success_position_threshold: float = receptive_meta.get("success_thresholds").get("position")
        self.success_orientation_threshold: float = receptive_meta.get("success_thresholds").get("orientation")
        self.symmetry_groups = None
        if cfg.symmetric_success:
            ins_sym = symmetry.read_symmetry({"symmetry": cfg.insertive_symmetry} if cfg.insertive_symmetry else insertive_meta)
            rec_sym = symmetry.read_symmetry({"symmetry": cfg.receptive_symmetry} if cfg.receptive_symmetry else receptive_meta)
            self.symmetry_groups = (symmetry.symmetry_group(ins_sym).to(self.device), symmetry.symmetry_group(rec_sym).to(self.device))
            self.receptive_offset_tensors = (
                torch.tensor(self.receptive_asset_offset.pos, device=self.device),
                torch.tensor(self.receptive_asset_offset.quat, device=self.device),
            )
        # several success targets (receptive metadata ``success_targets: all``): success at the best-scoring one
        offsets = symmetry.success_target_offsets(receptive_meta)
        self.success_targets = None
        if len(offsets) > 1:
            self.success_targets = [
                (torch.tensor(p, device=self.device), torch.tensor(q, device=self.device)) for p, q in offsets
            ]
            if self.symmetry_groups is None:
                identity = symmetry.symmetry_group(None).to(self.device)
                self.symmetry_groups = (identity, identity)

        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_xy_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_rot_align_error"][reset_env] = self.euler_xy_distance[reset_env]
        self.metrics["end_of_episode_pos_align_error"][reset_env] = self.xyz_distance[reset_env]
        last_episode_success = (self.orientation_aligned & self.position_aligned)[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

        # logs current data
        insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = self.insertive_asset_offset.apply(
            self.insertive_asset
        )
        if self.success_targets is not None:
            ins_pos_r, ins_quat_r = math_utils.subtract_frame_transforms(
                self.receptive_asset.data.root_pos_w.torch,
                self.receptive_asset.data.root_quat_w.torch,
                insertive_asset_alignment_pos_w,
                insertive_asset_alignment_quat_w,
            )
            self.xyz_distance[:], self.euler_xy_distance[:], _ = symmetry.multi_target_alignment_error(
                ins_pos_r, ins_quat_r, self.success_targets, *self.symmetry_groups,
                (self.success_position_threshold, self.success_orientation_threshold),
            )
        else:
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
            self.xyz_distance[:], self.euler_xy_distance[:] = self.alignment_error(
                insertive_asset_in_receptive_asset_frame_pos, insertive_asset_in_receptive_asset_frame_quat
            )
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < self.success_orientation_threshold
        self.metrics["average_rot_align_error"][:] = self.euler_xy_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

    def alignment_error(self, rel_pos: torch.Tensor, rel_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Position error and |roll| + |pitch| (yaw ignored) of the insertive alignment frame in the receptive one."""
        if self.symmetry_groups is not None:
            return symmetry.symmetric_alignment_error(
                rel_pos, rel_quat, self.receptive_offset_tensors, *self.symmetry_groups,
                (self.success_position_threshold, self.success_orientation_threshold),
            )
        e_x, e_y, _ = math_utils.euler_xyz_from_quat(rel_quat)
        return torch.norm(rel_pos, dim=1), math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class MultiStageTaskCommand(TaskCommand):
    """``TaskCommand`` whose success requires every stage aligned: the primary insertive/receptive pair plus each
    ``cfg.extra_stages`` pair (same thresholds, single offset, yaw ignored). ``position_aligned`` and
    ``orientation_aligned`` hold the all-stage flag, so success metrics, ``SuccessLatch`` and SGS see full assembly."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.stages = []
        for s in cfg.extra_stages:
            ins, rec = env.scene[s["insertive"]], env.scene[s["receptive"]]
            io = s.get("insertive_offset") or utils.read_metadata_from_usd_directory(ins.cfg.spawn.usd_path)["assembled_offset"]
            ro = s["receptive_offset"]
            self.stages.append((ins, rec, Offset(pos=tuple(io["pos"]), quat=tuple(io["quat"])),
                                Offset(pos=tuple(ro["pos"]), quat=tuple(ro["quat"]))))
        self.stage_aligned = torch.zeros(len(self.stages) + 1, self.num_envs, dtype=torch.bool, device=self.device)
        for k in range(len(self.stages) + 1):
            self.metrics[f"end_of_episode_stage{k}_success_rate"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        reset_env = self._env.episode_length_buf == 0
        for k in range(len(self.stages) + 1):
            self.metrics[f"end_of_episode_stage{k}_success_rate"][reset_env] = self.stage_aligned[k][reset_env].float()
        super()._update_metrics()
        all_ok = self.orientation_aligned & self.position_aligned
        self.stage_aligned[0] = all_ok
        for k, (ins, rec, io, ro) in enumerate(self.stages):
            ip, iq = io.apply(ins)
            rp, rq = ro.apply(rec)
            p, q = math_utils.subtract_frame_transforms(rp, rq, ip, iq)
            e_x, e_y, _ = math_utils.euler_xyz_from_quat(q)
            tilt = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
            ok = (p.norm(dim=1) < self.success_position_threshold) & (tilt < self.success_orientation_threshold)
            self.stage_aligned[k + 1] = ok
            all_ok = all_ok & ok
        self.position_aligned[:] = all_ok
        self.orientation_aligned[:] = all_ok
