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

    from .commands_cfg import TaskCommandCfg, TaskCommandReachingCfg, TaskDependentCommandCfg


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
        self.success_position_threshold: float = receptive_meta.get("success_thresholds").get("position")
        self.success_orientation_threshold: float = receptive_meta.get("success_thresholds").get("orientation")

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
        # symmetry-aware: closest of the receptive asset's assembled offsets; yaw ignored
        ins_pos_w, ins_quat_w = self.insertive_asset_offset.apply(self.insertive_asset)
        xyz, exy = utils.assembled_alignment_error(
            ins_pos_w, ins_quat_w, self.receptive_asset.data.root_pos_w, self.receptive_asset.data.root_quat_w,
            self.receptive_offsets_pos, self.receptive_offsets_quat,
        )
        self.euler_xy_distance[:] = exy
        self.xyz_distance[:] = xyz
        self.position_aligned[:] = self.xyz_distance < self.success_position_threshold
        self.orientation_aligned[:] = self.euler_xy_distance < self.success_orientation_threshold
        self.metrics["average_rot_align_error"][:] = self.euler_xy_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class TaskCommandReaching(TaskDependentCommand):
    """Command generator for the reaching task. Tracks EE-link → target-marker pose error.

    Mirrors the metrics produced by :class:`TaskCommand` (peg insertion) so that the same
    wandb panels (Episode/Metrics/task_command/...) are populated. Reaching success is
    position-only since the orientation Kp is too weak to track reliably.
    """

    cfg: TaskCommandReachingCfg

    def __init__(self, cfg: TaskCommandReachingCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.ee_asset: Articulation = env.scene[cfg.ee_asset_cfg.name]
        self.target_asset: RigidObject = env.scene[cfg.target_asset_cfg.name]
        self.ee_body_idx = (
            0 if isinstance(cfg.ee_asset_cfg.body_ids, slice) else cfg.ee_asset_cfg.body_ids[0]
        )
        self.success_position_threshold: float = cfg.success_position_threshold
        self.success_orientation_threshold: float = cfg.success_orientation_threshold

        self.metrics["average_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["average_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_rot_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_pos_align_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.orientation_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.position_aligned = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.euler_xy_distance = torch.zeros((self._env.num_envs), device=self._env.device)
        self.xyz_distance = torch.zeros((self._env.num_envs), device=self._env.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    def _update_metrics(self):
        # Read terminal-state values from the progress_context reward term.
        # The reward term runs BEFORE _reset_idx, so its fields reflect the
        # terminal state of the just-ended episode for envs that just reset.
        # If we computed pose error from scene assets here, we'd get the
        # post-reset state for resetting envs (since _update_metrics runs
        # AFTER reset), which is wrong.
        progress_ctx = self._env.reward_manager.get_term_cfg("progress_context").func

        # logs end of episode data (uses terminal state from progress_ctx)
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_rot_align_error"][reset_env] = progress_ctx.euler_xy_distance[reset_env]
        self.metrics["end_of_episode_pos_align_error"][reset_env] = progress_ctx.xyz_distance[reset_env]
        last_episode_success = (progress_ctx.orientation_aligned & progress_ctx.position_aligned)[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = last_episode_success.float()

        # logs current per-env state (mirrored from progress_ctx)
        self.euler_xy_distance[:] = progress_ctx.euler_xy_distance
        self.xyz_distance[:] = progress_ctx.xyz_distance
        self.position_aligned[:] = progress_ctx.position_aligned
        self.orientation_aligned[:] = progress_ctx.orientation_aligned
        self.metrics["average_rot_align_error"][:] = self.euler_xy_distance
        self.metrics["average_pos_align_error"][:] = self.xyz_distance

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
