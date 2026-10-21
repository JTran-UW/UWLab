# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .commands import TaskCommand, TaskDependentCommand


@configclass
class TaskDependentCommandCfg(CommandTermCfg):
    class_type: type = TaskDependentCommand

    reset_terms_when_resample: dict[str, EventTerm] = {}


@configclass
class TaskCommandCfg(TaskDependentCommandCfg):
    class_type: type = TaskCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    insertive_asset_cfg: SceneEntityCfg = MISSING

    receptive_asset_cfg: SceneEntityCfg = MISSING

    symmetric_success: bool = False
    """Score success at the best-aligned symmetric version of the assembly instead of the single assembled offset."""

    insertive_symmetry: dict | None = None
    """Symmetry of the insertive object (``symmetry`` block format); None reads it from the object's metadata."""

    receptive_symmetry: dict | None = None
    """Symmetry of the receptive object (``symmetry`` block format); None reads it from the object's metadata."""
