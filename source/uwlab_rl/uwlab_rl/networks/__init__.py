# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-agnostic network modules shared across the codebase.

Deliberately dependency-light (torch + torchvision only). This subpackage is a
leaf under the ``uwlab_rl`` namespace package, so importing it does **not**
trigger ``uwlab_rl.rsl_rl``'s heavy ``__init__`` (which pulls in isaaclab). That
lets the same module be imported by:

* the host-side PyTorch-Lightning BC trainer (``scripts/imitation_learning/
  point_cloud/``), which runs on bare ``patlab`` with no Isaac, and
* the in-container rsl_rl online-DAgger student (``uwlab_rl.rsl_rl.
  student_teacher_pointcloud``),

so the PointNet architecture has a single source of truth. See
``uwlab_rl.networks.point_net``.
"""

from .history_point_net import HistoryPointNet
from .point_net import MLP, PointNet, ResidualMLP, ResidualPointNet

__all__ = ["HistoryPointNet", "MLP", "PointNet", "ResidualMLP", "ResidualPointNet"]
