# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``...-State-v0`` with Success Guided Sampling over all reset states (per-env PPO episodes)."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ... import mdp as task_mdp

from .rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCTrainCfg

# Each env pushes its episode's success_once when it resets; every 16th env samples uniformly as an unbiased monitor.
SGS_PPO = dict(history=100, target=0.5, kappa=1.0, temperature=2.0, eps=1e-4, empty_rate=0.5,
               async_outcomes=True, uniform_monitor_every=16)


def sgs_sync_event() -> EventTerm:
    """Pools multi-GPU SGS outcomes every 1 s of sim time, on the same step on every rank (no-op on one GPU)."""
    return EventTerm(func=task_mdp.sgs_sync, mode="interval", interval_range_s=(1.0, 1.0), is_global_time=True)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCSGSTrainCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_reset_states.params["sgs"] = dict(SGS_PPO)
        self.rewards.success_latch = RewTerm(func=task_mdp.SuccessLatch, weight=1.0)
        self.events.sgs_sync = sgs_sync_event()
