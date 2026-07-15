# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env cfg for open-loop action playback (sim2real gap measurement).

Same dynamics as the env the demos / real rollouts were produced under
(:class:`Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg`: explicit actuator, stiff
OSC gains, fixed sysid params) so a real-robot action sequence replayed here is
directly comparable to what the real arm did.

Differences from the eval cfg:

* **Observations** -- one non-concatenated group whose only term is the trailing 6
  joint positions (the Robotiq 2F-85 gripper joints). ``concatenate_terms=False``
  so the driver script can dump each term to its own array; add more terms here
  and they show up as extra keys with no script change.
* **Terminations** -- disabled. Playback must run the full recorded action list; a
  success / abnormal-state termination would auto-reset the env mid-episode.

Mirrors the sim2real tooling pattern:
  camera_align_cfg.py  +  scripts_v2/tools/sim2real/align_cameras.py
  playback_cfg.py      +  scripts_v2/tools/sim2real/playback_actions.py
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg


@configclass
class PlaybackObservationsCfg:
    """One un-concatenated group; each term is recorded to its own array."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Trailing 6 DOFs of the 12-DOF articulation = the Robotiq 2F-85 gripper
        # joints (the 6 UR5e arm joints come first in DOF order).
        gripper_joint_pos = ObsTerm(
            func=task_mdp.joint_pos_last_n,
            params={"n": 6, "asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()


@configclass
class Ur5eRobotiq2f85PlaybackCfg(Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg):
    """Eval dynamics + minimal per-term observations, no terminations."""

    observations: PlaybackObservationsCfg = PlaybackObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Playback drives the env from a fixed action list; nothing may cut the
        # episode short or the recorded trajectory stops tracking the real one.
        # (``time_out`` is kept -- the TerminationManager needs a term -- but the
        # horizon is pushed far beyond any recorded episode.)
        self.terminations.abnormal_robot = None
        self.terminations.success = None
        self.episode_length_s = 1.0e6
