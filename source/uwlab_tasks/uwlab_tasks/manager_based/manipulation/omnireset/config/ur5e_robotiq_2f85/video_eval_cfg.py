# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video-eval env variants for the seed-diversity study.

Each variant subclasses the env its policies were trained on, but standardizes the
reset distribution so every seed (and configuration) is rolled out from the *same*
reset states given a fixed ``--seed``:

* resets are drawn ONLY from the ``"ObjectAnywhereEEAnywhere"`` reset type
  (``probs=[1.0]``) — the canonical full-randomization start that best exposes
  behavioral diversity; and
* a success termination ends each episode the instant the peg is inserted, so a
  fixed-length video shows several insertion attempts back-to-back rather than one
  long settle.

GPS curriculum routing is already disabled on the ZeroG uniform cfgs, so sampling
over the single reset type is plain uniform and therefore deterministic under a
fixed seed — identical reset states across all seeds of a configuration.
"""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from . import gravity_cfg, rl_state_cfg

# The single reset type all video-eval envs reset from.
_RESET_TYPE = "ObjectAnywhereEEAnywhere"


# ---------------------------------------------------------------------------
# State-based variants (State-v0 / State-Sequential-v0)
# ---------------------------------------------------------------------------
@configclass
class _StateVideoEvalTerminationsCfg(rl_state_cfg.TerminationsCfg):
    """State terminations + a success DoneTerm (the State train cfg has none)."""

    success = DoneTerm(func=task_mdp.success_termination, time_out=False)


@configclass
class StateVideoEvalCfg(rl_state_cfg.Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    """State-v0 policies (base_4resets, seq_resets/State) rolled out for video."""

    terminations: _StateVideoEvalTerminationsCfg = _StateVideoEvalTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        p = self.events.reset_from_reset_states.params
        p["reset_types"] = [_RESET_TYPE]
        p["probs"] = [1.0]


@configclass
class StateSequentialVideoEvalCfg(rl_state_cfg.Ur5eRobotiq2f85RelCartesianOSCTrainSequentialCfg):
    """State-Sequential-v0 policies (seq_resets/State-Sequential) rolled out for video."""

    terminations: _StateVideoEvalTerminationsCfg = _StateVideoEvalTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        p = self.events.reset_from_reset_states.params
        p["reset_types"] = [_RESET_TYPE]
        p["probs"] = [1.0]


# ---------------------------------------------------------------------------
# ZeroG ScenePC variants (ZeroG-ScenePC-Uniform-v0 / ZeroG-ScenePC-SysID-Train-v0)
# ZeroGTerminationsCfg already carries a success DoneTerm, so only the reset
# distribution needs standardizing.
# ---------------------------------------------------------------------------
@configclass
class ZeroGScenePCUniformVideoEvalCfg(gravity_cfg.ZeroGScenePCUniformTrainCfg):
    """ZeroG-ScenePC-Uniform-v0 policies (base_grav_2resets, base_grav_4resets)."""

    def __post_init__(self):
        super().__post_init__()
        p = self.events.reset_from_states.params
        p["reset_types"] = [_RESET_TYPE]
        p["probs"] = [1.0]


@configclass
class ZeroGScenePCSysidVideoEvalCfg(gravity_cfg.ZeroGScenePCSysidSim2RealTrainCfg):
    """ZeroG-ScenePC-SysID-Train-v0 policies (sysid_floor01, sysid_grav_4resets)."""

    def __post_init__(self):
        super().__post_init__()
        p = self.events.reset_from_states.params
        p["reset_types"] = [_RESET_TYPE]
        p["probs"] = [1.0]
