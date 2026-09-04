# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cluster entry point for eval_critic.py with the reaching-only per-reset eval overrides baked in.

The cluster launcher relays job args through two unquoted shell hops, which mangles Hydra list
overrides like ``reset_types=[...]``; this wrapper appends them here instead. Pass everything else
(--task, --checkpoint, --num_envs, --group_size, dynamics_gap scalars, ...) as normal.
"""

import os
import runpy
import sys

sys.argv[1:] = sys.argv[1:] + [
    "env.events.reset_from_reset_states.params.reset_types=[ObjectAnywhereEEAnywhere]",
    "env.events.reset_from_reset_states.params.probs=[1.0]",
    "env.terminations.first_episode_termination=null",
    "env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset",
]
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_critic.py")
runpy.run_path(sys.argv[0], run_name="__main__")
