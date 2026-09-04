# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""train.py cluster entry point: kp x1.5 / hold(p=0.05, K=3) / +1 cm world-z peg bias gap PLUS resets
restricted to ``ObjectAnywhereEEAnywhere`` ("ReachingOnly"). Hydra list overrides do not survive the
launcher's unquoted shell hops, so they are appended here. Select with
``CLUSTER_PYTHON_EXECUTABLE=scripts/reinforcement_learning/holosoma/train_gap_kp1p5_hold3_zbias1cm_reaching_cluster.py``.
"""

import os
import runpy
import sys

OVERRIDES = [
    "env.events.reset_from_reset_states.params.reset_types=[ObjectAnywhereEEAnywhere]",
    "env.events.reset_from_reset_states.params.probs=[1.0]",
    "env.events.dynamics_gap.params.osc_kp_xyz_scale=1.5",
    "env.events.dynamics_gap.params.osc_kp_rpy_scale=1.5",
]
for term in ("insertive_asset_pose", "insertive_asset_in_receptive_asset_frame"):
    OVERRIDES += [
        f"env.observations.policy.{term}.params.hold_prob=0.05",
        f"env.observations.policy.{term}.params.hold_steps=3",
        f"env.observations.policy.{term}.params.world_pos_bias=[0.0,0.0,0.01]",
    ]

sys.argv[1:] = sys.argv[1:] + OVERRIDES
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
runpy.run_path(sys.argv[0], run_name="__main__")
