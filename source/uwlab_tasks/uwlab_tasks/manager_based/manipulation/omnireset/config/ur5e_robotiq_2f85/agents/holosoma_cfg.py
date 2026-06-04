# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlOffPolicyRunnerCfg, RslRlFastSACAlgorithmCfg, RslRlFastSACActorCriticCfg

from uwlab_rl.holosoma.rl_cfg import (
    BehaviorCloningCfg,
    OffPolicyAlgorithmCfg,
    RslRlFancyActorCriticCfg,
    RslRlFancyPpoAlgorithmCfg,
)


@configclass
class Base_FastSACRunnerCfg(RslRlOffPolicyRunnerCfg):
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_omnireset_agent"
    wandb_project = "omnireset_fastsac"
    policy = RslRlFastSACActorCriticCfg(
    )
    algorithm = RslRlFastSACAlgorithmCfg(
    )
