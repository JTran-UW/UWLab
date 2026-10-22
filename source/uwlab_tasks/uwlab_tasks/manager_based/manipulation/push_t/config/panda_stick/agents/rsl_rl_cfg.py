# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PushT_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stock rsl_rl approximation of ManiSkill examples/baselines/ppo/ppo_fast.py (PushT flags) + obs normalization.

    Not reproduced: orthogonal init, target_kl early stopping, Adam eps 1e-5, a joint actor-critic grad clip.
    """

    num_steps_per_env = 16
    # 50M env steps at 4096 envs x 16 steps
    max_iterations = 763
    save_interval = 50
    resume = False
    experiment_name = "panda_stick_push_t"
    # ManiSkill clip_action: the env gets clamp(a, -1, 1)
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 256],
        activation="tanh",
        obs_normalization=True,
        # state-independent log-std initialized at 0 (ppo_fast)
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 256],
        activation="tanh",
        obs_normalization=True,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        max_grad_norm=0.5,
        # ManiSkill: vf_coef 0.5 * (0.5 * MSE); rsl_rl's unclipped value loss is the plain MSE
        value_loss_coef=0.25,
        use_clipped_value_loss=False,
        normalize_advantage_per_mini_batch=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="fixed",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.9,
    )
