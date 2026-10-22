# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from uwlab_rl.rsl_rl import RslRlOffPolicyRunnerCfg, RslRlSacActorCfg, RslRlSacAlgorithmCfg, RslRlSacCriticCfg

_ADAMW = {"weight_decay": 1e-3, "betas": (0.9, 0.95)}


@configclass
class PushT_FastSACRunnerCfg(RslRlOffPolicyRunnerCfg):
    """ManiSkill FastSAC (reinforcement_learning/fastsac, fastsac_pusht_control) on rsl_rl SAC; run with 4096 envs.

    8 critic updates of 8192 per 4096-env step, C51 [0, 100] LayerNorm/SiLU nets, gamma 0.97, tau 0.125, target
    entropy 0, AdamW 3e-4 / wd 1e-3, no grad clipping; 20M env steps. Episodes start in sync (partial_reset=False).
    """

    num_steps_per_env = 1
    max_iterations = 4883
    save_interval = 100
    experiment_name = "panda_stick_push_t_fastsac"
    init_at_random_ep_len = False
    # learning_starts 40960 env steps
    start_training = 10
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlSacActorCfg(
        hidden_dims=[512, 256, 128],
        activation="swish",
        obs_normalization=True,
        layer_norm=True,
        log_std_min=-5.0,
        log_std_max=0.0,
        log_std_mode="tanh",
    )
    critic = RslRlSacCriticCfg(
        hidden_dims=[768, 384, 192],
        activation="swish",
        obs_normalization=True,
        layer_norm=True,
        num_atoms=101,
        v_min=0.0,
        v_max=100.0,
    )
    algorithm = RslRlSacAlgorithmCfg(
        replay_buffer_size=4_096_000,
        num_learning_epochs=1,
        num_mini_batches=8,
        mini_batch_size=8192,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        actor_optimizer="adamw",
        critic_optimizer="adamw",
        actor_optimizer_kwargs=_ADAMW,
        critic_optimizer_kwargs=_ADAMW,
        auto_alpha=True,
        alpha=1e-3,
        tau=0.125,
        gamma=0.97,
        target_entropy_scale=0.0,
        max_grad_norm=None,
        policy_frequency=4,
        n_steps=1,
        q_aggregation="avg",
        action_bound="unit",
    )
