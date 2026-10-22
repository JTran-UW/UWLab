# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_flashsac.rl_cfg import FlashSACActorCfg, FlashSACCriticCfg, FlashSACRunnerCfg

from uwlab_rl.flashsac import FlashSACDemoAlgorithmCfg, FlashSACRGBAlgorithmCfg


@configclass
class PushT_FlashSACRunnerCfg(FlashSACRunnerCfg):
    """FlashSAC's shared benchmark hyperparameters (1024 envs, ~51M env steps)."""

    experiment_name = "panda_stick_push_t_flashsac"
    logger = "tensorboard"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    # the joint-delta action term already clamps to [-1, 1] and scales by 0.1
    action_bound = "scalar"
    action_bound_scale = 1.0


@configclass
class PushT_FlashSACDemoRunnerCfg(PushT_FlashSACRunnerCfg):
    """Same recipe + expert demo mixing; set ``agent.algorithm.expert_buffer_path``."""

    experiment_name = "panda_stick_push_t_flashsac_demo"
    algorithm = FlashSACDemoAlgorithmCfg()


@configclass
class PushT_FlashSACRGBRunnerCfg(FlashSACRunnerCfg):
    """ManiSkill FlashSAC-RGB (frgb_pusht_gray64_zoom_s1) on the ManiSkill-parity RGB task; run with 1024 envs and set
    ``agent.algorithm.expert_buffer_path``.

    16 updates of 1024 per 1024-env step, raw rewards on a [-5, 35] C51 support, gamma 0.97, 3-step returns, tau 0.01,
    actor every 2nd update, alpha 0.01 / target sigma 0.15, cosine LR 3e-4 -> 1.5e-4, 20% expert, 30M env steps.
    """

    experiment_name = "panda_stick_push_t_flashsac_rgb"
    logger = "tensorboard"
    max_iterations = 29297
    save_interval = 400
    init_at_random_ep_len = False
    obs_groups = {"actor": ["image", "state"], "critic": ["image", "state"]}
    action_bound = "scalar"
    action_bound_scale = 1.0
    actor = FlashSACActorCfg(num_blocks=2, hidden_dim=128, log_std_min=-10.0, log_std_max=2.0)
    critic = FlashSACCriticCfg(num_blocks=2, hidden_dim=256, num_bins=101, min_v=-5.0, max_v=35.0)
    algorithm = FlashSACRGBAlgorithmCfg(
        replay_buffer_size=1_024_000,
        buffer_min_length=10_240,
        buffer_obs_dtype="float16",
        num_learning_epochs=1,
        num_mini_batches=16,
        mini_batch_size=1024,
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_steps=0,
        actor_update_period=2,
        critic_target_update_tau=0.01,
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        gamma=0.97,
        n_steps=3,
        normalize_reward=False,
        use_amp=False,
        expert_ratio=0.2,
    )


@configclass
class PushT_FlashSACStateMSRunnerCfg(PushT_FlashSACRGBRunnerCfg):
    """Ablation: the FlashSAC-RGB recipe (hyperparameters, 20% demos) on the 31-D state instead of pixels."""

    experiment_name = "panda_stick_push_t_flashsac_state_ms"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    algorithm = FlashSACDemoAlgorithmCfg(
        replay_buffer_size=1_024_000,
        buffer_min_length=10_240,
        num_learning_epochs=1,
        num_mini_batches=16,
        mini_batch_size=1024,
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_steps=0,
        actor_update_period=2,
        critic_target_update_tau=0.01,
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        gamma=0.97,
        n_steps=3,
        normalize_reward=False,
        use_amp=False,
        expert_ratio=0.2,
    )


@configclass
class PushT_FlashSACRGBPrivStateRunnerCfg(PushT_FlashSACRGBRunnerCfg):
    """Debug: FlashSACRGB with the 31-D privileged state (incl. the tee pose) as its state input."""

    experiment_name = "panda_stick_push_t_flashsac_rgb_privstate"
    obs_groups = {"actor": ["image", "policy"], "critic": ["image", "policy"]}

    def __post_init__(self):
        self.algorithm.state_group = "policy"
