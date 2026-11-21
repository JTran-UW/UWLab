# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import uwlab_rl.flashsac  # noqa: F401
from isaaclab_flashsac.rl_cfg import FlashSACActorCfg, FlashSACCriticCfg, FlashSACRunnerCfg
from uwlab_rl.flashsac import FlashSACRGBAlgorithmCfg

# per action dim: p99 |action| (x y z rx ry rz gripper) of the PhysX State-Symmetry-SGS peg PPO expert (peg_tl/model_700,
# deterministic, 1024 envs x one 160-step episode, 96.5% success), so ~1% clips
PHYSX_PEG_EXPERT_P99_ACTION_BOUNDS = (5.79, 9.96, 7.11, 21.61, 19.12, 10.67, 11.52)


@configclass
class PointCloud_FlashSACRunnerCfg(FlashSACRunnerCfg):
    """FlashSAC with a shared PointNet + proprio encoder trained only by the critic loss (actor and next-action sample
    see detached features, the target critic an EMA encoder). Hyperparameters of optix-renderer's
    ``OmniReset_FlashSACRGB4096RunnerCfg``: 16 updates of 1024 per iteration, [-5, 35] C51 support, gamma 0.99, 3-step
    returns, 1.31M buffer; run with ``--num_envs 4096``. Expert demos are off unless
    ``agent.algorithm.expert_buffer_path`` is set (5% share).
    """

    experiment_name = "ur5e_robotiq_2f85_omnireset_flashsac_pointcloud"
    logger = "tensorboard"
    max_iterations = 29297
    save_interval = 400
    init_at_random_ep_len = False
    obs_groups = {"actor": ["pointcloud", "policy"], "critic": ["pointcloud", "policy"]}
    action_bound = "scalar"
    action_bound_scale = PHYSX_PEG_EXPERT_P99_ACTION_BOUNDS
    actor = FlashSACActorCfg(num_blocks=2, hidden_dim=128, log_std_min=-10.0, log_std_max=2.0)
    critic = FlashSACCriticCfg(num_blocks=2, hidden_dim=256, num_bins=101, min_v=-5.0, max_v=35.0)
    algorithm = FlashSACRGBAlgorithmCfg(
        replay_buffer_size=1_310_720,
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
        gamma=0.99,
        n_steps=3,
        normalize_reward=False,
        use_amp=False,
        expert_ratio=0.05,
        encoder_type="pointnet",
        image_group="pointcloud",
        state_group="policy",
        image_embed_dim=256,
        state_embed_dim=64,
        point_hidden_dims=[64, 128, 256],
    )
