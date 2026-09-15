# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg

from uwlab_rl.rsl_rl.rl_cfg import (
    BehaviorCloningCfg,
    GsdeDistributionCfg,
    OffPolicyAlgorithmCfg,
    RslRlFancyPpoAlgorithmCfg,
    RslRlGsdePpoAlgorithmCfg,
)


def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    resume = False
    experiment_name = "ur5e_robotiq_2f85_omnireset_agent"
    # Explicit `actor`/`critic` model cfgs rather than the legacy `policy` cfg: gSDE is a
    # distribution class, and the legacy path only forwards `std_type` ("scalar"/"log").
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        state_dependent_std=False,
        # learn_features=True keeps the gradient path from the state-dependent std into the
        # backbone, as in the rsl-rl 3.x recipe. With it detached (rsl_rl default) the
        # adaptive-KL schedule floors the LR at iteration 0 and the std runs away.
        distribution_cfg=GsdeDistributionCfg(init_std=1.0, learn_features=True),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
    )
    algorithm = RslRlGsdePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # rsl-rl 5.x gSDE holds the exploration matrix fixed for a whole rollout by default
        # (time-correlated noise). The recipe this task was tuned with (rsl-rl 3.x) drew fresh
        # noise every step; insertion relies on that dither to discover contacts, so resample
        # every step. Mean, std, log-prob and entropy are unaffected -- only the sample is.
        sde_sample_freq=1,
    )


@configclass
class Base_DAggerRunnerCfg(Base_PPORunnerCfg):
    algorithm = RslRlFancyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        offline_algorithm_cfg=OffPolicyAlgorithmCfg(
            behavior_cloning_cfg=BehaviorCloningCfg(
                experts_path=[""],
                experts_loader="torch.jit.load",
                experts_observation_group_cfg="uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg:ObservationsCfg.PolicyCfg",
                experts_observation_func=my_experts_observation_func,
                experts_action_group_cfg="uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions:Ur5eRobotiq2f85RelativeOSCAction",
                cloning_loss_coeff=1.0,
                loss_decay=1.0,
            )
        ),
    )
