# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import (
    BehaviorCloningCfg,
    OffPolicyAlgorithmCfg,
    RslRlFancyPpoAlgorithmCfg,
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
    # Declared as explicit `actor` / `critic` model configs rather than the legacy
    # `policy` config, because gSDE cannot survive the legacy path.
    #
    # isaaclab_rl's backward-compat shim maps `policy.noise_std_type` onto
    # `GaussianDistribution`'s `std_type`, which only accepts "scalar" or "log" --
    # so a `policy` config asking for "gsde" dies in rsl_rl/modules/distribution.py
    # with `ValueError: Unknown standard deviation type: gsde`. In rsl-rl >= 4.0
    # gSDE is not a std_type at all: it is its own distribution class
    # (`GsdeDistribution`), selected via `distribution_cfg.class_name`.
    #
    # Mirrors what the shim builds for the non-gSDE case (see
    # isaaclab_rl/rsl_rl/utils.py): actor stochastic, critic deterministic.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        state_dependent_std=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            # rsl-rl 5.x's `GsdeDistribution` is real gSDE: it holds the exploration
            # weight matrix fixed for a whole rollout, so noise is smooth in time
            # (lag-1 autocorrelation +0.86 in simulation, vs ~0 pre-3.0). The rsl-rl
            # this task was tuned against sampled the marginal directly, giving
            # noise independent across both steps and environments. Insertion needs
            # that dither to discover contacts, so use the subclass restoring it.
            # Only sampling changes -- stddev, log-probs and entropy are identical.
            # Verified equivalent in sim: scripts_v2/tools/diagnostics/audit_exploration_sim.py
            class_name="uwlab_rl.rsl_rl.distributions:LegacyGsdeDistribution",
            init_std=1.0,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
