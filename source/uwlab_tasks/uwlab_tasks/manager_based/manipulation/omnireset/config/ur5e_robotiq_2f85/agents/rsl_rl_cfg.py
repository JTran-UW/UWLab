# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import BehaviorCloningCfg, OffPolicyAlgorithmCfg, RslRlFancyPpoAlgorithmCfg


def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 100
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    resume = False
    experiment_name = "ur5e_robotiq_2f85_omnireset_agent"
    # Explicit `actor`/`critic` model cfgs preserve the distribution constructor options,
    # including the action-std bounds not exposed by the legacy `policy` conversion.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        # Bound normalized action std rather than feature-space noise weights.
        # Predict state-dependent log std alongside the action mean while keeping
        # the initial exploration scale identical to the previous recipe.
        distribution_cfg={
            "class_name": "HeteroscedasticGaussianDistribution",
            "init_std": 1.0,
            "std_type": "log",
            "std_range": [0.001, 2.0],
        },
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
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
        # Exploration is handled by the actor's distribution configuration.
        # Gaussian noise is drawn independently for each action sample.
        # There is no latent noise matrix to resample on environment steps.
        # Leave the optimizer and rollout settings unchanged.
    )


@configclass
class RslRlPointNetModelCfg(RslRlMLPModelCfg):
    """rsl_rl model whose (B, N, C) observation groups go through a PointNet set encoder."""

    class_name: str = "uwlab_rl.rsl_rl.point_net_model:PointNetModel"
    point_encoder_cfg: dict = {"hidden_dims": [64, 128, 256]}


@configclass
class PointCloud_PPORunnerCfg(Base_PPORunnerCfg):
    """Base recipe with a PointNet actor on (proprio, point cloud) and a privileged state critic."""

    obs_groups = {"actor": ["policy", "pointcloud"], "critic": ["critic"]}

    def __post_init__(self):
        a = self.actor
        self.actor = RslRlPointNetModelCfg(
            hidden_dims=a.hidden_dims, activation=a.activation, obs_normalization=a.obs_normalization,
            distribution_cfg=dict(a.distribution_cfg),
        )
        # PointNet activations are ~0.6 MB/sample: 16 minibatches keep 31744 envs/GPU near 40 GB instead of ~150 GB
        self.algorithm = self.algorithm.replace(num_mini_batches=16)


@configclass
class LongHorizon_PPORunnerCfg(Base_PPORunnerCfg):
    """Base recipe with gamma 0.995 (20 s horizon) for multi-stage episodes."""

    def __post_init__(self):
        self.algorithm = self.algorithm.replace(gamma=0.995)


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
