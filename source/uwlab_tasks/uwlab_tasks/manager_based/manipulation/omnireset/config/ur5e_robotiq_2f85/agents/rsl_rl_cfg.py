# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from uwlab_rl.rsl_rl.rl_cfg import (
    BehaviorCloningCfg,
    OffPolicyAlgorithmCfg,
    RslRlFancyActorCriticCfg,
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
    policy = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
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
class DistillationDAggerAlgorithmCfg:
    """Algorithm cfg for ``DistillationDAgger`` (β-annealed DAgger, MSE on mean)."""

    class_name: str = "DistillationDAgger"
    num_learning_epochs: int = 5
    learning_rate: float = 1.0e-4
    gradient_length: int = 15
    max_grad_norm: float | None = 1.0
    optimizer: str = "adam"
    loss_type: str = "mse"
    beta_anneal_iters: int = 0
    aux_coeff: float = 1.0


@configclass
class StudentTeacherVisionPolicyCfg:
    """Policy cfg for the depth CNN student + JIT teacher."""

    class_name: str = "StudentTeacherVision"
    # non-MISSING default so hydra CLI string overrides pass type-check
    teacher_jit_path: str = ""
    vision_groups: list[str] = MISSING
    embed_dim: int = 128
    student_hidden_dims: list[int] = MISSING
    activation: str = "elu"
    init_noise_std: float = 0.1
    noise_std_type: str = "scalar"
    student_obs_normalization: bool = True
    encoder_type: str = "depth_cnn"  # "depth_cnn" (slim 4-conv) or "resnet18"
    encoder_pretrained_path: str = ""  # path to ImageNet weights (.pth) for resnet18
    aux_enabled: bool = False  # train an aux pose-regression head on vision features
    aux_target_group: str = "aux_target"
    aux_hidden_dims: list[int] = [256, 128]


@configclass
class Depth_DAggerRunnerCfg(RslRlBaseRunnerCfg):
    """DAgger with depth student + JIT state teacher via rsl_rl ``DistillationRunner``."""

    class_name: str = "DistillationRunner"
    num_steps_per_env: int = 8
    max_iterations: int = 30000
    save_interval: int = 100
    # Leave empirical_normalization as MISSING so IsaacLab's deprecated-cfg shim
    # skips the auto-migration (it assumes actor/critic-style policies).
    experiment_name: str = "ur5e_robotiq_2f85_depth_dagger"
    obs_groups: dict = {
        "policy": ["proprio"],
        "teacher": ["teacher"],
    }
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(beta_anneal_iters=2000)
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_depth"],
        student_hidden_dims=[512, 256],
    )


@configclass
class Depth_DAggerSplitRunnerCfg(Depth_DAggerRunnerCfg):
    """Fixed-mask split variant with optional held-out eval pool.

    * ``student_fraction``: of the *train* envs, what fraction run student actions.
    * ``eval_fraction``: fraction of all envs reserved as student-driven, no-grad
      eval — contamination-free success signal even at tight train cadences
      (``num_steps_per_env=1``) where same-step gradients create echo-of-teacher bias.

    Defaults match the state→state DAgger recipe that succeeded on 2026-04-19
    (``student_fraction=1.0`` pure-student rollouts + DEXTRAH-matched Fast cadence
    ``num_steps_per_env=1``, ``num_learning_epochs=1``, ``gradient_length=1``).
    """

    class_name: str = "DistillationRunnerSplit"
    experiment_name: str = "ur5e_robotiq_2f85_depth_dagger_split"
    num_steps_per_env: int = 1
    max_iterations: int = 200000
    student_fraction: float = 1.0
    eval_fraction: float = 0.1
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(
        beta_anneal_iters=0,
        num_learning_epochs=1,
        gradient_length=1,
    )


@configclass
class Rgb_DAggerSplitRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """RGB variant of the split DAgger runner. Only ``vision_groups`` differs."""

    experiment_name: str = "ur5e_robotiq_2f85_rgb_dagger_split"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_rgb"],
        student_hidden_dims=[512, 256],
    )


@configclass
class Depth_DAggerResNet18RunnerCfg(Depth_DAggerSplitRunnerCfg):
    """1-cam depth split runner with ResNet18 student encoder (scratch)."""

    experiment_name: str = "ur5e_robotiq_2f85_depth_dagger_resnet18"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_depth"],
        student_hidden_dims=[512, 256],
        encoder_type="resnet18",
    )


@configclass
class Rgb_DAggerResNet18RunnerCfg(Depth_DAggerSplitRunnerCfg):
    """1-cam rgb split runner with ResNet18 student encoder (scratch)."""

    experiment_name: str = "ur5e_robotiq_2f85_rgb_dagger_resnet18"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_rgb"],
        student_hidden_dims=[512, 256],
        encoder_type="resnet18",
    )


@configclass
class Depth_DAggerAuxRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """1-cam depth split runner, ResNet18 encoder + aux pose-regression head."""

    experiment_name: str = "ur5e_robotiq_2f85_depth_dagger_aux"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_depth"],
        student_hidden_dims=[512, 256],
        encoder_type="resnet18",
        aux_enabled=True,
    )


@configclass
class Depth_DAggerPretrainedRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """1-cam depth split runner, ImageNet-pretrained ResNet18 encoder (no aux)."""

    experiment_name: str = "ur5e_robotiq_2f85_depth_dagger_pretrained"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_depth"],
        student_hidden_dims=[512, 256],
        encoder_type="resnet18",
        encoder_pretrained_path="teachers/resnet18_imagenet.pth",
    )


@configclass
class Depth_DAgger2CamSplitRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """2-camera depth: shared CNN over side + front depth streams (concat features)."""

    experiment_name: str = "ur5e_robotiq_2f85_depth_dagger_2cam_split"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_depth", "front_depth"],
        student_hidden_dims=[512, 256],
    )


@configclass
class Rgb_DAgger2CamSplitRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """2-camera RGB: shared CNN over side + front rgb streams (concat features)."""

    experiment_name: str = "ur5e_robotiq_2f85_rgb_dagger_2cam_split"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_rgb", "front_rgb"],
        student_hidden_dims=[512, 256],
    )


@configclass
class Rgb_DAgger2CamPretrainedRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """2-camera RGB (side + front) with ImageNet-pretrained ResNet18 encoder."""

    experiment_name: str = "ur5e_robotiq_2f85_rgb_dagger_2cam_pretrained"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_rgb", "front_rgb"],
        student_hidden_dims=[512, 256],
        encoder_type="resnet18",
        encoder_pretrained_path="teachers/resnet18_imagenet.pth",
    )


@configclass
class Rgb_DAggerWristSidePretrainedRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """2-camera RGB (side + wrist) with ImageNet-pretrained ResNet18 encoder."""

    experiment_name: str = "ur5e_robotiq_2f85_rgb_dagger_wristside_pretrained"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_rgb", "wrist_rgb"],
        student_hidden_dims=[512, 256],
        encoder_type="resnet18",
        encoder_pretrained_path="teachers/resnet18_imagenet.pth",
    )


@configclass
class Depth_DAggerPretrainedAux10xRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """1-cam depth ImageNet ResNet18 + aux head at aux_coeff=10.0.

    Rationale: with ``aux_coeff=1.0`` the aux loss (~0.64) contributes only ~3%
    of the summed loss because BC loss (~20) dominates the 30:1 ratio. Bumping
    to 10× gives aux ~25% contribution so features actually get shaped by
    object-pose supervision.
    """

    experiment_name: str = "ur5e_robotiq_2f85_depth_dagger_pretrained_aux10x"
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(
        beta_anneal_iters=0,
        num_learning_epochs=1,
        gradient_length=1,
        aux_coeff=10.0,
    )
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_depth"],
        student_hidden_dims=[512, 256],
        encoder_type="resnet18",
        encoder_pretrained_path="teachers/resnet18_imagenet.pth",
        aux_enabled=True,
    )


@configclass
class StudentTeacherMLPPolicyCfg:
    """Policy cfg for an MLP student + JIT teacher (state→state DAgger)."""

    class_name: str = "StudentTeacherMLP"
    teacher_jit_path: str = ""
    student_hidden_dims: list[int] = MISSING
    activation: str = "elu"
    init_noise_std: float = 0.1
    noise_std_type: str = "scalar"
    student_obs_normalization: bool = True


@configclass
class State_DAggerSplitRunnerCfg(RslRlBaseRunnerCfg):
    """State→state DAgger sanity check: MLP student reads the 215d teacher obs.

    Uses the same split-pool runner + DAgger algorithm as the depth variant so
    ``Metrics/success_student_only`` / ``Metrics/success_teacher_only`` remain
    directly comparable. Student hidden dims mirror the state PPO actor.
    """

    class_name: str = "DistillationRunnerSplit"
    num_steps_per_env: int = 8
    max_iterations: int = 5000
    save_interval: int = 100
    experiment_name: str = "ur5e_robotiq_2f85_state_dagger_split"
    student_fraction: float = 0.5
    eval_fraction: float = 0.0
    obs_groups: dict = {
        "policy": ["teacher"],
        "teacher": ["teacher"],
    }
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(beta_anneal_iters=0)
    policy: StudentTeacherMLPPolicyCfg = StudentTeacherMLPPolicyCfg(
        student_hidden_dims=[512, 256, 128, 64],
    )


@configclass
class State_DAggerFastRunnerCfg(State_DAggerSplitRunnerCfg):
    """DEXTRAH-cadence-matched state→state DAgger: 1 env step → 1 gradient step, no buffer replay.

    Matches ``DEXTRAH/dextrah_lab/distillation/distillation.py``:
    ``num_steps_per_env=1``, ``num_learning_epochs=1``, ``gradient_length=1``.
    Override ``agent.student_fraction`` on launch to ablate 0.5 (split-pool) vs
    1.0 (pure student rollouts, DEXTRAH's β=0 regime).
    """

    experiment_name: str = "ur5e_robotiq_2f85_state_dagger_fast"
    num_steps_per_env: int = 1
    # ~8× the prior max_iterations to match env-step volume (8 env steps/iter → 1)
    max_iterations: int = 40000
    # Reserve 10% of envs as held-out eval (student-driven, no-grad) by default.
    # Tight cadence (1 step/iter, grad_length=1) applies same-step gradients on
    # train envs' transitions; eval envs give a contamination-free student signal.
    eval_fraction: float = 0.1
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(
        beta_anneal_iters=0,
        num_learning_epochs=1,
        gradient_length=1,
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
