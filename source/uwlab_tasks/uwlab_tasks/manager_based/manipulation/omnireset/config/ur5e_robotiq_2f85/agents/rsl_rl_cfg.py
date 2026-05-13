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
    SuccessCriticCfg,
)


@configclass
class RslRlActorCriticWithEncoderCfg(RslRlFancyActorCriticCfg):
    class_name: str = "ActorCriticWithEncoder"
    encoder_groups: dict = dict()


def my_experts_observation_func(env):
    obs = env.unwrapped.obs_buf["expert_obs"]
    return obs


@configclass
class Base_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
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
    # Gripper-specific loss handling for DistillationDAggerWeighted only.
    # "shared": gripper dim treated like arm dims in weighted_l2 (default).
    # "mse":    arm uses weighted_l2 on dims 0-5; gripper uses MSE * weight.
    # "bce":    arm uses weighted_l2 on dims 0-5; gripper uses BCE_with_logits
    #           against (teacher > 0).float() — matches binary action semantics.
    gripper_loss_type: str = "shared"
    gripper_loss_weight: float = 1.0


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
    encoder_freeze_iters: int = 0  # DEXTRAH-style warmup: freeze ResNet18 backbone for first N algorithm updates
    aux_enabled: bool = False  # train an aux pose-regression head on vision features
    aux_target_group: str = "aux_target"
    aux_hidden_dims: list[int] = [256, 128]
    # Recurrent student params (only used when class_name="StudentTeacherVisionRecurrent")
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 128
    rnn_num_layers: int = 1
    # DEXTRAH-style (μ, σ) weighted-loss params
    predict_std: bool = False  # adds a student std_head (log_std output, exp'd on read)
    teacher_returns_std: bool = False  # teacher JIT returns (mean, std) tuple


@configclass
class Depth_DAggerRunnerCfg(RslRlBaseRunnerCfg):
    """DAgger with depth student + JIT state teacher via rsl_rl ``DistillationRunner``."""

    class_name: str = "DistillationRunner"
    num_steps_per_env: int = 8
    max_iterations: int = 30000
    # 5000 not 100: each ckpt is ~134MB (ResNet18 + AdamW state); at 100 we
    # accumulated 50GB+ per 30k-iter run and filled the cluster's 17TB disk.
    save_interval: int = 5000
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
    # Of the eval pool, fraction running teacher actions (no-grad teacher rate).
    # Default 0 = backward-compatible (eval is all student). Set to 0.5 to log
    # Metrics/success_teacher_eval alongside Metrics/success_student_eval.
    teacher_eval_fraction: float = 0.0
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(
        beta_anneal_iters=0,
        num_learning_epochs=1,
        gradient_length=1,
    )


@configclass
class Depth_DAggerWristSidePretrainedWeightedRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """2cam depth (side+wrist) + ImageNet ResNet18 + DEXTRAH-style weighted-L2 loss."""

    experiment_name: str = "ur5e_robotiq_2f85_depth_dagger_wristside_pretrained_weighted"
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(
        beta_anneal_iters=0,
        num_learning_epochs=1,
        gradient_length=1,
    )
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_depth", "wrist_depth"],
        student_hidden_dims=[512, 256],
        encoder_type="resnet18",
        encoder_pretrained_path="teachers/resnet18_imagenet.pth",
        predict_std=True,
        teacher_returns_std=True,
    )

    def __post_init__(self) -> None:
        self.algorithm.class_name = "DistillationDAggerWeighted"


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


@configclass
class StatePPORunnerCfg(Base_PPORunnerCfg):
    """Plain PPO on state-only obs. One ``policy`` group for actor + critic."""

    obs_groups = {"policy": ["policy"], "critic": ["policy"]}


@configclass
class SharedEncoder128PPORunnerCfg(Base_PPORunnerCfg):
    """128-pt PC with shared MLP encoder (both objects concatenated)."""

    class_name: str = "OnPolicyRunnerWithClassifier"
    classifier_num_epochs: int = 4
    classifier_minibatch_size: int = 256

    obs_groups = {
        "policy": ["proprio", "pointcloud"],
        "critic": ["proprio", "pointcloud", "time_left"],
        "success_classifier": ["success_classifier"],
    }
    policy = RslRlActorCriticWithEncoderCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
        encoder_groups={
            "pointcloud": {"hidden_dims": [256, 128], "output_dim": 32},
        },
    )
    success_critic: SuccessCriticCfg = SuccessCriticCfg()


@configclass
class ScenePCPPORunnerCfg(Base_PPORunnerCfg):
    """Plain PPO with shared MLP encoder for ScenePC obs (no V_success aux head).

    Symmetric AC: both actor and critic see ``proprio + pointcloud`` through
    the same shared PC encoder. Critic additionally gets ``time_left``.
    """

    obs_groups = {
        "policy": ["proprio", "pointcloud"],
        "critic": ["proprio", "pointcloud", "time_left"],
    }
    policy = RslRlActorCriticWithEncoderCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
        encoder_groups={
            "pointcloud": {"hidden_dims": [256, 128], "output_dim": 32},
        },
    )


# ===========================================================================
# BCPPO (PPO + BC auxiliary loss) for depth student.
# Asymmetric AC: depth-CNN actor on `proprio` + `wrist_depth` + `side_depth`
# (sees only what a real robot sees), state-only critic on `teacher` group
# (privileged 43d state for accurate value estimates).
# Pair with the Lean DAgger env (training events, no extra DR).
# ===========================================================================
@configclass
class _ActorCriticDepthCfg:
    class_name: str = "ActorCriticDepth"
    vision_groups: list = ["wrist_depth", "side_depth"]
    embed_dim: int = 128
    actor_hidden_dims: list = [512, 256]
    critic_hidden_dims: list = [256, 256, 256]
    activation: str = "elu"
    init_noise_std: float = 0.5
    noise_std_type: str = "scalar"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    encoder_type: str = "depth_cnn"
    encoder_pretrained_path: str = ""


@configclass
class _BCPPOAlgorithmCfg:
    class_name: str = "BCPPO"
    teacher_jit_path: str = ""
    teacher_obs_groups: list = ["teacher"]
    # bc_loss is MSE in OSC action space ~200-500 typical magnitude; surrogate
    # ~0.05-0.5 and value_loss ~0.002-0.5. coeff=0.001 brings BC into the same
    # scale as PPO terms so it acts as a regularizer, not a dominator.
    cloning_loss_coeff: float = 0.001
    cloning_loss_decay: float = 1.0
    # "mse" = plain MSE on action means (raw scale ~250 for peg teacher).
    # "weighted_mse" = DEXTRAH-style inverse-variance-weighted L2 (raw scale ~3-5
    # since 1/sigma^2 dampens). With weighted_mse, scale up coeff 50-100x.
    bc_loss_type: str = "mse"
    # Eval pool: fraction of envs reserved as eval-only (no gradient).
    # Half teacher-driven (Loss/success_teacher_eval) -- measures teacher in
    # the deployment env. Half student-driven (Loss/success_student_eval) --
    # measures deterministic student. Train pool (1 - eval_fraction) drives
    # PPO+BC normally and reports Loss/success_train.
    eval_fraction: float = 0.1
    teacher_eval_fraction: float = 0.5
    # Standard PPO knobs.
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    normalize_advantage_per_mini_batch: bool = False
    clip_param: float = 0.2
    entropy_coef: float = 0.006
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class Depth_BCPPORunnerCfg(RslRlBaseRunnerCfg):
    """PPO + BC for depth student on Lean env. Reward-bearing PPO update on
    student-driven rollouts, BC pull toward JIT teacher per minibatch.

    Storage: depth obs (224x224 x 1ch x 2 cams) blow GPU memory at large
    num_envs. Each transition holds ~400KB of obs; with num_steps_per_env=32
    and num_envs=256, that's ~3.3GB obs storage, plus PPO update activations.
    Hyak L40/L40s/A40 (48GB) comfortably fit num_envs=256-512 at this cadence.
    """

    class_name: str = "BCPPORunner"
    num_steps_per_env: int = 32
    max_iterations: int = 30000
    save_interval: int = 5000
    experiment_name: str = "ur5e_robotiq_2f85_bcppo"
    obs_groups: dict = {
        "policy": ["proprio"],
        "critic": ["teacher"],
    }
    policy: _ActorCriticDepthCfg = _ActorCriticDepthCfg()
    algorithm: _BCPPOAlgorithmCfg = _BCPPOAlgorithmCfg()


@configclass
class State_BCPPORunnerCfg(RslRlBaseRunnerCfg):
    """State→state BCPPO sanity check. Student reads same 43d teacher obs as
    teacher; no encoder, no asymmetry. Tests whether BCPPO algorithm itself is
    sound vs. depth-specific issues.

    Pair with state expert teacher (peg_sysid_full.pt) on a sysid env where
    the state PPO is known to converge to ~95% success.
    """

    class_name: str = "BCPPORunner"
    num_steps_per_env: int = 32
    max_iterations: int = 5000
    save_interval: int = 500
    experiment_name: str = "ur5e_robotiq_2f85_bcppo_state"
    # State env exposes a single ``policy`` group containing the 43d state obs
    # — same input that the teacher JIT also expects. Symmetric: actor+critic
    # both read it. Override at launch:
    #   agent.algorithm.teacher_obs_groups=[policy]
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    policy: RslRlFancyActorCriticCfg = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    algorithm: _BCPPOAlgorithmCfg = _BCPPOAlgorithmCfg(
        teacher_obs_groups=["policy"],
    )


@configclass
class _PPOPBRSAlgorithmCfg:
    """PPO + Potential-Based Reward Shaping using a frozen expert V function."""

    class_name: str = "PPOPBRS"
    expert_critic_path: str = ""  # path to RL expert ckpt with critic.* keys
    expert_obs_group: str = "policy"  # obs key fed to V_expert
    expert_obs_dim: int = 43
    expert_hidden_dims: list = [512, 256, 128, 64]
    expert_activation: str = "elu"
    init_critic_from_expert: bool = False  # also seed policy critic with expert weights
    pbrs_coef: float = 1.0  # scale on shaping term

    # Standard PPO knobs (rsl_rl PPO default-equivalent)
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    normalize_advantage_per_mini_batch: bool = False
    clip_param: float = 0.2
    entropy_coef: float = 0.006
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class State_PPOPBRSRunnerCfg(RslRlBaseRunnerCfg):
    """State→state PPO + PBRS sanity check.

    Pure PPO surrogate, no BC term. Reward shaped by V_expert from
    `teachers/peg_sysid_state_2400.pt`. Symmetric obs (state actor + state
    critic, both 43d). If this reaches teacher level (~95%), PBRS shaping is
    a viable alternative to BC and we can move to state→depth where BC was
    the bottleneck.
    """

    class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 32
    max_iterations: int = 5000
    save_interval: int = 500
    experiment_name: str = "ur5e_robotiq_2f85_ppo_pbrs_state"
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    policy: RslRlFancyActorCriticCfg = RslRlFancyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        noise_std_type="gsde",
        state_dependent_std=False,
    )
    algorithm: _PPOPBRSAlgorithmCfg = _PPOPBRSAlgorithmCfg(
        expert_critic_path="teachers/peg_sysid_state_2400.pt",
        expert_obs_group="policy",
    )


# ===========================================================================
# GRPO finetune for depth peg student (Doorman Phase 3).
# Loads DAgger checkpoint, applies PPO clipped surrogate with per-batch
# baseline (no V function) + KL anchor against frozen DAgger reference.
# ===========================================================================
@configclass
class _GRPOAlgorithmCfg:
    class_name: str = "GRPO"
    # Path to a DAgger student checkpoint (StudentTeacherVision keys).
    # If empty, GRPO starts from random + still snapshots that as reference.
    init_from_dagger_path: str = ""
    # KL penalty coefficient against the frozen reference policy.
    # DeepSeek default 0.04. Set kl_target>0 to adapt toward a target KL.
    kl_coeff: float = 0.04
    kl_target: float | None = 0.01
    # Optional: clamp per-batch advantages by quantile (0 disables). Helps if
    # rare huge returns dominate (e.g. one trajectory with success_reward=100
    # vs everything else 0).
    clamp_advantage_quantile: float | None = None
    # Standard PPO knobs (V loss disabled by forcing value_loss_coef=0 in code).
    value_loss_coef: float = 0.0
    use_clipped_value_loss: bool = True
    normalize_advantage_per_mini_batch: bool = False
    clip_param: float = 0.2
    entropy_coef: float = 0.006
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-5  # low; finetune from DAgger
    schedule: str = "fixed"  # don't adapt LR (no KL-LR tradeoff -- KL handled separately)
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float | None = None
    max_grad_norm: float = 1.0
    # Real grouped GRPO: K envs/group share a reset state (replicated by
    # MultiResetManager), per-group baseline isolates action quality from
    # reset luck. Set group_size > 1 + use the GRPOGroupedRunner.
    group_size: int = 1
    normalize_grouped_advantages: bool = False
    # Optional std overwrite at GRPO start (only when loading from DAgger).
    # 0.0 = disable. DAgger ckpts often collapse std to ~0.10, leaving no
    # exploration on tasks where the deterministic action fails (e.g.
    # ZeroGAnywhere with 0% DAgger success). Setting boost_init_std to 0.5
    # restores exploration so GRPO can find advantage signal on those resets.
    boost_init_std: float = 0.0
    # Lock depth_encoder.eval() at GRPO start so BN uses running stats and
    # stays in sync with the deepcopied reference. Default True for the
    # KL-explosion fix; turn False if the DAgger ckpt was trained in
    # batch-stats mode and the eval-mode switch corrupts the policy.
    lock_depth_encoder: bool = True


@configclass
class Depth_GRPORunnerCfg(RslRlBaseRunnerCfg):
    """GRPO finetune of DAgger student. Reuses ActorCriticDepth + BCPPO-Sysid env.

    Init policy from a DAgger checkpoint (set
    ``agent.algorithm.init_from_dagger_path=...`` on launch). Algorithm
    snapshots that as the frozen KL reference. Then runs PPO clipped surrogate
    with per-batch baseline (no V function) + KL anchor.
    """

    class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 32
    max_iterations: int = 10000
    save_interval: int = 1000
    experiment_name: str = "ur5e_robotiq_2f85_grpo"
    obs_groups: dict = {
        "policy": ["proprio"],
        "critic": ["teacher"],
    }
    policy: _ActorCriticDepthCfg = _ActorCriticDepthCfg()
    algorithm: _GRPOAlgorithmCfg = _GRPOAlgorithmCfg()


@configclass
class Depth_GRPOGroupedRunnerCfg(Depth_GRPORunnerCfg):
    """Real grouped GRPO. K envs in each group start each rollout from the same
    reset state (env-side replication via MultiResetManager.group_size). The
    GRPOGroupedRunner forces a full env.reset() at the start of every iter so
    groups stay synchronized; the algorithm uses the per-group return baseline.

    Set ``env.events.gravity_curriculum.params.group_size=K`` (or whichever event
    name MultiResetManager is registered under) to match the algorithm's
    ``group_size``.
    """

    class_name: str = "GRPOGroupedRunner"
    experiment_name: str = "ur5e_robotiq_2f85_grpo_grouped"

    def __post_init__(self):
        # Default: 8 envs per group (with num_envs=64 -> 8 groups). Override at launch
        # by setting both this and the env-side `group_size` param to the same K.
        self.algorithm.group_size = 8
