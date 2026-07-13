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
    encoder_per_view: bool = True  # resnet18 only: True=independent encoder per camera, False=shared encoder
    encoder_freeze_iters: int = 0  # DEXTRAH-style warmup: freeze ResNet18 backbone for first N algorithm updates
    aux_enabled: bool = False  # train aux pose-regression heads on vision features
    aux_hidden_dims: list[int] = [256, 128]
    # Aux target keys: names of top-level obs groups used as regression targets,
    # one per aux head. Each group must be a flat tensor (separate ObsGroup with
    # concatenate_terms=True). The policy infers each head's output dim from
    # obs[k].shape[-1] — no equal-dim assumption, any shape is valid.
    aux_target_keys: list[str] = []
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
class RGB_DAggerWristSidePretrainedWeightedRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """2cam RGB (side+wrist) + ImageNet ResNet18 + DEXTRAH-style weighted-L2 loss."""

    experiment_name: str = "ur5e_robotiq_2f85_rgb_dagger_wristside_pretrained_weighted"
    student_fraction: float = 0.5
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(
        beta_anneal_iters=0,
        num_learning_epochs=1,
        gradient_length=1,
    )
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_rgb", "wrist_rgb"],
        student_hidden_dims=[512, 512, 512, 512],
        encoder_type="resnet18",
        encoder_pretrained_path="teachers/resnet18_imagenet.pth",
        predict_std=True,
        teacher_returns_std=True,
    )

    def __post_init__(self) -> None:
        self.algorithm.class_name = "DistillationDAggerWeighted"


@configclass
class RGB_DAggerDataCollectionPCTeacherRunnerCfg(RGB_DAggerWristSidePretrainedWeightedRunnerCfg):
    """RGB DAgger from data-collection scene + ScenePC teacher + aux pose-regression loss.

    Same split-pool runner + weighted-L2 loss as the WristSide variant.
    Adds three DEXTRAH-style aux heads (insertive↔wrist, receptive↔wrist,
    insertive↔receptive) that force the CNN encoder to learn pose-aware features.
    Requires the env obs config to include an ``aux_target`` group (18d flat tensor,
    concatenate_terms=True) — see ``RGBDAggerDataCollectionPCTeacherObsCfg``.
    """

    experiment_name: str = "ur5e_robotiq_2f85_rgb_dagger_datacollection_pc_teacher"
    policy: StudentTeacherVisionPolicyCfg = StudentTeacherVisionPolicyCfg(
        vision_groups=["side_rgb", "wrist_rgb"],
        student_hidden_dims=[512, 512, 512, 512],
        encoder_type="resnet18",
        encoder_pretrained_path="teachers/resnet18_imagenet.pth",
        predict_std=True,
        teacher_returns_std=True,
        aux_enabled=True,
        # Each key is a separate top-level obs group — no concatenation, each
        # has its own shape. Defined in RGBDAggerDataCollectionPCTeacherObsCfg.
        aux_target_keys=["aux_insertive_in_wrist", "aux_receptive_in_wrist", "aux_insertive_in_receptive"],
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
class StudentTeacherPointCloudPolicyCfg:
    """Policy cfg for a PointNet student + JIT teacher (point-cloud -> point-cloud DAgger).

    The student is the SAME PointNet class as the offline BC trainer
    (``uwlab_rl.networks.point_net``); ``encoder_dims`` / ``action_dims`` /
    ``architecture`` mirror the BC config so a BC checkpoint can initialize it.
    ``pointcloud_groups`` lists which ``obs_groups['policy']`` groups are flat
    point clouds (reshaped to ``(N, point_dim)``); the rest are treated as proprio.
    """

    class_name: str = "StudentTeacherPointCloud"
    teacher_jit_path: str = ""
    pointcloud_groups: list[str] = MISSING
    point_dim: int = 4  # 3 = xyz; 4 = xyz + per-point segmentation label (matches CleanSeg BC)
    architecture: str = "residual_point_net"  # or "point_net"
    # Arch/sizing deferred — set to match the BC run we distill-init from before launching.
    encoder_dims: list[int] = MISSING
    action_dims: list[int] = MISSING
    activation: str = "elu"
    init_noise_std: float = 0.1
    noise_std_type: str = "scalar"
    student_obs_normalization: bool = True
    # DEXTRAH-style (μ, σ) weighted-loss params (only for DistillationDAggerWeighted).
    predict_std: bool = False  # student PointNet emits a second log-std head
    teacher_returns_std: bool = False  # teacher JIT returns (mean, std) tuple
    # BC-init: path to an offline PointNet-BC Lightning ckpt whose ``model.*`` weights warm-start the
    # student encoder/trunk (strict=False -> a predict_std head-shape mismatch is tolerated). "" = scratch.
    bc_init_path: str = ""


@configclass
class PC_DAggerSplitRunnerCfg(Depth_DAggerSplitRunnerCfg):
    """Online DAgger with a PointNet student + JIT PC teacher via ``DistillationRunnerSplit``.

    Same split-pool runner + DEXTRAH-cadence DAgger algorithm as the depth/state
    variants (so ``Metrics/success_student_*`` / ``success_teacher_*`` stay
    comparable), but the student consumes point clouds. Student obs = a proprio
    group + a flat ``scene_pc`` group; teacher obs = the ScenePC-expert group.

    Arch/sizing (``encoder_dims`` / ``action_dims``) intentionally left MISSING —
    set them (and ``teacher_jit_path``) on launch or in a subclass. Plain MSE-on-mean
    by default; flip to ``DistillationDAggerWeighted`` + ``predict_std=True`` +
    ``teacher_returns_std=True`` for the DEXTRAH inverse-variance-weighted loss.
    """

    class_name: str = "DistillationRunnerSplit"
    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_split"
    num_steps_per_env: int = 1
    max_iterations: int = 200000
    student_fraction: float = 0.95
    eval_fraction: float = 0.1
    obs_groups: dict = {
        "policy": ["proprio", "scene_pc"],
        "teacher": ["teacher"],
    }
    algorithm: DistillationDAggerAlgorithmCfg = DistillationDAggerAlgorithmCfg(
        beta_anneal_iters=0,
        num_learning_epochs=1,
        gradient_length=1,
    )
    policy: StudentTeacherPointCloudPolicyCfg = StudentTeacherPointCloudPolicyCfg(
        pointcloud_groups=["scene_pc"],
    )


@configclass
class PC_DAggerSplitWeightedXLRunnerCfg(PC_DAggerSplitRunnerCfg):
    """DEXTRAH inverse-variance-weighted online PC DAgger with an XL-residual student matching the
    offline BC baseline (``[1024]*3`` encoder / ``[2048]*5`` head, point_dim=4 seg). Bakes in the
    weighted loss (``DistillationDAggerWeighted`` + student ``predict_std`` + ``teacher_returns_std``)
    and the XL sizing so only scalar overrides remain on launch: a ``--std`` ScenePC teacher JIT
    (``agent.policy.teacher_jit_path``) and an optional BC warm-start (``agent.policy.bc_init_path``).
    Split out as a subclass because hydra can't cleanly merge a list override into a MISSING field."""

    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_weighted_xl"
    policy: StudentTeacherPointCloudPolicyCfg = StudentTeacherPointCloudPolicyCfg(
        pointcloud_groups=["scene_pc"],
        encoder_dims=[1024, 1024, 1024],
        action_dims=[2048, 2048, 2048, 2048, 2048],
        predict_std=True,
        teacher_returns_std=True,
    )

    def __post_init__(self) -> None:
        self.algorithm.class_name = "DistillationDAggerWeighted"


@configclass
class PC_DAggerSplitWeightedXLOccGripRunnerCfg(PC_DAggerSplitWeightedXLRunnerCfg):
    """WeightedXL DAgger runner for the OCCLUDED GRIPPER-ONLY 3D cloud env
    (``PC-DAgger-OccludedGripper-WeightedXL-v0``): identical to the CleanSeg WeightedXL
    runner except the student consumes a 3-channel cloud (no seg) -> ``point_dim=3``,
    and results log to a separate experiment dir."""

    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_weighted_xl_occgrip"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.policy.point_dim = 3


@configclass
class PC_DAggerSplitWeightedXLOccGripHardOnlyRunnerCfg(PC_DAggerSplitWeightedXLOccGripRunnerCfg):
    """Occgrip 3D WeightedXL runner, HARD-reset-only env variant (separate experiment dir).
    Also the runner for the BC-init finetune arm (pass ``agent.policy.bc_init_path``)."""

    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_weighted_xl_occgrip_hardonly"


@configclass
class PC_DAggerSplitWeightedXLOccFingersHardOnlyRunnerCfg(PC_DAggerSplitWeightedXLOccGripHardOnlyRunnerCfg):
    """Fingers-only (arm/wrist occluder) variant of the occgrip hard-only runner."""

    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_weighted_xl_occfingers_hardonly"


@configclass
class PC_DAggerSplitWeightedXLOccObjectsHardOnlyRunnerCfg(PC_DAggerSplitWeightedXLOccFingersHardOnlyRunnerCfg):
    """Objects-only cloud (whole robot = occluder) variant of the hard-only runner."""

    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_weighted_xl_occobjects_hardonly"


@configclass
class PC_DAggerSplitWeightedXLOccObjectsHardOnlyArm6RunnerCfg(PC_DAggerSplitWeightedXLOccObjectsHardOnlyRunnerCfg):
    """Objects-only hard-only runner on the ARM-ONLY-proprio env (6 arm joints, no gripper
    mimic joints — real-robot proprio). Only the experiment dir differs; the student's proprio
    dim (12) is inferred from the env obs."""

    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_weighted_xl_occobjects_arm6_hardonly"


@configclass
class StudentTeacherHistoryPointCloudPolicyCfg:
    """Policy cfg for a history-conditioned Transformer student + JIT teacher (PC DAgger).

    The student is the SAME :class:`uwlab_rl.networks.history_point_net.HistoryPointNet` the
    offline sequence BC trainer builds (``train_point_net_seq.py``); the arch fields mirror the
    BC yaml so a sequence-BC checkpoint can initialize it. The student keeps a rolling per-env
    window of the last ``history_len`` (state, executed-action) pairs — env obs stay single-frame.
    """

    class_name: str = "StudentTeacherHistoryPointCloud"
    teacher_jit_path: str = ""
    pointcloud_groups: list[str] = MISSING
    point_dim: int = 3
    architecture: str = "history_point_net"
    # Per-point set encoder + per-token action head — mirror the BC config exactly.
    encoder_dims: list[int] = MISSING
    action_dims: list[int] = MISSING
    # Causal-Transformer geometry (mirror the BC config exactly).
    history_len: int = 16
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    transformer_dropout: float = 0.1
    activation: str = "elu"
    init_noise_std: float = 0.1
    noise_std_type: str = "scalar"
    # z-scoring stats are baked from the BC ckpt (proprio/action mean+std buffers), not learned online.
    student_obs_normalization: bool = False
    predict_std: bool = False
    teacher_returns_std: bool = False
    bc_init_path: str = ""


@configclass
class PC_DAggerSplitWeightedHist16OccObjectsHardOnlyRunnerCfg(PC_DAggerSplitRunnerCfg):
    """Weighted online DAgger with a HISTORY-16 Transformer student on the objects-only cloud
    (hard-reset-only env). Same split runner / DEXTRAH weighted loss / cadence as the WeightedXL
    feed-forward runners; only the student is the sequence policy. Arch mirrors
    ``configs/pn_xl_residual_history_clean.yaml`` at ``history_len=16`` so the sequence-BC ckpt
    (``occobjects_hardonly_hist16``) BC-inits exactly."""

    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_weighted_hist16_occobjects_hardonly"
    policy: StudentTeacherHistoryPointCloudPolicyCfg = StudentTeacherHistoryPointCloudPolicyCfg(
        pointcloud_groups=["scene_pc"],
        point_dim=3,
        encoder_dims=[256, 512, 512],
        action_dims=[512, 256],
        history_len=16,
        d_model=512,
        n_heads=8,
        n_layers=4,
        predict_std=True,
        teacher_returns_std=True,
    )

    def __post_init__(self) -> None:
        self.algorithm.class_name = "DistillationDAggerWeighted"


@configclass
class PC_DAggerSplitWeightedXLOccGripSegHardOnlyRunnerCfg(PC_DAggerSplitWeightedXLRunnerCfg):
    """Occgrip 4D (seg) WeightedXL runner, HARD-reset-only: cloud is (1024, 4) so the
    inherited ``point_dim=4`` is correct; only the experiment dir differs."""

    experiment_name: str = "ur5e_robotiq_2f85_pc_dagger_weighted_xl_occgripseg_hardonly"


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


@configclass
class ScenePCSuccessCriticOnlyRunnerCfg(ScenePCPPORunnerCfg):
    """Fit V_success for ONE frozen expert — PPO never updates (critic-only).

    Used to build the per-strategy value ensemble for the BAMDP
    value-feedback layer: run once per expert with
    ``--resume_path <expert ckpt>``; the saved checkpoints carry the critic
    under a ``success_critic`` key (see ``SuccessCriticOnlyRunner.save``).
    ``V(s, z=i)`` at deployment = critic from run i.

    ``gamma=0.95`` on purpose (parent default is 1.0): with terminal-success
    reward, gamma<1 makes V_success a *progress-graded* signal
    (~gamma^steps-to-success), the analog of the dm_control prototype's
    discounted-proximity V(s,z). The BAMDP failure cue is "value stops
    rising", which needs a value that rises in the first place; at gamma=1
    a near-optimal expert's V is ~flat (its success prob) and a synthetic
    plateau would be invisible.
    """

    class_name: str = "SuccessCriticOnlyRunner"
    experiment_name = "bamdp_success_critic"
    max_iterations = 300
    save_interval = 50
    obs_groups = {
        "policy": ["proprio", "pointcloud"],
        "critic": ["proprio", "pointcloud", "time_left"],
        "success_classifier": ["success_classifier"],
    }
    deterministic_rollout: bool = True
    success_critic: SuccessCriticCfg = SuccessCriticCfg(gamma=0.95)


# ===========================================================================
# RMA (Rapid Motor Adaptation) on ScenePC + Sysid Sim2Real.
# Actor conditions on phi(privileged_rma) -> z (16d). A small bidirectional
# transformer psi is trained in parallel to predict z from a 32-step history
# of proprio + last_action via MSE (stop-grad on z).
# ===========================================================================
@configclass
class RslRlActorCriticRMACfg(RslRlActorCriticWithEncoderCfg):
    class_name: str = "ActorCriticRMA"
    privileged_group: str = "privileged_rma"
    latent_dim: int = 16
    history_length: int = 32
    history_obs_keys: tuple = ("proprio",)
    history_include_actions: bool = True
    transformer_d_model: int = 64
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_ff: int = 128


@configclass
class RslRlPpoRMAAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPO_RMA"
    aux_coeff: float = 1.0
    history_learning_rate: float = 1.0e-4
    history_num_learning_epochs: int = 5
    history_num_mini_batches: int = 4
    # Cap per-minibatch size for the transformer aux pass. Without this,
    # 16k envs/GPU × 16 rollout steps = 262144 transitions, and divided by
    # ``history_num_mini_batches=4`` gives a 65k-batch SDPA forward that
    # trips CUDA's invalid-configuration limit. 4096 is safe; tune up if
    # you have memory headroom and want lower variance per update.
    history_minibatch_size: int = 4096
    # Optional: only train psi on a random subset of transitions per epoch.
    # ``None`` uses all transitions. Set if you want psi training cost
    # decoupled from policy num_envs.
    history_max_samples_per_epoch: int | None = None
    history_max_grad_norm: float = 1.0


@configclass
class ScenePCRMARunnerCfg(ScenePCPPORunnerCfg):
    """RMA: ScenePC PPO + privileged latent + history-encoder MSE auxiliary loss."""

    class_name: str = "OnPolicyRunnerRMA"

    obs_groups = {
        "policy": ["proprio", "pointcloud", "privileged_rma"],
        "critic": ["proprio", "pointcloud", "privileged_rma", "time_left"],
    }
    policy: RslRlActorCriticRMACfg = RslRlActorCriticRMACfg(
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
            # privileged_rma encoder (phi) is auto-injected by ActorCriticRMA
            # with output_dim=latent_dim; override hidden_dims here if desired.
        },
    )
    algorithm: RslRlPpoRMAAlgorithmCfg = RslRlPpoRMAAlgorithmCfg(
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
