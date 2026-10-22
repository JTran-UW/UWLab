# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configs for rsl_rl's off-policy SAC (``rsl_rl.algorithms.SAC`` / ``rsl_rl.runners.OffPolicyRunner``)."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


@configclass
class RslRlSacActorCfg:
    class_name: str = "SACActorModel"
    hidden_dims: list[int] = MISSING
    activation: str = "elu"
    obs_normalization: bool = False
    layer_norm: bool = False
    init_noise_std: float = 1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    log_std_mode: str = "clamp"
    """"clamp" or "tanh" (FastSAC: log-std = min + (max - min) * (tanh(x) + 1) / 2)."""


@configclass
class RslRlSacCriticCfg:
    class_name: str = "SACCriticModel"
    hidden_dims: list[int] = MISSING
    activation: str = "elu"
    obs_normalization: bool = False
    layer_norm: bool = False
    num_atoms: int = 1
    """> 1: C51 distributional critics over ``num_atoms`` atoms in [v_min, v_max]."""
    v_min: float = -10.0
    v_max: float = 10.0


@configclass
class RslRlSacAlgorithmCfg:
    class_name: str = "SAC"
    replay_buffer_size: int = 1_000_000
    """Total transitions across all envs."""
    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    """Gradient updates per iteration (with num_learning_epochs=1)."""
    mini_batch_size: int = 256
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    alpha_learning_rate: float = 1e-3
    actor_optimizer: str = "adam"
    critic_optimizer: str = "adam"
    actor_optimizer_kwargs: dict | None = None
    critic_optimizer_kwargs: dict | None = None
    auto_alpha: bool = True
    alpha: float = 0.05
    tau: float = 0.005
    gamma: float = 0.99
    target_entropy_scale: float = 1.0
    """Target entropy = -scale * action_dim."""
    max_grad_norm: float | None = 1.0
    policy_frequency: int = 2
    n_steps: int = 1
    q_aggregation: str = "min"
    action_bound: str = "joint_limit"
    """"joint_limit" (scale to soft joint limits) or "unit" (actions in [-1, 1])."""


@configclass
class RslRlOffPolicyRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "OffPolicyRunner"
    empirical_normalization: bool = False
    start_training: int = 0
    """Iterations of pure collection before the first update."""
    actor: RslRlSacActorCfg = MISSING
    critic: RslRlSacCriticCfg = MISSING
    algorithm: RslRlSacAlgorithmCfg = MISSING
