# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Gravity + reward-weight curriculums for OmniReset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GravityCurriculum(ManagerTermBase):
    """Curriculum that ramps gravity from zero-g to full gravity based on success rate.

    Supports two reduction modes for turning per-reset outcomes into a single
    ``difficulty_frac`` in ``[0, 1]``:

    - ``mean`` (default, legacy): per-env integer counter in ``[0, max_difficulty]``,
      ``+1`` on success-aligned-at-reset, ``-1`` otherwise. ``difficulty_frac`` is
      the mean counter across envs divided by ``max_difficulty``.
    - ``min_per_bucket``: one float counter per reset bucket (rt0, rt1, ...).
      Each counter accumulates ``wins - losses`` within that bucket, normalized
      by total ``num_envs`` per batch so the scale matches the legacy path.
      ``difficulty_frac`` is the ``min`` of per-bucket counters divided by
      ``max_difficulty`` — gravity ramps only when the slowest bucket is
      accumulating net wins. Decouples gravity from reset-sampling distribution.
    - ``monitor_mean``: reads the per-state success rates maintained by
      ``SuccessMonitor`` (uniform across all tracked reset states — not biased
      by GPS sampling distribution). Mean over every state is compared to
      ``monitor_threshold`` (default 0.8); if above, bump ``difficulty_frac``
      up by ``1 / max_difficulty``; if below, bump down. Unvisited states
      have success_rate=0 and drag the mean down, which is intentional —
      forces GPS to spread coverage before gravity ramps.

    Sets the physics scene gravity to ``(0, 0, full_gravity * difficulty_frac)``
    each call. Returns the current ``difficulty_frac`` for wandb logging.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Legacy per-env counter.
        self.current_difficulties = torch.zeros(env.num_envs, device=env.device)
        # Per-bucket counters (lazy: shape set on first call in min_per_bucket mode).
        self.bucket_difficulties: torch.Tensor | None = None
        self._reset_manager = None
        self.difficulty_frac = 0.0

    def _find_reset_manager(self, env):
        """Walk ``env.event_manager`` to find the first ``MultiResetManager``.

        Imports inside the function to avoid import-time cycles with uwlab_tasks.
        """
        event_mgr = getattr(env, "event_manager", None)
        if event_mgr is None:
            return None
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import MultiResetManager

        for mode_cfgs in event_mgr._mode_class_term_cfgs.values():
            for term_cfg in mode_cfgs:
                func = term_cfg.func
                if isinstance(func, MultiResetManager):
                    return func
        return None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        success_str: str = "env.reward_manager.get_term_cfg('progress_context').func.success",
        max_difficulty: int = 10,
        full_gravity: float = -9.81,
        reduction: str = "mean",
        monitor_threshold: float = 0.8,
    ) -> dict[str, float]:
        if reduction not in ("mean", "min_per_bucket", "monitor_mean"):
            raise ValueError(
                f"Unknown reduction '{reduction}', expected 'mean', 'min_per_bucket', or 'monitor_mean'"
            )

        log: dict[str, float] = {}

        if env_ids is not None and len(env_ids) > 0:
            success = eval(success_str)  # (num_envs,) bool tensor
            success_mask = success[env_ids]

            if reduction == "mean":
                self.current_difficulties[env_ids] = torch.where(
                    success_mask,
                    self.current_difficulties[env_ids] + 1,
                    self.current_difficulties[env_ids] - 1,
                ).clamp(min=0, max=max_difficulty)
                self.difficulty_frac = self.current_difficulties.mean().item() / max(max_difficulty, 1)

            elif reduction == "min_per_bucket":
                if self._reset_manager is None:
                    self._reset_manager = self._find_reset_manager(env)
                if self._reset_manager is None or not hasattr(self._reset_manager, "task_id"):
                    # Reset manager not ready yet (first-ever reset before MultiResetManager._lazy_init).
                    # Fall back: no update this step. difficulty_frac stays at 0.
                    pass
                else:
                    num_reset_types = self._reset_manager.num_reset_types
                    if self.bucket_difficulties is None:
                        self.bucket_difficulties = torch.zeros(
                            num_reset_types, dtype=torch.float, device=env.device
                        )
                    task_ids = self._reset_manager.task_id[env_ids]
                    per_env_step = 1.0 / env.num_envs
                    for k in range(num_reset_types):
                        mask = task_ids == k
                        if not mask.any():
                            continue
                        wins = success_mask[mask].float().sum()
                        losses = (~success_mask[mask]).float().sum()
                        delta = (wins - losses) * per_env_step
                        self.bucket_difficulties[k] = (
                            self.bucket_difficulties[k] + delta
                        ).clamp(min=0, max=max_difficulty)
                    self.difficulty_frac = (
                        self.bucket_difficulties.min().item() / max(max_difficulty, 1)
                    )
                    for k in range(num_reset_types):
                        log[f"gravity_bucket_{k}_frac"] = (
                            self.bucket_difficulties[k].item() / max(max_difficulty, 1)
                        )

            else:  # monitor_mean
                if self._reset_manager is None:
                    self._reset_manager = self._find_reset_manager(env)
                monitor = getattr(self._reset_manager, "success_monitor", None) if self._reset_manager else None
                if monitor is None:
                    # Reset manager / monitor not ready yet. Hold difficulty_frac.
                    pass
                else:
                    mean_rate = monitor.success_rate.mean().item()
                    visited_frac = (monitor.success_size > 0).float().mean().item()
                    step = 1.0 / max(max_difficulty, 1)
                    if mean_rate > monitor_threshold:
                        self.difficulty_frac = min(self.difficulty_frac + step, 1.0)
                    elif mean_rate < monitor_threshold:
                        self.difficulty_frac = max(self.difficulty_frac - step, 0.0)
                    log["monitor_mean_rate"] = mean_rate
                    log["monitor_visited_frac"] = visited_frac

        # Set gravity based on current difficulty
        import carb
        from isaaclab.sim import SimulationContext

        gravity = carb.Float3(0.0, 0.0, full_gravity * self.difficulty_frac)
        physics_sim_view = SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(gravity)

        log["gravity_frac"] = self.difficulty_frac
        return log


class RewardWeightCurriculum(ManagerTermBase):
    """Curriculum that anneals reward term weights toward zero as gravity difficulty increases.

    Tracks the gravity curriculum's difficulty fraction and linearly decays the
    specified reward weights from their initial values to zero.
    At difficulty_frac=0 (zero-g), weights are at full value.
    At difficulty_frac=1 (full gravity), weights are zero.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initial_weights: dict[str, float] = {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        reward_term_names: list[str] | None = None,
        gravity_curriculum_name: str = "gravity_curriculum",
    ) -> dict[str, float]:
        if reward_term_names is None:
            reward_term_names = []

        # Capture initial weights on first call
        if not self._initial_weights:
            for name in reward_term_names:
                self._initial_weights[name] = env.reward_manager.get_term_cfg(name).weight

        # Get gravity difficulty fraction from gravity curriculum
        # CurriculumManager doesn't have get_term_cfg, so look up by name directly
        cm = env.curriculum_manager
        gravity_cfg = cm._term_cfgs[cm._term_names.index(gravity_curriculum_name)]
        gravity_term = gravity_cfg.func
        difficulty_frac = getattr(gravity_term, "difficulty_frac", 0.0)
        decay = 1.0 - difficulty_frac

        log = {}
        for name in reward_term_names:
            cfg = env.reward_manager.get_term_cfg(name)
            cfg.weight = self._initial_weights[name] * decay
            env.reward_manager.set_term_cfg(name, cfg)
            log[f"dense_reward_decay/{name}_weight"] = cfg.weight

        log["dense_reward_decay/decay_factor"] = decay
        return log
