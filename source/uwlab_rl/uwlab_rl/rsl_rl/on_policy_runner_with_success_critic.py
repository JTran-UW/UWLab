# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner variant that also trains a V_success auxiliary value head.

The V_success head takes a dedicated `success_classifier` obs group and is
trained via GAE on a reward stream `r_t = 1 on success-termination else 0`.
Its predictions are injected into MultiResetManager for GPS curriculum sampling.
"""

from __future__ import annotations

import torch
from rsl_rl.runners import OnPolicyRunner

from .success_critic import SuccessCritic, compute_success_gae


class OnPolicyRunnerWithSuccessCritic(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        sc_cfg = train_cfg["success_critic"]
        self.sc_num_epochs = int(sc_cfg["num_learning_epochs"])
        self.sc_num_mini_batches = int(sc_cfg["num_mini_batches"])
        self.sc_gamma = float(sc_cfg["gamma"])
        self.sc_lam = float(sc_cfg["lam"])
        self.sc_clip_param = float(sc_cfg["clip_param"])
        self.sc_use_clipped_value_loss = bool(sc_cfg["use_clipped_value_loss"])
        self.sc_max_grad_norm = float(sc_cfg["max_grad_norm"])

        obs = self.env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        sc_obs_example = obs["success_classifier"]

        self.success_critic = SuccessCritic(
            input_dim=sc_obs_example.shape[-1],
            hidden_dims=list(sc_cfg["hidden_dims"]),
            activation=sc_cfg["activation"],
            obs_normalization=bool(sc_cfg["obs_normalization"]),
            lr=float(sc_cfg["lr"]),
            device=device,
        )

        T = self.num_steps_per_env
        N = self.env.num_envs
        D = sc_obs_example.shape[-1]
        self._sc_obs = torch.zeros(T, N, D, device=device)
        self._sc_values = torch.zeros(T, N, 1, device=device)
        self._sc_rewards = torch.zeros(T, N, 1, device=device)
        self._sc_dones = torch.zeros(T, N, 1, device=device)
        self._sc_time_outs = torch.zeros(T, N, 1, device=device)
        self._sc_last_obs: torch.Tensor | None = None
        self._sc_step = 0

        self._reset_manager = None
        self._reset_manager_injected = False

        self._wrap_alg_methods()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _wrap_alg_methods(self) -> None:
        inner_act = self.alg.act
        inner_process = self.alg.process_env_step
        inner_compute_returns = self.alg.compute_returns
        inner_update = self.alg.update

        def act_wrapped(obs):
            t = self._sc_step
            if t < self.num_steps_per_env:
                sc_obs = obs["success_classifier"]
                self._sc_obs[t] = sc_obs
                with torch.no_grad():
                    self._sc_values[t] = self.success_critic(sc_obs).detach()
                self.success_critic.update_normalization(sc_obs)
            return inner_act(obs)

        def process_env_step_wrapped(obs, rewards, dones, extras):
            t = self._sc_step
            if t < self.num_steps_per_env:
                progress = self.env.unwrapped.reward_manager.get_term_cfg("progress_context").func
                success_bool = progress.success.float()
                dones_f = dones.float()
                self._sc_rewards[t, :, 0] = success_bool * dones_f
                self._sc_dones[t, :, 0] = dones_f
                if "time_outs" in extras:
                    self._sc_time_outs[t, :, 0] = extras["time_outs"].float()
                else:
                    self._sc_time_outs[t, :, 0] = 0.0
            self._sc_step += 1
            return inner_process(obs, rewards, dones, extras)

        def compute_returns_wrapped(obs):
            self._sc_last_obs = obs["success_classifier"]
            return inner_compute_returns(obs)

        def update_wrapped():
            ppo_loss = inner_update()
            sc_metrics = self._update_success_critic()
            ppo_loss.update({f"SuccessCritic/{k}": v for k, v in sc_metrics.items()})
            self._sc_step = 0
            return ppo_loss

        self.alg.act = act_wrapped
        self.alg.process_env_step = process_env_step_wrapped
        self.alg.compute_returns = compute_returns_wrapped
        self.alg.update = update_wrapped

    # ------------------------------------------------------------------
    # V_success update
    # ------------------------------------------------------------------
    def _update_success_critic(self) -> dict[str, float]:
        if self._sc_last_obs is None:
            return {"value_loss": 0.0, "skipped": 1.0}

        with torch.no_grad():
            last_values = self.success_critic(self._sc_last_obs).detach()

        returns, _ = compute_success_gae(
            self._sc_rewards,
            self._sc_values,
            self._sc_dones,
            self._sc_time_outs,
            last_values,
            self.sc_gamma,
            self.sc_lam,
        )
        returns = returns.clamp(0.0, 1.0)

        flat_obs = self._sc_obs.reshape(-1, self._sc_obs.shape[-1])
        flat_returns = returns.reshape(-1, 1)
        flat_values = self._sc_values.reshape(-1, 1)

        metrics = self.success_critic.update(
            flat_obs,
            flat_returns,
            flat_values,
            num_epochs=self.sc_num_epochs,
            num_mini_batches=self.sc_num_mini_batches,
            use_clipped_value_loss=self.sc_use_clipped_value_loss,
            clip_param=self.sc_clip_param,
            max_grad_norm=self.sc_max_grad_norm,
        )
        metrics["rollout_success_rate"] = float(self._sc_rewards.sum().item() / max(1, (self._sc_dones > 0).sum().item()))

        if not self._reset_manager_injected:
            self._reset_manager = self._find_reset_manager()
            if self._reset_manager is not None:
                self._reset_manager.set_success_critic(self.success_critic)
                self._reset_manager_injected = True

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_reset_manager(self):
        base_env = self.env.unwrapped
        event_mgr = getattr(base_env, "event_manager", None)
        if event_mgr is None:
            return None

        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import MultiResetManager

        for mode_cfgs in event_mgr._mode_class_term_cfgs.values():
            for term_cfg in mode_cfgs:
                func = term_cfg.func
                if isinstance(func, MultiResetManager) and getattr(func, "use_success_critic", False):
                    return func
        return None
