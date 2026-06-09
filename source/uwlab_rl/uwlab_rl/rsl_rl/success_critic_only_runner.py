# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Critic-only variant of :class:`OnPolicyRunnerWithSuccessCritic`.

Fits the auxiliary ``V_success`` head on rollouts of a FROZEN policy — PPO
never updates. Use case: per-strategy success critics for the BAMDP
value-feedback layer (one run per frozen expert checkpoint, loaded via
``--resume_path``). The collection of resulting critics IS the
skill-conditioned value V(s, z): ``V(s, z=i) = critic_i(s)``.

Differences from the parent:

  * ``alg.act`` returns ``policy.act_inference(obs)`` (deterministic mean,
    matching how the expert ensemble dispatches actions at deployment) and
    never populates the PPO rollout storage.
  * ``alg.process_env_step`` / ``alg.compute_returns`` only do the
    success-critic bookkeeping; PPO storage stays empty.
  * ``alg.update`` trains ONLY the success critic. Policy and PPO optimizer
    are untouched, so the loaded expert is bit-identical at every iteration.
  * ``save`` embeds the critic (weights + arch + normalizer) under a
    ``success_critic`` key in the checkpoint; ``load`` restores it when
    present and resets ``current_learning_iteration`` to 0 so resuming from
    an expert checkpoint (e.g. ``model_4000.pt``) does not consume the
    iteration budget in train.py's remaining-iterations arithmetic.

Single-GPU only (no parameter broadcast is needed since nothing about the
policy ever changes; just don't launch with ``--distributed``).

train_cfg keys (beyond the parent's ``success_critic`` block):

  ``deterministic_rollout``  default True — drive with the policy mean.
"""

from __future__ import annotations

import torch

from .on_policy_runner_with_success_critic import OnPolicyRunnerWithSuccessCritic


class SuccessCriticOnlyRunner(OnPolicyRunnerWithSuccessCritic):
    # ------------------------------------------------------------------
    # Hooks — replaces the parent's wrapping entirely (called from the
    # parent's __init__, so read flags from self.cfg, not attributes).
    # ------------------------------------------------------------------
    def _wrap_alg_methods(self) -> None:
        deterministic = bool(self.cfg.get("deterministic_rollout", True))

        def act_wrapped(obs):
            t = self._sc_step
            if t < self.num_steps_per_env:
                sc_obs = obs["success_classifier"]
                self._sc_obs[t] = sc_obs
                with torch.no_grad():
                    self._sc_values[t] = self.success_critic(sc_obs).detach()
                self.success_critic.update_normalization(sc_obs)
            # Frozen policy drives; PPO storage is never touched.
            if deterministic:
                return self.alg.policy.act_inference(obs)
            return self.alg.policy.act(obs)

        def process_env_step_wrapped(obs, rewards, dones, extras):
            t = self._sc_step
            if t < self.num_steps_per_env:
                progress = self._env_unwrapped().reward_manager.get_term_cfg("progress_context").func
                success_bool = progress.success.float()
                dones_f = dones.float()
                self._sc_rewards[t, :, 0] = success_bool * dones_f
                self._sc_dones[t, :, 0] = dones_f
                if "time_outs" in extras:
                    self._sc_time_outs[t, :, 0] = extras["time_outs"].float()
                else:
                    self._sc_time_outs[t, :, 0] = 0.0
            self._sc_step += 1

        def compute_returns_wrapped(obs):
            self._sc_last_obs = obs["success_classifier"]

        def update_wrapped():
            sc_metrics = self._update_success_critic()
            self._sc_step = 0
            return {f"SuccessCritic/{k}": v for k, v in sc_metrics.items()}

        self.alg.act = act_wrapped
        self.alg.process_env_step = process_env_step_wrapped
        self.alg.compute_returns = compute_returns_wrapped
        self.alg.update = update_wrapped

    def _env_unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

    # ------------------------------------------------------------------
    # Persistence — the parent never saves the critic; we embed it in the
    # regular checkpoint so a single .pt carries (frozen policy, critic).
    # ------------------------------------------------------------------
    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        ckpt = torch.load(path, weights_only=False)  # default map keeps devices
        sc_cfg = self.cfg["success_critic"]
        ckpt["success_critic"] = {
            "state_dict": {k: v.detach().cpu() for k, v in self.success_critic.state_dict().items()},
            "input_dim": int(self._sc_obs.shape[-1]),
            "hidden_dims": list(sc_cfg["hidden_dims"]),
            "activation": str(sc_cfg["activation"]),
            "obs_normalization": bool(sc_cfg["obs_normalization"]),
        }
        torch.save(ckpt, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        loaded = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(loaded, dict) and "success_critic" in loaded:
            self.success_critic.load_state_dict(loaded["success_critic"]["state_dict"])
            print("[SuccessCriticOnlyRunner] restored success critic from checkpoint.")
        # Expert checkpoints carry their own training iteration (e.g. 4000);
        # don't let it eat our --max_iterations budget.
        self.current_learning_iteration = 0
        return infos
