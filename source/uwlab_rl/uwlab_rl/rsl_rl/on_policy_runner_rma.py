# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner that maintains a per-env rolling (B, T, D) history of
proprio + last_action and snapshots it on every rollout step so PPO_RMA can
align the auxiliary MSE loss with the per-transition obs in storage.

The buffer is allocated once based on observed obs shapes. On env reset the
corresponding rows are zeroed so the history encoder doesn't span an episode
boundary.
"""

from __future__ import annotations

import torch

from rsl_rl.runners import OnPolicyRunner


class OnPolicyRunnerRMA(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Make our policy + algorithm classes visible to upstream eval() resolution.
        import rsl_rl.runners.on_policy_runner as _opr

        from .actor_critic_rma import ActorCriticRMA
        from .ppo_rma import PPO_RMA

        _opr.ActorCriticRMA = ActorCriticRMA
        _opr.PPO_RMA = PPO_RMA

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # Read history config off the policy (built from policy cfg during super().__init__).
        policy = self.alg.policy
        if not isinstance(policy, ActorCriticRMA):
            raise TypeError(
                "OnPolicyRunnerRMA requires policy.class_name=ActorCriticRMA in the runner cfg."
            )
        self._history_length = int(policy.history_length)
        self._history_input_dim = int(policy.history_input_dim)
        self._history_obs_keys = tuple(policy.history_obs_keys)
        self._history_include_actions = bool(policy.history_include_actions)
        self._num_actions = int(env.num_actions)
        self._num_envs = int(env.num_envs)

        # Rolling buffer (one window per env) and per-step snapshots over the rollout.
        self._hist_buf = torch.zeros(
            self._num_envs, self._history_length, self._history_input_dim, device=self.device
        )
        self._hist_snapshots = torch.zeros(
            self.num_steps_per_env,
            self._num_envs,
            self._history_length,
            self._history_input_dim,
            device=self.device,
        )
        self._last_actions = torch.zeros(self._num_envs, self._num_actions, device=self.device)
        self._rollout_step = 0

        # Hand the snapshot buffer to the algorithm for the auxiliary loss.
        self.alg.attach_history_snapshots(self._hist_snapshots)

        # Hook into PPO's act / process_env_step / update to manage the buffer.
        self._inner_act = self.alg.act
        self._inner_process_env_step = self.alg.process_env_step
        self._inner_update = self.alg.update
        self.alg.act = self._act_with_history
        self.alg.process_env_step = self._process_env_step_with_history
        self.alg.update = self._update_with_history

    def _resolve_obs(self, obs, key):
        # Tolerate both flat ``obs[key]`` and the upstream nested form
        # ``obs['policy'][key]`` used by some rsl_rl paths.
        if key in obs.keys():
            return obs[key]
        if "policy" in obs.keys() and key in obs["policy"].keys():
            return obs["policy"][key]
        raise KeyError(f"OnPolicyRunnerRMA: history obs '{key}' not found in obs.")

    def _build_history_frame(self, obs) -> torch.Tensor:
        parts = [self._resolve_obs(obs, k) for k in self._history_obs_keys]
        if self._history_include_actions:
            parts.append(self._last_actions)
        return torch.cat(parts, dim=-1)

    def _act_with_history(self, obs):
        # Push the current frame (proprio + last action) onto the buffer BEFORE acting,
        # so the snapshot represents what psi would see at deploy time.
        frame = self._build_history_frame(obs)
        self._hist_buf = torch.roll(self._hist_buf, shifts=-1, dims=1)
        self._hist_buf[:, -1] = frame
        # Snapshot this rollout step's history window for the auxiliary loss.
        self._hist_snapshots[self._rollout_step].copy_(self._hist_buf)
        actions = self._inner_act(obs)
        self._last_actions = actions.detach()
        return actions

    def _process_env_step_with_history(self, obs, rewards, dones, extras):
        self._rollout_step += 1
        # Zero history rows for envs that just terminated, so the next episode's
        # transformer input doesn't reach across the reset.
        if dones.any():
            done_mask = dones.bool()
            self._hist_buf[done_mask] = 0.0
            self._last_actions[done_mask] = 0.0
        self._inner_process_env_step(obs, rewards, dones, extras)

    def _update_with_history(self):
        loss_dict = self._inner_update()
        self._rollout_step = 0
        return loss_dict
