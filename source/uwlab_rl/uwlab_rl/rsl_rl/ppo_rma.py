# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + RMA history-encoder auxiliary loss.

Runs upstream PPO unchanged for the actor/critic update, then runs a separate
auxiliary pass that minimizes ``MSE(psi(history), sg(phi(privileged_rma)))``
to train the history encoder. ``phi`` keeps a stop-gradient at the MSE so the
privileged latent is shaped purely by PPO, matching the original RMA setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO

from .actor_critic_rma import ActorCriticRMA


class PPO_RMA(PPO):
    def __init__(
        self,
        policy: ActorCriticRMA,
        aux_coeff: float = 1.0,
        history_learning_rate: float = 1.0e-4,
        history_num_learning_epochs: int | None = None,
        history_num_mini_batches: int | None = None,
        history_minibatch_size: int | None = 4096,
        history_max_samples_per_epoch: int | None = None,
        history_max_grad_norm: float = 1.0,
        **kwargs,
    ) -> None:
        if not isinstance(policy, ActorCriticRMA):
            raise TypeError("PPO_RMA requires an ActorCriticRMA policy.")

        # Build the parent optimizer over policy params EXCLUDING the history encoder.
        # Easiest way: build PPO normally, then replace its optimizer with one that
        # excludes psi, and separately create a psi-only optimizer.
        weight_decay = kwargs.get("weight_decay", 0.0)
        learning_rate = kwargs.get("learning_rate", 1.0e-4)
        super().__init__(policy=policy, **kwargs)

        history_params = list(policy.history_encoder.parameters())
        history_param_ids = {id(p) for p in history_params}
        main_params = [p for p in policy.parameters() if id(p) not in history_param_ids]
        self.optimizer = optim.AdamW(main_params, lr=learning_rate, weight_decay=weight_decay)
        self.history_optimizer = optim.AdamW(history_params, lr=history_learning_rate)

        self.aux_coeff = float(aux_coeff)
        self.history_max_grad_norm = float(history_max_grad_norm)
        self.history_num_learning_epochs = (
            int(history_num_learning_epochs)
            if history_num_learning_epochs is not None
            else int(self.num_learning_epochs)
        )
        self.history_num_mini_batches = (
            int(history_num_mini_batches)
            if history_num_mini_batches is not None
            else int(self.num_mini_batches)
        )
        # Cap minibatch size to keep SDPA / transformer memory under control.
        # If set, recomputes num_mini_batches per epoch so each minibatch is
        # at most this many (step, env) samples.
        self.history_minibatch_size = (
            int(history_minibatch_size) if history_minibatch_size is not None else None
        )
        # Cap total samples per aux epoch — for the user's "subsample envs we
        # train the history encoder on" knob. If set, draws this many random
        # (step, env) indices per epoch instead of using all transitions.
        self.history_max_samples_per_epoch = (
            int(history_max_samples_per_epoch)
            if history_max_samples_per_epoch is not None
            else None
        )

        # Set by the runner once the rollout buffer is sized.
        # Shape: (num_steps_per_env, num_envs, history_length, history_input_dim)
        self.history_snapshots: torch.Tensor | None = None

    def attach_history_snapshots(self, history_snapshots: torch.Tensor) -> None:
        """Called by the runner to share its rolling history snapshot buffer."""
        self.history_snapshots = history_snapshots

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self.history_snapshots is None:
            return loss_dict

        aux_loss_value, aux_mse_value = self._update_history_encoder()
        loss_dict["aux_history_mse"] = aux_mse_value
        loss_dict["aux_loss"] = aux_loss_value
        return loss_dict

    def _update_history_encoder(self) -> tuple[float, float]:
        """Auxiliary pass: psi(hist) -> z_hat, MSE against sg(phi(priv))."""
        privileged_group = self.policy.privileged_group  # type: ignore[attr-defined]
        privileged_obs = self._fetch_privileged_from_storage(privileged_group)
        privileged_flat = privileged_obs.reshape(-1, privileged_obs.shape[-1])
        history_flat = self.history_snapshots.reshape(
            -1, self.history_snapshots.shape[-2], self.history_snapshots.shape[-1]
        )

        total_samples = privileged_flat.shape[0]
        # Optional sub-sampling: when the rollout is huge (e.g. 16k envs/GPU
        # × 16 steps = 262144 transitions), feeding all of it through the
        # transformer trips SDPA / attention kernel limits and is wasteful.
        # Honor both ``history_max_samples_per_epoch`` (subsample envs) and
        # ``history_minibatch_size`` (cap minibatch size for the SDPA kernel).
        epoch_samples = (
            min(self.history_max_samples_per_epoch, total_samples)
            if self.history_max_samples_per_epoch is not None
            else total_samples
        )
        if self.history_minibatch_size is not None:
            mini_batch_size = min(self.history_minibatch_size, epoch_samples)
            num_minibatches = max(1, epoch_samples // mini_batch_size)
        else:
            num_minibatches = max(1, self.history_num_mini_batches)
            mini_batch_size = epoch_samples // num_minibatches
        if mini_batch_size <= 0:
            return 0.0, 0.0

        mse = nn.MSELoss()
        running_loss = 0.0
        running_mse = 0.0
        num_updates = 0
        for _ in range(self.history_num_learning_epochs):
            # If subsampling, draw `epoch_samples` from `total_samples` without replacement.
            if epoch_samples < total_samples:
                indices = torch.randperm(total_samples, device=self.device)[:epoch_samples]
            else:
                indices = torch.randperm(epoch_samples, device=self.device)
            indices = indices[: num_minibatches * mini_batch_size]
            for i in range(num_minibatches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                priv_batch = privileged_flat[batch_idx]
                hist_batch = history_flat[batch_idx]

                with torch.no_grad():
                    z_priv = self.policy.encode_privileged(priv_batch)  # type: ignore[attr-defined]
                z_hat = self.policy.encode_history(hist_batch)  # type: ignore[attr-defined]
                mse_loss = mse(z_hat, z_priv)
                loss = self.aux_coeff * mse_loss

                self.history_optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self._reduce_history_grads()
                nn.utils.clip_grad_norm_(
                    self.policy.history_encoder.parameters(),  # type: ignore[attr-defined]
                    self.history_max_grad_norm,
                )
                self.history_optimizer.step()

                running_loss += loss.item()
                running_mse += mse_loss.item()
                num_updates += 1

        if num_updates == 0:
            return 0.0, 0.0
        return running_loss / num_updates, running_mse / num_updates

    def _fetch_privileged_from_storage(self, group: str) -> torch.Tensor:
        """Return the (T_roll, N, D) tensor for ``group`` from whichever rollout
        bucket holds it — upstream rsl_rl splits per-step obs between
        ``self.storage.policy_observations`` (groups listed under obs_groups['policy'])
        and ``self.storage.observations`` (everything else).
        """
        policy_obs = getattr(self.storage, "policy_observations", None)
        if policy_obs is not None and group in policy_obs.keys():
            return policy_obs[group]
        if group in self.storage.observations.keys():
            return self.storage.observations[group]
        raise KeyError(
            f"PPO_RMA: privileged group '{group}' not present in rollout storage."
            f" policy_observations keys: {list(policy_obs.keys()) if policy_obs is not None else None};"
            f" observations keys: {list(self.storage.observations.keys())}."
        )

    def _reduce_history_grads(self) -> None:
        """All-reduce psi gradients across GPUs (mirrors PPO.reduce_parameters)."""
        for p in self.policy.history_encoder.parameters():  # type: ignore[attr-defined]
            if p.grad is None:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad /= self.gpu_world_size
