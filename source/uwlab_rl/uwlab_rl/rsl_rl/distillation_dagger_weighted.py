# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DEXTRAH-style inverse-variance-weighted distillation loss.

Replaces the plain MSE-on-mean loss of :class:`DistillationDAgger` with::

    loss = weighted_l2(μ_student, μ_teacher, weights = (1/σ_teacher)^2)
         + l2(σ_student, σ_teacher)

where:

* ``weighted_l2(m, t, w) = sqrt(Σ_i w_i (m_i - t_i)^2)`` — per-sample L2 norm
  with per-action-dim weights, matching
  ``dextrah_lab/distillation/distillation.py:53-54``.
* ``l2(m, t) = ||m - t||_2`` — unweighted per-sample L2 norm.

Teacher ``σ_teacher`` is detached from the weighting factor (we don't want the
weighting itself to create a gradient path back into the teacher — it's frozen
anyway, but explicit detach preserves semantic clarity).

Requires:

* Policy built with ``teacher_returns_std=True`` + teacher JIT that returns
  ``(mean, std)`` (see :func:`scripts_v2/tools/convert_state_expert_to_jit.py`
  with ``--std``).
* Policy built with ``predict_std=True`` — adds a student ``std_head`` head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from .distillation_dagger import DistillationDAgger


def _weighted_l2(model: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-sample weighted L2 norm. ``sqrt(Σ_i w_i (m_i - t_i)²)``. Returns (B,)."""
    return torch.sqrt(torch.sum(weights * (model - target) ** 2, dim=-1))


def _l2(model: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm. ``||m - t||_2``. Returns (B,)."""
    return torch.norm(model - target, p=2, dim=-1)


class DistillationDAggerWeighted(DistillationDAgger):
    """DAgger with inverse-variance-weighted L2 on mean + L2 on std."""

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Rollout: same routing logic as parent, but evaluate teacher (μ, σ)
        and stash ``teacher_std`` in the transition's observation dict so the
        update loop can recover it without re-running the teacher JIT."""
        student_action = self.policy.act(obs).detach()
        teacher_mean, teacher_std = self.policy.evaluate_with_std(obs)
        teacher_mean = teacher_mean.detach()
        teacher_std = teacher_std.detach()

        if self.student_mask is not None:
            mask = self.student_mask.unsqueeze(-1)
            action = torch.where(mask, student_action, teacher_mean)
        else:
            beta = self.beta
            if beta > 0.0:
                num_envs = student_action.shape[0]
                use_teacher = torch.rand(num_envs, device=student_action.device) < beta
                action = torch.where(use_teacher.unsqueeze(-1), teacher_mean, student_action)
            else:
                action = student_action

        self.transition.actions = action
        self.transition.privileged_actions = teacher_mean
        # Stash teacher std alongside the obs so the update-loop replay can use it.
        # ``observations`` is a TensorDict; adding a ``_teacher_std`` key is safe
        # since obs_groups only inspect specific group names.
        obs_with_std = obs
        obs_with_std["_teacher_std"] = teacher_std
        self.transition.observations = obs_with_std
        return action

    def update(self) -> dict[str, float]:
        """Weighted BC update. Mirrors the aux-branch of :class:`DistillationDAgger.update`
        but replaces ``behavior_loss = mse(μ_s, μ_t)`` with the DEXTRAH-style
        weighted L2 + σ-term loss.

        Must re-run the student forward per step inside this loop to get
        ``(μ_s, σ_s)``; cannot reuse ``privileged_actions`` alone since σ
        supervision needs both ends.
        """
        self.num_updates += 1
        mean_mu_loss = 0.0
        mean_sigma_loss = 0.0
        mean_aux_loss = 0.0
        loss = 0
        cnt = 0

        aux_enabled = bool(getattr(self.policy, "aux_enabled", False))
        train_mask = (~self.eval_mask).to(self.device) if self.eval_mask is not None else None

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_mu, dones in self.storage.generator():
                # Student (μ, σ).
                if aux_enabled:
                    # forward_with_aux returns (mean, aux_pred); need std separately,
                    # which requires a second encoder call. For now: no aux here.
                    raise NotImplementedError(
                        "DistillationDAggerWeighted + aux head not implemented; "
                        "share the encoder between action-head/std-head/aux-head forward."
                    )
                student_mu, student_sigma = self.policy.act_inference_with_std(obs)

                teacher_sigma = obs["_teacher_std"]
                teacher_mu = privileged_mu

                # Apply eval mask (mask rows out of the gradient).
                if train_mask is not None:
                    s_mu = student_mu[train_mask]
                    s_sig = student_sigma[train_mask]
                    t_mu = teacher_mu[train_mask]
                    t_sig = teacher_sigma[train_mask]
                else:
                    s_mu, s_sig = student_mu, student_sigma
                    t_mu, t_sig = teacher_mu, teacher_sigma

                # weights = (1/σ_t)² per dim, detached.
                weights = (1.0 / t_sig.detach().clamp_min(1e-6)) ** 2
                mu_loss = _weighted_l2(s_mu, t_mu, weights).mean()
                sigma_loss = _l2(s_sig, t_sig).mean()
                step_loss = mu_loss + sigma_loss

                loss = loss + step_loss
                mean_mu_loss += mu_loss.item()
                mean_sigma_loss += sigma_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_mu_loss /= cnt
        mean_sigma_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        loss_dict = {
            # Use "behavior" as the shared key for rsl_rl's logger / our plot script.
            "behavior": mean_mu_loss + mean_sigma_loss,
            "behavior_mu": mean_mu_loss,
            "behavior_sigma": mean_sigma_loss,
        }
        if self.student_mask is None:
            loss_dict["beta"] = self.beta
        else:
            loss_dict["student_fraction"] = float(self.student_mask.float().mean().item())
        if self.eval_mask is not None:
            loss_dict["eval_fraction"] = float(self.eval_mask.float().mean().item())
        return loss_dict
