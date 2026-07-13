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
* Policy built with ``predict_std=True`` — adds a student ``log_std_head`` head.
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
    """DAgger with inverse-variance-weighted L2 on mean + L2 on std.

    Optional separate loss treatment for the binary gripper dim (last action
    dim) — addresses the failure mode where MSE'd binary gripper command
    smooths out into indecisive sub-threshold values that mistime the grasp.
    """

    def __init__(
        self,
        *args,
        gripper_loss_type: str = "shared",
        gripper_loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # gripper_loss_type:
        #   "shared" — treat gripper dim like arm dims in weighted_l2 (original)
        #   "mse"    — split arm vs gripper; add λ·MSE(student_grip, teacher_grip)
        #   "bce"    — split arm vs gripper; add λ·BCE_with_logits(student_grip,
        #              (teacher_grip > 0).float()) — treats binary action as
        #              classification, more faithful to BinaryJointPositionAction
        if gripper_loss_type not in ("shared", "mse", "bce"):
            raise ValueError(f"Unknown gripper_loss_type: {gripper_loss_type}")
        self.gripper_loss_type = gripper_loss_type
        self.gripper_loss_weight = float(gripper_loss_weight)

    # Uses parent's ``act`` unchanged — teacher mean is stored as ``privileged_actions``.
    # Teacher std is re-computed on demand in ``update`` via a fresh ``evaluate_with_std``
    # call (teacher JIT is frozen + small, cheap per-step forward).

    def update(self) -> dict[str, float]:
        """Weighted BC update. Mirrors the aux-branch of :class:`DistillationDAgger.update`
        but replaces ``behavior_loss = mse(μ_s, μ_t)`` with the DEXTRAH-style
        weighted L2 + σ-term loss.

        Must re-run the student forward per step inside this loop to get
        ``(μ_s, σ_s)``; cannot reuse ``privileged_actions`` alone since σ
        supervision needs both ends.
        """
        # Unfreeze vision backbone once the warmup window ends.
        if hasattr(self.policy, "maybe_unfreeze_backbone"):
            self.policy.maybe_unfreeze_backbone(self.num_updates + 1)

        self.num_updates += 1
        mean_mu_loss = 0.0
        mean_sigma_loss = 0.0
        mean_aux_loss = 0.0
        loss = 0
        cnt = 0

        aux_enabled = bool(getattr(self.policy, "aux_enabled", False))
        has_action_history = hasattr(self.policy, "record_executed_action")
        train_mask = (~self.eval_mask).to(self.device) if self.eval_mask is not None else None

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, stored_actions, privileged_mu, dones in self.storage.generator():
                # Student (μ, σ) — single encoder pass, optionally with aux.
                if aux_enabled:
                    student_mu, student_sigma, aux_pred = self.policy.forward_with_aux_and_std(obs)
                    aux_target = self.policy.get_aux_target(obs)
                else:
                    student_mu, student_sigma = self.policy.act_inference_with_std(obs)
                    aux_pred, aux_target = None, None
                # Replay is time-ordered: re-feed the stored EXECUTED action so a stateful history
                # student's window rebuilds exactly as it stood during rollout.
                if has_action_history:
                    self.policy.record_executed_action(stored_actions)

                # Re-run teacher to get σ (teacher μ is already in privileged_actions).
                # Frozen JIT forward — cheap (215d → 7d MLP + gSDE head).
                _, teacher_sigma = self.policy.evaluate_with_std(obs)
                teacher_mu = privileged_mu

                # Apply eval mask (mask rows out of the gradient).
                if train_mask is not None:
                    s_mu = student_mu[train_mask]
                    s_sig = student_sigma[train_mask]
                    t_mu = teacher_mu[train_mask]
                    t_sig = teacher_sigma[train_mask]
                    if aux_enabled:
                        aux_pred = aux_pred[train_mask]
                        aux_target = aux_target[train_mask]
                else:
                    s_mu, s_sig = student_mu, student_sigma
                    t_mu, t_sig = teacher_mu, teacher_sigma

                # weights = (1/σ_t)² per dim, detached.
                weights = (1.0 / t_sig.detach().clamp_min(1e-6)) ** 2
                if self.gripper_loss_type == "shared":
                    mu_loss = _weighted_l2(s_mu, t_mu, weights).mean()
                    arm_loss_val = mu_loss.item()
                    gripper_loss_val = 0.0
                else:
                    # Split arm dims (0:6) from gripper dim (6).
                    arm_loss = _weighted_l2(s_mu[:, :6], t_mu[:, :6], weights[:, :6]).mean()
                    if self.gripper_loss_type == "mse":
                        gripper_loss = (s_mu[:, 6] - t_mu[:, 6]).pow(2).mean()
                    else:  # bce
                        # Teacher's binary intent: gripper output > 0 means CLOSE.
                        binary_target = (t_mu[:, 6] > 0).float()
                        gripper_loss = nn.functional.binary_cross_entropy_with_logits(
                            s_mu[:, 6], binary_target
                        )
                    mu_loss = arm_loss + self.gripper_loss_weight * gripper_loss
                    arm_loss_val = arm_loss.item()
                    gripper_loss_val = gripper_loss.item()
                sigma_loss = _l2(s_sig, t_sig).mean()
                step_loss = mu_loss + sigma_loss
                if aux_enabled:
                    aux_loss = self.loss_fn(aux_pred, aux_target)
                    step_loss = step_loss + self.aux_coeff * aux_loss
                    mean_aux_loss += aux_loss.item()

                loss = loss + step_loss
                mean_mu_loss += mu_loss.item()
                mean_sigma_loss += sigma_loss.item()
                if not hasattr(self, "_mean_arm_loss"):
                    self._mean_arm_loss = 0.0
                    self._mean_gripper_loss = 0.0
                self._mean_arm_loss += arm_loss_val
                self._mean_gripper_loss += gripper_loss_val
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
        mean_arm_loss = self._mean_arm_loss / cnt if hasattr(self, "_mean_arm_loss") else 0.0
        mean_gripper_loss = self._mean_gripper_loss / cnt if hasattr(self, "_mean_gripper_loss") else 0.0
        if hasattr(self, "_mean_arm_loss"):
            self._mean_arm_loss = 0.0
            self._mean_gripper_loss = 0.0
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        loss_dict = {
            # Use "behavior" as the shared key for rsl_rl's logger / our plot script.
            "behavior": mean_mu_loss + mean_sigma_loss,
            "behavior_mu": mean_mu_loss,
            "behavior_sigma": mean_sigma_loss,
            "behavior_arm": mean_arm_loss,
            "behavior_gripper": mean_gripper_loss,
        }
        if aux_enabled:
            loss_dict["aux"] = mean_aux_loss / cnt
        if self.student_mask is None:
            loss_dict["beta"] = self.beta
        else:
            loss_dict["student_fraction"] = float(self.student_mask.float().mean().item())
        if self.eval_mask is not None:
            loss_dict["eval_fraction"] = float(self.eval_mask.float().mean().item())
        return loss_dict
