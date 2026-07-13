# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation algorithm with two teacher-injection modes.

Two mutually exclusive mixing strategies on top of
``rsl_rl.algorithms.Distillation``:

1. β-annealed per-step coin flip (default): each env each step, teacher acts
   with probability ``beta = max(0, 1 - num_updates / beta_anneal_iters)``.
   Mixes teacher into rollouts early, anneals to pure student.

2. Fixed per-env pool split: caller provides ``student_mask`` of shape
   ``(num_envs,)``; student-pool envs always run student actions, teacher-pool
   envs always run teacher actions. Enables clean per-pool success logging.

Optional ``eval_mask`` (shape ``(num_envs,)``): envs where the student drives
rollouts but whose transitions are excluded from the gradient. Intended as a
contamination-free eval signal at tight train cadences (e.g. ``num_steps_per_env=1``
with ``gradient_length=1``) where same-step gradient updates on a given env's
transition bias the student's next action on that same env ("echo-of-teacher").

Loss is MSE-on-mean over every non-eval transition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation


class DistillationDAgger(Distillation):
    """DAgger with β-annealed mixing or fixed per-env pool split + optional eval pool."""

    def __init__(
        self,
        *args,
        beta_anneal_iters: int = 0,
        student_mask: torch.Tensor | None = None,
        eval_mask: torch.Tensor | None = None,
        aux_coeff: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta_anneal_iters = int(beta_anneal_iters)
        if student_mask is not None:
            student_mask = student_mask.to(self.device).bool()
        self.student_mask = student_mask
        if eval_mask is not None:
            eval_mask = eval_mask.to(self.device).bool()
        self.eval_mask = eval_mask
        self.aux_coeff = float(aux_coeff)

    @property
    def beta(self) -> float:
        if self.beta_anneal_iters <= 0:
            return 0.0
        return max(0.0, 1.0 - self.num_updates / self.beta_anneal_iters)

    def act(self, obs: TensorDict) -> torch.Tensor:
        student_action = self.policy.act(obs).detach()
        teacher_action = self.policy.evaluate(obs).detach()

        if self.student_mask is not None:
            mask = self.student_mask.unsqueeze(-1)
            action = torch.where(mask, student_action, teacher_action)
        else:
            beta = self.beta
            if beta > 0.0:
                num_envs = student_action.shape[0]
                use_teacher = torch.rand(num_envs, device=student_action.device) < beta
                action = torch.where(use_teacher.unsqueeze(-1), teacher_action, student_action)
            else:
                action = student_action

        # Stateful history students keep a rolling (state, executed-action) window: tell the
        # policy which action was ACTUALLY applied (student / teacher / mixed) this step.
        if hasattr(self.policy, "record_executed_action"):
            self.policy.record_executed_action(action)

        self.transition.actions = action
        self.transition.privileged_actions = teacher_action
        self.transition.observations = obs
        return action

    def update(self) -> dict[str, float]:
        """BC update, excluding eval-pool envs from the gradient.

        Copied from ``Distillation.update`` because we need to mask rows
        (eval-pool envs) out of each minibatch before the MSE (parent uses
        a single-path loss), AND to optionally add an aux-pose loss via
        ``policy.evaluate_aux`` + ``policy.get_aux_target`` when the student
        has an aux head.
        """
        aux_enabled = bool(getattr(self.policy, "aux_enabled", False))
        has_action_history = hasattr(self.policy, "record_executed_action")
        use_custom = (self.eval_mask is not None) or aux_enabled or has_action_history

        # Unfreeze vision backbone once the warmup window ends (no-op if
        # ``encoder_freeze_iters <= 0`` or already unfrozen).
        if hasattr(self.policy, "maybe_unfreeze_backbone"):
            self.policy.maybe_unfreeze_backbone(self.num_updates + 1)

        if not use_custom:
            loss_dict = super().update()
        else:
            self.num_updates += 1
            mean_behavior_loss = 0.0
            mean_aux_loss = 0.0
            loss = 0
            cnt = 0

            train_mask = (~self.eval_mask).to(self.device) if self.eval_mask is not None else None

            for epoch in range(self.num_learning_epochs):
                self.policy.reset(hidden_states=self.last_hidden_states)
                self.policy.detach_hidden_states()
                for obs, stored_actions, privileged_actions, dones in self.storage.generator():
                    if aux_enabled:
                        actions, aux_pred = self.policy.forward_with_aux(obs)
                        aux_target = self.policy.get_aux_target(obs)
                    else:
                        actions = self.policy.act_inference(obs)
                        aux_pred = None
                    # Replay is time-ordered: re-feed the stored EXECUTED action so the history
                    # window rebuilds exactly as it stood during rollout.
                    if has_action_history:
                        self.policy.record_executed_action(stored_actions)

                    if train_mask is not None:
                        actions_m = actions[train_mask]
                        privileged_m = privileged_actions[train_mask]
                    else:
                        actions_m = actions
                        privileged_m = privileged_actions
                    behavior_loss = self.loss_fn(actions_m, privileged_m)
                    step_loss = behavior_loss
                    mean_behavior_loss += behavior_loss.item()

                    if aux_enabled:
                        if train_mask is not None:
                            aux_pred_m = aux_pred[train_mask]
                            aux_target_m = aux_target[train_mask]
                        else:
                            aux_pred_m = aux_pred
                            aux_target_m = aux_target
                        aux_loss = self.loss_fn(aux_pred_m, aux_target_m)
                        step_loss = step_loss + self.aux_coeff * aux_loss
                        mean_aux_loss += aux_loss.item()

                    loss = loss + step_loss
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

            mean_behavior_loss /= cnt
            mean_aux_loss = mean_aux_loss / cnt if aux_enabled else 0.0
            self.storage.clear()
            self.last_hidden_states = self.policy.get_hidden_states()
            self.policy.detach_hidden_states()
            loss_dict = {"behavior": mean_behavior_loss}
            if aux_enabled:
                loss_dict["aux"] = mean_aux_loss

        if self.student_mask is None:
            loss_dict["beta"] = self.beta
        else:
            loss_dict["student_fraction"] = float(self.student_mask.float().mean().item())
        if self.eval_mask is not None:
            loss_dict["eval_fraction"] = float(self.eval_mask.float().mean().item())
        return loss_dict
