# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recurrent vision student for DAgger — LSTM between CNN/proprio embedding and action head.

DEXTRAH-style recurrent student (see ``DEXTRAH/dextrah_lab/distillation/distillation.py``):

* Fused features ``[proprio ; per-cam image features]`` → LSTM(hidden_dim) →
  action-head MLP → action mean.
* Per-env LSTM hidden state stored on ``self.memory_s``; detached each iter
  (so gradient only flows through the current cell — "1-step BPTT"), zeroed
  on ``reset(dones)`` for done envs.
* Trained with the existing ``DistillationDAgger`` loop. rsl_rl's
  ``Distillation.update`` already calls ``policy.reset`` / ``detach_hidden_states``
  / ``get_hidden_states`` around each epoch and each ``gradient_length`` step —
  those ops are no-ops on the non-recurrent parent and become real ops here.

Proprio ``history_length`` is expected to be 1 for this variant (the LSTM
supplies temporal context). Keeping history=5 on proprio also works but is
redundant and doubles input dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, Memory

from .student_teacher_vision import StudentTeacherVision


class StudentTeacherVisionRecurrent(StudentTeacherVision):
    """Vision student + LSTM + JIT teacher.

    Inherits CNN encoder, proprio normalization, JIT teacher, and optional aux
    head from ``StudentTeacherVision``. Rewires the action head: LSTM over
    fused features, then MLP to actions.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        *args,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 1,
        **kwargs,
    ) -> None:
        # Parent builds ``self.student = MLP(fused_dim, num_actions, hidden_dims)``.
        # We replace that with LSTM(fused_dim, rnn_hidden_dim) → MLP(rnn_hidden_dim, num_actions).
        super().__init__(*args, **kwargs)

        # Infer fused_dim from parent's MLP input layer, then swap the MLP.
        first_linear = next(m for m in self.student.modules() if isinstance(m, nn.Linear))
        fused_dim = first_linear.in_features
        # The original MLP's hidden dims: strip input + output dims from the layer list.
        hidden_dims = []
        for m in self.student.modules():
            if isinstance(m, nn.Linear):
                hidden_dims.append(m.out_features)
        hidden_dims = hidden_dims[:-1]  # drop action-head output dim
        # Use the parent's activation string (stored on the MLP's activation modules).
        activation = "elu"
        for m in self.student.modules():
            if isinstance(m, nn.ELU):
                activation = "elu"; break
            if isinstance(m, nn.ReLU):
                activation = "relu"; break
            if isinstance(m, nn.Tanh):
                activation = "tanh"; break

        self.memory_s = Memory(fused_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self.student = MLP(rnn_hidden_dim, self.num_actions, hidden_dims, activation)
        print(f"Recurrent student: LSTM({fused_dim} -> {rnn_hidden_dim}) -> MLP({rnn_hidden_dim} -> {self.num_actions})")
        print(f"Memory: {self.memory_s}")

        # Rebuild the optional std head on top of rnn features (not raw fused) so
        # predicted std is state-dependent through the LSTM context.
        if self.predict_std:
            init_bias = float(self.std_head.bias[0].detach().cpu())
            self.std_head = nn.Linear(rnn_hidden_dim, self.num_actions)
            nn.init.zeros_(self.std_head.weight)
            nn.init.constant_(self.std_head.bias, init_bias)

    # ------------------------------------------------------------------ forward

    def _encode_and_memory(self, obs: TensorDict) -> torch.Tensor:
        """Encode obs through CNN + proprio concat, then LSTM. Returns (B, rnn_hidden_dim)."""
        fused = self._encode_student(obs)  # (B, fused_dim)
        # Memory.__call__ expects (B, D); internally unsqueezes to (1, B, D).
        out_mem = self.memory_s(fused).squeeze(0)  # (B, rnn_hidden_dim)
        return out_mem

    def _update_distribution(self, feat: torch.Tensor) -> None:
        # ``feat`` is the LSTM output here, not the raw fused vector.
        mean = self.student(feat)
        if self.predict_std:
            std = torch.exp(self.std_head(feat))
        elif self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        feat = self._encode_and_memory(obs)
        self._update_distribution(feat)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        feat = self._encode_and_memory(obs)
        return self.student(feat)

    def act_inference_with_std(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.predict_std:
            raise RuntimeError("act_inference_with_std called but predict_std=False")
        feat = self._encode_and_memory(obs)
        mean = self.student(feat)
        std = torch.exp(self.std_head(feat))
        return mean, std

    def forward_with_aux(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Shared CNN forward between action head (via LSTM) and aux head (no LSTM).

        Aux head always operates on raw image features — temporal context is
        irrelevant for per-step pose regression. Must not re-run the CNN, so we
        compute image features once and reuse for both paths.
        """
        proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        proprio = self.student_obs_normalizer(proprio)
        img_feats = torch.cat([self.depth_encoder(obs[g]) for g in self.vision_groups], dim=-1)
        fused = torch.cat([proprio, img_feats], dim=-1)
        out_mem = self.memory_s(fused).squeeze(0)
        action = self.student(out_mem)
        aux_pred = self.aux_head(img_feats) if self.aux_enabled else None
        return action, aux_pred

    # --------------------------------------------------------- hidden state API

    def reset(self, dones: torch.Tensor | None = None, hidden_states=(None, None)) -> None:
        self.memory_s.reset(dones, hidden_states[0])

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        self.memory_s.detach_hidden_state(dones)

    def get_hidden_states(self):
        return self.memory_s.hidden_state, None
