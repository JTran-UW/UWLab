# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RMA actor-critic.

Extends ``ActorCriticWithEncoder`` with a small bidirectional transformer that
maps a window of past (proprio + last_action) onto the privileged latent
``z = phi(privileged_rma)``. During PPO, the actor conditions on ``z`` via the
existing per-group MLP encoder pipeline; the transformer is trained against
``sg(z)`` with MSE inside the runner's auxiliary loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from .actor_critic_encoder import ActorCriticWithEncoder


class _BidirectionalTransformerEncoder(nn.Module):
    """Tiny bidirectional (no causal mask) transformer that pools a (B, T, D) window to (B, latent_dim)."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        latent_dim: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, latent_dim)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist: (B, T, D_in)
        x = self.input_proj(hist) + self.pos_embed[:, : hist.shape[1]]
        x = self.encoder(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


class ActorCriticRMA(ActorCriticWithEncoder):
    """ActorCriticWithEncoder + RMA history encoder.

    The privileged group is encoded by the existing per-group MLP (phi) and
    concatenated into the actor/critic inputs. The history encoder (psi) is
    used only by the PPO auxiliary loss to predict the privileged latent from
    a (B, T, D_hist) window.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict,
        num_actions: int,
        privileged_group: str = "privileged_rma",
        latent_dim: int = 16,
        history_length: int = 32,
        history_obs_keys: tuple[str, ...] = ("proprio",),
        history_include_actions: bool = True,
        transformer_d_model: int = 64,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 4,
        transformer_ff: int = 128,
        encoder_groups: dict | None = None,
        **kwargs,
    ) -> None:
        # Phi (privileged -> latent) is the per-group MLP for ``privileged_group``;
        # inject a default encoder cfg for it if the caller didn't provide one.
        encoder_groups = dict(encoder_groups or {})
        if privileged_group not in encoder_groups:
            encoder_groups[privileged_group] = {"hidden_dims": [128, 64], "output_dim": latent_dim}
        else:
            encoder_groups[privileged_group]["output_dim"] = latent_dim
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            encoder_groups=encoder_groups,
            **kwargs,
        )

        # Privileged group must exist in obs and in the policy group.
        if privileged_group not in obs.keys():
            raise ValueError(
                f"ActorCriticRMA: obs is missing required group '{privileged_group}'."
            )
        if privileged_group not in obs_groups.get("policy", []):
            raise ValueError(
                f"ActorCriticRMA: '{privileged_group}' must appear in obs_groups['policy']."
            )

        # Sanity-check history obs keys live in obs.
        for k in history_obs_keys:
            if k not in obs.keys():
                raise ValueError(f"ActorCriticRMA: history_obs_keys references missing obs '{k}'.")

        self._privileged_group = privileged_group
        self._latent_dim = int(latent_dim)
        self._history_length = int(history_length)
        self._history_obs_keys = tuple(history_obs_keys)
        self._history_include_actions = bool(history_include_actions)

        # Compute history input dim from obs shapes + action dim.
        hist_dim = int(sum(obs[k].shape[-1] for k in self._history_obs_keys))
        if self._history_include_actions:
            hist_dim += int(num_actions)
        self._history_input_dim = hist_dim

        self.history_encoder = _BidirectionalTransformerEncoder(
            input_dim=hist_dim,
            seq_len=self._history_length,
            latent_dim=self._latent_dim,
            d_model=transformer_d_model,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            dim_feedforward=transformer_ff,
        )

        print(
            f"ActorCriticRMA: privileged_group='{privileged_group}' latent_dim={latent_dim}"
            f"  history_length={history_length} history_input_dim={hist_dim}"
            f"  transformer(layers={transformer_num_layers}, heads={transformer_num_heads},"
            f" d_model={transformer_d_model}, ff={transformer_ff})"
        )

    @property
    def privileged_group(self) -> str:
        return self._privileged_group

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def history_input_dim(self) -> int:
        return self._history_input_dim

    @property
    def history_obs_keys(self) -> tuple[str, ...]:
        return self._history_obs_keys

    @property
    def history_include_actions(self) -> bool:
        return self._history_include_actions

    def encode_privileged(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        """Apply phi (actor-side privileged encoder, with normalization) to raw privileged_rma obs."""
        x = privileged_obs
        if self._privileged_group in self._group_normalizers:
            x = self._group_normalizers[self._privileged_group](x)
        return self.actor_encoders[self._privileged_group](x)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """Apply psi (bidirectional transformer) to a (B, T, D_hist) history."""
        return self.history_encoder(history)
