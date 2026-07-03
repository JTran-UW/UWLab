# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""History-conditioned causal Transformer for point-cloud BC (:class:`HistoryPointNet`).

Where the feed-forward PointNet policies map a SINGLE (point-cloud, proprio) to an action, this
policy conditions on a short *history* of the trajectory. Every timestep contributes TWO tokens:

  * a **state** token  -- the pooled PointNet PC feature fused with the proprio feature, and
  * an **action** token -- the action that was executed at that state.

The tokens are interleaved in time and run through a causal Transformer encoder::

    [ s_0, a_0, s_1, a_1, ..., s_{H-2}, a_{H-2}, s_{H-1} ]      (2H-1 tokens)

The action for a timestep is read out from that timestep's **state** token: under the causal mask
a state token has attended to every earlier state and action but NOT its own action, so its hidden
state is exactly the "predict the next action given the history so far" representation. At deploy
the policy reads the LAST state token -> the action for the current step (this is the "prediction
from the last hidden state of the most recent history" the design calls for).

Training predicts an action at *every* state token (dense BC supervision over the whole window,
each position being the last-hidden-state of its own prefix); the loss is masked to the valid
(non-padded) positions so episode-start left-padding never contributes.

Interface parity with :class:`PointNet` for the trainer/deploy: ``is_sequence = True`` flags the
sequence path (mirroring diffusion's ``is_diffusion``). The trainer calls ``forward`` with batched
histories ``points (B,H,N,point_dim)``, ``proprio (B,H,proprio_dim)``, ``actions (B,H,action_dim)``
(all already z-scored by the caller) and a ``valid (B,H)`` mask; deploy uses ``predict`` for the
single last-step action. ``pooled_dim`` is kept only so the aux-probe plumbing has a value to read
(the aux probe is disabled for the sequence policy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP


class HistoryPointNet(nn.Module):
    """Causal Transformer over interleaved (state, action) tokens of a trajectory history."""

    is_sequence = True

    def __init__(
        self,
        encoder_hidden_dims: list[int],
        action_hidden_dims: list[int],
        proprio_dim: int,
        action_dim: int,
        predict_std: bool,
        point_dim: int = 3,
        history_len: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        transformer_dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.predict_std = predict_std
        self.point_dim = point_dim  # 3 = xyz; 4 = xyz + per-point segmentation label
        self.action_dim = action_dim
        self.history_len = int(history_len)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        # Width of the pooled global PC feature. Kept for interface parity with PointNet (the aux
        # probe reads model.pooled_dim); the sequence policy disables the aux probe.
        self.pooled_dim = encoder_hidden_dims[-1]

        # --- per-timestep encoders (weights SHARED across every step of the history) ---
        # PointNet set encoder + max-pool, mirroring PointNet exactly so a state token carries the
        # same pooled PC feature the feed-forward policies use.
        self.set_encoder = MLP(in_channels=point_dim, hidden_channels=encoder_hidden_dims, norm_layer=nn.LayerNorm)
        self.point_norm = nn.LayerNorm(self.pooled_dim)
        self.proprio_encoder = nn.Sequential(nn.Linear(proprio_dim, self.pooled_dim), nn.LayerNorm(self.pooled_dim))
        # State token: fuse [pooled PC feature, proprio feature] -> d_model.
        self.state_proj = nn.Linear(self.pooled_dim * 2, self.d_model)
        # Action token: embed the executed action -> d_model.
        self.action_embed = nn.Linear(action_dim, self.d_model)

        # Token embeddings: a learned position over the interleaved 2H-1 token slots plus a learned
        # modality (state vs action) embedding. Position already distinguishes even/odd slots, but the
        # explicit modality embedding makes the token type unambiguous and is cheap.
        self.max_tokens = 2 * self.history_len - 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, self.d_model))
        self.modality_embed = nn.Parameter(torch.zeros(1, 2, self.d_model))  # [state, action]
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * ff_mult,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable to train than post-LN
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(self.d_model)

        out_dim = action_dim * 2 if predict_std else action_dim
        self.action_head = MLP(in_channels=self.d_model, hidden_channels=[*action_hidden_dims, out_dim])

    # ------------------------------------------------------------------ helpers
    def _encode_states(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """(B,H,N,point_dim), (B,H,proprio_dim) -> state tokens (B,H,d_model)."""
        B, H = points.shape[:2]
        N, C = points.shape[2], points.shape[3]
        feats = self.set_encoder(points.reshape(B * H, N, C))       # (B*H, N, pooled)
        pooled = self.point_norm(feats.amax(dim=1))                 # (B*H, pooled) global PC feature
        pfeat = self.proprio_encoder(proprio.reshape(B * H, -1))    # (B*H, pooled)
        tok = self.state_proj(torch.cat([pooled, pfeat], dim=-1))   # (B*H, d_model)
        return tok.reshape(B, H, self.d_model)

    def _build_sequence(self, state_tok: torch.Tensor, action_tok: torch.Tensor) -> torch.Tensor:
        """Interleave into [s_0, a_0, ..., s_{H-1}] (2H-1 tokens) + modality/pos embeddings."""
        B, H, D = state_tok.shape
        L = 2 * H - 1
        seq = state_tok.new_zeros(B, L, D)
        seq[:, 0::2] = state_tok                    # even slots: the H state tokens
        if H > 1:
            seq[:, 1::2] = action_tok[:, : H - 1]   # odd slots: the first H-1 action tokens (drop a_{H-1})
        mod = state_tok.new_zeros(1, L, D)
        mod[:, 0::2] = self.modality_embed[:, 0:1]
        mod[:, 1::2] = self.modality_embed[:, 1:2]
        return seq + mod + self.pos_embed[:, :L]

    def _token_valid(self, valid: torch.Tensor) -> torch.Tensor:
        """Per-timestep validity (B,H) -> per-token validity (B, 2H-1). Action a_i is valid iff its
        state s_i is valid (both come from the same real timestep)."""
        B, H = valid.shape
        L = 2 * H - 1
        tv = valid.new_zeros(B, L, dtype=torch.bool)
        tv[:, 0::2] = valid
        if H > 1:
            tv[:, 1::2] = valid[:, : H - 1]
        return tv

    def _attn_mask(self, valid: torch.Tensor, device) -> torch.Tensor:
        """Combined causal + padding attention mask (B*n_heads, L, L); ``True`` = blocked.

        A query may attend to a key only if the key is causally in the past (or itself) AND the key
        is a valid (non-pad) token. The self/diagonal entry is always left OPEN so a query never has
        an all-masked row (which would make the softmax NaN) -- pad queries thus produce garbage that
        the loss mask discards, while valid queries never see a pad key."""
        B, H = valid.shape
        L = 2 * H - 1
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)  # True above diag
        tok_valid = self._token_valid(valid)                                                  # (B, L)
        block = causal.unsqueeze(0) | (~tok_valid).unsqueeze(1)                                # (B, L, L)
        eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
        block = block & ~eye                                                                   # keep diagonal open
        return block.repeat_interleave(self.n_heads, dim=0)                                    # (B*nheads, L, L)

    # ------------------------------------------------------------------ forward
    def forward(self, points: torch.Tensor, proprio: torch.Tensor, actions: torch.Tensor,
                valid: torch.Tensor | None = None):
        """Predict an action at EVERY state token of the window.

        ``points`` (B,H,N,point_dim), ``proprio`` (B,H,proprio_dim), ``actions`` (B,H,action_dim)
        are already z-scored by the caller; ``valid`` (B,H) marks real vs left-padded timesteps
        (default: all real). Returns ``(B,H,action_dim)`` -- or ``(mean, log_std)`` each that shape
        with ``predict_std``. Only ``actions[:, :H-1]`` are consumed as input tokens (the last
        action is what a full-window rollout would predict, never fed in)."""
        B, H = points.shape[:2]
        if valid is None:
            valid = torch.ones(B, H, dtype=torch.bool, device=points.device)
        state_tok = self._encode_states(points, proprio)     # (B,H,D)
        action_tok = self.action_embed(actions)              # (B,H,D)
        seq = self._build_sequence(state_tok, action_tok)    # (B,L,D)
        mask = self._attn_mask(valid, points.device)         # (B*nheads,L,L)
        h = self.ln_f(self.transformer(seq, mask=mask))      # (B,L,D)
        state_h = h[:, 0::2]                                 # (B,H,D) state-token outputs
        out = self.action_head(state_h)                      # (B,H,out_dim)
        if self.predict_std:
            return out.chunk(2, dim=-1)                      # (mean, log_std)
        return out

    @torch.no_grad()
    def predict(self, points: torch.Tensor, proprio: torch.Tensor, actions: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        """Deploy read-out: the action at the LAST (most-recent) state token, ``(B, action_dim)``."""
        out = self.forward(points, proprio, actions, valid)
        mean = out[0] if self.predict_std else out
        return mean[:, -1]

    def calculate_loss(self, points, proprio, actions, targets, valid=None):
        """Masked BC loss over the window (MSE, or Gaussian NLL with ``predict_std``). ``targets``
        is the per-timestep z-scored action (B,H,action_dim); ``valid`` masks padded positions."""
        B, H = points.shape[:2]
        if valid is None:
            valid = torch.ones(B, H, dtype=torch.bool, device=points.device)
        m = valid.float()
        denom = m.sum().clamp_min(1.0)
        out = self.forward(points, proprio, actions, valid)
        if self.predict_std:
            mean, log_std = out
            per = F.gaussian_nll_loss(mean, targets, torch.exp(2 * log_std), reduction="none").mean(-1)
        else:
            per = F.mse_loss(out, targets, reduction="none").mean(-1)  # (B,H)
        return (per * m).sum() / denom
