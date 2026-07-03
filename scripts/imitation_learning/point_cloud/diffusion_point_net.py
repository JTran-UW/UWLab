# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion action head on a PointNet set encoder (:class:`DiffusionActionPointNet`).

A small DDPM/DDIM policy for single-step action prediction: the PointNet set encoder (residual)
+ proprio encoder produce a conditioning vector, and an MLP denoiser predicts the v-target
(Salimans & Ho, 2022) for the (z-scored) action. The schedule is rescaled to ZERO terminal SNR
(Lin et al., WACV 2024) so the pure-noise sampling prior matches training -- v-prediction keeps this
well-posed at SNR=0. Inference runs a few DDIM steps. Targets the action-multimodality ceiling that
unimodal MSE BC can't pass -- a diffusion head can represent a multimodal expert action distribution
instead of averaging its modes.

Interface mirrors :class:`PointNet` enough for the trainer/deploy to treat it uniformly:
``is_diffusion=True`` flags the diffusion path; ``loss(points, proprio, action_n)`` returns the
training loss and ``sample(points, proprio)`` returns a (z-scored) action. Both ``points`` and
``proprio``/``action_n`` are already normalized by the caller (PointNetBC / bc_utils), exactly as
for the MSE policies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from residual_point_net import ResidualMLP


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal embedding of integer timesteps ``t`` (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:  # odd dim -> pad one column
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _betas_with_zero_terminal_snr(num_timesteps: int) -> torch.Tensor:
    """Linear-beta schedule rescaled so terminal SNR is exactly zero (Lin et al., WACV 2024).

    The standard ``linspace(1e-4, 2e-2, T)`` betas were tuned for T=1000; at T=100 they leave
    ~60% signal in the "fully noised" state (sqrt(alpha_bar_T) ~ 0.6). But ``sample()`` starts from
    PURE Gaussian noise (assumes sqrt(alpha_bar_T)=0), so the denoiser is queried out of
    distribution on its first reverse step and pulls samples toward the data mean -- defeating the
    point of a multimodal head. Here we rescale sqrt(alpha_bar) so the first value is preserved and
    the last is exactly 0, then back-solve betas. Pairs with v-prediction (well-behaved at SNR=0)."""
    betas = torch.linspace(1e-4, 2e-2, num_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_acp = alphas_cumprod.sqrt()
    sqrt_acp_0, sqrt_acp_T = sqrt_acp[0].clone(), sqrt_acp[-1].clone()
    sqrt_acp -= sqrt_acp_T                                   # last -> 0
    sqrt_acp *= sqrt_acp_0 / (sqrt_acp_0 - sqrt_acp_T)       # first -> preserved
    acp = sqrt_acp**2
    alphas = torch.cat([acp[:1], acp[1:] / acp[:-1]])
    return 1.0 - alphas


class DiffusionActionPointNet(nn.Module):
    """PointNet-conditioned DDPM action head (v-prediction, zero terminal SNR; samples with DDIM).

    The conditioning stack reproduces :class:`PointNet`'s encoder exactly (residual set encoder +
    max-pool + LayerNorm, residual proprio encoder), so the only architectural change vs the MSE
    policy is the action head: instead of regressing the action, an MLP denoiser maps
    ``(noisy_action, timestep_embed, condition) -> v`` (the velocity target).
    """

    is_diffusion = True

    def __init__(
        self,
        encoder_hidden_dims: list[int],
        action_hidden_dims: list[int],
        proprio_dim: int,
        action_dim: int,
        predict_std: bool,  # ignored (diffusion has no std head); kept for a uniform signature
        point_dim: int = 3,
        num_train_timesteps: int = 100,
        num_sample_steps: int = 10,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.predict_std = False
        self.point_dim = point_dim
        self.action_dim = action_dim
        self.pooled_dim = encoder_hidden_dims[-1]
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_sample_steps = int(num_sample_steps)
        self.time_embed_dim = int(time_embed_dim)

        # --- conditioning encoder (mirrors PointNet / ResidualPointNet) ---
        self.set_encoder = ResidualMLP(in_channels=point_dim, hidden_channels=encoder_hidden_dims)
        self.point_norm = nn.LayerNorm(self.pooled_dim)
        self.proprio_encoder = nn.Sequential(nn.Linear(proprio_dim, self.pooled_dim), nn.LayerNorm(self.pooled_dim))
        cond_dim = self.pooled_dim * 2  # [pooled PC feature, proprio feature]

        # --- denoiser: (noisy action, time embed, condition) -> predicted noise ---
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim), nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.denoiser = ResidualMLP(
            in_channels=action_dim + self.time_embed_dim + cond_dim,
            hidden_channels=[*action_hidden_dims, action_dim],
        )

        # --- DDPM schedule (linear betas rescaled to ZERO terminal SNR) ---
        # Zero terminal SNR makes the most-noised state pure noise (sqrt(alpha_bar_T)=0), matching the
        # pure-Gaussian prior sample() draws from -> no train/sample mismatch, no mean bias. Requires
        # v-prediction (eps-pred is singular at SNR=0). Only alphas_cumprod is needed downstream.
        alphas_cumprod = torch.cumprod(1.0 - _betas_with_zero_terminal_snr(self.num_train_timesteps), dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _condition(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        feats = self.set_encoder(points)
        pooled = self.point_norm(feats.amax(dim=1))
        proprio_feats = self.proprio_encoder(proprio)
        return torch.cat([pooled, proprio_feats], dim=-1)

    def _denoise(self, noisy_action: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Network output, interpreted as the v-prediction target (Salimans & Ho, 2022)."""
        t_emb = self.time_mlp(_sinusoidal_embedding(t, self.time_embed_dim))
        return self.denoiser(torch.cat([noisy_action, t_emb, cond], dim=-1))

    def loss(self, points: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """v-prediction MSE. ``action`` is the z-scored target action (B, action_dim). v-pred (not
        eps) so it stays well-posed at the zero-SNR terminal step (where v = -x0)."""
        cond = self._condition(points, proprio)
        B = action.shape[0]
        t = torch.randint(0, self.num_train_timesteps, (B,), device=action.device)
        acp = self.alphas_cumprod[t].unsqueeze(-1)  # (B, 1)
        noise = torch.randn_like(action)
        noisy = acp.sqrt() * action + (1.0 - acp).sqrt() * noise
        v_target = acp.sqrt() * noise - (1.0 - acp).sqrt() * action   # v = sqrt(acp)*eps - sqrt(1-acp)*x0
        v_pred = self._denoise(noisy, t, cond)
        return torch.nn.functional.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """DDIM sampling (deterministic, eta=0) over ``num_sample_steps`` -> z-scored action.

        v-prediction: from v, recover x0 = sqrt(acp)*x - sqrt(1-acp)*v and eps = sqrt(1-acp)*x +
        sqrt(acp)*v (no division by sqrt(acp), so the zero-SNR first step is well-defined)."""
        cond = self._condition(points, proprio)
        B = cond.shape[0]
        x = torch.randn(B, self.action_dim, device=cond.device)            # pure noise == zero-SNR prior
        steps = torch.linspace(self.num_train_timesteps - 1, 0, self.num_sample_steps, device=cond.device).long()
        for i in range(self.num_sample_steps):
            t = steps[i]
            acp_t = self.alphas_cumprod[t]
            t_batch = torch.full((B,), int(t), device=cond.device, dtype=torch.long)
            v = self._denoise(x, t_batch, cond)
            x0 = acp_t.sqrt() * x - (1.0 - acp_t).sqrt() * v              # predicted clean action
            if i == self.num_sample_steps - 1:
                x = x0
            else:
                eps = (1.0 - acp_t).sqrt() * x + acp_t.sqrt() * v
                acp_next = self.alphas_cumprod[steps[i + 1]]
                x = acp_next.sqrt() * x0 + (1.0 - acp_next).sqrt() * eps  # DDIM eta=0 step
        return x

    def forward(self, x: torch.Tensor, proprio: torch.Tensor, return_pooled: bool = False):
        """Generic call -> a sampled action (so non-diffusion-aware callers still work).

        The trainer/deploy use ``loss``/``sample`` directly via the ``is_diffusion`` flag; this is a
        convenience fallback. ``return_pooled`` exposes the pooled feature for parity with PointNet."""
        action = self.sample(x, proprio)
        if return_pooled:
            with torch.no_grad():
                feats = self.set_encoder(x)
                pooled = self.point_norm(feats.amax(dim=1))
            return action, pooled
        return action
