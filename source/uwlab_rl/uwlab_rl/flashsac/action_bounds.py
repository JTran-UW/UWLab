# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-dimension scalar action bounds for ``FlashSACVecEnvWrapper`` (upstream takes one scalar)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from uwlab_rl.env_guards import apply_newton_env_guards


def make_flashsac_env(env, wrapper_cls, clip_actions, action_bound: str, action_bound_scale: float | Sequence[float]):
    """Wrap ``env``; a sequence ``action_bound_scale`` sets one ``action_bound="scalar"`` half-width per action dim.

    The policy acts in tanh space [-1, 1] and the wrapper applies ``clamp(a, -1, 1) * scale`` at the env boundary;
    FlashSAC copies the wrapper's ``action_scale`` into the actor when the algorithm is built, so set it here.
    """
    per_dim = isinstance(action_bound_scale, Sequence) and not isinstance(action_bound_scale, str)
    wrapped = wrapper_cls(
        env, clip_actions=clip_actions, action_bound=action_bound,
        action_bound_scale=1.0 if per_dim else action_bound_scale,
    )
    if per_dim:
        assert action_bound == "scalar", "per-dimension action_bound_scale needs action_bound='scalar'"
        scale = torch.tensor([float(v) for v in action_bound_scale], device=wrapped.device)
        assert scale.numel() == wrapped.num_actions, (scale.numel(), wrapped.num_actions)
        wrapped.action_scale = scale
        wrapped.action_bias = torch.zeros_like(scale)
        wrapped.unwrapped._action_scale = scale
        print(f"[flashsac] per-dimension action bounds: {scale.tolist()}", flush=True)
    apply_newton_env_guards(wrapped)
    return wrapped

