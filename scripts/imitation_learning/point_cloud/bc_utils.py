# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-time helpers for a behavior-cloned :class:`PointNet`.

Reconstructs the raw :class:`PointNet` (+ the proprio/action z-scoring stats) from a
Lightning checkpoint written by ``train_point_net.py``, and runs it on a live
observation group -- so rolling out a BC PointNet (e.g. in ``play.py``) needs no
Lightning dependency, only ``torch`` + ``point_net.py``.
"""

from __future__ import annotations

import os
import sys

import torch

# PointNet lives next to this file; make it importable regardless of caller cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from point_net import PointNet  # noqa: E402

# Obs terms that are point clouds (stored flat as (N, num_points*3)); everything else -> proprio.
# Mirrors train_point_net.py so eval proprio is built exactly as the model was trained.
_BC_PC_TERMS = {"scene_pc"}


def load_bc_pointnet(path: str, device: str) -> dict:
    """Reconstruct a PointNet (+ saved normalization stats) from a Lightning BC checkpoint.

    The checkpoint is the plain ``torch.save`` dict Lightning writes: ``hyper_parameters``
    holds the PointNet shape, ``state_dict`` holds ``model.*`` weights plus the
    proprio/action z-scoring buffers. We rebuild the raw :class:`PointNet` so eval needs
    no Lightning dependency.

    Returns a dict with ``model`` (eval-mode PointNet on ``device``), ``hp`` (the saved
    hyper-parameters), and the four normalization tensors (``proprio_mean/std``,
    ``action_mean/std``).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt["hyper_parameters"]
    sd = ckpt["state_dict"]
    model = PointNet(
        encoder_hidden_dims=hp["encoder_dims"],
        action_hidden_dims=hp["action_dims"],
        proprio_dim=hp["proprio_dim"],
        action_dim=hp["action_dim"],
        predict_std=hp["predict_std"],
    )
    model.load_state_dict({k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")})
    model.to(device).eval()
    norm = {k: sd[k].to(device) for k in ("proprio_mean", "proprio_std", "action_mean", "action_std")}
    return {"model": model, "hp": hp, **norm}


def bc_actions(bc: dict, obs, group_name: str) -> torch.Tensor:
    """Run the BC PointNet on the ``group_name`` obs group and return denormalized actions.

    The group is a per-term dict (``concatenate_terms=False``): the point cloud term is
    reshaped to ``(N, num_points, 3)`` and the remaining terms are concatenated -- in
    declaration order -- into proprio, exactly as ``train_point_net.py`` did. Proprio is
    z-scored with the saved stats; the predicted (normalized) action is denormalized back
    to env action units.
    """
    g = obs[group_name]
    keys = list(g.keys())
    pc_key = next(k for k in keys if k in _BC_PC_TERMS)
    pc = g[pc_key].reshape(g[pc_key].shape[0], -1, 3)
    proprio = torch.cat([g[k] for k in keys if k not in _BC_PC_TERMS], dim=-1)
    proprio = (proprio - bc["proprio_mean"]) / bc["proprio_std"]
    out = bc["model"](pc, proprio)
    mean = out[0] if isinstance(out, tuple) else out  # predict_std -> (mean, log_std)
    return mean * bc["action_std"] + bc["action_mean"]
