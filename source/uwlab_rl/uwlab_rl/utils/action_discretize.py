# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for discretizing continuous robot actions into a fixed vocabulary.

The scheme targets the 7-DoF UR5e + Robotiq 2F-85 action space:
  dims 0-5  — continuous arm OSC deltas (XYZ + axis-angle)
  dim  6    — binary gripper (open / close)

Arm dims are snapped to uniform bin centers over a symmetric clip range.
The gripper is always sign-thresholded to {-1, +1} regardless of num_bins.

Typical usage in data collection:
    spec = make_discretize_spec(num_bins=20, clip_val=2.0)
    save_discretize_spec(spec, output_dir)
    disc_actions = discretize_actions(env_actions, num_bins=20, clip_val=2.0)

Typical usage at training time:
    spec = load_discretize_spec(dataset_dir)
    # build Categorical(num_bins) head per arm dim + Categorical(2) for gripper
    arm_centers = torch.tensor(spec["arm_bin_centers"])   # (num_bins,)

Typical usage at inference:
    spec = load_discretize_spec(checkpoint_dir)
    arm_centers = torch.tensor(spec["arm_bin_centers"], device=device)
    raw_action = decode_discretized_action(indices, arm_centers)
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


def discretize_actions(actions: torch.Tensor, num_bins: int, clip_val: float) -> torch.Tensor:
    """Snap the 6 continuous arm dims to the nearest uniform bin center; threshold gripper at 0.

    Args:
        actions:  Raw policy actions, shape ``(N, 7)``. Dims 0-5 are continuous arm OSC,
                  dim 6 is the binary gripper signal.
        num_bins: Number of uniform bins for dims 0-5.  Must be >= 2.
        clip_val: Symmetric clip range ``[-clip_val, +clip_val]`` for arm dims.

    Returns:
        New tensor of the same shape with arm dims snapped and gripper thresholded.
    """
    result = actions.clone()
    bin_centers = torch.linspace(-clip_val, clip_val, num_bins, device=actions.device, dtype=actions.dtype)
    for d in range(6):
        vals = result[:, d].clamp(-clip_val, clip_val)
        result[:, d] = bin_centers[(vals.unsqueeze(-1) - bin_centers.unsqueeze(0)).abs().argmin(dim=-1)]
    result[:, 6] = torch.where(
        actions[:, 6] >= 0,
        torch.ones_like(actions[:, 6]),
        -torch.ones_like(actions[:, 6]),
    )
    return result


def make_discretize_spec(num_bins: int, clip_val: float, action_dim: int = 7) -> dict:
    """Build a serialisable spec that fully describes the discretization scheme.

    The spec should be saved alongside every dataset and policy checkpoint so
    downstream code can reconstruct the categorical head and decode actions at
    inference without any hardcoded constants.

    Keys:
        num_bins        — number of uniform bins for the 6 continuous arm dims
        clip_val        — symmetric clip range used for binning
        arm_bin_centers — list of ``num_bins`` floats (same grid for dims 0-5)
        gripper_bins    — two values, ``[-1.0, 1.0]`` for the binary gripper (dim 6)
        action_dim      — total action dimensionality (7 for this robot)
        arm_dims        — list of continuous arm dimension indices ``[0, 1, 2, 3, 4, 5]``
        gripper_dim     — index of the binary gripper dimension (6)
    """
    arm_centers = np.linspace(-clip_val, clip_val, num_bins).tolist()
    return {
        "num_bins": num_bins,
        "clip_val": clip_val,
        "arm_bin_centers": arm_centers,
        "gripper_bins": [-1.0, 1.0],
        "action_dim": action_dim,
        "arm_dims": list(range(6)),
        "gripper_dim": 6,
    }


def save_discretize_spec(spec: dict, directory: str, filename: str = "discretize_spec.json") -> str:
    """Write the discretization spec as JSON. Returns the full path written."""
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    return path


def load_discretize_spec(directory: str, filename: str = "discretize_spec.json") -> dict:
    """Load a previously saved discretization spec from ``directory/filename``."""
    path = os.path.join(directory, filename)
    with open(path) as f:
        return json.load(f)


def decode_discretized_action(
    arm_indices: torch.Tensor,
    gripper_indices: torch.Tensor,
    arm_bin_centers: torch.Tensor,
) -> torch.Tensor:
    """Map per-dim bin indices back to raw action values.

    Args:
        arm_indices:     Integer tensor of shape ``(N, 6)`` with bin indices for arm dims.
        gripper_indices: Integer tensor of shape ``(N, 1)`` with 0 (open) or 1 (close).
        arm_bin_centers: Float tensor of shape ``(num_bins,)`` — the bin center values.

    Returns:
        Float action tensor of shape ``(N, 7)``.
    """
    arm_values = arm_bin_centers[arm_indices]  # (N, 6)
    gripper_values = torch.where(
        gripper_indices == 1,
        torch.ones_like(gripper_indices, dtype=arm_bin_centers.dtype),
        -torch.ones_like(gripper_indices, dtype=arm_bin_centers.dtype),
    )  # (N, 1)
    return torch.cat([arm_values, gripper_values], dim=-1)  # (N, 7)
