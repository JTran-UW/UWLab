# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object symmetry declarations and orientation features that are identical for every identical-looking pose.

.. code:: yaml

   symmetry:
     axis: [0.0, 0.0, 1.0]   # spin axis in the object frame, through the origin
     order: 4                # identical-looking positions per full turn
     flip: true              # also identical after a half turn about the object's x axis (y if axis is x)

Pure torch (no simulator imports). Quaternions are xyzw (Isaac Lab 3).
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils

# Unique index multisets of a symmetric rank-4 tensor in 3-D (15 entries): xxxx xxxy ... zzzz
TENSOR4_INDEX = [(i, j, k, l) for i in range(3) for j in range(i, 3) for k in range(j, 3) for l in range(k, 3)]
_I4 = torch.tensor(TENSOR4_INDEX)


def read_symmetry(metadata: dict) -> dict | None:
    """The validated ``symmetry`` block of an asset's metadata, or None if the object declares none."""
    sym = metadata.get("symmetry")
    if sym is None:
        return None
    axis = [float(a) for a in sym.get("axis", (0.0, 0.0, 1.0))]
    norm = math.sqrt(sum(a * a for a in axis))
    if len(axis) != 3 or norm == 0.0:
        raise ValueError(f"symmetry.axis must be a non-zero 3-vector, got {sym.get('axis')}")
    order = int(sym.get("order", 1))
    if order < 1:
        raise NotImplementedError("symmetry.order must be >= 1 (continuous symmetry, order 0, is not supported yet)")
    return {"axis": [a / norm for a in axis], "order": order, "flip": bool(sym.get("flip", False))}


def tensor4_features(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[N, 15]: unique components of ``u^{(x)4} + v^{(x)4}`` for unit vectors ``u``, ``v`` [N, 3].

    With ``u``/``v`` the two face normals of a 4-fold object this is invariant to 90-degree spins (which swap them)
    and flips (which negate them), and is a smooth function of the pose.
    """
    idx = _I4.to(u.device)
    return u[:, idx].prod(dim=-1) + v[:, idx].prod(dim=-1)


def orientation_features(quat: torch.Tensor, sym: dict | None) -> torch.Tensor:
    """Continuous features of an orientation that are identical for every identical-looking version of it.

    ``quat`` [N, 4] is the object's orientation in some reference frame. With ``u``, ``v`` the object's axes
    perpendicular to its symmetry axis ``a`` (rotated into that frame):

    * no symmetry: the first two columns of the rotation matrix [6];
    * ``order: 4, flip: true``: ``u^4 + v^4`` [15];
    * ``order: 4, flip: false``: the same plus the axis ``a`` [18], since the object is not the same upside down.
    """

    def axis(local: torch.Tensor) -> torch.Tensor:
        return math_utils.quat_apply(quat, local.to(quat).expand(quat.shape[0], 3))

    if sym is None:
        return torch.cat([axis(torch.tensor([1.0, 0.0, 0.0])), axis(torch.tensor([0.0, 1.0, 0.0]))], dim=-1)
    if sym["order"] != 4:
        raise NotImplementedError(f"orientation features for symmetry order {sym['order']} are not implemented yet")
    a = torch.tensor(sym["axis"], dtype=torch.float32)
    u = torch.tensor([1.0, 0.0, 0.0]) if abs(sym["axis"][0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    u = u - torch.dot(u, a) * a
    u = u / u.norm()
    v = torch.linalg.cross(a, u)
    t4 = tensor4_features(axis(u), axis(v))
    return t4 if sym["flip"] else torch.cat([t4, axis(a)], dim=-1)
