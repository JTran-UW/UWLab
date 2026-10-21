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


def _axes(sym: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The symmetry axis ``a`` and the flip axis ``u`` perpendicular to it (object x, or y if ``a`` is x)."""
    a = torch.tensor(sym["axis"], dtype=torch.float32)
    u = torch.tensor([1.0, 0.0, 0.0]) if abs(sym["axis"][0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    u = u - torch.dot(u, a) * a
    return a, u / u.norm()


def symmetry_group(sym: dict | None) -> torch.Tensor:
    """[K, 4] rotations (object frame) that map the object onto itself, identity first: the spins, then each flipped."""
    if sym is None:
        return torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    a, u = _axes(sym)
    n = sym["order"]
    angles = torch.arange(n, dtype=torch.float32) * (2.0 * math.pi / n)
    group = math_utils.quat_from_angle_axis(angles, a.expand(n, 3))
    if sym["flip"]:
        flip = math_utils.quat_from_angle_axis(torch.tensor([math.pi]), u[None]).expand(n, 4)
        group = torch.cat([group, math_utils.quat_mul(group, flip)])
    return group


def symmetric_alignment_error(
    rel_pos: torch.Tensor,
    rel_quat: torch.Tensor,
    receptive_offset: tuple[torch.Tensor, torch.Tensor],
    insertive_group: torch.Tensor,
    receptive_group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position error [N] and |roll| + |pitch| [N] of the best-aligned symmetric version of an assembly.

    ``rel_pos``/``rel_quat`` [N, 3]/[N, 4]: the insertive alignment frame in the receptive alignment frame.
    ``receptive_offset``: the receptive alignment offset (pos [3], quat [4]) in its object frame, about which
    ``receptive_group`` acts (``insertive_group`` acts in the insertive alignment frame). Yaw is ignored, as in the
    single-offset check, and the position error is taken at the version with the smallest tilt.
    """
    off_p, off_q = (t.to(rel_pos) for t in receptive_offset)
    h_inv = math_utils.quat_inv(receptive_group.to(rel_pos))
    k_h, k_g, n = h_inv.shape[0], insertive_group.shape[0], rel_pos.shape[0]
    # receptive alignment frame of version h, expressed in the nominal one: off^-1 * h^-1 * off
    off_inv_q = math_utils.quat_inv(off_q)[None].expand(k_h, 4)
    t_q = math_utils.quat_mul(math_utils.quat_mul(off_inv_q, h_inv), off_q[None].expand(k_h, 4))
    t_p = math_utils.quat_apply(off_inv_q, math_utils.quat_apply(h_inv, off_p[None].expand(k_h, 3)) - off_p)
    pos = t_p[None] + math_utils.quat_apply(t_q[None].expand(n, k_h, 4).reshape(-1, 4), rel_pos.repeat_interleave(k_h, 0)).view(n, k_h, 3)
    quat = math_utils.quat_mul(t_q[None].expand(n, k_h, 4).reshape(-1, 4), rel_quat.repeat_interleave(k_h, 0))
    quat = math_utils.quat_mul(
        quat.repeat_interleave(k_g, 0), insertive_group.to(rel_pos).repeat(n * k_h, 1)
    )
    e_x, e_y, _ = math_utils.euler_xyz_from_quat(quat)
    tilt = (math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()).view(n, k_h * k_g)
    tilt, best = tilt.min(dim=1)
    xyz = pos.norm(dim=-1).gather(1, (best // k_g)[:, None]).squeeze(1)
    return xyz, tilt


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
    a, u = _axes(sym)
    v = torch.linalg.cross(a, u)
    t4 = tensor4_features(axis(u), axis(v))
    return t4 if sym["flip"] else torch.cat([t4, axis(a)], dim=-1)
