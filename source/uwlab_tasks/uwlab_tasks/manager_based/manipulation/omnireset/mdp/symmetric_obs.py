# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry-invariant, continuous encodings of a rectangular peg's pose.

Pure torch (no Isaac imports) so the invariance can be unit-tested outside the simulator.

The peg is treated as an unoriented axis segment: its symmetry group is the 4 yaws about its long
axis (90 deg steps) composed with the end-over-end flip, i.e. the ``assembled_offsets`` of the
yandabao peg hole. Every feature below is exactly invariant under all 8 elements and continuous
in the peg pose (no canonical-representative jumps):

* ``segment_features``: segment centre in a frame (3) + the axis as an unoriented line, the outer
  product ``d d^T`` (6 unique entries). Spin leaves ``d`` fixed; the flip maps ``d -> -d`` which
  leaves ``d d^T`` unchanged.
* ``spin_features``: the peg's spin about its axis relative to a reference frame, as
  ``|r| * (cos 4phi, sin 4phi * (d . z_ref))``. ``phi`` is measured from the reference x-axis
  projected onto the plane normal to ``d``; the ``|r|`` weight makes the feature vanish smoothly
  where that projection is degenerate (axis parallel to the reference x), and the ``(d . z_ref)``
  factor makes the sine flip-invariant (the flip reverses ``d`` and mirrors ``phi``).
* ``yaw_mod90_features``: ``(cos 4psi, sin 4psi)`` of a frame's yaw -- for the (static) hole.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

_Z = (0.0, 0.0, 1.0)
_X = (1.0, 0.0, 0.0)


def _axis(quat: torch.Tensor, local: tuple[float, float, float]) -> torch.Tensor:
    v = torch.tensor(local, dtype=quat.dtype, device=quat.device).expand(quat.shape[0], 3)
    return math_utils.quat_apply(quat, v)


def line_encoding(d: torch.Tensor) -> torch.Tensor:
    """Unoriented-line encoding of unit vectors ``d`` [N,3] -> [N,6]: (xx, yy, zz, xy, xz, yz)."""
    x, y, z = d.unbind(-1)
    return torch.stack([x * x, y * y, z * z, x * y, x * z, y * z], dim=-1)


def segment_features(
    peg_pos_w: torch.Tensor, peg_quat_w: torch.Tensor, frame_pos_w: torch.Tensor, frame_quat_w: torch.Tensor
) -> torch.Tensor:
    """[N,9]: peg centre in ``frame`` (3) + line encoding of the peg's local z axis in ``frame`` (6)."""
    rel_pos, rel_quat = math_utils.subtract_frame_transforms(frame_pos_w, frame_quat_w, peg_pos_w, peg_quat_w)
    d = _axis(rel_quat, _Z)
    return torch.cat([rel_pos, line_encoding(d)], dim=-1)


def spin_features(
    peg_pos_w: torch.Tensor, peg_quat_w: torch.Tensor, frame_pos_w: torch.Tensor, frame_quat_w: torch.Tensor
) -> torch.Tensor:
    """[N,2]: peg spin about its axis relative to ``frame``, modulo 90 deg, flip-invariant, continuous."""
    _, rel_quat = math_utils.subtract_frame_transforms(frame_pos_w, frame_quat_w, peg_pos_w, peg_quat_w)
    d = _axis(rel_quat, _Z)  # peg axis in frame
    e = _axis(rel_quat, _X)  # a peg face normal in frame (any of the 4 gives the same 4phi)
    x_ref = torch.tensor(_X, dtype=d.dtype, device=d.device).expand_as(d)
    r_raw = x_ref - (x_ref * d).sum(-1, keepdim=True) * d  # frame x projected onto the plane normal to d
    w = torch.norm(r_raw, dim=-1, keepdim=True)
    r = r_raw / w.clamp_min(1e-8)
    s = torch.cross(d, r, dim=-1)  # completes a right-handed basis (r, s, d) in the plane
    phi = torch.atan2((e * s).sum(-1), (e * r).sum(-1))
    flip_sign = d[:, 2:3]  # d . z_frame: reverses with the flip, ~+-1 when the peg is upright
    return torch.cat([w * torch.cos(4 * phi).unsqueeze(-1), w * torch.sin(4 * phi).unsqueeze(-1) * flip_sign], dim=-1)


def yaw_mod90_features(quat_in_frame: torch.Tensor) -> torch.Tensor:
    """[N,2]: (cos 4psi, sin 4psi) of the yaw of a quaternion expressed in some frame."""
    _, _, yaw = math_utils.euler_xyz_from_quat(quat_in_frame)
    return torch.stack([torch.cos(4 * yaw), torch.sin(4 * yaw)], dim=-1)


def symmetry_group_local() -> torch.Tensor:
    """The 8 group elements as quaternions acting in the peg's OWN frame (wxyz): 4 yaws x flip about x."""
    import math

    quats = []
    for k in range(4):
        a = k * math.pi / 2
        q_yaw = torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
        quats.append(q_yaw)
        q_flip = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 180 deg about x
        quats.append(math_utils.quat_mul(q_yaw.unsqueeze(0), q_flip.unsqueeze(0)).squeeze(0))
    return torch.stack(quats)


# Unique index multisets of a symmetric rank-4 tensor in 3-D, in the order used everywhere (15 entries):
# xxxx xxxy xxxz xxyy xxyz xxzz xyyy xyyz xyzz xzzz yyyy yyyz yyzz yzzz zzzz
TENSOR4_INDEX = [
    (i, j, k, l) for i in range(3) for j in range(i, 3) for k in range(j, 3) for l in range(k, 3)
]
TENSOR4_NAMES = ["".join("xyz"[i] for i in c) for c in TENSOR4_INDEX]
_I4 = torch.tensor(TENSOR4_INDEX)  # [15, 4]


def tensor4_features(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[N,15]: unique components of ``u^{(x)4} + v^{(x)4}`` for unit vectors ``u``, ``v`` [N,3].

    With ``u``/``v`` the peg's two face normals, this is invariant under the peg's 8-element symmetry
    group (90-degree spins swap ``u``/``v``; flips negate them; 4th powers absorb both) and is a smooth
    function of the pose with no canonical-representative jumps. Its partial trace recovers the axis:
    ``T_ijkk = delta_ij - d_i d_j``.
    """
    idx = _I4.to(u.device)
    uu = u[:, idx]  # [N,15,4]
    vv = v[:, idx]
    return uu.prod(dim=-1) + vv.prod(dim=-1)


def tensor4_pose_features(
    peg_pos_w: torch.Tensor, peg_quat_w: torch.Tensor, frame_pos_w: torch.Tensor, frame_quat_w: torch.Tensor
) -> torch.Tensor:
    """[N,18]: peg centre in ``frame`` (3) + ``tensor4_features`` of the peg's x and y axes in ``frame`` (15)."""
    rel_pos, rel_quat = math_utils.subtract_frame_transforms(frame_pos_w, frame_quat_w, peg_pos_w, peg_quat_w)
    u = _axis(rel_quat, _X)
    v = _axis(rel_quat, (0.0, 1.0, 0.0))
    return torch.cat([rel_pos, tensor4_features(u, v)], dim=-1)
