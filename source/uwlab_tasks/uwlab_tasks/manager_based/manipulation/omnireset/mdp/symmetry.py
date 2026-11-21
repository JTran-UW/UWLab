# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object symmetry declarations and alignment errors scored at the best-matching identical-looking pose.

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
    flip = bool(sym.get("flip", False))
    if order == 1 and not flip:
        return None  # trivial group: same as declaring no symmetry
    return {"axis": [a / norm for a in axis], "order": order, "flip": flip}


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
    thresholds: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position error [N] and |roll| + |pitch| [N] of the best-aligned symmetric version of an assembly.

    ``rel_pos``/``rel_quat`` [N, 3]/[N, 4]: the insertive alignment frame in the receptive alignment frame.
    ``receptive_offset``: the receptive alignment offset (pos [3], quat [4]) in its object frame, about which
    ``receptive_group`` acts (``insertive_group`` acts in the insertive alignment frame). Yaw is ignored, as in the
    single-offset check. The version returned minimizes max(xyz / thresholds[0], tilt / thresholds[1]), so a pose
    passes both thresholds iff some version does (a receptive group that moves the target, e.g. a tabletop's four
    holes, must be chosen by position, not tilt).
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
    xyz = pos.norm(dim=-1).repeat_interleave(k_g, 1)
    best = torch.maximum(xyz / thresholds[0], tilt / thresholds[1]).argmin(dim=1, keepdim=True)
    return xyz.gather(1, best).squeeze(1), tilt.gather(1, best).squeeze(1)


def success_target_offsets(receptive_meta: dict) -> list[tuple[list[float], list[float]]]:
    """Receptive alignment offsets (pos, quat xyzw) that count as success: every ``targets`` entry when the metadata
    says ``success_targets: all`` (e.g. a desk top whose 4 holes are not related by its symmetry), else just
    ``assembled_offset``."""
    if receptive_meta.get("success_targets") == "all" and receptive_meta.get("targets"):
        return [(list(t["pos"]), list(t["quat"])) for t in receptive_meta["targets"]]
    o = receptive_meta["assembled_offset"]
    return [(list(o["pos"]), list(o["quat"]))]


def multi_target_alignment_error(
    ins_pos_r: torch.Tensor,
    ins_quat_r: torch.Tensor,
    targets: list[tuple[torch.Tensor, torch.Tensor]],
    insertive_group: torch.Tensor,
    receptive_group: torch.Tensor,
    thresholds: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Best ``symmetric_alignment_error`` over several receptive targets.

    ``ins_pos_r``/``ins_quat_r`` [N, 3]/[N, 4]: the insertive alignment frame in the receptive ROOT frame. Returns
    position error [N], |roll| + |pitch| [N] and the index [N] of the target that scored best (by
    max(xyz / pos_thr, tilt / ori_thr)).
    """
    errs = []
    for t_p, t_q in targets:
        t_p, t_q = t_p.to(ins_pos_r), t_q.to(ins_pos_r)
        n = ins_pos_r.shape[0]
        rel_p, rel_q = math_utils.subtract_frame_transforms(t_p.expand(n, 3), t_q.expand(n, 4), ins_pos_r, ins_quat_r)
        errs.append(symmetric_alignment_error(rel_p, rel_q, (t_p, t_q), insertive_group, receptive_group, thresholds))
    xyz = torch.stack([e[0] for e in errs], dim=1)
    tilt = torch.stack([e[1] for e in errs], dim=1)
    best = torch.maximum(xyz / thresholds[0], tilt / thresholds[1]).argmin(dim=1, keepdim=True)
    return xyz.gather(1, best).squeeze(1), tilt.gather(1, best).squeeze(1), best.squeeze(1)
