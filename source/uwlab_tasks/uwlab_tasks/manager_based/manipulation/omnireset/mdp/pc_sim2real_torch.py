# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched, GPU-resident torch port of the sim2real point-cloud augmentation.

This is the data-collection-scale version of :mod:`pc_sim2real` (which is
numpy/scipy and runs one env at a time). Every stage here operates on a whole
batch ``(N, M, 3)`` of clouds at once and stays on-device -- no per-env Python
loop, no host syncs -- so it can run inside the obs pipeline at thousands of
parallel envs.

Two algorithmic substitutions vs the numpy reference make the batch tractable:

* **HPR occlusion -> z-buffer.** scipy's Katz convex-hull HPR has no batched GPU
  analogue. We replace it with a depth-buffer test: project every point to a
  camera grid and keep only the nearest point per cell (a batched scatter-min).
  This is O(N*M), fully vectorized, and is arguably a *more* faithful model of a
  real depth camera (one depth sample per pixel) than Katz spherical inversion.

* **Edge-bleed boundary detection -> depth-image edges.** The numpy version finds
  silhouette points via per-point PCA normals (a batched kNN, expensive). Here a
  point is a boundary point iff a neighbouring z-buffer cell is empty or much
  farther -- i.e. it sits on a depth discontinuity. That reuses the z-buffer we
  already built and needs no normals.

Variable-length stages are handled with a fixed-size buffer plus a per-point
``valid`` mask rather than ragged tensors, so shapes stay static and batched.

All clouds are in the camera **optical frame** (z forward, x right, y down); the
camera sits at the origin. ``plane_point`` / ``plane_normal`` describe the
background (table) plane in that same frame, per env, shape ``(N, 3)``.
"""

from __future__ import annotations

import torch

from .pc_sim2real import AugParams

# Default z-buffer / occlusion grid (Hz, Wz), 4:3 to match the 640x480 camera.
# Coarser than the real image so the scatter-min buffer (N * Hz * Wz floats)
# stays small; raise it for finer occlusion at a memory cost.
DEFAULT_ZBUF_HW: tuple[int, int] = (192, 256)


def _intrinsics(params: AugParams) -> tuple[float, float, float, float]:
    fx = params.focal_length_mm / params.horizontal_aperture_mm * params.img_width
    return fx, fx, params.img_width / 2.0, params.img_height / 2.0


def frustum_cull_mask(points: torch.Tensor, params: AugParams) -> torch.Tensor:
    """``(N, M)`` bool mask of points inside the camera FOV and in front of it."""
    x, y, z = points.unbind(-1)
    fx, fy, cx, cy = _intrinsics(params)
    in_front = z > params.near
    zc = torch.where(in_front, z, torch.ones_like(z))
    u = fx * x / zc + cx
    v = fy * y / zc + cy
    return in_front & (u >= 0) & (u < params.img_width) & (v >= 0) & (v < params.img_height)


def _zbuffer(points: torch.Tensor, valid: torch.Tensor, params: AugParams, zbuf_hw: tuple[int, int]):
    """Scatter-min depth buffer. Returns ``(buf, iu, iv, env_off, (Hz, Wz))``.

    ``buf`` is a flat ``(N*Hz*Wz,)`` tensor holding the nearest valid depth in
    each (env, cell); ``iu, iv`` are each point's integer grid coords; ``env_off``
    is the per-env flat-cell offset for indexing ``buf``.
    """
    N, M, _ = points.shape
    Hz, Wz = zbuf_hw
    x, y, z = points.unbind(-1)
    fx, fy, cx, cy = _intrinsics(params)
    sx = Wz / params.img_width
    sy = Hz / params.img_height
    zc = torch.where(z > 1e-6, z, torch.ones_like(z))
    iu = ((fx * x / zc + cx) * sx).long().clamp_(0, Wz - 1)
    iv = ((fy * y / zc + cy) * sy).long().clamp_(0, Hz - 1)
    ncell = Hz * Wz
    env_off = torch.arange(N, device=points.device).unsqueeze(1) * ncell  # (N, 1)
    key = (env_off + iv * Wz + iu).reshape(-1)  # (N*M,)
    inf = torch.finfo(points.dtype).max
    zfill = torch.where(valid, z, torch.full_like(z, inf)).reshape(-1)
    buf = torch.full((N * ncell,), inf, device=points.device, dtype=points.dtype)
    buf.scatter_reduce_(0, key, zfill, reduce="amin", include_self=True)
    return buf, iu, iv, env_off, (Hz, Wz)


def zbuffer_visible_mask(
    points: torch.Tensor, valid: torch.Tensor, params: AugParams,
    zbuf_hw: tuple[int, int] = DEFAULT_ZBUF_HW, eps: float = 1e-3,
) -> torch.Tensor:
    """``(N, M)`` bool mask: a point is visible iff it is the nearest (within
    ``eps``) at its z-buffer cell. Occluded points (something nearer at the same
    cell) are dropped -- this is the GPU occlusion test."""
    buf, iu, iv, env_off, (Hz, Wz) = _zbuffer(points, valid, params, zbuf_hw)
    min_z = buf[(env_off + iv * Wz + iu).reshape(-1)].reshape(points.shape[:2])
    return valid & (points[..., 2] <= min_z + eps)


def edge_bleed_batched(
    points: torch.Tensor, valid: torch.Tensor, params: AugParams,
    plane_point: torch.Tensor, plane_normal: torch.Tensor,
    zbuf_hw: tuple[int, int], generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One skirt candidate per point, bridging depth-edge points toward the bg plane.

    Returns ``(skirt (N, M, 3), skirt_valid (N, M))``. A candidate is kept iff its
    source point is on a depth discontinuity (a neighbour z-buffer cell is empty
    or much farther), the background plane is genuinely behind it and within
    ``max_bleed_depth``, and a Bernoulli(``bleed_frac``) draw passes. Generating
    at most one skirt per point (vs ``n_skirt`` in numpy) keeps the buffer fixed
    at ``2M``; the kept fraction reproduces the thin edge skirt statistically.
    """
    N, M, _ = points.shape
    buf, iu, iv, env_off, (Hz, Wz) = _zbuffer(points, valid, params, zbuf_hw)
    z = points[..., 2]
    inf = torch.finfo(points.dtype).max

    def nbr_minz(du: int, dv: int) -> torch.Tensor:
        nu = (iu + du).clamp(0, Wz - 1)
        nv = (iv + dv).clamp(0, Hz - 1)
        return buf[(env_off + nv * Wz + nu).reshape(-1)].reshape(N, M)

    is_boundary = torch.zeros_like(valid)
    for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nz = nbr_minz(du, dv)
        is_boundary = is_boundary | (nz >= inf * 0.5) | ((nz - z) > params.edge_depth_jump)
    is_boundary = is_boundary & valid

    # ray geometry (camera at optical-frame origin)
    rnorm = points.norm(dim=-1).clamp(min=1e-9)  # euclidean depth (N, M)
    dirs = points / rnorm.unsqueeze(-1)
    pn = plane_normal.unsqueeze(1)  # (N, 1, 3)
    denom = (dirs * pn).sum(-1)  # (N, M)
    num = (plane_point * plane_normal).sum(-1, keepdim=True)  # (N, 1)
    safe = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, float("nan")))
    z_bg = num / safe  # (N, M)
    z_obj = rnorm
    ok = torch.isfinite(z_bg) & (z_bg > z_obj) & ((z_bg - z_obj) <= params.max_bleed_depth)

    a = 0.2 + 0.7 * torch.rand(N, M, generator=generator, device=points.device)
    zi = (1.0 - a) * z_obj + a * z_bg
    skirt = dirs * zi.unsqueeze(-1)
    rkeep = torch.rand(N, M, generator=generator, device=points.device) < params.bleed_frac
    skirt_valid = is_boundary & ok & rkeep
    skirt = torch.nan_to_num(skirt, nan=0.0, posinf=0.0, neginf=0.0)
    return skirt, skirt_valid


def surface_bias_batched(
    points: torch.Tensor, valid: torch.Tensor, params: AugParams, generator: torch.Generator | None,
) -> torch.Tensor:
    """Low-frequency depth bias (scale error + gentle bend + tiny jitter), per env.

    Only valid points are modified; invalid slots are returned unchanged (they get
    resampled away anyway)."""
    rnorm = points.norm(dim=-1, keepdim=True).clamp(min=1e-9)  # (N, M, 1)
    dirs = points / rnorm
    z = rnorm
    vmask = valid.unsqueeze(-1)
    cnt = valid.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (N, 1, 1)
    zmean = torch.where(vmask, z, torch.zeros_like(z)).sum(1, keepdim=True) / cnt  # (N, 1, 1)
    bias = params.axial_scale * z + params.bend * (z - zmean) ** 2 / (zmean**2 + 1e-9)
    pts = dirs * (z + bias)
    pts = pts + torch.randn(points.shape, generator=generator, device=points.device) * params.residual_jitter
    return torch.where(vmask, pts, points)


def resample_batched(
    points: torch.Tensor, valid: torch.Tensor, num_points: int, generator: torch.Generator | None,
) -> torch.Tensor:
    """Resample each env's valid points to exactly ``num_points`` (with replacement
    when there are fewer). Fully batched: random-index into each env's valid set.

    Returns ``(N, num_points, 3)``. Envs with no valid points return that env's
    slot-0 point (a rare degenerate case)."""
    N, M, _ = points.shape
    device = points.device
    # Valid positions first per row (argsort of the bool, descending).
    order = torch.argsort(valid.to(torch.int8), dim=1, descending=True)  # (N, M)
    counts = valid.sum(1).clamp(min=1)  # (N,)
    r = torch.rand(N, num_points, generator=generator, device=device)
    pick = (r * counts.unsqueeze(1)).long().clamp_(max=M - 1)  # (N, num_points) in [0, counts)
    gather_idx = torch.gather(order, 1, pick)  # (N, num_points) actual point indices
    return torch.gather(points, 1, gather_idx.unsqueeze(-1).expand(-1, -1, 3))


def apply_fliers(points: torch.Tensor, params: AugParams, generator: torch.Generator | None) -> torch.Tensor:
    """Turn a small fraction of points into isolated fliers (wrong-depth outliers).

    Each selected point is displaced along its camera ray by a random depth offset
    (floating off the surface) plus small lateral jitter -- models learned-stereo
    false matches (the detached points a real cloud has). Batched, in-place-safe.
    """
    N, P, _ = points.shape
    device = points.device
    norm = points.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dirs = points / norm
    depth_off = (torch.rand(N, P, 1, generator=generator, device=device) * 2 - 1) * params.flier_depth
    lateral = torch.randn(N, P, 3, generator=generator, device=device) * params.flier_lateral
    flier = points + dirs * depth_off + lateral
    mask = (torch.rand(N, P, generator=generator, device=device) < params.flier_frac).unsqueeze(-1)
    return torch.where(mask, flier, points)


def augment_pointcloud_batched(
    points: torch.Tensor,
    params: AugParams,
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
    num_points: int,
    zbuf_hw: tuple[int, int] = DEFAULT_ZBUF_HW,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full batched pipeline: frustum -> z-buffer occlusion -> edge bleed ->
    surface bias -> dropout -> resample.

    Args:
        points: ``(N, M, 3)`` dense scene cloud in the camera optical frame.
        params: augmentation knobs (shared :class:`AugParams`).
        plane_point, plane_normal: ``(N, 3)`` background plane in the camera frame.
        num_points: fixed output size per env.
        zbuf_hw: occlusion grid resolution.
        generator: torch RNG (device-resident) for reproducible noise.

    Returns:
        ``(N, num_points, 3)`` augmented cloud in the camera optical frame.
    """
    N, M, _ = points.shape
    valid = torch.ones(N, M, dtype=torch.bool, device=points.device)

    if params.enable_frustum_cull:
        valid = valid & frustum_cull_mask(points, params)
    if params.enable_hpr:
        valid = valid & zbuffer_visible_mask(points, valid, params, zbuf_hw)

    all_pts, all_valid = points, valid
    if params.enable_edge_bleed:
        skirt, skirt_valid = edge_bleed_batched(
            points, valid, params, plane_point, plane_normal, zbuf_hw, generator
        )
        all_pts = torch.cat([points, skirt], dim=1)
        all_valid = torch.cat([valid, skirt_valid], dim=1)

    if params.enable_surface_bias:
        all_pts = surface_bias_batched(all_pts, all_valid, params, generator)

    if params.enable_dropout:
        keep = torch.rand(all_valid.shape, generator=generator, device=points.device) < params.dropout_keep
        all_valid = all_valid & keep

    out = resample_batched(all_pts, all_valid, num_points, generator)
    if params.enable_flier:
        out = apply_fliers(out, params, generator)
    return out
