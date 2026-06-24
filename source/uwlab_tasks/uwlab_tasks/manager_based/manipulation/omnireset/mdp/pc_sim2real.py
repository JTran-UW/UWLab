# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2real point-cloud augmentation for a single fixed RealSense D455 + FoundationStereo.

Numpy/scipy implementation of the pipeline described in ``PC_DESIGN.MD``. The
camera is a single fixed stereo IR pair whose depth comes from a learned
matcher (FoundationStereo), *not* the hardware block-matching ASIC. That flips
the noise model: the output is dense and smooth, so we model **edge bleed** and
**low-frequency surface bias** rather than textureless dropout / disparity
quantization.

Pipeline (order matters -- visibility first defines which points survive, then
noise/dropout act only on the survivors)::

    CAD cloud (camera optical frame)
      -> 0. frustum cull: keep points inside the camera FOV  (frustum_cull)
      -> 1. HPR self-occlusion from the camera              (occlude_hpr)
      -> 2. edge bleed (boundary skirt) + surface bias      (edge_bleed, surface_bias)
      -> 3. light random dropout                            (light_dropout)
      -> partial, realistic cloud

Both visibility steps -- frustum projection and the Katz HPR convex hull -- are
pure geometry, **no ray tracing**, matching what a real fixed camera physically
captures (out-of-FOV geometry isn't imaged; occluded geometry isn't seen).

The cloud is expressed in the camera **optical frame** (z forward, x right, y
down): the camera sits at the origin, so ``camera_center`` is the zero vector in
:func:`augment_pointcloud`. The background (table) plane for edge bleed is passed
as ``(plane_point, plane_normal)`` in that same frame. All functions operate on
``(N, 3)`` numpy arrays.

Open3D is intentionally *not* a dependency here (it is not installed in the
runtime env): HPR uses ``scipy.spatial.ConvexHull`` (Katz-Tal-Basri spherical
inversion) and normals are estimated via local-PCA over a kd-tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


@dataclass
class AugParams:
    """Knobs for the sim2real point-cloud augmentation pipeline.

    Defaults follow the priority ranking in ``PC_DESIGN.MD``: edge bleed is the
    dominant FoundationStereo artifact, HPR is necessary, surface bias is low
    effort, dropout is minor. Per-stage ``enable_*`` flags let a sanity-check
    toggle each stage independently.
    """

    # -- stage toggles --
    enable_frustum_cull: bool = True
    enable_hpr: bool = True
    enable_edge_bleed: bool = True
    enable_surface_bias: bool = True
    enable_dropout: bool = True

    # -- (0) frustum cull: keep only points inside the camera FOV, in front --
    # Intrinsics derived from the sim front-camera (camera_align_cfg.py): a
    # PinholeCamera with focal_length / horizontal_aperture (mm) and an image
    # size. Pixels are square (vertical aperture auto-scaled), so fy == fx.
    focal_length_mm: float = 13.20
    horizontal_aperture_mm: float = 20.955
    img_width: int = 640
    img_height: int = 480
    near: float = 0.02  # drop points closer than this (m) along the optical axis

    # -- (1) HPR self-occlusion --
    # Katz radius = scene_diameter * hpr_radius_scale. Larger => more permissive
    # (only clearly back-facing points removed); smaller => over-prunes concave
    # regions. Tune once against real frames (see PC_DESIGN.MD section 1).
    hpr_radius_scale: float = 100.0

    # -- normals (for edge-bleed boundary detection) --
    normal_k: int = 16

    # -- (2a) edge bleed: skirt of interpolated points bridging object->background --
    grazing_thresh: float = 0.3  # |normal . viewdir| < thresh => boundary point
    n_skirt: int = 3  # interpolation samples between object depth and bg depth
    bleed_frac: float = 0.4  # fraction of (boundary x skirt) points actually kept
    max_bleed_depth: float = 0.3  # cap skirt length (m); skip when bg is farther
    # (e.g. the table plane grazing the camera rays -> intersection at infinity)
    edge_depth_jump: float = 0.02  # (torch only) depth-image edge threshold (m): a
    # point is a boundary point if a neighbour z-buffer cell is >= this much farther
    # or empty. Replaces the numpy normal-grazing boundary test.

    # -- (2b) low-frequency surface bias (gentle plane bend + scale error) --
    axial_scale: float = 0.005  # systematic depth scale error (~0.5% of range)
    bend: float = 0.003  # gentle plane bend coefficient
    residual_jitter: float = 0.0005  # tiny per-point jitter only

    # -- (3) light random dropout --
    dropout_keep: float = 0.92

    # -- (4) fliers: isolated outlier points floating off the surface --
    # Models learned-stereo false matches (the scattered points the real cloud
    # has that aren't part of any structure). A small fraction of points are
    # displaced along their camera ray (wrong depth) plus a little lateral jitter.
    enable_flier: bool = True
    flier_frac: float = 0.01  # fraction of output points turned into fliers
    flier_depth: float = 0.08  # max +/- along-ray (depth) displacement (m)
    flier_lateral: float = 0.02  # lateral jitter std (m)

    def intrinsics(self) -> tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) in pixels from the pinhole camera parameters."""
        fx = self.focal_length_mm / self.horizontal_aperture_mm * self.img_width
        return fx, fx, self.img_width / 2.0, self.img_height / 2.0


def frustum_cull(
    points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float = 0.02,
) -> np.ndarray:
    """Return indices of points inside the camera frustum (FOV + in front).

    ``points`` are in the optical frame (z forward, x right, y down). A point is
    kept iff it is in front of the camera (``z > near``) and its pinhole
    projection lands inside the image. Pure projection -- no ray tracing -- so it
    matches what the real camera physically captures (out-of-FOV geometry simply
    isn't imaged).
    """
    if len(points) == 0:
        return np.arange(0)
    z = points[:, 2]
    in_front = z > near
    zc = np.where(in_front, z, 1.0)
    u = fx * points[:, 0] / zc + cx
    v = fy * points[:, 1] / zc + cy
    inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.nonzero(inside)[0]


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Per-point normals via local PCA over k nearest neighbors.

    Orientation is arbitrary (we only use ``|normal . viewdir|``), so we skip the
    expensive normal-orientation propagation that Open3D does.
    """
    n = len(points)
    if n < 3:
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=points.dtype), (n, 1))
    k = int(min(k, n))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)  # (n, k)
    nbrs = points[idx]  # (n, k, 3)
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    # smallest-eigenvalue eigenvector is the surface normal
    _, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    return vecs[:, :, 0].astype(points.dtype)


def occlude_hpr(points: np.ndarray, camera_center: np.ndarray, radius_scale: float = 100.0) -> np.ndarray:
    """Katz-Tal-Basri Hidden Point Removal. Returns indices of *visible* points.

    Spherical inversion about the camera center followed by a convex hull; a
    point is visible iff its inverted image lies on the hull of the inverted set
    (plus the camera origin). No meshing or ray tracing.
    """
    n = len(points)
    if n < 4:
        return np.arange(n)
    p = points - camera_center
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-9)
    diameter = float(np.linalg.norm(points.max(0) - points.min(0)))
    radius = diameter * radius_scale
    # spherical flip: reflect each point across a sphere of the given radius
    flipped = p + 2.0 * (radius - norm) * (p / norm)
    aug = np.vstack([flipped, np.zeros((1, 3), dtype=flipped.dtype)])  # + camera origin
    try:
        hull = ConvexHull(aug)
    except Exception:
        # degenerate (e.g. coplanar) input -- treat everything as visible
        return np.arange(n)
    visible = np.unique(hull.vertices)
    return visible[visible != n]  # drop the appended camera-origin index


def edge_bleed(
    points: np.ndarray,
    normals: np.ndarray,
    camera_center: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    n_skirt: int = 3,
    frac: float = 0.4,
    grazing_thresh: float = 0.3,
    max_bleed_depth: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add a skirt of points bridging object boundaries down to the background.

    The dominant learned-stereo artifact: at depth discontinuities the matcher
    interpolates between foreground and background, producing points strung along
    the camera ray between the object edge and the surface behind it. The
    background is modeled as a plane (``plane_point``, ``plane_normal``, e.g. the
    tabletop expressed in the working frame); we add interpolated points along the
    ray from each boundary point toward that plane.
    """
    rng = rng or np.random.default_rng()
    if len(points) == 0:
        return points
    view = points - camera_center
    z = np.linalg.norm(view, axis=1, keepdims=True)
    z = np.maximum(z, 1e-9)
    dirs = view / z
    grazing = np.abs((normals * dirs).sum(1)) < grazing_thresh  # boundary points
    edge = points[grazing]
    if len(edge) == 0:
        return points
    edge_dirs = dirs[grazing]
    z_obj = z[grazing]  # (m, 1) depth of object boundary along its ray

    # background depth = ray/plane intersection distance. Ray = camera_center +
    # t * dir (dir unit), so t (== euclidean depth) = dot(plane_point - cam, n) /
    # dot(dir, n).
    denom = edge_dirs @ plane_normal  # (m,)
    num = float((plane_point - camera_center) @ plane_normal)
    safe_denom = np.where(np.abs(denom) > 1e-6, denom, np.nan)
    z_bg = (num / safe_denom)[:, None]  # (m, 1)

    # only bridge where the background is genuinely behind the object boundary
    # AND within a sensible distance (a grazing plane intersects the ray at near
    # infinity -- those skirts are meaningless and must be dropped).
    valid = (
        np.isfinite(z_bg).ravel()
        & (z_bg > z_obj).ravel()
        & ((z_bg - z_obj) <= max_bleed_depth).ravel()
    )
    edge_dirs, z_obj, z_bg = edge_dirs[valid], z_obj[valid], z_bg[valid]
    if len(edge_dirs) == 0:
        return points

    skirts = []
    for a in np.linspace(0.2, 0.9, n_skirt):
        zi = (1.0 - a) * z_obj + a * z_bg
        skirts.append(camera_center + edge_dirs * zi)
    skirts = np.concatenate(skirts, axis=0)

    keep = int(frac * len(edge))
    if keep <= 0:
        return points
    sel = rng.choice(len(skirts), size=min(keep, len(skirts)), replace=False)
    return np.vstack([points, skirts[sel].astype(points.dtype)])


def surface_bias(
    points: np.ndarray,
    camera_center: np.ndarray,
    axial_scale: float = 0.005,
    bend: float = 0.003,
    jitter: float = 0.0005,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Low-frequency depth bias: small systematic scale error + gentle plane bend.

    Learned stereo gets the *global* scale/offset slightly wrong and bends flat
    surfaces gently, rather than adding white noise. Modeled as a smooth field
    along the camera ray plus a tiny residual jitter.
    """
    rng = rng or np.random.default_rng()
    if len(points) == 0:
        return points
    rays = points - camera_center
    z = np.linalg.norm(rays, axis=1, keepdims=True)
    z = np.maximum(z, 1e-9)
    dirs = rays / z
    zmean = float(z.mean())
    bias = axial_scale * z
    bias = bias + bend * (z - zmean) ** 2 / (zmean**2 + 1e-9)
    pts = camera_center + dirs * (z + bias)
    pts = pts + rng.standard_normal(pts.shape) * jitter
    return pts.astype(points.dtype)


def light_dropout(points: np.ndarray, keep: float = 0.92, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mild random dropout (keep ~90-95%). Minor regularization for a simple scene."""
    rng = rng or np.random.default_rng()
    if len(points) == 0:
        return points
    mask = rng.random(len(points)) < keep
    if not mask.any():
        return points
    return points[mask]


def flier_noise(points: np.ndarray, params: AugParams, rng: np.random.Generator | None = None) -> np.ndarray:
    """Turn a small fraction of points into isolated fliers (wrong-depth outliers).

    Each selected point is pushed along its camera ray by a random depth offset
    (so it floats in front of / behind the surface) plus a small lateral jitter --
    matching the detached stereo-mismatch points a real learned-stereo cloud has.
    """
    rng = rng or np.random.default_rng()
    n = len(points)
    if n == 0:
        return points
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-6)
    dirs = points / norm
    depth_off = (rng.random((n, 1)) * 2 - 1) * params.flier_depth
    lateral = rng.standard_normal((n, 3)) * params.flier_lateral
    flier = points + dirs * depth_off + lateral
    mask = rng.random(n) < params.flier_frac
    out = points.copy()
    out[mask] = flier[mask].astype(points.dtype)
    return out


def augment_pointcloud(
    points_cam: np.ndarray,
    params: AugParams,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run the full frustum-cull -> HPR -> edge-bleed -> surface-bias -> dropout
    pipeline on a cloud expressed in the camera optical frame.

    The camera sits at the origin of this frame, so ``camera_center`` is the zero
    vector throughout. ``plane_point`` / ``plane_normal`` describe the background
    (table) plane in the same camera frame, for the edge-bleed skirt.

    Returns a variable-length ``(M, 3)`` cloud; use :func:`resample_to` to get a
    fixed-size observation vector.
    """
    rng = rng or np.random.default_rng()
    pts = np.asarray(points_cam, dtype=np.float32)
    cam = np.zeros(3, dtype=np.float32)  # camera at origin of the optical frame

    if params.enable_frustum_cull:
        fx, fy, cx, cy = params.intrinsics()
        keep = frustum_cull(pts, fx, fy, cx, cy, params.img_width, params.img_height, params.near)
        pts = pts[keep]
    if len(pts) == 0:
        return pts

    if params.enable_hpr:
        vis = occlude_hpr(pts, cam, params.hpr_radius_scale)
        pts = pts[vis]
    if len(pts) == 0:
        return pts

    if params.enable_edge_bleed:
        normals = estimate_normals(pts, params.normal_k)
        pts = edge_bleed(
            pts, normals, cam, plane_point, plane_normal,
            n_skirt=params.n_skirt, frac=params.bleed_frac,
            grazing_thresh=params.grazing_thresh, max_bleed_depth=params.max_bleed_depth, rng=rng,
        )
    if params.enable_surface_bias:
        pts = surface_bias(
            pts, cam,
            axial_scale=params.axial_scale, bend=params.bend,
            jitter=params.residual_jitter, rng=rng,
        )
    if params.enable_dropout:
        pts = light_dropout(pts, keep=params.dropout_keep, rng=rng)
    if params.enable_flier:
        pts = flier_noise(pts, params, rng)
    return pts


def resample_to(points: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Resample a variable-length cloud to exactly ``n`` points (for a fixed obs)."""
    rng = rng or np.random.default_rng()
    m = len(points)
    if m == 0:
        return np.zeros((n, 3), dtype=np.float32)
    replace = m < n
    idx = rng.choice(m, size=n, replace=replace)
    return points[idx].astype(np.float32)
