# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-kernel depth augmentation, lifted from DEXTRAH (`depth_augs.py`).

Targets the train-eval generalization gap on depth-DAgger students. Each kernel
operates on raw depth in meters (B, H, W); call before normalization in
``process_image``.

Augmentations included (defaults from DEXTRAH's __main__ example):
- pixel dropout + random-uniform depth blobs (correlated 2x2 patches)
- random sticks (long thin foreground occluders)
- correlated bilinear shift + Kinect-style depth quantization
"""
from __future__ import annotations

import threading
from typing import Optional

import torch
import warp as wp


# ---- Warp kernels (verbatim from DEXTRAH/dextrah_lab/distillation/depth_augs.py) ----

@wp.kernel
def _add_pixel_dropout_and_randu_kernel(
    depths: wp.array(dtype=float, ndim=3),
    rand_dropout: wp.array(dtype=float, ndim=3),
    rand_u: wp.array(dtype=float, ndim=3),
    rand_u_values: wp.array(dtype=float, ndim=3),
    p_dropout: float,
    p_randu: float,
    d_min: float,
    d_max: float,
    kernel_size: int,
    seed: int,
):
    batch_index, pixel_row, pixel_column = wp.tid()

    if rand_dropout[batch_index, pixel_row, pixel_column] <= p_dropout:
        depths[batch_index, pixel_row, pixel_column] = 0.0

    if rand_u[batch_index, pixel_row, pixel_column] <= p_randu:
        rand_depth = rand_u_values[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
        depths[batch_index, pixel_row, pixel_column] = rand_depth

        for i in range(kernel_size):
            for j in range(kernel_size):
                state = wp.rand_init(seed, batch_index + pixel_row + pixel_column + i + j)
                if wp.randf(state) < 0.25:
                    depths[batch_index, pixel_row + i, pixel_column + j] = rand_depth


@wp.kernel
def _add_sticks_kernel(
    depths: wp.array(dtype=float, ndim=3),
    rand_sticks: wp.array(dtype=float, ndim=3),
    rand_sticks_depths: wp.array(dtype=float, ndim=3),
    p_stick: float,
    max_stick_len: float,
    max_stick_width: float,
    height: int,
    width: int,
    d_min: float,
    d_max: float,
    seed: int,
):
    batch_index, pixel_row, pixel_column = wp.tid()

    stick_width = float(0.0)
    stick_len = float(0.0)
    stick_rot = float(0.0)

    rand_depth = rand_sticks_depths[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min

    if rand_sticks[batch_index, pixel_row, pixel_column] <= p_stick:
        for i in range(3):
            state = wp.rand_init(seed, batch_index + pixel_row + pixel_column + i)
            if i == 0:
                stick_width = wp.randf(state) * max_stick_width
            if i == 1:
                stick_len = wp.randf(state) * max_stick_len + 1.0
            if i == 2:
                stick_rot = wp.randf(state) * (3.14 * 2.0)

        for i in range(int(wp.rint(stick_len))):
            hor_coord = float(pixel_column + i)
            vert_coord = wp.floor(float(i) * wp.sin(stick_rot)) + float(pixel_row)

            if hor_coord > float(width):
                hor_coord = float(width)
            if hor_coord < 0.0:
                hor_coord = 0.0
            if vert_coord > float(height):
                vert_coord = float(height)
            if vert_coord < 0.0:
                vert_coord = 0.0

            depths[batch_index, int(vert_coord), int(hor_coord)] = rand_depth

            for j in range(1, int(max_stick_width)):
                if stick_rot > (3.14 / 4.0) and stick_rot < (3.0 * 3.14 / 4.0):
                    depths[batch_index, int(vert_coord) + j, int(hor_coord)] = rand_depth
                elif stick_rot > (5.0 * 3.14 / 4.0) and stick_rot < (7.0 * 3.14 / 4.0):
                    depths[batch_index, int(vert_coord) + j, int(hor_coord)] = rand_depth
                else:
                    depths[batch_index, int(vert_coord), int(hor_coord) + j] = rand_depth


@wp.kernel
def _add_correlated_noise_kernel(
    depths: wp.array(dtype=float, ndim=3),
    rand_sigma_s_x: wp.array(dtype=float, ndim=3),
    rand_sigma_s_y: wp.array(dtype=float, ndim=3),
    rand_sigma_d: wp.array(dtype=float, ndim=3),
    height: int,
    width: int,
    d_min: float,
    d_max: float,
    noisy_depths: wp.array(dtype=float, ndim=3),
):
    batch_index, pixel_row, pixel_column = wp.tid()

    nx = rand_sigma_s_x[batch_index, pixel_row, pixel_column]
    ny = rand_sigma_s_y[batch_index, pixel_row, pixel_column]

    u = nx + float(pixel_column)
    v = ny + float(pixel_row)

    u0 = int(u)
    v0 = int(v)
    u1 = u0 + 1
    v1 = v0 + 1

    fu = u - float(u0)
    fv = v - float(v0)

    u0 = wp.max(0, wp.min(u0, width - 1))
    u1 = wp.max(0, wp.min(u1, width - 1))
    v0 = wp.max(0, wp.min(v0, height - 1))
    v1 = wp.max(0, wp.min(v1, height - 1))

    w_00 = (1.0 - fu) * (1.0 - fv)
    w_01 = (1.0 - fu) * fv
    w_10 = fu * (1.0 - fv)
    w_11 = fu * fv

    noisy_depths[batch_index, pixel_row, pixel_column] = (
        depths[batch_index, v0, u0] * w_00
        + depths[batch_index, v0, u1] * w_01
        + depths[batch_index, v1, u0] * w_10
        + depths[batch_index, v1, u1] * w_11
    )

    baseline = float(35130.0)
    ref = float(8.0)
    denominator = (
        baseline / noisy_depths[batch_index, pixel_row, pixel_column]
        + rand_sigma_d[batch_index, pixel_row, pixel_column]
        + 0.5
    )
    noisy_depths[batch_index, pixel_row, pixel_column] = baseline / (wp.rint(denominator / ref) * ref)


# ---- Python entry-point ----

# DEXTRAH defaults (depth_augs.py:__main__).
# Comment from upstream: dropout/randu/stick rates were divided by 4 to tame DR magnitude.
DEFAULT_PARAMS = dict(
    p_dropout=0.0125 / 4,
    p_randu=0.0125 / 4,
    p_stick=0.001 / 4,
    max_stick_len=18.0,
    max_stick_width=3.0,
    sigma_s=0.5,
    sigma_d=1.0 / 6.0,
    kernel_size=2,
)


_singletons: dict[str, "DepthAug"] = {}
_singletons_lock = threading.Lock()
_warp_inited = False


def _ensure_warp_inited():
    global _warp_inited
    if not _warp_inited:
        wp.init()
        _warp_inited = True


class DepthAug:
    """Per-device cached aug invoker. Use ``DepthAug.get(device)`` to fetch the singleton."""

    def __init__(self, device: str, seed: int = 42, params: Optional[dict] = None):
        _ensure_warp_inited()
        self.device = device
        self.seed = seed
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    @classmethod
    def get(cls, device: str) -> "DepthAug":
        with _singletons_lock:
            if device not in _singletons:
                _singletons[device] = cls(device)
            return _singletons[device]

    def apply(self, depths: torch.Tensor, d_min: float, d_max: float) -> torch.Tensor:
        """Apply correlated noise -> dropout/randu -> sticks in raw-meters depth.

        ``depths`` is mutated in place AND returned (so callers can keep their
        existing chained ops). Expects shape (B, H, W) float32 contiguous on CUDA.
        """
        if depths.dim() != 3:
            raise ValueError(f"DepthAug.apply expects (B,H,W); got {tuple(depths.shape)}")
        if not depths.is_contiguous():
            depths = depths.contiguous()

        B, H, W = depths.shape
        p = self.params

        noisy = torch.empty_like(depths)
        rand_sigma_s_x = p["sigma_s"] * torch.randn(B, H, W, device=self.device)
        rand_sigma_s_y = p["sigma_s"] * torch.randn(B, H, W, device=self.device)
        rand_sigma_d = p["sigma_d"] * torch.randn(B, H, W, device=self.device)

        wp.launch(
            kernel=_add_correlated_noise_kernel,
            dim=[B, H, W],
            inputs=[
                wp.torch.from_torch(depths),
                wp.torch.from_torch(rand_sigma_s_x),
                wp.torch.from_torch(rand_sigma_s_y),
                wp.torch.from_torch(rand_sigma_d),
                H,
                W,
                d_min,
                d_max,
                wp.torch.from_torch(noisy),
            ],
            device=self.device,
        )
        depths.copy_(noisy)

        rand_dropout = torch.rand(B, H, W, device=self.device)
        rand_u = torch.rand(B, H, W, device=self.device)
        rand_u_values = torch.rand(B, H, W, device=self.device)
        wp.launch(
            kernel=_add_pixel_dropout_and_randu_kernel,
            dim=[B, H, W],
            inputs=[
                wp.torch.from_torch(depths),
                wp.torch.from_torch(rand_dropout),
                wp.torch.from_torch(rand_u),
                wp.torch.from_torch(rand_u_values),
                p["p_dropout"],
                p["p_randu"],
                d_min,
                d_max,
                p["kernel_size"],
                self.seed,
            ],
            device=self.device,
        )

        rand_stick = torch.rand(B, H, W, device=self.device)
        rand_stick_depths = torch.rand(B, H, W, device=self.device)
        wp.launch(
            kernel=_add_sticks_kernel,
            dim=[B, H, W],
            inputs=[
                wp.torch.from_torch(depths),
                wp.torch.from_torch(rand_stick),
                wp.torch.from_torch(rand_stick_depths),
                p["p_stick"],
                p["max_stick_len"],
                p["max_stick_width"],
                H,
                W,
                d_min,
                d_max,
                self.seed,
            ],
            device=self.device,
        )

        return depths
