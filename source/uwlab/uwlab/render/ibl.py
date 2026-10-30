# SPDX-License-Identifier: BSD-3-Clause
"""Precompute diffuse irradiance and GGX-filtered HDRI for stable real-time IBL."""

from functools import lru_cache

import numpy as np

from .types import IBL_ROUGHNESS_LEVELS

IBL_WIDTH = 128
IBL_HEIGHT = 64


def _directions(width: int, height: int) -> np.ndarray:
    u = (np.arange(width, dtype=np.float64) + 0.5) / width
    v = (np.arange(height, dtype=np.float64) + 0.5) / height
    phi = 2.0 * np.pi * (u[None, :] - 0.5)
    theta = np.pi * v[:, None]
    return np.stack(
        np.broadcast_arrays(
            np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
        ),
        axis=-1,
    )


def _sample_latlong(pixels: np.ndarray, directions: np.ndarray) -> np.ndarray:
    height, width = pixels.shape[:2]
    direction = directions / np.maximum(
        np.linalg.norm(directions, axis=-1, keepdims=True), 1.0e-12
    )
    u = np.arctan2(direction[..., 1], direction[..., 0]) / (2.0 * np.pi) + 0.5
    v = np.arccos(np.clip(direction[..., 2], -1.0, 1.0)) / np.pi
    x = u * width - 0.5
    y = np.clip(v * height - 0.5, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    a = pixels[y0, x0 % width] * (1.0 - fx) + pixels[y0, (x0 + 1) % width] * fx
    b = pixels[np.minimum(y0 + 1, height - 1), x0 % width] * (1.0 - fx)
    b += pixels[np.minimum(y0 + 1, height - 1), (x0 + 1) % width] * fx
    return a * (1.0 - fy) + b * fy


def _sh_basis(directions: np.ndarray) -> np.ndarray:
    x, y, z = np.moveaxis(directions, -1, 0)
    return np.stack(
        (
            np.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ),
        axis=-1,
    )


def _hammersley(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.uint32)
    bits = ((indices << 16) | (indices >> 16)) & 0xFFFFFFFF
    bits = (((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
    bits = (((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
    bits = (((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF
    bits = (((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)) & 0xFFFFFFFF
    return np.stack(
        ((indices + 0.5) / count, bits.astype(np.float64) / 2.0**32), axis=-1
    )


def _ggx_half_vectors(roughness: float, samples: np.ndarray) -> np.ndarray:
    alpha = max(roughness * roughness, 0.0025)
    phi = 2.0 * np.pi * samples[:, 0]
    cosine = np.sqrt(
        (1.0 - samples[:, 1]) / (1.0 + (alpha * alpha - 1.0) * samples[:, 1])
    )
    sine = np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))
    return np.stack((sine * np.cos(phi), sine * np.sin(phi), cosine), axis=-1)


def precompute_ibl(
    pixels: np.ndarray, samples: int = 256
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return a cosine-convolved diffuse map and GGX reflection maps, all linear RGB."""
    rgb = np.asarray(pixels[..., :3], dtype=np.float64)
    height, width = rgb.shape[:2]
    source_dirs = _directions(width, height)
    solid_angle = (2.0 * np.pi / width) * (np.pi / height)
    solid_angle *= np.sin(np.pi * (np.arange(height) + 0.5) / height)[:, None]
    coefficients = np.einsum("hwc,hwb,hw->bc", rgb, _sh_basis(source_dirs), solid_angle)
    coefficients *= np.array([np.pi, *([2.0 * np.pi / 3.0] * 3), *([np.pi / 4.0] * 5)])[
        :, None
    ]
    normals = _directions(IBL_WIDTH, IBL_HEIGHT)
    irradiance = np.maximum(
        np.einsum("hwb,bc->hwc", _sh_basis(normals), coefficients), 0.0
    )

    dirs = normals.reshape(-1, 3)
    axis = np.tile(np.array([0.0, 0.0, 1.0]), (len(dirs), 1))
    axis[np.abs(dirs[:, 2]) > 0.99] = (0.0, 1.0, 0.0)
    tangent = np.cross(axis, dirs)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    bitangent = np.cross(dirs, tangent)
    sequence = _hammersley(samples)
    maps = [np.maximum(_sample_latlong(rgb, normals), 0.0).astype(np.float32)]
    for level in range(1, IBL_ROUGHNESS_LEVELS):
        half = _ggx_half_vectors(level / (IBL_ROUGHNESS_LEVELS - 1), sequence)
        filtered = np.empty((len(dirs), 3), dtype=np.float32)
        for start in range(0, len(dirs), 128):
            stop = min(start + 128, len(dirs))
            h = tangent[start:stop, None] * half[None, :, 0, None]
            h += bitangent[start:stop, None] * half[None, :, 1, None]
            h += dirs[start:stop, None] * half[None, :, 2, None]
            view_dot_h = np.sum(dirs[start:stop, None] * h, axis=-1)
            light = 2.0 * view_dot_h[..., None] * h - dirs[start:stop, None]
            cosine = np.maximum(np.sum(dirs[start:stop, None] * light, axis=-1), 0.0)
            radiance = _sample_latlong(rgb, light)
            filtered[start:stop] = (
                np.sum(radiance * cosine[..., None], axis=1)
                / np.maximum(cosine.sum(axis=1)[:, None], 1.0e-8)
            ).astype(np.float32)
        maps.append(filtered.reshape(IBL_HEIGHT, IBL_WIDTH, 3))
    return irradiance.astype(np.float32), maps


@lru_cache(maxsize=1)
def precompute_brdf_lut(size: int = 64, samples: int = 256) -> np.ndarray:
    """Split-sum GGX scale/bias for Schlick Fresnel, indexed by N.V and roughness."""
    sequence = _hammersley(samples)
    result = np.zeros((size, size, 4), dtype=np.float32)
    for row in range(size):
        roughness = (row + 0.5) / size
        half = _ggx_half_vectors(roughness, sequence)
        nh = half[:, 2]
        alpha2 = max(roughness**4, 0.0025**2)
        for col in range(size):
            nv = (col + 0.5) / size
            view = np.array([np.sqrt(1.0 - nv * nv), 0.0, nv])
            vh = np.maximum(half @ view, 0.0)
            light = 2.0 * vh[:, None] * half - view
            nl = np.maximum(light[:, 2], 0.0)
            gv = 2.0 * nv / (nv + np.sqrt(alpha2 + (1.0 - alpha2) * nv * nv))
            gl = (
                2.0
                * nl
                / np.maximum(nl + np.sqrt(alpha2 + (1.0 - alpha2) * nl * nl), 1.0e-8)
            )
            visibility = gv * gl * vh / np.maximum(nh * nv, 1.0e-8)
            fresnel = (1.0 - vh) ** 5
            result[row, col, 0] = np.mean(
                np.where(nl > 0.0, (1.0 - fresnel) * visibility, 0.0)
            )
            result[row, col, 1] = np.mean(np.where(nl > 0.0, fresnel * visibility, 0.0))
    return result
