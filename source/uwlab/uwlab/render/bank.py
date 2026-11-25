# SPDX-License-Identifier: BSD-3-Clause
"""Texture and environment-map banks for appearance randomization: many source images reduced to render size once
and cached on disk, so that large sets (hundreds of 4k maps) load in seconds and fit GPU memory."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

TEXTURE_SIZES = (256, 128, 64)
"""Square mip levels kept per bank texture; the randomizer picks one by the texture's projected scale."""
ENVIRONMENT_SIZE = (128, 256)
"""Lat-long (height, width) of bank environment maps: lighting only needs their low frequencies."""


def _cache_file(kind: str, paths, extra) -> Path:
    key = hashlib.sha1(repr((kind, extra, [(str(p), os.path.getmtime(p), os.path.getsize(p)) for p in paths])).encode())
    root = Path(os.environ.get("UWLAB_RENDER_BANK_CACHE", Path.home() / ".cache" / "uwlab" / "render_bank"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{kind}_{key.hexdigest()[:16]}.npz"


def _save(path: Path, **arrays) -> None:
    scratch = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(scratch, **arrays)
    os.replace(scratch, path)


def load_texture_bank(paths, sizes=TEXTURE_SIZES) -> np.ndarray:
    """sRGB albedo images as ``(count, levels)`` object array of ``(size, size, 4)`` uint8 arrays."""
    from PIL import Image

    paths = [str(p) for p in paths]
    cache = _cache_file("textures", paths, tuple(sizes))
    if cache.exists():
        with np.load(cache) as data:
            levels = [data[f"level{i}"] for i in range(len(sizes))]
    else:
        levels = [np.zeros((len(paths), size, size, 4), dtype=np.uint8) for size in sizes]
        for index, path in enumerate(paths):
            image = Image.open(path).convert("RGBA")
            for level, size in zip(levels, sizes):
                level[index] = np.asarray(image.resize((size, size), Image.BOX if image.width >= size else Image.BILINEAR))
        _save(cache, **{f"level{i}": level for i, level in enumerate(levels)})
    bank = np.empty((len(paths), len(sizes)), dtype=object)
    for index in range(len(paths)):
        for i, level in enumerate(levels):
            bank[index, i] = level[index]
    return bank


def _read_hdr(path: str) -> np.ndarray:
    """Linear RGB float image from a Radiance .hdr / OpenEXR file (or an sRGB image as a fallback)."""
    if path.lower().endswith(".npy"):
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)[..., :3]
    try:
        import cv2

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is not None and image.dtype != np.uint8:
            return np.asarray(image[..., 2::-1], dtype=np.float32)
    except ImportError:
        pass
    import imageio.v3 as iio

    image = np.asarray(iio.imread(path))
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        image = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    return np.asarray(image[..., :3], dtype=np.float32)


def reduce_environment(pixels: np.ndarray, size=ENVIRONMENT_SIZE) -> np.ndarray:
    """Area-average a lat-long map to ``size`` (energy preserving, so suns keep their flux)."""
    height, width = size
    pixels = np.nan_to_num(np.asarray(pixels[..., :3], dtype=np.float64), nan=0.0, posinf=0.0).clip(min=0.0)
    rows = np.minimum(np.arange(pixels.shape[0]) * height // pixels.shape[0], height - 1)
    cols = np.minimum(np.arange(pixels.shape[1]) * width // pixels.shape[1], width - 1)
    total = np.zeros((height, width, 3))
    np.add.at(total, (rows[:, None], cols[None, :]), pixels)
    count = np.zeros((height, width, 1))
    np.add.at(count, (rows[:, None], cols[None, :]), 1.0)
    return (total / np.maximum(count, 1.0)).astype(np.float32)


def load_environment_bank(paths, size=ENVIRONMENT_SIZE) -> np.ndarray:
    """Lat-long environment maps as ``(count, height, width, 3)`` float32 linear radiance."""
    paths = [str(p) for p in paths]
    cache = _cache_file("environments", paths, tuple(size))
    if cache.exists():
        with np.load(cache) as data:
            return data["maps"]
    maps = np.stack([reduce_environment(_read_hdr(path), size) for path in paths])
    _save(cache, maps=maps)
    return maps
