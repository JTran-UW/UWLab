# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OmniResetImageAugV2: torchvision equivalence of each op, per-sample draws, paired obs/next_obs draws."""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import torch
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("rgb_aug", ROOT / "source/uwlab_rl/uwlab_rl/flashsac/rgb.py")


def load_aug_class():
    # rgb.py imports the training stack; pull only the class source to keep this test dependency-free
    src = (ROOT / "source/uwlab_rl/uwlab_rl/flashsac/rgb.py").read_text()
    start = src.index("class OmniResetImageAugV2(")
    end = src.index("\nclass ", start + 10)
    ns = {"torch": torch, "nn": torch.nn}
    exec(src[start:end], ns)
    return ns["OmniResetImageAugV2"]


AugV2 = load_aug_class()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def imgs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 6, 64, 64, generator=g) * 255).round().to(DEV)


def test_torchvision_equivalence():
    """Pin every uniform draw to 0.75 and the op order to b,c,s,h; compare to torchvision functional ops."""
    aug = AugV2(noise_std=0.0)
    x = imgs(1)
    real_rand = torch.rand
    with mock.patch("torch.rand", lambda *a, **k: torch.full(a if isinstance(a[0], int) else a[0], 0.75, device=k.get("device"))), \
         mock.patch("torch.randperm", lambda n: torch.arange(n)):
        out = aug(x)
    del real_rand
    f = lambda lo, hi: lo + 0.75 * (hi - lo)
    for cam in range(2):
        ref = x[:, 3 * cam:3 * cam + 3] / 255.0
        ref = TF.adjust_brightness(ref, f(0.9, 1.1))
        ref = TF.adjust_contrast(ref, f(0.9, 1.1))
        ref = TF.adjust_saturation(ref, f(0.9, 1.1))
        ref = TF.adjust_hue(ref, f(-0.05, 0.05))
        ref = TF.gaussian_blur(ref, [3, 3], [f(0.1, 0.6)] * 2)
        err = (out[:, 3 * cam:3 * cam + 3] / 255.0 - ref).abs().max().item()
        assert err < 2e-5, f"cam {cam}: max err {err}"


def test_per_sample_draws():
    """Identical images in one batch must come out different (v1 would make them identical)."""
    aug = AugV2(noise_std=0.0, grayscale_p=0.0)
    x = imgs(1).repeat(64, 1, 1, 1)
    out = aug(x)
    d = (out - out[:1]).abs().flatten(1).amax(1)
    assert (d[1:] > 1.0).float().mean() > 0.95, d


def test_paired_draws():
    """[obs; next_obs] with identical content: each transition's two halves get the same draw, different transitions differ."""
    aug = AugV2(noise_std=0.0, grayscale_p=0.5)
    x = imgs(1).repeat(64, 1, 1, 1)
    out = aug(torch.cat([x, x]), paired=True)
    a, b = out.chunk(2)
    assert (a - b).abs().max().item() < 1e-4
    assert (a - a[:1]).abs().flatten(1).amax(1)[1:].gt(1.0).float().mean() > 0.9


def test_paired_negative_control():
    """Mutant that ignores `paired` must fail the pairing check."""
    aug = AugV2(noise_std=0.0)
    x = imgs(1).repeat(64, 1, 1, 1)
    out = aug(torch.cat([x, x]), paired=False)
    a, b = out.chunk(2)
    assert (a - b).abs().max().item() > 1.0


def test_shape_range_dtype():
    aug = AugV2()
    x = imgs(32)
    out = aug(x, paired=True)
    assert out.shape == x.shape and out.dtype == torch.float32
    assert out.min() > -20 and out.max() < 275  # unclamped noise (std 2.55), as v1


def test_noncontiguous_cpu_input():
    """Camera slices of a (N, 6, H, W) stack are non-contiguous; must work on CPU and GPU."""
    for dev in ("cpu", DEV):
        x = imgs(8).to(dev)[:, :, ::1, :].transpose(2, 3).transpose(2, 3)
        out = AugV2()(x, paired=True)
        assert out.shape == x.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
