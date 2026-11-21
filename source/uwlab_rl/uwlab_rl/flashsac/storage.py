# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl_flashsac replay buffers with a storage dtype per observation group (upstream takes one for all groups).

Storing images as uint8 next to a float32 state keeps the state exact; every read casts back to float32.
"""

from __future__ import annotations

import torch
from rsl_rl_flashsac.storage import MemoryEfficientTorchUniformBuffer, TorchUniformBuffer


def parse_storage_dtypes(spec) -> torch.dtype | dict[str, torch.dtype] | None:
    """None, a dtype name for every group ("float16"), or per group ({"image": "uint8"} or "image=uint8,state=float32")."""
    if spec is None or isinstance(spec, torch.dtype):
        return spec
    if isinstance(spec, str) and "=" in spec:
        spec = dict(item.split("=", 1) for item in spec.split(","))
    if isinstance(spec, dict):
        return {group: getattr(torch, name) if isinstance(name, str) else name for group, name in spec.items()}
    return getattr(torch, spec)


class _PerGroupDtype:
    def _make_obs_storage(self, pin: bool) -> dict[str, torch.Tensor]:
        spec = self._obs_storage_dtype
        return {
            key: torch.empty(
                (self._max_length, *shape),
                dtype=spec.get(key, torch.float32) if isinstance(spec, dict) else (spec or torch.float32),
                device=self._device,
                pin_memory=pin,
            )
            for key, shape in self._obs_shapes.items()
        }


class PerGroupDtypeTorchUniformBuffer(_PerGroupDtype, TorchUniformBuffer):
    pass


class PerGroupDtypeMemoryEfficientBuffer(_PerGroupDtype, MemoryEfficientTorchUniformBuffer):
    pass
