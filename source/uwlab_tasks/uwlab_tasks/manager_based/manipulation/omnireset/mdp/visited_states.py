# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ring buffer of visited scene states (pure torch) for resetting envs to states other envs visited."""

from __future__ import annotations

import torch


def leaf_paths(tree: dict, prefix: tuple = ()) -> list[tuple]:
    """Paths to the tensor leaves of a nested dict, in insertion order."""
    paths = []
    for key, value in tree.items():
        if isinstance(value, dict):
            paths += leaf_paths(value, prefix + (key,))
        else:
            paths.append(prefix + (key,))
    return paths


def get_path(tree: dict, path: tuple):
    for key in path:
        tree = tree[key]
    return tree


def build_tree(paths: list[tuple], values: list) -> dict:
    tree: dict = {}
    for path, value in zip(paths, values):
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return tree


class VisitedStateBuffer:
    """Ring of scene states laid out like ``template`` (a batch-first nested dict, e.g. ``scene.get_state`` restricted
    to the reset datasets' leaves). Every leaf shares one slot index, so a sampled state is one consistent snapshot."""

    def __init__(self, capacity: int, template: dict, device: str | torch.device):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.paths = leaf_paths(template)
        self.storage = [
            torch.zeros((self.capacity, *get_path(template, p).shape[1:]), dtype=get_path(template, p).dtype, device=self.device)
            for p in self.paths
        ]
        self.src_env = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.src_step = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.next, self.filled = 0, 0

    def push(self, state: dict, mask: torch.Tensor, env_ids: torch.Tensor, steps: torch.Tensor) -> int:
        """Append the ``mask`` rows of ``state`` (with their env ids / episode steps); returns how many were stored."""
        rows = mask.nonzero().flatten()
        n = rows.numel()
        if n == 0:
            return 0
        if n > self.capacity:  # only the newest capacity rows survive
            rows, n = rows[-self.capacity :], self.capacity
        slots = (self.next + torch.arange(n, device=self.device)) % self.capacity
        for path, buf in zip(self.paths, self.storage):
            buf[slots] = get_path(state, path)[rows].to(buf)
        self.src_env[slots] = env_ids[rows].to(self.device, torch.long)
        self.src_step[slots] = steps[rows].to(self.device, torch.long)
        self.next = int((self.next + n) % self.capacity)
        self.filled = min(self.filled + n, self.capacity)
        return n

    def sample(self, n: int, generator: torch.Generator | None = None) -> tuple[dict, torch.Tensor]:
        """``n`` states drawn uniformly (with replacement) over the filled slots, and their slot indices."""
        assert self.filled > 0, "sample() on an empty VisitedStateBuffer"
        idx = torch.randint(0, self.filled, (n,), device=self.device, generator=generator)
        return self.at(idx), idx

    def at(self, idx: torch.Tensor) -> dict:
        return build_tree(self.paths, [buf[idx] for buf in self.storage])
