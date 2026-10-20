# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD file spawners."""

from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


def spawn_from_usd_with_collision_approximation(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a USD file, then set ``physics:approximation`` on every collision mesh beneath it.

    Instanced collision meshes are de-instanced so the override can be authored.
    """
    from pxr import Usd, UsdPhysics

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root in sim_utils.find_matching_prims(prim_path) or [prim]:
        stage = root.GetStage()
        paths = [
            p.GetPath()
            for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.MeshCollisionAPI)
        ]
        for path in paths:
            while stage.GetPrimAtPath(path).IsInstanceProxy():
                ancestor = stage.GetPrimAtPath(path).GetParent()
                while ancestor.IsInstanceProxy():
                    ancestor = ancestor.GetParent()
                ancestor.SetInstanceable(False)
            UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(path)).CreateApproximationAttr().Set(
                cfg.collision_approximation
            )
    return prim


@configclass
class UsdFileWithCollisionApproximationCfg(sim_utils.UsdFileCfg):
    """:class:`~isaaclab.sim.UsdFileCfg` that overrides the mesh-collision approximation of the asset."""

    func: Callable = spawn_from_usd_with_collision_approximation

    collision_approximation: str = "convexHull"
    """``physics:approximation`` token written to every collision mesh (e.g. ``"convexHull"``)."""
