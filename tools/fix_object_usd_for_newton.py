#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Patched copies of object USDs for the Newton backend.

``physics:approximation = sdf`` is not mapped by Newton's importer (raw triangle-mesh shapes).
Convex objects get ``convexHull``; non-convex ones get an explicit CoACD decomposition baked in
as one ``convexHull`` collider prim per piece (Newton's own convexDecomposition path silently
falls back to a single hull, which would seal the peghole).

Usage: fix_object_usd_for_newton.py --src <asset root> --dst <asset root>
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

OBJECTS = {
    "Props/Custom/Peg/peg.usd": "convexHull",
    "Props/Custom/PegHole/peg_hole.usd": "coacd",
}


def triangles(mesh: UsdGeom.Mesh) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    tris, k = [], 0
    for c in counts:
        face = idx[k : k + c]
        for j in range(1, c - 1):
            tris.append([face[0], face[j], face[j + 1]])
        k += c
    return pts, np.array(tris, dtype=np.int64)


def bake_coacd(stage: Usd.Stage, prim: Usd.Prim, threshold: float = 0.05) -> int:
    import coacd

    mesh = UsdGeom.Mesh(prim)
    pts, tris = triangles(mesh)
    parts = coacd.run_coacd(coacd.Mesh(pts, tris), threshold=threshold)
    parent = prim.GetParent()
    xform = UsdGeom.Xformable(prim).GetLocalTransformation()
    for i, (v, f) in enumerate(parts):
        hull = UsdGeom.Mesh.Define(stage, parent.GetPath().AppendChild(f"hull_{i:03d}"))
        hull.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in v]))
        hull.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
        hull.CreateFaceVertexIndicesAttr(Vt.IntArray([int(x) for x in np.asarray(f).reshape(-1)]))
        hull.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision only, not rendered
        UsdGeom.Xformable(hull).AddTransformOp().Set(xform)
        UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(hull.GetPrim()).CreateApproximationAttr().Set("convexHull")
    # keep the original mesh for its material/mass but stop it from colliding
    prim.GetAttribute("physics:collisionEnabled").Set(False) if prim.GetAttribute("physics:collisionEnabled") else UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    return len(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    for rel, mode in OBJECTS.items():
        src, dst = os.path.join(args.src, rel), os.path.join(args.dst, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Sdf.Layer.FindOrOpen(src).Export(dst)
        stage = Usd.Stage.Open(dst)
        for prim in list(stage.Traverse()):
            a = prim.GetAttribute("physics:approximation")
            if not (a and a.Get() == "sdf"):
                continue
            if mode == "coacd":
                n = bake_coacd(stage, prim)
                print(f"  {rel}: {prim.GetPath()} sdf -> {n} baked convex hulls")
            else:
                a.Set(mode)
                print(f"  {rel}: {prim.GetPath()} sdf -> {mode}")
        stage.GetRootLayer().Save()
        meta = os.path.join(os.path.dirname(src), "metadata.yaml")
        if os.path.exists(meta):
            shutil.copy(meta, os.path.join(os.path.dirname(dst), "metadata.yaml"))


if __name__ == "__main__":
    main()
