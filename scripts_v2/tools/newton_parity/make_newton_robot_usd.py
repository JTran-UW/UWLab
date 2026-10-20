# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write a Newton-loadable copy of the UR5e + Robotiq 2F-85 USD that is physically unchanged on PhysX.

Edits (all authored in the root layer):

1. ``*_inner_finger_knuckle_joint``: swap body0/body1 and rotate both joint frames 180 deg about an axis
   perpendicular to the joint axis, which keeps the joint-angle sign. ``*_inner_finger_joint`` gets
   ``excludeFromArticulation`` (the loop closers PhysX infers).
2. ``right_inner_knuckle_joint`` mimic gearing -1 -> +1 so every mimic rule agrees with the four-bar.
3. CollisionAPI/MeshCollisionAPI authored on Xforms (inside instances) move to their Mesh descendants.
4. Bodies listed in ``--mass_from`` get an explicit body-level MassAPI with the reference backend's values.
5. A world-anchored root joint with a non-identity child frame is rewritten so the parent frame carries the full pose
   (T0' = T0 * inv(T1), T1' = identity). Bodies stay put; Newton then reports the root body pose as the root pose.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/make_newton_robot_usd.py src.usd out.usd --mass_from physx.pt
"""

from __future__ import annotations

import argparse
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics

parser = argparse.ArgumentParser(description="Write the Newton-loadable UR5e + Robotiq USD.")
parser.add_argument("src", type=str)
parser.add_argument("out", type=str)
parser.add_argument("--mass_from", type=str, required=True, help="dump_model.py output of the reference backend.")
parser.add_argument(
    "--mass_bodies", nargs="*", default=["left_outer_knuckle", "right_outer_knuckle", "robotiq_base_link"]
)
parser.add_argument("--mimic_gearing", nargs="*", default=["right_inner_knuckle_joint=1"])
args = parser.parse_args()

_PERP_FLIP = {"X": (0.0, 0.0, 1.0, 0.0), "Y": (0.0, 0.0, 0.0, 1.0), "Z": (0.0, 1.0, 0.0, 0.0)}


def unreverse_joint(prim):
    joint = UsdPhysics.Joint(prim)
    b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
    p0, p1 = joint.GetLocalPos0Attr().Get(), joint.GetLocalPos1Attr().Get()
    r0, r1 = joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()
    w, x, y, z = _PERP_FLIP[UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()]
    flip = Gf.Quatf(w, Gf.Vec3f(x, y, z))
    joint.GetBody0Rel().SetTargets(b1)
    joint.GetBody1Rel().SetTargets(b0)
    joint.GetLocalPos0Attr().Set(p1)
    joint.GetLocalPos1Attr().Set(p0)
    joint.GetLocalRot0Attr().Set((r1 * flip).GetNormalized())
    joint.GetLocalRot1Attr().Set((r0 * flip).GetNormalized())


def normalize_root_joints(root):
    """Move a world-anchored joint's child frame into its parent frame (T0' = T0 * inv(T1), T1' = identity)."""
    changed = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetBody0Rel().GetTargets():
            continue
        p0, r0 = Gf.Vec3d(joint.GetLocalPos0Attr().Get()), Gf.Quatd(joint.GetLocalRot0Attr().Get())
        p1, r1 = Gf.Vec3d(joint.GetLocalPos1Attr().Get()), Gf.Quatd(joint.GetLocalRot1Attr().Get())
        if r1.GetReal() == 1.0 and p1 == Gf.Vec3d(0, 0, 0):
            continue
        r1_inv = r1.GetInverse()
        rot = (r0 * r1_inv).GetNormalized()
        pos = p0 + r0.Transform(r1_inv.Transform(-p1))
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(pos))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(rot))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        changed.append(prim.GetName())
    return changed


def conform_colliders(stage, root):
    targets = [
        p.GetPath()
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsA(UsdGeom.Gprim)
    ]
    for path in targets:
        prim = stage.GetPrimAtPath(path)
        while prim.IsInstanceProxy():
            ancestor = prim.GetParent()
            while ancestor.IsInstanceProxy():
                ancestor = ancestor.GetParent()
            ancestor.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
        approx = prim.GetAttribute("physics:approximation").Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(child)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(child)
                if approx:
                    mesh_api.CreateApproximationAttr().Set(approx)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
    return [str(p) for p in targets]


def reference_mass(path, bodies):
    ref = torch.load(path, weights_only=False)["articulations"]["robot"]
    props = {}
    for body in bodies:
        i = ref["body_names"].index(body)
        inertia = ref["body_inertia"][0, i].double().reshape(3, 3)
        evals, evecs = torch.linalg.eigh(0.5 * (inertia + inertia.T))
        if torch.det(evecs) < 0:
            evecs[:, 0] = -evecs[:, 0]
        rot = Gf.Matrix3d(*evecs.T.flatten().tolist()).ExtractRotation().GetQuat()
        props[body] = (float(ref["body_mass"][0, i]), ref["body_com_pose_b"][0, i, :3].tolist(), evals.tolist(), rot)
    return props


def main():
    stage = Usd.Stage.Open(args.src)
    root = stage.GetDefaultPrim()
    gearing = {k: float(v) for k, v in (item.split("=") for item in args.mimic_gearing)}
    mass = reference_mass(args.mass_from, args.mass_bodies)
    done = {"unreversed": [], "excluded": [], "gearing": [], "mass": []}
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name in ("left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"):
            unreverse_joint(prim)
            done["unreversed"].append(name)
        elif name in ("left_inner_finger_joint", "right_inner_finger_joint"):
            UsdPhysics.Joint(prim).CreateExcludeFromArticulationAttr().Set(True)
            done["excluded"].append(name)
        if name in gearing:
            # match by attribute: GetAppliedSchemas() hides PhysxSchema APIs when the plugin is not loaded
            for attr in prim.GetAttributes():
                if attr.GetName().startswith("physxMimicJoint:") and attr.GetName().endswith(":gearing"):
                    attr.Set(gearing[name])
                    done["gearing"].append(f"{name}={gearing[name]}")
        if name in mass and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m, com, diag, rot = mass[name]
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr().Set(m)
            api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com))
            api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*diag))
            api.CreatePrincipalAxesAttr().Set(Gf.Quatf(rot.GetReal(), Gf.Vec3f(rot.GetImaginary())))
            done["mass"].append(f"{name}={m:.4f}kg")
    done["conformed_colliders"] = conform_colliders(stage, root)
    done["normalized_root_joints"] = normalize_root_joints(root)
    for key, value in done.items():
        print(f"[make_newton_robot_usd] {key}: {value}")
    assert len(done["unreversed"]) == 2 and len(done["excluded"]) == 2, done
    assert len(done["gearing"]) == len(gearing) and len(done["mass"]) == len(mass), done
    stage.GetRootLayer().Export(args.out)


if __name__ == "__main__":
    main()
