#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Produce patched copies of the calibrated Robotiq 2F-85 USDs (gripper-only and UR5e+gripper).

Default edits (validated at parity on PhysX, see ISAACLAB_3_GRASP_HANDOFF.md section 11):
  1. left/right_inner_finger_knuckle_joint: body0/body1 (and localPos/localRot 0/1) swapped so the
     articulation parent is body0. Newton's USD importer rejects "reversed" joints; PhysX tolerated
     them. Swapping flips the joint sign, so the mimic gearing/offset flip with it.
  2. MassAPI on left/right_outer_knuckle (no collider -> PhysX had mass -1, inertia {1,1,1}).
Kept as authored: PhysxMimicJointAPI on the five passive joints (PhysX logs them as rejected for
lacking finite limits, but REMOVING them collapses the loop closure), and the +-inf limits.
Options that were tried and regress PhysX grasping: --limits finite (activates the mimics, whose
zero offsets fight the calibrated joint frames), --mimic strip (linkage collapses),
--mimic fix (corrected gearing on right_inner_knuckle_joint; still regresses).

Usage: fix_robotiq_usd.py --src <cloud cache dir with Robots/...> --dst <output root>
"""

from __future__ import annotations

import argparse
import os
import shutil

from pxr import Gf, Sdf, Usd, UsdPhysics

REL = "Robots/UniversalRobots"
FILES = {
    "2f85RobotiqGripperCalibrated": "robotiq_2f85_gripper_calibrated.usd",
    "Ur5e2f85RobotiqGripperCalibrated": "ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd",
}
REVERSED_JOINTS = ["left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"]
GEARING_FIX = {"right_inner_knuckle_joint": 1.0}
INF_JOINTS = ["right_inner_knuckle_joint", "left_inner_knuckle_joint", "right_inner_finger_knuckle_joint", "left_inner_finger_knuckle_joint"]
LIMIT_DEG = 100.0
MIMIC_MODE, SWAP, LIMITS, MASS, GRAVCOMP, LOOP_STRIP, ZERO_ROOT = "keep", True, False, True, False, False, False
ROOT_BODIES = {"robotiq_2f85_gripper_calibrated.usd": "robotiq_base_link", "ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd": "base_link"}
LOOP_JOINTS = ["left_inner_finger_joint", "right_inner_finger_joint"]
KNUCKLES = {"left_outer_knuckle": 0.012734640389680862, "right_outer_knuckle": 0.012734640389680862}


def find(stage: Usd.Stage, name: str) -> Usd.Prim:
    hits = [p for p in stage.Traverse() if p.GetName() == name]
    assert len(hits) == 1, (name, hits)
    return hits[0]


def swap_bodies(prim: Usd.Prim) -> None:
    j = UsdPhysics.Joint(prim)
    b0, b1 = list(j.GetBody0Rel().GetTargets()), list(j.GetBody1Rel().GetTargets())
    j.GetBody0Rel().SetTargets(b1)
    j.GetBody1Rel().SetTargets(b0)
    for a in ("localPos", "localRot"):
        a0, a1 = prim.GetAttribute(f"physics:{a}0"), prim.GetAttribute(f"physics:{a}1")
        v0, v1 = a0.Get(), a1.Get()
        a0.Set(v1)
        a1.Set(v0)
    g = prim.GetAttribute("physxMimicJoint:rotZ:gearing")
    if g and g.Get() is not None:
        g.Set(-float(g.Get()))
    o = prim.GetAttribute("physxMimicJoint:rotZ:offset")
    if o and o.Get() is not None:
        o.Set(-float(o.Get()))
    lo, hi = prim.GetAttribute("physics:lowerLimit"), prim.GetAttribute("physics:upperLimit")
    if lo.Get() is not None and hi.Get() is not None and abs(lo.Get()) != float("inf"):
        l, h = lo.Get(), hi.Get()
        lo.Set(-h)
        hi.Set(-l)
    print(f"    swapped body0/body1 on {prim.GetName()} -> body0={b1[0].name} body1={b0[0].name}")


def patch(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Sdf.Layer.FindOrOpen(src).Export(dst)
    stage = Usd.Stage.Open(dst)
    print(f"  {os.path.basename(dst)}")
    if SWAP:
        for n in REVERSED_JOINTS:
            swap_bodies(find(stage, n))
    if MIMIC_MODE == "keep":
        pass
    elif MIMIC_MODE == "fix":
        for n, g in GEARING_FIX.items():
            a = find(stage, n).GetAttribute("physxMimicJoint:rotZ:gearing")
            print(f"    {n} gearing {a.Get()} -> {g}")
            a.Set(g)
    else:
        # PhysxMimicJointAPI is a multi-apply schema; the physx schema plugin is not
        # loaded here, so GetAppliedSchemas() is empty -- edit the apiSchemas metadata
        # and remove the properties by name.
        from pxr import Sdf as _Sdf
        for prim in list(stage.Traverse()):
            props = [x.GetName() for x in prim.GetProperties() if x.GetName().startswith("physxMimicJoint")]
            if not props:
                continue
            if MIMIC_MODE == "strip_inner" and prim.GetName() not in INF_JOINTS:
                continue  # keep right_outer_knuckle_joint's mimic: the only one PhysX ever enforced
            for name in props:
                prim.RemoveProperty(name)
            lo = prim.GetMetadata("apiSchemas")
            if lo is not None:
                keep = [t for t in lo.GetAddedOrExplicitItems() if not str(t).startswith("PhysxMimicJointAPI")]
                prim.SetMetadata("apiSchemas", _Sdf.TokenListOp.CreateExplicit(keep))
            print(f"    stripped mimic from {prim.GetName()}")
    if LIMITS:
        for n in INF_JOINTS:
            p = find(stage, n)
            p.GetAttribute("physics:lowerLimit").Set(-LIMIT_DEG)
            p.GetAttribute("physics:upperLimit").Set(LIMIT_DEG)
        print(f"    finite limits +-{LIMIT_DEG} deg on {INF_JOINTS}")
    for n, m in (KNUCKLES.items() if MASS else []):
        p = find(stage, n)
        mass = UsdPhysics.MassAPI.Apply(p)
        mass.CreateMassAttr().Set(m)
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(2e-6, 2e-6, 2e-6))
        print(f"    MassAPI on {n}: mass={m} inertia=2e-6")
    if MASS_FROM:
        # Author the PhysX-effective mass properties (nominal, un-randomized dump of
        # `data.default_mass/default_inertia/body_com_pos_b`) so Newton's density-derived
        # values are replaced by what the recipe was tuned on.
        import json
        import numpy as np
        bodies = json.load(open(MASS_FROM))["bodies"]["robot"]
        for n, v in bodies.items():
            hits = [x for x in stage.Traverse() if x.GetName() == n and x.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(hits) != 1:
                continue
            p = hits[0]
            I = np.array(v["inertia"], dtype=np.float64).reshape(3, 3)
            I = 0.5 * (I + I.T)
            w, R = np.linalg.eigh(I)
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            q = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()  # Gf uses row-vector convention
            w = np.maximum(w, 1e-9)
            mass = UsdPhysics.MassAPI.Apply(p)
            mass.CreateMassAttr().Set(float(v["mass"]))
            mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(c) for c in v["com"]]))
            mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(c) for c in w]))
            mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(c) for c in q.GetImaginary()])))
            if p.GetAttribute("physics:density"):
                p.GetAttribute("physics:density").Clear()
            print(f"    MassAPI(from dump) on {n}: mass={v['mass']:.4f} Idiag={[round(float(c),6) for c in w]} com={[round(float(c),4) for c in v['com']]}")
    if LOOP_STRIP:
        for n in LOOP_JOINTS:
            prim = find(stage, n)
            stage.RemovePrim(prim.GetPath())
            print(f"    removed loop joint {n}")
    if ZERO_ROOT:
        from pxr import UsdGeom
        root_name = ROOT_BODIES[os.path.basename(dst)]
        xc = UsdGeom.XformCache()
        m_root = xc.GetLocalTransformation(find(stage, root_name))[0]
        inv = m_root.GetInverse()
        n = 0
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            xf = UsdGeom.Xformable(prim)
            m = xc.GetLocalTransformation(prim)[0]
            m_new = m * inv  # row-vector convention: apply inv after m
            for op in xf.GetOrderedXformOps():
                prim.RemoveProperty(op.GetOpName())
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(m_new)
            n += 1
        print(f"    zeroed root body {root_name}: rigidly re-based {n} bodies")
    if GRAVCOMP:
        n = 0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float).Set(1.0)
                n += 1
        print(f"    mjc:gravcomp=1 on {n} rigid bodies")
    stage.GetRootLayer().Save()


MASS_FROM = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="asset root containing Robots/UniversalRobots/...")
    ap.add_argument("--dst", required=True, help="output asset root")
    ap.add_argument("--mimic", choices=["strip", "strip_inner", "fix", "keep"], default="keep",
                    help="strip: remove PhysxMimicJointAPI (loop closure only, matches validated PhysX behaviour); "
                         "fix: keep mimics with the corrected gearing (regresses grasp hold under PhysX)")
    ap.add_argument("--swap", choices=["on", "off"], default="on", help="swap body0/body1 on the reversed joints")
    ap.add_argument("--limits", choices=["finite", "inf"], default="inf", help="limits on the four +-inf inner joints")
    ap.add_argument("--mass", choices=["on", "off"], default="on")
    ap.add_argument("--loop", choices=["keep", "strip"], default="keep",
                    help="strip: delete the two loop-closing joints (left/right_inner_finger_joint) so the gripper is a "
                         "pure tree driven by mimic couplings (Newton/MuJoCo style)")
    ap.add_argument("--zero_root", action="store_true",
                    help="rigidly transform all bodies so the root body's authored xform is identity (Newton keeps the "
                         "authored offset under the articulation root; PhysX overrides it with the root-pose write)")
    ap.add_argument("--gravcomp", action="store_true",
                    help="author mjc:gravcomp=1 on every rigid body (Newton/MuJoCo equivalent of PhysX disableGravity, which Newton ignores)")
    ap.add_argument("--mass_from", default=None,
                    help="JSON dump of per-body mass/inertia/com (jtran_mass_dump.py on PhysX, randomization disabled); "
                         "authored as MassAPI on every listed rigid body so Newton matches PhysX's effective values")
    args = ap.parse_args()
    global MIMIC_MODE, SWAP, LIMITS, MASS, GRAVCOMP, LOOP_STRIP, ZERO_ROOT, MASS_FROM
    MASS_FROM = args.mass_from
    LOOP_STRIP = args.loop == "strip"
    ZERO_ROOT = args.zero_root
    MIMIC_MODE, SWAP, LIMITS, MASS = args.mimic, args.swap == "on", args.limits == "finite", args.mass == "on"
    GRAVCOMP = args.gravcomp
    for d, f in FILES.items():
        src_dir, dst_dir = os.path.join(args.src, REL, d), os.path.join(args.dst, REL, d)
        patch(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        meta = os.path.join(src_dir, "metadata.yaml")
        if os.path.exists(meta):
            shutil.copy(meta, os.path.join(dst_dir, "metadata.yaml"))
            print(f"    copied metadata.yaml")
    print(f"\nPatched assets under {args.dst}. Use: export UWLAB_ROBOT_ASSETS_DIR={args.dst}")


if __name__ == "__main__":
    main()
