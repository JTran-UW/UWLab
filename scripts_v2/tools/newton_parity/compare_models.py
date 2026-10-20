# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare two ``dump_model.py`` outputs by name and report PASS/FAIL per quantity.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/compare_models.py physx.pt newton.pt --report report.md
"""

from __future__ import annotations

import argparse
import math
import torch

parser = argparse.ArgumentParser(description="Compare two model dumps.")
parser.add_argument("ref", type=str, help="Reference dump (usually PhysX).")
parser.add_argument("test", type=str, help="Dump under test.")
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--atol", type=float, default=1e-6)
parser.add_argument("--pos_tol", type=float, default=1e-5, help="Pose position tolerance [m].")
parser.add_argument("--rot_tol", type=float, default=1e-4, help="Pose rotation tolerance [rad].")
parser.add_argument(
    "--dyn_rtol", type=float, default=1e-3, help="Mass matrix / gravity / Jacobian tolerance (relative to max |ref|)."
)
parser.add_argument("--top", type=int, default=6, help="Worst offenders listed per failing check.")
parser.add_argument("--report", type=str, default=None, help="Optional markdown report path.")
args = parser.parse_args()

rows: list[tuple[str, str, str]] = []


def record(name, ok, detail="", info=False):
    if isinstance(ok, torch.Tensor):
        ok = bool(ok)
    status = "SKIP" if ok is None else ("PASS" if ok else ("INFO" if info else "FAIL"))
    rows.append((status, name, detail))
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""))


def _num(x):
    return x.float() if isinstance(x, torch.Tensor) else None


UNBOUNDED = 1e6


def _close(a, b, rtol, atol):
    a = torch.where(a.abs() >= UNBOUNDED, torch.sign(a) * math.inf, a)
    b = torch.where(b.abs() >= UNBOUNDED, torch.sign(b) * math.inf, b)
    both_inf = torch.isinf(a) & torch.isinf(b) & (torch.sign(a) == torch.sign(b))
    err = (a - b).abs()
    err = torch.where(both_inf, torch.zeros_like(err), err)
    tol = atol + rtol * b.abs().nan_to_num(posinf=0.0, neginf=0.0)
    return err, err <= tol


def compare_per_name(label, names_ref, names_test, ref, test, rtol=None, atol=None, env=0, only=None, info=False):
    """Compare (num_envs, n, ...) tensors matched by element names."""
    rtol = args.rtol if rtol is None else rtol
    atol = args.atol if atol is None else atol
    a, b = _num(ref), _num(test)
    if a is None or b is None:
        record(label, None, f"missing (ref={a is not None}, test={b is not None})")
        return
    common = [n for n in names_ref if n in names_test and (only is None or n in only)]
    ia = torch.tensor([names_ref.index(n) for n in common])
    ib = torch.tensor([names_test.index(n) for n in common])
    a, b = a[env][ia], b[env][ib]
    if a.shape != b.shape:
        record(label, False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    err, ok = _close(a, b, rtol, atol)
    bad = [
        (common[i], a[i].flatten().tolist(), b[i].flatten().tolist())
        for i in range(len(common))
        if not bool(ok[i].all())
    ]
    detail = f"{len(common)} matched"
    if bad:
        detail += f", {len(bad)} differ: " + "; ".join(
            f"{n} ref={_fmt(x)} test={_fmt(y)}" for n, x, y in bad[: args.top]
        )
    record(label, not bad, detail, info=info)


def _fmt(v):
    v = v if isinstance(v, list) else [v]
    s = ",".join(f"{x:.4g}" for x in v[:9])
    return f"[{s}{',...' if len(v) > 9 else ''}]"


def _quat_angle(qa, qb):
    """Rotation angle of qa^-1 qb (xyzw); atan2 form stays accurate near zero."""
    qa = qa / qa.norm(dim=-1, keepdim=True)
    qb = qb / qb.norm(dim=-1, keepdim=True)
    xa, wa = qa[..., :3], qa[..., 3:]
    xb, wb = qb[..., :3], qb[..., 3:]
    w = wa * wb + (xa * xb).sum(-1, keepdim=True)
    xyz = wa * xb - wb * xa - torch.cross(xa, xb, dim=-1)
    return 2.0 * torch.atan2(xyz.norm(dim=-1), w.abs().squeeze(-1))


def compare_poses(label, names_ref, names_test, ref, test, env=0, position_only=False, info=False):
    common = [n for n in names_ref if n in names_test]
    a = ref[env][[names_ref.index(n) for n in common]].double()
    b = test[env][[names_test.index(n) for n in common]].double()
    pos_err = (a[:, :3] - b[:, :3]).norm(dim=-1)
    rot_err = _quat_angle(a[:, 3:7], b[:, 3:7])
    if position_only:
        rot_err = torch.zeros_like(rot_err)
    bad = [i for i in range(len(common)) if pos_err[i] > args.pos_tol or rot_err[i] > args.rot_tol]
    worst = sorted(bad, key=lambda i: -(pos_err[i] / args.pos_tol + rot_err[i] / args.rot_tol))
    detail = f"max pos {pos_err.max():.2e} m, max rot {math.degrees(rot_err.max()):.3g} deg"
    if bad:
        detail += f"; {len(bad)} bodies off: " + "; ".join(
            f"{common[i]} ({pos_err[i] * 1e3:.3g} mm, {math.degrees(rot_err[i]):.3g} deg)" for i in worst[: args.top]
        )
    record(label, not bad, detail, info=info)


def compare_matrix(label, ref, test, rows_ref, rows_test, cols_ref=None, cols_test=None, env=0):
    """Compare (num_envs, R, C) by row/column names; tolerance relative to max |ref|."""
    a, b = _num(ref), _num(test)
    if a is None or b is None:
        record(label, None, f"missing (ref={a is not None}, test={b is not None})")
        return
    cols_ref, cols_test = cols_ref or rows_ref, cols_test or rows_test
    rc = [n for n in rows_ref if n in rows_test]
    cc = [n for n in cols_ref if n in cols_test]
    a = a[env][[rows_ref.index(n) for n in rc]][:, [cols_ref.index(n) for n in cc]]
    b = b[env][[rows_test.index(n) for n in rc]][:, [cols_test.index(n) for n in cc]]
    scale = a.abs().max().clamp(min=1e-9)
    err = (a - b).abs() / scale
    worst = torch.topk(err.flatten(), min(args.top, err.numel()))
    entries = []
    for e, idx in zip(worst.values.tolist(), worst.indices.tolist()):
        r, c = divmod(idx, len(cc))
        if e > args.dyn_rtol:
            entries.append(f"[{rc[r]},{cc[c]}] ref={a[r, c]:.4g} test={b[r, c]:.4g}")
    detail = f"max rel err {err.max():.2e} (scale {scale:.3g})"
    if entries:
        detail += "; " + "; ".join(entries)
    record(label, err.max() <= args.dyn_rtol, detail)


def compare_articulation(name, A, B):
    record(
        f"{name}.is_fixed_base",
        A["is_fixed_base"] == B["is_fixed_base"],
        f"{A['is_fixed_base']} vs {B['is_fixed_base']}",
    )
    ja, jb = A["joint_names"], B["joint_names"]
    record(
        f"{name}.joint_set",
        set(ja) == set(jb),
        f"ref-only={sorted(set(ja) - set(jb))} test-only={sorted(set(jb) - set(ja))}",
    )
    record(f"{name}.joint_order", ja == jb, "" if ja == jb else f"ref={ja} test={jb}", info=True)
    ba, bb = A["body_names"], B["body_names"]
    record(
        f"{name}.body_set",
        set(ba) == set(bb),
        f"ref-only={sorted(set(ba) - set(bb))} test-only={sorted(set(bb) - set(ba))}",
    )
    record(f"{name}.body_order", ba == bb, "" if ba == bb else f"ref={ba} test={bb}", info=True)
    for key in ["body_mass", "body_inertia"]:
        compare_per_name(f"{name}.{key}", ba, bb, A[key], B[key])
    compare_per_name(
        f"{name}.body_com_pos_b",
        ba,
        bb,
        _slice(A["body_com_pose_b"], 0, 3),
        _slice(B["body_com_pose_b"], 0, 3),
        atol=args.pos_tol,
    )
    for key in [
        "joint_stiffness",
        "joint_damping",
        "joint_armature",
        "joint_friction_coeff",
        "joint_dynamic_friction_coeff",
        "joint_viscous_friction_coeff",
        "joint_pos_limits",
        "joint_vel_limits",
        "default_joint_pos",
    ]:
        compare_per_name(f"{name}.{key}", ja, jb, A[key], B[key])
    actuated = {j for act in A["actuators"].values() for j in act["joint_names"]}
    compare_per_name(
        f"{name}.joint_effort_limits[actuated]",
        ja,
        jb,
        A["joint_effort_limits"],
        B["joint_effort_limits"],
        only=actuated,
    )
    compare_per_name(
        f"{name}.joint_effort_limits[passive]",
        ja,
        jb,
        A["joint_effort_limits"],
        B["joint_effort_limits"],
        only=set(ja) - actuated,
        info=True,
    )
    for act, cfg in A["actuators"].items():
        other = B["actuators"].get(act)
        if other is None:
            record(f"{name}.actuator[{act}]", False, "missing in test")
            continue
        same = cfg["class"] == other["class"] and cfg["joint_names"] == other["joint_names"]
        record(
            f"{name}.actuator[{act}].layout",
            same,
            f"{cfg['class']}{cfg['joint_names']} vs {other['class']}{other['joint_names']}",
        )
        for key in [
            "stiffness",
            "damping",
            "armature",
            "effort_limit",
            "effort_limit_sim",
            "velocity_limit",
            "velocity_limit_sim",
        ]:
            if isinstance(cfg[key], torch.Tensor) and isinstance(other[key], torch.Tensor):
                compare_per_name(
                    f"{name}.actuator[{act}].{key}", cfg["joint_names"], other["joint_names"], cfg[key], other[key]
                )
    for label, dump in (("ref", A), ("test", B)):
        sweep = dump.get("gripper_sweep")
        if not sweep:
            continue
        for loop, res in sweep["residual"].items():
            pos = max(r[0] for r in res)
            ang = max(r[1] for r in res)
            per = ", ".join(f"{t:.2f}:{r[0] * 1e3:.2f}mm" for t, r in zip(sweep["theta"], res))
            record(
                f"{name}.mimic_loop_consistency[{label}].{loop}",
                pos <= 1e-4 and ang <= 1e-3,
                f"max {pos * 1e3:.3f} mm / {math.degrees(ang):.3f} deg over finger_joint [{per}]",
            )
    for k, (ca, cb) in enumerate(zip(A["configs"], B["configs"])):
        tag = f"{name}.config[{k}]"
        compare_per_name(
            f"{tag}.joint_pos_readback", ja, jb, ca["joint_pos_readback"], cb["joint_pos_readback"], atol=1e-5
        )
        compare_poses(f"{tag}.body_link_pose", ba, bb, ca["body_link_pose_r"], cb["body_link_pose_r"])
        compare_poses(f"{tag}.body_com_pos", ba, bb, ca["body_com_pose_r"], cb["body_com_pose_r"], position_only=True)
        compare_matrix(f"{tag}.mass_matrix", ca["mass_matrix"], cb["mass_matrix"], ja, jb)
        g_a, g_b = ca["gravity_compensation_forces"], cb["gravity_compensation_forces"]
        if g_a is not None and g_b is not None:
            compare_matrix(f"{tag}.gravity_compensation", g_a[..., None], g_b[..., None], ja, jb, ["g"], ["g"])
        else:
            record(f"{tag}.gravity_compensation", None, "missing")
        _compare_jacobian(tag, A, B, ca["body_link_jacobian_w"], cb["body_link_jacobian_w"])


def _compare_jacobian(tag, A, B, Ja, Jb):
    if Ja is None or Jb is None:
        record(f"{tag}.jacobian", None, "missing")
        return
    ba, bb, ja, jb = A["body_names"], B["body_names"], A["joint_names"], B["joint_names"]
    # fixed-base PhysX omits the root body row; align by the trailing bodies
    ba_j = ba[len(ba) - Ja.shape[1] :]
    bb_j = bb[len(bb) - Jb.shape[1] :]
    ja_j = ja[len(ja) - Ja.shape[-1] :] if Ja.shape[-1] <= len(ja) else None
    jb_j = jb[len(jb) - Jb.shape[-1] :] if Jb.shape[-1] <= len(jb) else None
    if ja_j is None or jb_j is None:
        record(f"{tag}.jacobian", None, f"dof columns {Ja.shape[-1]} / {Jb.shape[-1]} exceed joint count")
        return
    common = [n for n in ba_j if n in bb_j]
    a = Ja[0][[ba_j.index(n) for n in common]].flatten(0, 1)
    b = Jb[0][[bb_j.index(n) for n in common]].flatten(0, 1)
    rows = [f"{n}:{c}" for n in common for c in ("vx", "vy", "vz", "wx", "wy", "wz")]
    compare_matrix(f"{tag}.jacobian", a[None], b[None], rows, rows, ja_j, jb_j)


def _slice(x, lo, hi):
    return x[..., lo:hi] if isinstance(x, torch.Tensor) else None


def compare_rigid(name, A, B, kinematic=False):
    for key in ["body_mass", "body_inertia"]:
        compare_per_name(f"{name}.{key}", A["body_names"], B["body_names"], A[key], B[key])
    if A["body_com_pose_b"] is not None and B["body_com_pose_b"] is not None:
        compare_poses(
            f"{name}.body_com_pos_b",
            A["body_names"],
            B["body_names"],
            A["body_com_pose_b"],
            B["body_com_pose_b"],
            position_only=True,
            info=kinematic,
        )
    compare_poses(
        f"{name}.default_root_pose",
        ["root"],
        ["root"],
        A["default_root_pose"][:, None],
        B["default_root_pose"][:, None],
    )
    compare_poses(
        f"{name}.root_link_pose_w_after_reset",
        ["root"],
        ["root"],
        A["root_link_pose_w"][:, None],
        B["root_link_pose_w"][:, None],
        info=True,
    )


def _is_kinematic(dump, name):
    prim = {"insertive_object": "/InsertiveObject", "receptive_object": "/ReceptiveObject", "table": "/Table"}.get(name)
    for e in dump["usd"].get("masses", []):
        if prim and e["path"] == prim:
            return bool(e["kinematic"])
    return False


def main():
    ref = torch.load(args.ref, weights_only=False)
    test = torch.load(args.test, weights_only=False)
    print(f"ref:  {ref['meta']}\ntest: {test['meta']}\n")
    for name, A in ref["articulations"].items():
        if name in test["articulations"]:
            compare_articulation(name, A, test["articulations"][name])
        else:
            record(f"{name}", False, "articulation missing in test")
    for name, A in ref["rigid_objects"].items():
        if name in test["rigid_objects"]:
            compare_rigid(name, A, test["rigid_objects"][name], kinematic=_is_kinematic(ref, name))
        else:
            record(f"{name}", False, "rigid object missing in test")
    n_fail = sum(r[0] == "FAIL" for r in rows)
    n_pass = sum(r[0] == "PASS" for r in rows)
    n_skip = sum(r[0] == "SKIP" for r in rows)
    n_info = sum(r[0] == "INFO" for r in rows)
    print(f"\n{n_pass} PASS, {n_fail} FAIL, {n_info} INFO, {n_skip} SKIP")
    if args.report:
        with open(args.report, "w") as f:
            f.write(
                f"# Model comparison\n\n- ref: `{args.ref}` {ref['meta']}\n- test: `{args.test}` {test['meta']}\n\n"
            )
            f.write(
                f"**{n_pass} PASS, {n_fail} FAIL, {n_info} INFO, {n_skip} SKIP**\n\n| status | check | detail"
                " |\n|---|---|---|\n"
            )
            for status, check, detail in rows:
                f.write(f"| {status} | `{check}` | {detail.replace('|', '/')} |\n")


if __name__ == "__main__":
    main()
