# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare two ``mdp_probe.py`` outputs and report PASS/FAIL per MDP quantity.

Joint/body quantities are matched by name. Observations are compared element-wise in the order the policy
sees them. The reset check is absolute: each backend's post-reset state must equal the forced dataset entry,
read in the reference backend's joint order (the order the datasets were recorded in).

.. code-block:: bash

    python scripts_v2/tools/newton_parity/compare_mdp.py physx_mdp.pt newton_mdp.pt --report mdp.md
"""

from __future__ import annotations

import argparse
import torch

parser = argparse.ArgumentParser(description="Compare two MDP probe dumps.")
parser.add_argument("ref", type=str)
parser.add_argument("test", type=str)
parser.add_argument("--atol", type=float, default=1e-4)
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--top", type=int, default=5)
parser.add_argument("--report", type=str, default=None)
parser.add_argument(
    "--nominal_ref", type=str, default=None, help="dump_model.py output of ref backend (event range checks)."
)
parser.add_argument("--nominal_test", type=str, default=None, help="dump_model.py output of test backend.")
args = parser.parse_args()

rows: list[tuple[str, str, str]] = []


def record(name, ok, detail="", info=False):
    ok = bool(ok) if isinstance(ok, torch.Tensor) else ok
    status = "SKIP" if ok is None else ("PASS" if ok else ("INFO" if info else "FAIL"))
    rows.append((status, name, detail))
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""))


def close(label, a, b, labels=None, atol=None, rtol=None, info=False):
    """Compare (num_envs, ...) tensors; labels name the trailing axis."""
    atol = args.atol if atol is None else atol
    rtol = args.rtol if rtol is None else rtol
    if a is None or b is None:
        record(label, None, f"missing (ref={a is not None}, test={b is not None})")
        return
    a, b = a.double(), b.double()
    if a.shape != b.shape:
        record(label, False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}", info=info)
        return
    err = (a - b).abs()
    bad = err > atol + rtol * b.abs()
    if a.dim() == 1:
        a, b, err, bad = a[:, None], b[:, None], err[:, None], bad[:, None]
    flat = bad.reshape(bad.shape[0], -1)
    n_env = int(flat.any(1).sum())
    detail = f"max abs err {err.max():.3g}"
    if n_env:
        cols = flat.any(0).nonzero().flatten().tolist()
        width = flat.shape[1]
        names = labels if labels is not None and len(labels) == width else [str(c) for c in range(width)]
        e0 = int(flat.any(1).nonzero()[0])
        a0, b0 = a.reshape(a.shape[0], -1)[e0], b.reshape(b.shape[0], -1)[e0]
        detail += (
            f"; {n_env}/{flat.shape[0]} envs, {len(cols)}/{width} elements differ: "
            + "; ".join(f"{names[c]} ref={a0[c]:.4g} test={b0[c]:.4g}" for c in cols[: args.top])
            + f" (env {e0})"
        )
    record(label, n_env == 0, detail, info=info)


def by_name(x, names, order):
    idx = [names.index(n) for n in order]
    return x[:, idx] if x.dim() == 2 else x[:, idx, ...]


def compare_state(tag, A, B, info=False, physics_info=False, mass_info=False):
    for name, ra in A["articulations"].items():
        rb = B["articulations"][name]
        ja, jb, ba, bb = ra["joint_names"], rb["joint_names"], ra["body_names"], rb["body_names"]
        record(f"{tag}.{name}.public_joint_order", ja == jb, "" if ja == jb else f"ref={ja} test={jb}", info=True)
        record(f"{tag}.{name}.public_body_order", ba == bb, "" if ba == bb else f"ref={ba} test={bb}", info=True)
        for key in (
            "joint_pos",
            "joint_vel",
            "joint_pos_target",
            "joint_vel_target",
            "joint_stiffness",
            "joint_damping",
        ):
            phys = key in ("joint_pos", "joint_vel") or (tag == "after_step")
            close(f"{tag}.{name}.{key}", ra[key], by_name(rb[key], jb, ja), ja, info=info or (physics_info and phys))
        close(f"{tag}.{name}.root_link_pose", ra["root_link_pose"], rb["root_link_pose"], atol=1e-5)
        close(
            f"{tag}.{name}.body_link_pos",
            ra["body_link_pose"][..., :3],
            by_name(rb["body_link_pose"], bb, ba)[..., :3],
            [f"{n}.{c}" for n in ba for c in "xyz"],
            atol=1e-5,
            info=physics_info,
        )
        close(
            f"{tag}.{name}.base_link_pose",
            ra["body_link_pose"][:, ba.index("base_link")],
            rb["body_link_pose"][:, bb.index("base_link")],
            atol=1e-5,
        )
        close(f"{tag}.{name}.body_mass", ra["body_mass"], by_name(rb["body_mass"], bb, ba), ba, info=info or mass_info)
    for name, oa in A["rigid_objects"].items():
        ob = B["rigid_objects"][name]
        close(f"{tag}.{name}.root_link_pose", oa["root_link_pose"], ob["root_link_pose"], atol=1e-5, info=physics_info)
        close(f"{tag}.{name}.body_mass", oa["body_mass"], ob["body_mass"], info=info or mass_info)


def compare_reset_fidelity(label, dump, ref_joint_order):
    """Post-reset state must equal the forced dataset entry (absolute check)."""
    st = dump["after_reset"]
    robot = st["articulations"]["robot"]
    names = robot["joint_names"]
    for task, (env_ids, entry) in sorted(dump["forced"]["written"].items()):
        ds = entry["articulation"]["robot"]
        expected = ds["joint_position"]
        got = by_name(robot["joint_pos"][env_ids], names, ref_joint_order)
        close(f"reset_fidelity[{label}].task{task}.robot.joint_pos", got, expected, ref_joint_order, atol=1e-5)
        close(
            f"reset_fidelity[{label}].task{task}.robot.root_pose",
            robot["root_link_pose"][env_ids],
            ds["root_pose"],
            atol=1e-5,
        )
        for obj, od in entry["rigid_object"].items():
            close(
                f"reset_fidelity[{label}].task{task}.{obj}.root_pose",
                st["rigid_objects"][obj]["root_link_pose"][env_ids],
                od["root_pose"],
                atol=1e-5,
            )


def compare_materials(A, B):
    """PhysX (static, dynamic, restitution) per shape vs Newton mu per shape, as per-env sorted multisets."""
    ma, mb = A["after_startup"]["materials"], B["after_startup"]["materials"]
    if "shape_label" not in mb:
        return
    labels = mb["shape_label"]
    prefix = {
        "robot": "/Robot/",
        "insertive_object": "/InsertiveObject/",
        "receptive_object": "/ReceptiveObject/",
        "table": "/Table/",
    }
    for asset, pat in prefix.items():
        if asset not in ma:
            continue
        va = ma[asset]["values"]
        n_env = va.shape[0]
        mu = mb["shape_material_mu"]
        per_env = []
        for e in range(n_env):
            idx = [i for i, l in enumerate(labels) if f"/env_{e}{pat}" in l]
            per_env.append(mu[idx])
        if len({len(p) for p in per_env}) != 1:
            record(f"startup.material[{asset}]", False, "shape count varies across envs")
            continue
        mb_t = torch.stack(per_env)
        record(
            f"startup.material[{asset}].shape_count", va.shape[1] == mb_t.shape[1], f"{va.shape[1]} vs {mb_t.shape[1]}"
        )
        if va.shape[1] == mb_t.shape[1]:
            close(f"startup.material[{asset}].static_vs_mu(sorted)", va[..., 0].sort(1).values, mb_t.sort(1).values)
            close(
                f"startup.material[{asset}].dynamic_vs_mu(sorted)",
                va[..., 1].sort(1).values,
                mb_t.sort(1).values,
                info=True,
            )


MASS_EVENTS = {
    "robot": ("scale", 0.7, 1.3),
    "insertive_object": ("abs", 0.02, 0.2),
    "receptive_object": ("scale", 0.5, 1.5),
    "table": ("scale", 0.5, 1.5),
}
STATIC_FRICTION = {
    "robot": (0.3, 1.2),
    "insertive_object": (1.0, 2.0),
    "receptive_object": (0.2, 0.6),
    "table": (0.3, 0.6),
}
DYNAMIC_FRICTION = {
    "robot": (0.2, 1.0),
    "insertive_object": (0.9, 1.9),
    "receptive_object": (0.15, 0.5),
    "table": (0.2, 0.5),
}


def check_event_ranges(label, dump, nominal):
    """Startup randomization written values lie in the configured ranges (RNG-independent)."""
    st = dump["after_startup"]["state"]
    for asset, (op, lo, hi) in MASS_EVENTS.items():
        group = "articulations" if asset in st["articulations"] else "rigid_objects"
        mass = st[group][asset]["body_mass"].double()
        if op == "scale":
            nom = nominal["articulations" if group == "articulations" else "rigid_objects"][asset]["body_mass"][
                0
            ].double()
            if group == "articulations":
                names = st[group][asset]["body_names"]
                nom_names = nominal["articulations"][asset]["body_names"]
                nom = nom[[nom_names.index(n) for n in names]]
            value = mass / nom
        else:
            value = mass
        ok = bool(((value >= lo - 1e-4) & (value <= hi + 1e-4)).all())
        record(
            f"event_range[{label}].{asset}.mass_{op}",
            ok,
            f"min {value.min():.4f} max {value.max():.4f} mean {value.mean():.4f} (cfg [{lo}, {hi}])",
        )
    mats = dump["after_startup"]["materials"]
    if "shape_label" in mats:
        labels, mu = mats["shape_label"], mats["shape_material_mu"].double()
        prefix = {
            "robot": "/Robot/",
            "insertive_object": "/InsertiveObject/",
            "receptive_object": "/ReceptiveObject/",
            "table": "/Table/",
        }
        for asset, (lo, hi) in STATIC_FRICTION.items():
            idx = [i for i, l in enumerate(labels) if prefix[asset] in l]
            value = mu[idx]
            ok = bool(((value >= lo - 1e-4) & (value <= hi + 1e-4)).all())
            record(
                f"event_range[{label}].{asset}.mu_in_static_range",
                ok,
                f"min {value.min():.3f} max {value.max():.3f} mean {value.mean():.3f} (cfg static [{lo}, {hi}])",
            )
    else:
        for asset, (lo, hi) in STATIC_FRICTION.items():
            v = mats[asset]["values"].double()
            dlo, dhi = DYNAMIC_FRICTION[asset]
            ok_s = bool(((v[..., 0] >= lo - 1e-4) & (v[..., 0] <= hi + 1e-4)).all())
            ok_d = bool(((v[..., 1] >= dlo - 1e-4) & (v[..., 1] <= dhi + 1e-4)).all())
            record(
                f"event_range[{label}].{asset}.static_friction",
                ok_s,
                f"min {v[..., 0].min():.3f} max {v[..., 0].max():.3f} (cfg [{lo}, {hi}])",
            )
            record(
                f"event_range[{label}].{asset}.dynamic_friction",
                ok_d,
                f"min {v[..., 1].min():.3f} max {v[..., 1].max():.3f} (cfg [{dlo}, {dhi}])",
            )
            record(f"event_range[{label}].{asset}.dynamic_le_static", bool((v[..., 1] <= v[..., 0] + 1e-6).all()))


def main():
    A = torch.load(args.ref, weights_only=False)
    B = torch.load(args.test, weights_only=False)
    print(f"ref:  {A['meta']}\ntest: {B['meta']}\n")
    record(
        "forced_entries_identical",
        torch.equal(A["forced"]["task_ids"], B["forced"]["task_ids"])
        and torch.equal(A["forced"]["state_ids"], B["forced"]["state_ids"]),
    )
    ref_order = A["after_reset"]["articulations"]["robot"]["joint_names"]
    compare_reset_fidelity("ref", A, ref_order)
    compare_reset_fidelity("test", B, ref_order)
    rng_info = args.nominal_ref is not None and args.nominal_test is not None
    if rng_info:
        check_event_ranges("ref", A, torch.load(args.nominal_ref, weights_only=False))
        check_event_ranges("test", B, torch.load(args.nominal_test, weights_only=False))
    compare_state(
        "startup", A["after_startup"]["state"], B["after_startup"]["state"], physics_info=True, mass_info=rng_info
    )
    if not rng_info:
        compare_materials(A, B)
    compare_state("after_reset", A["after_reset"], B["after_reset"], mass_info=rng_info)
    for group, ga in A["observations"].items():
        gb = B["observations"][group]
        for term, va in ga["terms"].items():
            close(f"obs.{group}.{term}", va, gb["terms"].get(term), atol=1e-4)
        if "vector" in ga:
            close(f"obs.{group}.<full vector>", ga["vector"], gb.get("vector"), atol=1e-4)
    for name, ca in A["commands"].items():
        for key, va in ca.items():
            close(f"command.{name}.{key}", va, B["commands"][name].get(key))
    ja = A["after_reset"]["articulations"]["robot"]["joint_names"]
    jb = B["after_reset"]["articulations"]["robot"]["joint_names"]
    for k, (aa, ab) in enumerate(zip(A["actions"], B["actions"])):
        for term, pa in aa["processed"].items():
            close(f"action[{k}].{term}.processed", pa, ab["processed"][term])
        for key in ("joint_effort_target", "joint_pos_target", "joint_vel_target"):
            close(f"action[{k}].{key}", aa[key], by_name(ab[key], jb, ja), ja, rtol=1e-3, atol=1e-3)
    for k, (ra, rb) in enumerate(zip(A["rewards_terminations"], B["rewards_terminations"])):
        for term, va in ra["reward_terms"].items():
            close(f"reward[{k}].{term}", va, rb["reward_terms"][term])
        close(f"reward[{k}].<total>", ra["reward_total"], rb["reward_total"])
        for term, va in ra["termination_terms"].items():
            close(f"termination[{k}].{term}", va.float(), rb["termination_terms"][term].float())
    if "after_step" in A and "after_step" in B:
        compare_state("after_step", A["after_step"], B["after_step"], physics_info=True, mass_info=rng_info)
        for label, dump in (("ref", A), ("test", B)):
            st = dump["after_step"]
            if "terminated" in st:
                terms = {k: int(v.sum()) for k, v in st["termination_terms"].items()}
                record(
                    f"after_step[{label}].terminations",
                    None if False else int(st["terminated"].sum()) == 0,
                    f"terminated={int(st['terminated'].sum())}/{st['terminated'].numel()} by term {terms}",
                    info=True,
                )
            robot = dump["after_step"]["articulations"]["robot"]
            base = robot["body_link_pose"][:, robot["body_names"].index("base_link")]
            close(
                f"after_step[{label}].robot.base_link_pose_vs_reset_root",
                base,
                dump["after_reset"]["articulations"]["robot"]["root_link_pose"],
                atol=1e-5,
            )
    n = {s: sum(r[0] == s for r in rows) for s in ("PASS", "FAIL", "INFO", "SKIP")}
    print(f"\n{n['PASS']} PASS, {n['FAIL']} FAIL, {n['INFO']} INFO, {n['SKIP']} SKIP")
    if args.report:
        with open(args.report, "w") as f:
            f.write(f"# MDP comparison\n\n- ref: `{args.ref}` {A['meta']}\n- test: `{args.test}` {B['meta']}\n\n")
            f.write(f"**{n['PASS']} PASS, {n['FAIL']} FAIL, {n['INFO']} INFO, {n['SKIP']} SKIP**\n\n")
            f.write("| status | check | detail |\n|---|---|---|\n")
            for status, check, detail in rows:
                f.write(f"| {status} | `{check}` | {detail.replace('|', '/')} |\n")


if __name__ == "__main__":
    main()
