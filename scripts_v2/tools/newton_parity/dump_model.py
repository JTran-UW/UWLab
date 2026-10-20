# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump the nominal physics model of a task's scene on one backend for cross-backend comparison.

Builds only the simulation context and the scene (no managers, no events), so the dump reflects the
assets and configs as authored. Pair with ``compare_models.py``.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/dump_model.py --headless --backend physx --out physx.pt
    python scripts_v2/tools/newton_parity/dump_model.py --headless --backend newton --out newton.pt
"""

from __future__ import annotations

import argparse
import contextlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump the nominal physics model of a task's scene.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--out", type=str, required=True, help="Output .pt file.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--num_configs", type=int, default=6, help="Joint configurations for kinematics/dynamics.")
parser.add_argument("--config_scale", type=float, default=0.3, help="Joint offsets from default (rad or m).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--explicit_gripper_topology",
    action="store_true",
    help=(
        "Author the Robotiq loop topology PhysX infers (un-reverse inner_finger_knuckle joints, exclude inner_finger"
        " joints)."
    ),
)
parser.add_argument(
    "--pin_robot_mass_from",
    type=str,
    default=None,
    help="Author body-level MassAPI on robot links from a reference dump (mass, COM, principal inertia).",
)
parser.add_argument(
    "--gripper_sweep", type=int, default=9, help="finger_joint samples for the mimic/loop consistency sweep."
)
parser.add_argument(
    "--sweep_gearing", nargs="*", default=[], help="joint=gearing overrides for the sweep only (USD untouched)."
)
parser.add_argument(
    "--conform_colliders",
    action="store_true",
    help="Move CollisionAPI from non-geometry prims (Xform) to their Mesh descendants, de-instancing as needed.",
)
parser.add_argument("--mimic_gearing", nargs="*", default=[], help="joint=gearing written to the robot USD mimic API.")
parser.add_argument("--robot_usd", type=str, default=None, help="Replace the robot spawn usd_path.")
parser.add_argument("--mjcf", type=str, default=None, help="Newton only: also write the solver model as MJCF.")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.backend == "newton" and not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from uwlab_tasks.utils.hydra import hydra_task_compose


def _t(x):
    if x is None:
        return None
    if hasattr(x, "torch"):
        x = x.torch
    elif not isinstance(x, torch.Tensor):
        import warp as wp

        if isinstance(x, wp.array):
            x = wp.to_torch(x)
        else:
            return x
    return x.detach().clone().cpu()


def _get(data, *names):
    for name in names:
        try:
            value = getattr(data, name)
        except Exception:
            continue
        if value is not None:
            return _t(value)
    return None


def _refresh_kinematics(sim, scene, backend):
    if backend == "physx":
        from isaaclab_physx.physics import PhysxManager

        PhysxManager._view.update_articulations_kinematic()
    else:
        sim.forward()
    scene.update(sim.get_physics_dt())


def _relative_pose(root, poses):
    """Express world poses (..., 7, xyzw) in the root frame."""
    from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

    rp, rq = root[..., :3], root[..., 3:7]
    rq = rq.expand(poses.shape[:-1] + (4,))
    pos = quat_apply_inverse(rq, poses[..., :3] - rp)
    quat = quat_mul(quat_inv(rq), poses[..., 3:7])
    return torch.cat([pos, quat], dim=-1)


def _joint_configs(joint_names, default_q, n, scale, seed):
    """Name-keyed offsets so both backends get identical configs regardless of joint order."""
    rng = np.random.default_rng(seed)
    order = sorted(joint_names)
    configs = [dict(zip(joint_names, default_q.tolist()))]
    for _ in range(n - 1):
        offsets = dict(zip(order, rng.uniform(-scale, scale, len(order)).tolist()))
        configs.append({j: q + offsets[j] for j, q in zip(joint_names, default_q.tolist())})
    return configs


def dump_articulation(sim, scene, backend, asset):
    d = asset.data
    out = {
        "body_names": list(asset.body_names),
        "joint_names": list(asset.joint_names),
        "is_fixed_base": bool(asset.is_fixed_base),
        "num_joints": asset.num_joints,
        "num_fixed_tendons": getattr(asset, "num_fixed_tendons", None),
        "body_mass": _get(d, "body_mass", "default_mass"),
        "body_inertia": _get(d, "body_inertia", "default_inertia"),
        "body_com_pose_b": _get(d, "body_com_pose_b"),
        "joint_stiffness": _get(d, "joint_stiffness"),
        "joint_damping": _get(d, "joint_damping"),
        "joint_armature": _get(d, "joint_armature"),
        "joint_friction_coeff": _get(d, "joint_friction_coeff"),
        "joint_dynamic_friction_coeff": _get(d, "joint_dynamic_friction_coeff"),
        "joint_viscous_friction_coeff": _get(d, "joint_viscous_friction_coeff"),
        "joint_pos_limits": _get(d, "joint_pos_limits"),
        "joint_vel_limits": _get(d, "joint_vel_limits"),
        "joint_effort_limits": _get(d, "joint_effort_limits"),
        "default_joint_pos": _get(d, "default_joint_pos"),
        "default_root_pose": _get(d, "default_root_pose"),
        "root_link_pose_w": _get(d, "root_link_pose_w"),
        "actuators": {
            name: {
                "class": type(act).__name__,
                "joint_names": list(act.joint_names),
                "stiffness": _t(getattr(act, "stiffness", None)),
                "damping": _t(getattr(act, "damping", None)),
                "armature": _t(getattr(act, "armature", None)),
                "effort_limit": _t(getattr(act, "actuator_effort_limit", None)),
                "effort_limit_sim": _t(getattr(act, "effort_limit_sim", None)),
                "velocity_limit": _t(getattr(act, "actuator_velocity_limit", None)),
                "velocity_limit_sim": _t(getattr(act, "velocity_limit_sim", None)),
            }
            for name, act in asset.actuators.items()
        },
    }
    default_q = out["default_joint_pos"][0]
    configs = _joint_configs(out["joint_names"], default_q, args_cli.num_configs, args_cli.config_scale, args_cli.seed)
    samples = []
    for cfg in configs:
        q = torch.tensor([cfg[j] for j in out["joint_names"]], device=asset.device).repeat(scene.num_envs, 1)
        asset.write_joint_state_to_sim(q, torch.zeros_like(q))
        _refresh_kinematics(sim, scene, backend)
        root = _t(d.root_link_pose_w)[:, None]
        samples.append({
            "q": cfg,
            "joint_pos_readback": _get(d, "joint_pos"),
            "body_link_pose_r": _relative_pose(root, _t(d.body_link_pose_w)),
            "body_com_pose_r": _relative_pose(root, _t(d.body_com_pose_w)),
            "mass_matrix": _get(d, "mass_matrix"),
            "gravity_compensation_forces": _get(d, "gravity_compensation_forces"),
            "body_link_jacobian_w": _get(d, "body_link_jacobian_w"),
        })
    out["configs"] = samples
    if "finger_joint" in out["joint_names"] and args_cli.gripper_sweep > 0:
        out["gripper_sweep"] = _gripper_sweep(sim, scene, backend, asset, out, default_q)
    return out


def _mimic_rules(env_root="/World/envs/env_0/Robot"):
    """joint -> (reference joint, gearing, offset) from PhysxMimicJointAPI (q + gearing * q_ref + offset = 0)."""
    import isaaclab.sim as sim_utils
    from pxr import Usd

    rules, loops = {}, {}
    stage = sim_utils.get_current_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(env_root)):
        for schema in prim.GetAppliedSchemas():
            if schema.startswith("PhysxMimicJointAPI:"):
                axis = schema.split(":", 1)[1]
                ref = prim.GetRelationship(f"physxMimicJoint:{axis}:referenceJoint").GetTargets()
                rules[prim.GetName()] = (
                    ref[0].name if ref else "finger_joint",
                    float(prim.GetAttribute(f"physxMimicJoint:{axis}:gearing").Get()),
                    float(prim.GetAttribute(f"physxMimicJoint:{axis}:offset").Get()),
                )
        exclude = prim.GetAttribute("physics:excludeFromArticulation")
        if exclude and exclude.Get():
            b0 = prim.GetRelationship("physics:body0").GetTargets()[0].name
            b1 = prim.GetRelationship("physics:body1").GetTargets()[0].name
            loops[prim.GetName()] = (
                b0,
                b1,
                list(prim.GetAttribute("physics:localPos0").Get()),
                list(prim.GetAttribute("physics:localPos1").Get()),
                _py(prim.GetAttribute("physics:localRot0").Get()),
                _py(prim.GetAttribute("physics:localRot1").Get()),
            )
    return rules, loops


def _gripper_sweep(sim, scene, backend, asset, out, default_q):
    """FK with every mimic joint set by its authored rule; loop-closure residual per finger_joint sample."""
    from isaaclab.utils.math import quat_apply, quat_mul

    rules, loops = _mimic_rules()
    for item in args_cli.sweep_gearing:
        joint, gearing = item.split("=")
        ref, _, offset = rules[joint]
        rules[joint] = (ref, float(gearing), offset)
    names, bodies = out["joint_names"], out["body_names"]
    lo, hi = out["joint_pos_limits"][0, names.index("finger_joint")].tolist()
    result = {"rules": rules, "loops": {k: v[:2] for k, v in loops.items()}, "theta": [], "residual": {}}
    for theta in torch.linspace(lo, min(hi, 0.8), args_cli.gripper_sweep).tolist():
        q = default_q.clone()
        q[names.index("finger_joint")] = theta
        for joint, (ref, gearing, offset) in rules.items():
            if joint in names and ref in names:
                q[names.index(joint)] = -(gearing * q[names.index(ref)] + offset)
        qe = q.to(asset.device).repeat(scene.num_envs, 1)
        asset.write_joint_state_to_sim(qe, torch.zeros_like(qe))
        _refresh_kinematics(sim, scene, backend)
        poses = _t(asset.data.body_link_pose_w)[0]
        result["theta"].append(theta)
        for loop, (b0, b1, p0, p1, r0, r1) in loops.items():
            x0, x1 = poses[bodies.index(b0)], poses[bodies.index(b1)]
            a0 = x0[:3] + quat_apply(x0[3:7], torch.tensor(p0, dtype=x0.dtype))
            a1 = x1[:3] + quat_apply(x1[3:7], torch.tensor(p1, dtype=x1.dtype))
            # USD localRot is (w, x, y, z); joint axis is Z for these joints
            f0 = quat_mul(x0[3:7], torch.tensor([*r0[1:], r0[0]], dtype=x0.dtype))
            f1 = quat_mul(x1[3:7], torch.tensor([*r1[1:], r1[0]], dtype=x1.dtype))
            z = torch.tensor([0.0, 0.0, 1.0], dtype=x0.dtype)
            cos = (quat_apply(f0, z) * quat_apply(f1, z)).sum().clamp(-1, 1)
            result["residual"].setdefault(loop, []).append(((a0 - a1).norm().item(), torch.acos(cos.abs()).item()))
    return result


def dump_rigid_object(asset):
    d = asset.data
    return {
        "body_names": list(asset.body_names),
        "body_mass": _get(d, "body_mass", "default_mass"),
        "body_inertia": _get(d, "body_inertia", "default_inertia"),
        "body_com_pose_b": _get(d, "body_com_pose_b"),
        "default_root_pose": _get(d, "default_root_pose"),
        "root_link_pose_w": _get(d, "root_link_pose_w"),
    }


def _py(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "real") and hasattr(v, "imaginary"):
        return [float(v.real), *[float(c) for c in v.imaginary]]
    if hasattr(v, "__len__"):
        with contextlib.suppress(TypeError):
            return [_py(x) for x in v]
    return str(v)


def dump_usd(env_root="/World/envs/env_0"):
    """Authored physics on the stage of env 0 (identical input for both backends)."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    colliders, joints, materials, masses = [], [], [], []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(env_root), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())[len(env_root) :]
        schemas = list(prim.GetAppliedSchemas())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            approx = prim.GetAttribute("physics:approximation")
            enabled = prim.GetAttribute("physics:collisionEnabled")
            points = prim.GetAttribute("points")
            colliders.append({
                "path": path,
                "type": prim.GetTypeName(),
                "active": prim.IsActive(),
                "enabled": _py(enabled.Get()) if enabled else None,
                "approximation": _py(approx.Get()) if approx else None,
                "num_points": len(points.Get()) if points and points.Get() is not None else None,
                "instance_proxy": prim.IsInstanceProxy(),
                "schemas": [str(x) for x in schemas],
            })
        if prim.IsA(UsdPhysics.Joint):
            attrs = {
                a.GetName(): a.Get()
                for a in prim.GetAttributes()
                if a.GetName().startswith(("physics:", "physxMimicJoint", "physxJoint", "drive:", "mimic"))
            }
            joints.append({
                "path": path,
                "type": prim.GetTypeName(),
                "body0": [str(t) for t in prim.GetRelationship("physics:body0").GetTargets()],
                "body1": [str(t) for t in prim.GetRelationship("physics:body1").GetTargets()],
                "schemas": [str(x) for x in schemas],
                "attrs": {k: _py(v) for k, v in attrs.items()},
            })
        if prim.HasAPI(UsdPhysics.MassAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            rb = UsdPhysics.RigidBodyAPI(prim)
            masses.append({
                "path": path,
                "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                "kinematic": _py(rb.GetKinematicEnabledAttr().Get()) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else None,
                "mass_api": prim.HasAPI(UsdPhysics.MassAPI),
                "mass": _py(m.GetMassAttr().Get()),
                "density": _py(m.GetDensityAttr().Get()),
                "center_of_mass": _py(m.GetCenterOfMassAttr().Get()),
                "diagonal_inertia": _py(m.GetDiagonalInertiaAttr().Get()),
                "principal_axes": _py(m.GetPrincipalAxesAttr().Get()),
                "xform_ops": {
                    a.GetName(): _py(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith("xformOp")
                },
            })
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            m = UsdPhysics.MaterialAPI(prim)
            materials.append({
                "path": path,
                "active": prim.IsActive(),
                "static_friction": _py(m.GetStaticFrictionAttr().Get()),
                "dynamic_friction": _py(m.GetDynamicFrictionAttr().Get()),
                "restitution": _py(m.GetRestitutionAttr().Get()),
                "schemas": [str(x) for x in schemas],
            })
    return {"colliders": colliders, "joints": joints, "materials": materials, "masses": masses}


def _arrays(obj):
    import warp as wp

    out = {}
    for name in dir(obj):
        if name.startswith("__"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if isinstance(value, wp.array):
            try:
                out[name] = wp.to_torch(value).detach().clone().cpu() if value.size else torch.empty(0)
            except Exception:
                out[name] = value.numpy()
        elif isinstance(value, (int, float, bool, str)):
            out[name] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float, bool, str)) for v in value):
            out[name] = list(value)
    return out


def dump_native(backend, scene):
    if backend == "newton":
        from isaaclab_newton.physics import NewtonManager

        native = {"model": _arrays(NewtonManager._model)}
        solver = NewtonManager._solver
        mjw_model = getattr(solver, "mjw_model", None)
        if mjw_model is not None:
            native["mjw_model"] = _arrays(mjw_model)
            native["mjw_opt"] = _arrays(mjw_model.opt)
        return native
    native = {}
    for name, asset in scene.articulations.items():
        view = asset.root_view
        entry = {}
        for getter in [
            "get_masses",
            "get_inertias",
            "get_coms",
            "get_material_properties",
            "get_contact_offsets",
            "get_rest_offsets",
            "get_dof_stiffnesses",
            "get_dof_dampings",
            "get_dof_armatures",
            "get_dof_friction_properties",
            "get_dof_max_velocities",
            "get_dof_max_forces",
            "get_dof_limits",
        ]:
            try:
                entry[getter] = _t(getattr(view, getter)())
            except Exception as e:
                entry[getter] = f"unavailable: {e}"
        for attr in ["max_shapes", "max_links", "max_dofs", "shared_metatype"]:
            v = getattr(view, attr, None)
            entry[attr] = v if isinstance(v, (int, float, str)) else None
        meta = getattr(view, "shared_metatype", None)
        if meta is not None:
            entry["metatype"] = {
                "link_names": list(getattr(meta, "link_names", [])),
                "dof_names": list(getattr(meta, "dof_names", [])),
                "dof_types": [str(t) for t in getattr(meta, "dof_types", [])],
                "link_parents": list(getattr(meta, "link_parents", [])),
            }
        native[name] = entry
    for name, asset in scene.rigid_objects.items():
        view = asset.root_view
        entry = {}
        for getter in [
            "get_masses",
            "get_inertias",
            "get_coms",
            "get_material_properties",
            "get_contact_offsets",
            "get_rest_offsets",
        ]:
            try:
                entry[getter] = _t(getattr(view, getter)())
            except Exception as e:
                entry[getter] = f"unavailable: {e}"
        native[name] = entry
    return native


_PERP_FLIP = {"X": (0.0, 0.0, 1.0, 0.0), "Y": (0.0, 0.0, 0.0, 1.0), "Z": (0.0, 1.0, 0.0, 0.0)}


def _make_explicit_gripper_topology(prim):
    """Un-reverse the inner_finger_knuckle joints (angle sign kept) and exclude the inner_finger loop closers."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, Usd, UsdPhysics

    for root in sim_utils.find_matching_prims(str(prim.GetPath())) or [prim]:
        for child in Usd.PrimRange(root):
            name = child.GetName()
            if name in ("left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"):
                joint = UsdPhysics.Joint(child)
                b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                p0, p1 = joint.GetLocalPos0Attr().Get(), joint.GetLocalPos1Attr().Get()
                r0, r1 = joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()
                w, x, y, z = _PERP_FLIP[UsdPhysics.RevoluteJoint(child).GetAxisAttr().Get()]
                flip = Gf.Quatf(w, Gf.Vec3f(x, y, z))
                joint.GetBody0Rel().SetTargets(b1)
                joint.GetBody1Rel().SetTargets(b0)
                joint.GetLocalPos0Attr().Set(p1)
                joint.GetLocalPos1Attr().Set(p0)
                joint.GetLocalRot0Attr().Set((r1 * flip).GetNormalized())
                joint.GetLocalRot1Attr().Set((r0 * flip).GetNormalized())
            elif name in ("left_inner_finger_joint", "right_inner_finger_joint"):
                UsdPhysics.Joint(child).CreateExcludeFromArticulationAttr().Set(True)


def _pin_robot_mass(reference_path):
    """Spawn hook writing each robot link's reference mass properties as an explicit body-level MassAPI."""
    ref = torch.load(reference_path, weights_only=False)["articulations"]["robot"]
    props = {}
    for i, body in enumerate(ref["body_names"]):
        inertia = ref["body_inertia"][0, i].double().reshape(3, 3)
        evals, evecs = torch.linalg.eigh(0.5 * (inertia + inertia.T))
        if torch.det(evecs) < 0:
            evecs[:, 0] = -evecs[:, 0]
        props[body] = (float(ref["body_mass"][0, i]), ref["body_com_pose_b"][0, i, :3].tolist(), evals.tolist(), evecs)

    def post(prim):
        import isaaclab.sim as sim_utils
        from isaaclab.utils.math import quat_from_matrix
        from pxr import Gf, Usd, UsdPhysics

        for root in sim_utils.find_matching_prims(str(prim.GetPath())) or [prim]:
            for child in Usd.PrimRange(root):
                if child.GetName() not in props or not child.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                mass, com, diag, axes = props[child.GetName()]
                x, y, z, w = quat_from_matrix(axes.float()[None])[0].tolist()
                api = UsdPhysics.MassAPI.Apply(child)
                api.CreateMassAttr().Set(mass)
                api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com))
                api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*diag))
                api.CreatePrincipalAxesAttr().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    return post


def _conform_colliders(prim):
    """UsdPhysics expects CollisionAPI on gprims; PhysX tolerates it on an Xform, Newton skips it."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdPhysics

    for root in sim_utils.find_matching_prims(str(prim.GetPath())) or [prim]:
        targets = [
            p
            for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsA(UsdGeom.Gprim)
        ]
        for target in targets:
            path = target.GetPath()
            ancestor = root.GetStage().GetPrimAtPath(path)
            while ancestor.IsInstanceProxy() or ancestor.IsInstance():
                parent = ancestor
                while parent.IsInstanceProxy():
                    parent = parent.GetParent()
                parent.SetInstanceable(False)
                ancestor = root.GetStage().GetPrimAtPath(path)
            approx = (
                ancestor.GetAttribute("physics:approximation").Get()
                if ancestor.HasAPI(UsdPhysics.MeshCollisionAPI)
                else None
            )
            for child in Usd.PrimRange(ancestor):
                if child.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(child)
                    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(child)
                    if approx:
                        mesh_api.CreateApproximationAttr().Set(approx)
            ancestor.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            ancestor.RemoveAPI(UsdPhysics.CollisionAPI)


def _set_mimic_gearing(overrides):
    def post(prim):
        import isaaclab.sim as sim_utils
        from pxr import Usd

        for root in sim_utils.find_matching_prims(str(prim.GetPath())) or [prim]:
            for child in Usd.PrimRange(root):
                if child.GetName() in overrides:
                    for schema in child.GetAppliedSchemas():
                        if schema.startswith("PhysxMimicJointAPI:"):
                            axis = schema.split(":", 1)[1]
                            child.GetAttribute(f"physxMimicJoint:{axis}:gearing").Set(overrides[child.GetName()])

    return post


def _wrap_spawn(spawn_cfg, post):
    original = spawn_cfg.func

    def spawn(prim_path, cfg, *args, **kwargs):
        prim = original(prim_path, cfg, *args, **kwargs)
        post(prim)
        return prim

    spawn_cfg.func = spawn


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.backend == "newton" and args_cli.mjcf:
        env_cfg.sim.physics.solver_cfg.save_to_mjcf = args_cli.mjcf

    if args_cli.robot_usd:
        env_cfg.scene.robot.spawn.usd_path = args_cli.robot_usd
    if args_cli.explicit_gripper_topology:
        _wrap_spawn(env_cfg.scene.robot.spawn, _make_explicit_gripper_topology)
    if args_cli.mimic_gearing:
        overrides = {k: float(v) for k, v in (item.split("=") for item in args_cli.mimic_gearing)}
        _wrap_spawn(env_cfg.scene.robot.spawn, _set_mimic_gearing(overrides))
    if args_cli.conform_colliders:
        _wrap_spawn(env_cfg.scene.robot.spawn, _conform_colliders)
    if args_cli.pin_robot_mass_from:
        _wrap_spawn(env_cfg.scene.robot.spawn, _pin_robot_mass(args_cli.pin_robot_mass_from))

    sim = SimulationContext(env_cfg.sim)
    scene = InteractiveScene(env_cfg.scene)
    sim.reset()
    scene.update(sim.get_physics_dt())

    out = {  # noqa: SIM904
        "meta": {
            "task": args_cli.task,
            "backend": args_cli.backend,
            "physics_cfg": repr(env_cfg.sim.physics),
            "dt": sim.get_physics_dt(),
            "num_envs": scene.num_envs,
            "explicit_gripper_topology": args_cli.explicit_gripper_topology,
            "pin_robot_mass_from": args_cli.pin_robot_mass_from,
            "conform_colliders": args_cli.conform_colliders,
            "mimic_gearing": args_cli.mimic_gearing,
            "robot_usd": args_cli.robot_usd,
        },
        "usd": dump_usd(),
        "articulations": {
            n: dump_articulation(sim, scene, args_cli.backend, a) for n, a in scene.articulations.items()
        },
        "rigid_objects": {n: dump_rigid_object(a) for n, a in scene.rigid_objects.items()},
    }
    out["native"] = dump_native(args_cli.backend, scene)
    torch.save(out, args_cli.out)
    print(
        f"[dump_model] wrote {args_cli.out}: articulations={list(out['articulations'])} "
        f"rigid_objects={list(out['rigid_objects'])}"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
