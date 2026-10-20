# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record every physics substep of chosen envs from forced dataset states, with the abnormal_robot signal.

Uses the same forced (reset type, dataset index) entries as ``mdp_probe.py``. The abnormal_robot and time-out
terminations are disabled so the clip continues past the point where the episode would have ended.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/record_abnormal.py --visualizer none --backend newton \\
        --envs 0 1 2 3 --out_dir clips/newton
"""

from __future__ import annotations

import argparse
import contextlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record forced-state rollouts substep by substep.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--envs", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--env_steps", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--probe_num_envs", type=int, default=64, help="Env count of the mdp_probe run whose states to reuse."
)
parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480])
parser.add_argument(
    "--gravcomp_from_disable_gravity",
    action="store_true",
    help=(
        "Author mjc:gravcomp=1 on robot bodies with physxRigidBody:disableGravity (Newton has no per-body gravity"
        " flag)."
    ),
)
parser.add_argument(
    "--newton", nargs="*", default=[], help="MJWarpSolverCfg overrides key=value (or newton.<NewtonCfg field>=value)."
)
parser.add_argument("--gripper_action", type=float, default=0.0, help="Constant gripper action (< 0 closes).")
parser.add_argument("--contacts", action="store_true", help="Log gripper-peg contact points every substep.")
parser.add_argument(
    "--peg_contacts", action="store_true", help="Newton: log every peg contact (other body, separation)."
)
parser.add_argument("--peg_convex", action="store_true", help="Collide the insertive object as its convex hull.")
parser.add_argument(
    "--eq_solref",
    type=float,
    nargs=2,
    default=None,
    help="Newton only: (timeconst, dampratio) for all equality constraints.",
)
parser.add_argument("--renderer", type=str, choices=["default", "newton_warp"], default="default")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Drive actions with this RSL-RL policy instead of a constant action."
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--frame_stride", type=int, default=1, help="Keep every Nth physics substep (frames and traces).")
parser.add_argument(
    "--preset", action="store_true", help="Newton: use the task's presets=newton_mjwarp instead of --newton flags."
)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.preset:
    remaining_args.append("presets=newton_mjwarp")
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import numpy as np
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import events as omnireset_events
from uwlab_tasks.utils.hydra import hydra_task_compose

_FORCED: dict = {}


def _forced_call(self, env, env_ids, dataset_dir, reset_types, probs, success=None):
    if env_ids is None:
        env_ids = torch.arange(self.num_envs, device=self._env.device)
    task_ids = _FORCED["task_ids"].to(self.device)[env_ids]
    state_ids = _FORCED["state_ids"].to(self.device)[env_ids]
    self.task_id[env_ids] = task_ids
    for dataset_idx in range(self.num_tasks):
        mask = task_ids == dataset_idx
        if mask.any():
            idx = state_ids[mask] % self.num_states[dataset_idx]
            states = omnireset_events.sample_from_nested_dict(self.datasets[dataset_idx], idx)
            self._reset_to(states["initial_state"], env_ids=env_ids[mask], is_relative=True)
    robot = self._env.scene["robot"]
    robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)


def _set_collision_approximation(spawn_cfg, approximation):
    original = spawn_cfg.func

    def spawn(prim_path, cfg, *args, **kwargs):
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdPhysics

        prim = original(prim_path, cfg, *args, **kwargs)
        for root in sim_utils.find_matching_prims(prim_path) or [prim]:
            for child in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if child.HasAPI(UsdPhysics.MeshCollisionAPI):
                    stage = child.GetStage()
                    path = child.GetPath()
                    while stage.GetPrimAtPath(path).IsInstanceProxy():
                        anc = stage.GetPrimAtPath(path).GetParent()
                        while anc.IsInstanceProxy():
                            anc = anc.GetParent()
                        anc.SetInstanceable(False)
                    UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(path)).CreateApproximationAttr().Set(approximation)
        return prim

    spawn_cfg.func = spawn


def _author_gravcomp(spawn_cfg):
    original = spawn_cfg.func

    def spawn(prim_path, cfg, *args, **kwargs):
        import isaaclab.sim as sim_utils
        from pxr import Sdf, Usd

        prim = original(prim_path, cfg, *args, **kwargs)
        for root in sim_utils.find_matching_prims(prim_path) or [prim]:
            for child in Usd.PrimRange(root):
                attr = child.GetAttribute("physxRigidBody:disableGravity")
                if attr and attr.Get():
                    child.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float).Set(1.0)
        return prim

    spawn_cfg.func = spawn


GRIPPER_BODIES = [
    "left_inner_finger",
    "right_inner_finger",
    "left_outer_finger",
    "right_outer_finger",
    "left_inner_knuckle",
    "right_inner_knuckle",
    "robotiq_base_link",
]


class _PhysxGripperContacts:
    def __init__(self, env):
        from isaaclab_physx.physics import PhysxManager

        self.n = env.num_envs
        patterns = [f"/World/envs/env_*/Robot/{b}" for b in GRIPPER_BODIES]
        self.view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            patterns,
            filter_patterns=[["/World/envs/env_*/InsertiveObject"]] * len(patterns),
            max_contact_data_count=64 * len(patterns) * self.n,
        )
        self.dt = env.physics_dt

    def read(self):
        import warp as wp

        forces, points, normals, seps, counts, starts = self.view.get_contact_data(dt=self.dt)

        def t(a):
            return (wp.to_torch(a) if isinstance(a, wp.array) else torch.as_tensor(a)).cpu()

        forces, points, normals, seps, counts, starts = map(t, (forces, points, normals, seps, counts, starts))
        out = [{b: [] for b in GRIPPER_BODIES} for _ in range(self.n)]
        for bi, b in enumerate(GRIPPER_BODIES):
            for e in range(self.n):
                row = bi * self.n + e
                c, s0 = int(counts[row].reshape(-1)[0]), int(starts[row].reshape(-1)[0])
                for i in range(s0, s0 + c):
                    out[e][b].append((
                        points[i].reshape(3),
                        normals[i].reshape(3),
                        float(seps[i].reshape(-1)[0]),
                        float(forces[i].reshape(-1)[0]),
                    ))
        return out


class _NewtonGripperContacts:
    def __init__(self, env):
        from isaaclab_newton.physics import NewtonManager

        self.mgr = NewtonManager
        model = NewtonManager._model
        labels = list(model.shape_label)
        body_labels = list(model.body_label)
        shape_body = model.shape_body.numpy()
        self.n = env.num_envs
        self.peg = {}
        self.finger = {}
        for i, label in enumerate(labels):
            for e in range(self.n):
                if f"/env_{e}/InsertiveObject/" in label:
                    self.peg[i] = e
            body = body_labels[shape_body[i]] if shape_body[i] >= 0 else ""
            for b in GRIPPER_BODIES:
                if body.endswith(f"/Robot/{b}"):
                    self.finger[i] = (b, int(body.split("/env_")[1].split("/")[0]), int(shape_body[i]))

    def read(self):
        from isaaclab.utils.math import quat_apply

        c = self.mgr._contacts
        n = int(c.rigid_contact_count.numpy()[0])
        s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
        p0, p1 = c.rigid_contact_point0.numpy()[:n], c.rigid_contact_point1.numpy()[:n]
        nrm = c.rigid_contact_normal.numpy()[:n]
        body_q = torch.as_tensor(self.mgr._state_0.body_q.numpy())
        out = [{b: [] for b in GRIPPER_BODIES} for _ in range(self.n)]
        for k in range(n):
            a, b = int(s0[k]), int(s1[k])
            if a in self.finger and b in self.peg:
                (name, e, body), pt, sign = self.finger[a], p0[k], 1.0
            elif b in self.finger and a in self.peg:
                (name, e, body), pt, sign = self.finger[b], p1[k], -1.0
            else:
                continue
            q = body_q[body]
            world = q[:3] + quat_apply(q[None, 3:7], torch.as_tensor(pt, dtype=q.dtype)[None])[0]
            out[e][name].append((world, torch.as_tensor(nrm[k]) * sign, float("nan"), float("nan")))
        return out


class _NewtonPegContacts:
    def __init__(self, env):
        from isaaclab_newton.physics import NewtonManager

        self.mgr = NewtonManager
        model = NewtonManager._model
        labels, body_labels = list(model.shape_label), list(model.body_label)
        shape_body = model.shape_body.numpy()
        self.n = env.num_envs
        self.peg, self.other = {}, {}
        for i, label in enumerate(labels):
            b = int(shape_body[i])
            body = body_labels[b].split("/")[-1] if b >= 0 else label.split("/")[2]
            for e in range(self.n):
                if f"/env_{e}/" in label:
                    (self.peg if "/InsertiveObject/" in label else self.other)[i] = (e, b, body)
            if i not in self.peg and i not in self.other:
                self.other[i] = (-1, b, body)

    def read(self):
        from isaaclab.utils.math import quat_apply

        c = self.mgr._contacts
        n = int(c.rigid_contact_count.numpy()[0])
        s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
        p0, p1 = torch.as_tensor(c.rigid_contact_point0.numpy()[:n]), torch.as_tensor(
            c.rigid_contact_point1.numpy()[:n]
        )
        nrm = torch.as_tensor(c.rigid_contact_normal.numpy()[:n])
        body_q = torch.as_tensor(self.mgr._state_0.body_q.numpy())

        def world(body, pt):
            if body < 0:
                return pt
            q = body_q[body]
            return q[:3] + quat_apply(q[None, 3:7], pt[None])[0]

        out = [dict() for _ in range(self.n)]
        for k in range(n):
            a, b = int(s0[k]), int(s1[k])
            if a in self.peg and b in self.other:
                e, other = self.peg[a][0], self.other[b]
            elif b in self.peg and a in self.other:
                e, other = self.peg[b][0], self.other[a]
            else:
                continue
            ba = (self.peg.get(a) or self.other.get(a))[1]
            bb = (self.peg.get(b) or self.other.get(b))[1]
            sep = float(((world(bb, p1[k]) - world(ba, p0[k])) * nrm[k]).sum())
            entry = out[e].setdefault(other[2], {"count": 0, "min_sep": 1e9})
            entry["count"] += 1
            entry["min_sep"] = min(entry["min_sep"], sep)
        return out


@hydra_task_compose(
    args_cli.task, args_cli.agent if args_cli.checkpoint else "env_cfg_entry_point", hydra_args=remaining_args
)
def main(env_cfg, agent_cfg) -> None:  # noqa: C901
    probe_envs = torch.tensor(args_cli.envs)
    env_cfg.scene.num_envs = len(args_cli.envs)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.terminations.abnormal_robot = None
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.enable_corruption = False
    width, height = args_cli.resolution
    camera_cfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/viz_camera",
        update_period=0,
        width=width,
        height=height,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.46613815, 0.25072644, 0.47107948, 0.70564552),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    )
    if args_cli.renderer == "newton_warp":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        camera_cfg.renderer_cfg = NewtonWarpRendererCfg()
    env_cfg.scene.viz_camera = camera_cfg
    if args_cli.gravcomp_from_disable_gravity:
        _author_gravcomp(env_cfg.scene.robot.spawn)
    if args_cli.contacts:
        env_cfg.scene.robot.spawn.activate_contact_sensors = True
    if args_cli.peg_convex:
        _set_collision_approximation(env_cfg.scene.insertive_object.spawn, "convexHull")
    if args_cli.backend == "newton" and not args_cli.preset:
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        solver_kw, newton_kw, shape_kw, collision_kw = {}, {}, {}, {}
        for item in args_cli.newton:
            key, value = item.split("=", 1)
            value = {"true": True, "false": False}.get(value.lower(), value)
            if isinstance(value, str) and value.replace(".", "", 1).replace("e", "", 1).replace("-", "", 1).isdigit():
                value = float(value) if any(c in value for c in ".e") else int(value)
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    value = float(value) if "." in value or "e" in value else int(value)
            if key.startswith("shape."):
                shape_kw[key.removeprefix("shape.")] = value
            elif key.startswith("collision."):
                collision_kw[key.removeprefix("collision.")] = value
            else:
                (newton_kw if key.startswith("newton.") else solver_kw)[key.removeprefix("newton.")] = value
        if shape_kw:
            from isaaclab_newton.physics import NewtonShapeCfg

            newton_kw["default_shape_cfg"] = NewtonShapeCfg(**shape_kw)
        if collision_kw:
            from isaaclab_newton.physics import NewtonCollisionPipelineCfg

            newton_kw["collision_cfg"] = NewtonCollisionPipelineCfg(**collision_kw)
        env_cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(**solver_kw), **newton_kw)
    reset_types = env_cfg.events.reset_from_reset_states.params["reset_types"]
    gen = torch.Generator().manual_seed(args_cli.seed)
    task_ids = torch.arange(args_cli.probe_num_envs) % len(reset_types)
    state_ids = torch.randint(0, 1 << 30, (args_cli.probe_num_envs,), generator=gen)
    _FORCED["task_ids"], _FORCED["state_ids"] = task_ids[probe_envs], state_ids[probe_envs]
    omnireset_events.MultiResetManager.__call__ = _forced_call

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = gym_env.unwrapped
    robot, camera = env.scene["robot"], env.scene["viz_camera"]
    if args_cli.eq_solref is not None and args_cli.backend == "newton":
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        mjw = NewtonManager._solver.mjw_model
        wp.to_torch(mjw.eq_solref)[..., 0] = args_cli.eq_solref[0]
        wp.to_torch(mjw.eq_solref)[..., 1] = args_cli.eq_solref[1]
        src = getattr(getattr(NewtonManager._model, "mujoco", None), "eq_solref", None)
        if src is not None:
            t = wp.to_torch(src)
            t[..., 0], t[..., 1] = args_cli.eq_solref
        print(f"[record_abnormal] eq_solref set to {args_cli.eq_solref} for {mjw.eq_solref.shape} constraints")
    names = list(robot.joint_names)
    physics_dt = env.physics_dt
    os.makedirs(args_cli.out_dir, exist_ok=True)
    frames, signal, trace = [], [], []
    peg = env.scene["insertive_object"]
    finger = names.index("finger_joint")
    arm = [names.index(n) for n in names[:6]]
    hand = robot.body_names.index("robotiq_base_link")

    def peg_in_hand():
        from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

        h = robot.data.body_link_pose_w.torch[:, hand]
        p = peg.data.root_link_pose_w.torch
        rel_pos = quat_apply_inverse(h[:, 3:7], p[:, :3] - h[:, :3])
        rel_quat = quat_mul(quat_inv(h[:, 3:7]), p[:, 3:7])
        return torch.cat([rel_pos, rel_quat], dim=1)

    sim_step, scene_update = env.sim.step, env.scene.update
    contact_reader = None
    contact_log = []
    peg_contact_reader = _NewtonPegContacts(env) if args_cli.peg_contacts and args_cli.backend == "newton" else None
    peg_contact_log = []
    if args_cli.contacts:
        contact_reader = (_PhysxGripperContacts if args_cli.backend == "physx" else _NewtonGripperContacts)(env)

    def log_contacts():
        from isaaclab.utils.math import quat_apply_inverse

        hand_pose = robot.data.body_link_pose_w.torch[:, hand].cpu()
        frame = []
        for e, per_body in enumerate(contact_reader.read()):
            entry = {}
            for b, items in per_body.items():
                if items:
                    pts = torch.stack([it[0].float() for it in items])
                    nrm = torch.stack([it[1].float() for it in items])
                    rel = quat_apply_inverse(hand_pose[e : e + 1, 3:7].expand(len(pts), 4), pts - hand_pose[e, :3])
                    nrel = quat_apply_inverse(hand_pose[e : e + 1, 3:7].expand(len(pts), 4), nrm)
                    entry[b] = {
                        "p": rel.tolist(),
                        "n": nrel.tolist(),
                        "sep": [it[2] for it in items],
                        "f": [it[3] for it in items],
                    }
            frame.append(entry)
        contact_log.append(frame)

    def grab():
        env.sim.forward()
        env.sim.render()
        camera.update(physics_dt, force_recompute=True)
        rgb = camera.data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        frames.append(rgb[..., :3].detach().cpu().numpy().astype(np.uint8))

    counter = {"substep": 0}

    def step(render=True):
        sim_step(render=False)
        counter["substep"] += 1
        if counter["substep"] % args_cli.frame_stride == 0:
            grab()

    def update(dt):
        scene_update(dt)
        if counter["substep"] % args_cli.frame_stride != 0:
            return
        if contact_reader is not None:
            log_contacts()
        if peg_contact_reader is not None:
            peg_contact_log.append(peg_contact_reader.read())
        qd = robot.data.joint_vel.torch.abs()
        signal.append((qd / (2.0 * robot.data.joint_vel_limits.torch)).cpu())
        trace.append(
            torch.cat(
                [
                    robot.data.joint_pos.torch[:, arm],
                    robot.data.joint_pos.torch[:, finger : finger + 1],
                    robot.data.joint_pos_target.torch[:, finger : finger + 1],
                    peg.data.root_link_pose_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3],
                    peg.data.root_lin_vel_w.torch[:, 2:3],
                    peg_in_hand(),
                    robot.data.joint_pos.torch[:, 6:],
                ],
                dim=1,
            ).cpu()
        )

    torch.manual_seed(args_cli.seed)
    env.reset(seed=args_cli.seed)
    for _ in range(3):
        env.sim.render()
    grab()
    signal.append(torch.zeros(env.num_envs, len(names)))
    scene_update(0.0)
    trace.append(
        torch.cat(
            [
                robot.data.joint_pos.torch[:, arm],
                robot.data.joint_pos.torch[:, finger : finger + 1],
                robot.data.joint_pos_target.torch[:, finger : finger + 1],
                peg.data.root_link_pose_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3],
                peg.data.root_lin_vel_w.torch[:, 2:3],
                peg_in_hand(),
                robot.data.joint_pos.torch[:, 6:],
            ],
            dim=1,
        ).cpu()
    )
    env.sim.step, env.scene.update = step, update
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    gripper_slice = sum(env.action_manager.action_term_dim[: env.action_manager.active_terms.index("gripper")])
    action[:, gripper_slice:] = args_cli.gripper_action
    if args_cli.checkpoint:
        import sys

        from isaaclab.utils.assets import retrieve_file_path
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "reinforcement_learning", "rsl_rl")
        )
        import cli_args

        agent = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
        vec = RslRlVecEnvWrapper(gym_env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(vec, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(retrieve_file_path(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        obs = vec.get_observations()
        with torch.inference_mode():
            for _ in range(args_cli.env_steps):
                obs, _, _, _ = vec.step(policy(obs))
    else:
        for _ in range(args_cli.env_steps):
            env.step(action)
    video = np.stack(frames)
    ratio = torch.stack(signal)
    summary = {
        "physics_dt": physics_dt,
        "frame_stride": args_cli.frame_stride,
        "joint_names": names,
        "envs": {},
        "joint_vel_limits": robot.data.joint_vel_limits.torch[0].cpu().tolist(),
    }
    for k, e in enumerate(args_cli.envs):
        np.savez_compressed(os.path.join(args_cli.out_dir, f"env{e}.npz"), frames=video[:, k])
        over = (ratio[:, k] > 1.0).any(dim=1).nonzero().flatten()
        first = int(over[0]) if len(over) else None
        summary["envs"][e] = {
            "reset_type": reset_types[int(_FORCED["task_ids"][k])],
            "dataset_index": int(_FORCED["state_ids"][k]),
            "first_abnormal_frame": first,
            "max_ratio_per_frame": ratio[:, k].max(dim=1).values.tolist(),
            "argmax_joint_per_frame": [names[j] for j in ratio[:, k].argmax(dim=1).tolist()],
            "ratio": ratio[:, k].tolist(),
            "trace_columns": [
                *names[:6],
                "finger_joint",
                "finger_joint_target",
                "peg_z",
                "peg_vz",
                "peg_in_hand_x",
                "peg_in_hand_y",
                "peg_in_hand_z",
                "peg_in_hand_qx",
                "peg_in_hand_qy",
                "peg_in_hand_qz",
                "peg_in_hand_qw",
                *names[6:],
            ],
            "gripper_action": args_cli.gripper_action,
            "trace": torch.stack(trace)[:, k].tolist(),
            "peak_joint_vel": (
                (ratio[:, k] * 2.0 * robot.data.joint_vel_limits.torch[k].cpu()).max(dim=0).values.tolist()
            ),
        }
        print(
            f"[record_abnormal] env {e} {summary['envs'][e]['reset_type']}: {video.shape[0]} frames, first abnormal"
            f" frame={first} {summary['envs'][e]['argmax_joint_per_frame'][first] if first is not None else ''}"
        )
    if contact_log:
        summary["contacts"] = contact_log
    if peg_contact_log:
        summary["peg_contacts"] = peg_contact_log
    with open(os.path.join(args_cli.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
