# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton NaN-persistence probe: poison one world, then report every env-0 buffer that stays non-finite across the
blow-up termination and the reset that follows."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--inject", type=str, default="warmstart", choices=["warmstart", "joint_qd", "none"])
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--trace", action="store_true", help="After the first post-inject step, trace one step phase by phase.")
parser.add_argument("--clean_mjw", action="store_true", help="Zero every non-finite mjw_data entry before the traced solve.")
parser.add_argument("--clean_only", type=str, default="", help="With --clean_mjw: only fields starting with this prefix.")
parser.add_argument("--clean_except", type=str, default="", help="With --clean_mjw: skip fields starting with this prefix.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
import warp as wp

from isaaclab.utils.warp.proxy_array import ProxyArray
from isaaclab_newton.physics.newton_manager import NewtonManager

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

BAD = 0


def as_tensor(x):
    if isinstance(x, ProxyArray):
        x = x.torch
    if isinstance(x, wp.array):
        try:
            x = wp.to_torch(x)
        except Exception:
            return None
    return x if torch.is_tensor(x) else None


def bad_rows(x, n):
    """Non-finite flag of row BAD for a per-env float tensor, else None."""
    t = as_tensor(x)
    if t is None or not t.is_floating_point() or t.dim() < 1 or t.shape[0] != n:
        return None
    return not bool(torch.isfinite(t[BAD]).all())


def scan(obj, name, n, out, depth=0, seen=None):
    """Recursively collect attribute paths of isaaclab/uwlab objects whose env-BAD row is non-finite."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return
    seen.add(id(obj))
    items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, (list, tuple)) else (
        vars(obj).items() if hasattr(obj, "__dict__") else [])
    for k, v in list(items):
        r = bad_rows(v, n)
        if r:
            out.append(f"{name}.{k}")
        elif r is None and (isinstance(v, (dict, list, tuple)) or type(v).__module__.split(".")[0] in (
                "isaaclab", "isaaclab_newton", "uwlab", "uwlab_tasks", "uwlab_assets")):
            scan(v, f"{name}.{k}", n, out, depth + 1, seen)


def report(env, tag):
    n = env.num_envs
    out = []
    for key in ("articulations", "rigid_objects"):
        for aname, asset in getattr(env.scene, key, {}).items():
            scan(asset, f"scene.{aname}", n, out)
    for mname in ("action_manager", "command_manager", "event_manager", "observation_manager", "reward_manager",
                  "termination_manager"):
        scan(getattr(env, mname), mname, n, out)
    solver = NewtonManager._solver
    nworld = n
    mj = []
    for k, v in vars(solver.mjw_data).items():
        t = as_tensor(v)
        if t is not None and t.is_floating_point() and t.dim() >= 1 and t.shape[0] == nworld:
            if not bool(torch.isfinite(t[BAD]).all()):
                mj.append(k)
    st = []
    model = NewtonManager._model
    for sname, s in (("state_0", NewtonManager._state_0), ("state_1", NewtonManager._state_1),
                     ("control", NewtonManager._control)):
        for k, v in vars(s).items():
            t = as_tensor(v)
            if t is None or not t.is_floating_point() or t.numel() == 0:
                continue
            if not bool(torch.isfinite(t).all()):
                st.append(f"{sname}.{k}")
    print(f"[probe] {tag}: mjw_data non-finite (world {BAD}): {mj}", flush=True)
    print(f"[probe] {tag}: newton state/control non-finite (any world): {st}", flush=True)
    print(f"[probe] {tag}: env-{BAD} manager/asset non-finite: {out}", flush=True)


def obs_report(env, obs, tag):
    om = env.observation_manager
    for group, terms in om._group_obs_term_names.items():
        g = obs[group]
        if isinstance(g, dict):
            continue
        finite = bool(torch.isfinite(g[BAD]).all()) if g.is_floating_point() else True
        if not finite:
            bad_terms = []
            for tname, tcfg in zip(terms, om._group_obs_term_cfgs[group]):
                v = tcfg.func(env, **tcfg.params)
                if v.is_floating_point() and not bool(torch.isfinite(v[BAD]).all()):
                    bad_terms.append(tname)
            print(f"[probe] {tag}: obs group {group} non-finite in env {BAD}; terms {bad_terms}", flush=True)


def cols(x):
    t = as_tensor(x)
    if t is None or not t.is_floating_point():
        return None
    r = t[BAD].reshape(t[BAD].shape[0], -1) if t[BAD].dim() > 1 else t[BAD].reshape(-1, 1)
    return [i for i in range(r.shape[0]) if not bool(torch.isfinite(r[i]).all())]


def presolve(env, tag):
    solver = NewtonManager._solver
    n = env.num_envs
    def world_bad(obj):
        out = []
        for k, v in vars(obj).items():
            t = as_tensor(v)
            if t is not None and t.is_floating_point() and t.dim() >= 1 and t.shape[0] == n and not bool(torch.isfinite(t[BAD]).all()):
                out.append(k)
        return out
    def any_bad(obj):
        out = []
        for k, v in vars(obj).items():
            t = as_tensor(v)
            if t is not None and t.is_floating_point() and t.numel() and not bool(torch.isfinite(t).all()):
                out.append(f"{k}{tuple(t.shape)}:{int((~torch.isfinite(t)).sum())}")
        return out
    print(f"[trace] {tag}: mjw_data world-{BAD} non-finite {world_bad(solver.mjw_data)}", flush=True)
    print(f"[trace] {tag}: mjw_model non-finite (any) {any_bad(solver.mjw_model)}", flush=True)
    print(f"[trace] {tag}: newton model non-finite (any) {any_bad(NewtonManager._model)}", flush=True)
    print(f"[trace] {tag}: contacts non-finite (any) {any_bad(NewtonManager._contacts) if NewtonManager._contacts is not None else None}", flush=True)
    for name, obj in vars(NewtonManager).items():
        if obj is not None and hasattr(obj, "__dict__") and "collision" in type(obj).__name__.lower():
            print(f"[trace] {tag}: {name} ({type(obj).__name__}) non-finite {any_bad(obj)}", flush=True)
    def deep(obj, prefix, out, clean, depth=0):
        for k, v in vars(obj).items():
            t = as_tensor(v)
            if t is not None and t.is_floating_point() and t.numel():
                bad = ~torch.isfinite(t)
                if bad.any():
                    out.append(f"{prefix}{k}{tuple(t.shape)}:{int(bad.sum())}")
                    name = f"{prefix}{k}"
                    keep = (not args_cli.clean_only or name.startswith(tuple(args_cli.clean_only.split(",")))) and not (
                        args_cli.clean_except and name.startswith(args_cli.clean_except))
                    if clean and keep:
                        t[bad] = 0.0
            elif t is None and depth < 2 and hasattr(v, "__dict__") and "mujoco_warp" in type(v).__module__:
                deep(v, f"{prefix}{k}.", out, clean, depth + 1)
    out = []
    deep(solver.mjw_data, "", out, args_cli.clean_mjw)
    print(f"[trace] {tag}: mjw_data non-finite anywhere (clean={args_cli.clean_mjw}) {out}", flush=True)
    model = NewtonManager._model
    solver._update_mjc_data(solver.mjw_data, model, NewtonManager._state_0)
    qpos = wp.to_torch(solver.mjw_data.qpos)[BAD]
    qvel = wp.to_torch(solver.mjw_data.qvel)[BAD]
    print(f"[trace] {tag}: after manual _update_mjc_data world-{BAD} qpos bad idx {torch.nonzero(~torch.isfinite(qpos)).flatten().tolist()} "
          f"qvel bad idx {torch.nonzero(~torch.isfinite(qvel)).flatten().tolist()} (nq {qpos.numel()})", flush=True)


def trace_step(env, action):
    robot = env.scene["robot"]
    model = NewtonManager._model
    def phys(tag):
        q = wp.to_torch(NewtonManager._state_0.joint_q)
        qd = wp.to_torch(NewtonManager._state_0.joint_qd)
        f = wp.to_torch(NewtonManager._control.joint_f)
        bq = wp.to_torch(NewtonManager._state_0.body_q)
        print(f"[trace] {tag}: newton joint_q nonfinite={int((~torch.isfinite(q)).sum())} joint_qd={int((~torch.isfinite(qd)).sum())} "
              f"body_q={int((~torch.isfinite(bq)).sum())} control.joint_f={int((~torch.isfinite(f)).sum())} "
              f"(joint_f idx {torch.nonzero(~torch.isfinite(f)).flatten().tolist()[:20]})", flush=True)
    def data(tag):
        d = robot.data
        print(f"[trace] {tag}: robot env0 joint_pos bad={cols(d.joint_pos)} joint_vel bad={cols(d.joint_vel)} "
              f"body_pos_w bad={cols(d.body_pos_w)} root_pos_w bad={cols(d.root_pos_w)} "
              f"joint_effort_target bad={cols(d._joint_effort_target) if hasattr(d, '_joint_effort_target') else 'n/a'} "
              f"joint_pos_target bad={cols(d.joint_pos_target)} joint_vel_target bad={cols(d.joint_vel_target)} "
              f"computed_torque bad={cols(d.computed_torque)} applied_torque bad={cols(d.applied_torque)}", flush=True)
        for name, term in env.action_manager._terms.items():
            bad = {k: cols(v) for k, v in vars(term).items() if cols(v)}
            print(f"[trace] {tag}: action term {name} bad buffers {bad}", flush=True)
    print(f"[trace] joint names {robot.joint_names}", flush=True)
    phys("after reset step"); data("after reset step")
    env.action_manager.process_action(action)
    data("after process_action")
    for i in range(env.cfg.decimation):
        env.action_manager.apply_action()
        data(f"substep {i} after apply_action")
        env.scene.write_data_to_sim()
        phys(f"substep {i} after write_data_to_sim")
        presolve(env, f"substep {i} before sim.step")
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        phys(f"substep {i} after sim.step"); data(f"substep {i} after sim.step")
        q = wp.to_torch(NewtonManager._state_0.joint_q)
        labels = getattr(model, "joint_label", None) or getattr(model, "joint_key", None)
        qs = model.joint_q_start.numpy()
        badq = torch.nonzero(~torch.isfinite(q)).flatten().tolist()
        names = sorted({labels[int((qs <= b).nonzero()[0][-1])] for b in badq}) if labels is not None else badq
        print(f"[trace] substep {i}: non-finite joint_q coords {badq} -> joints {names}", flush=True)
        print(f"[trace] substep {i}: mjw qacc world-{BAD} bad {int((~torch.isfinite(wp.to_torch(NewtonManager._solver.mjw_data.qacc)[BAD])).sum())} "
              f"qvel bad {int((~torch.isfinite(wp.to_torch(NewtonManager._solver.mjw_data.qvel)[BAD])).sum())} "
              f"qpos bad {int((~torch.isfinite(wp.to_torch(NewtonManager._solver.mjw_data.qpos)[BAD])).sum())}", flush=True)
        if i >= 1:
            break


@hydra_task_config(args_cli.task, "flashsac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    import time
    for _ in range(3):
        obs, *_ = env.step(zero)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args_cli.warmup):
        obs, *_ = env.step(zero)
    torch.cuda.synchronize()
    print(f"[probe] warmup {args_cli.warmup} steps x {env.num_envs} envs: {(time.time() - t0) / args_cli.warmup * 1e3:.2f} ms/step", flush=True)
    report(env, "before inject")
    if args_cli.inject == "warmstart":
        wp.to_torch(NewtonManager._solver.mjw_data.qacc_warmstart)[BAD] = float("nan")
    elif args_cli.inject == "joint_qd":
        qd = wp.to_torch(NewtonManager._solver.mjw_data.qvel)
        qd[BAD] = float("nan")
    print(f"[probe] injected {args_cli.inject} NaN into world {BAD}", flush=True)
    for k in range(args_cli.steps):
        obs, rew, term, trunc, extras = env.step(zero)
        terms = {name: bool(env.termination_manager.get_term(name)[BAD]) for name in env.termination_manager.active_terms}
        print(f"[probe] step {k}: reward finite={bool(torch.isfinite(rew[BAD]))} terminated={bool(term[BAD])} "
              f"term terms={terms} env1 reward finite={bool(torch.isfinite(rew[1]))}", flush=True)
        obs_report(env, obs, f"step {k}")
        report(env, f"step {k}")
        if args_cli.trace and k == 0:
            trace_step(env, zero)
            break
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
