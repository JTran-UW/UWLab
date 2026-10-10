# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a task's MDP (reset, observations, commands, actions, rewards, terminations) at forced states.

Every env is reset to a fixed (reset type, dataset index) entry instead of a random one, so both backends
start from the same dataset states. Quantities are recorded before any physics step. Pair with
``compare_mdp.py``.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/mdp_probe.py --visualizer none --backend physx --out physx_mdp.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a task's MDP at forced dataset states.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_actions", type=int, default=4, help="Fixed raw actions evaluated at each state.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--step_check", action="store_true", help="Record robot/object poses after one zero-action env step."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.backend == "newton" and not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import events as omnireset_events
from uwlab_tasks.utils.hydra import hydra_task_compose

_FORCED: dict = {}


def _forced_call(self, env, env_ids, dataset_dir, reset_types, probs, success=None):
    """MultiResetManager.__call__ with the (task, state) entries taken from _FORCED."""
    if env_ids is None:
        env_ids = torch.arange(self.num_envs, device=self._env.device)
    task_ids = _FORCED["task_ids"].to(self.device)[env_ids]
    state_ids = _FORCED["state_ids"].to(self.device)[env_ids]
    self.task_id[env_ids] = task_ids
    for dataset_idx in range(self.num_tasks):
        mask = task_ids == dataset_idx
        if not mask.any():
            continue
        idx = state_ids[mask] % self.num_states[dataset_idx]
        states = omnireset_events.sample_from_nested_dict(self.datasets[dataset_idx], idx)
        self._reset_to(states["initial_state"], env_ids=env_ids[mask], is_relative=True)
        _FORCED.setdefault("written", {})[dataset_idx] = (env_ids[mask].cpu(), states["initial_state"])
    robot = self._env.scene["robot"]
    robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)


def _t(x):
    if x is None:
        return None
    if hasattr(x, "torch"):
        x = x.torch
    if not isinstance(x, torch.Tensor):
        import warp as wp

        if isinstance(x, wp.array):
            x = wp.to_torch(x)
    return x.detach().clone().cpu() if isinstance(x, torch.Tensor) else x


def _cpu_tree(x):
    if isinstance(x, dict):
        return {k: _cpu_tree(v) for k, v in x.items()}
    return _t(x)


def _state(env):
    scene = env.scene
    origins = scene.env_origins
    out = {"articulations": {}, "rigid_objects": {}}
    for name, a in scene.articulations.items():
        d = a.data
        root = _t(d.root_link_pose_w)
        root[:, :3] -= origins.cpu()
        bodies = _t(d.body_link_pose_w)
        bodies[..., :3] -= origins.cpu()[:, None]
        out["articulations"][name] = {
            "joint_names": list(a.joint_names),
            "body_names": list(a.body_names),
            "joint_pos": _t(d.joint_pos),
            "joint_vel": _t(d.joint_vel),
            "joint_pos_target": _t(d.joint_pos_target),
            "joint_vel_target": _t(d.joint_vel_target),
            "joint_stiffness": _t(d.joint_stiffness),
            "joint_damping": _t(d.joint_damping),
            "root_link_pose": root,
            "body_link_pose": bodies,
            "body_mass": _t(d.body_mass),
        }
    for name, o in scene.rigid_objects.items():
        root = _t(o.data.root_link_pose_w)
        root[:, :3] -= origins.cpu()
        out["rigid_objects"][name] = {
            "root_link_pose": root,
            "root_vel": _t(o.data.root_vel_w),
            "body_mass": _t(o.data.body_mass),
        }
    return out


def _observations(env):
    om = env.observation_manager
    obs = om.compute()
    out = {}
    for group, value in obs.items():
        if isinstance(value, dict):
            out[group] = {"terms": {k: _t(v) for k, v in value.items()}}
            continue
        value = _t(value)
        terms, start = {}, 0
        for term, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
            width = int(torch.tensor(dims).prod())
            terms[term] = value[:, start : start + width]
            start += width
        assert start == value.shape[1], (group, start, value.shape)
        out[group] = {"vector": value, "terms": terms}
    return out


def _commands(env):
    out = {}
    for name, term in env.command_manager._terms.items():
        entry = {"command": _t(term.command)}
        for attr, value in vars(term).items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (env.num_envs,):
                entry[attr] = _t(value)
        out[name] = entry
    return out


def _actions(env, raw_actions):
    robot = env.scene["robot"]
    am = env.action_manager
    out = []
    for raw in raw_actions:
        a = raw.to(env.device).repeat(env.num_envs, 1)
        am.process_action(a)
        am.apply_action()
        entry = {
            "raw": raw,
            "processed": {name: _t(am.get_term(name).processed_actions) for name in am.active_terms},
            "joint_effort_target": _t(robot.data.joint_effort_target),
            "joint_pos_target": _t(robot.data.joint_pos_target),
            "joint_vel_target": _t(robot.data.joint_vel_target),
        }
        out.append(entry)
    return out


def _rewards_terminations(env, raw_action):
    env.action_manager.process_action(raw_action.to(env.device).repeat(env.num_envs, 1))
    rm, tm = env.reward_manager, env.termination_manager
    total = _t(rm.compute(dt=env.step_dt))
    rewards = {name: _t(rm._step_reward[:, i]) for i, name in enumerate(rm.active_terms)}
    tm.compute()
    terms = {name: _t(tm.get_term(name)) for name in tm.active_terms}
    return {"reward_total": total, "reward_terms": rewards, "termination_terms": terms}


def _native_materials(env, backend):
    out = {}
    if backend == "physx":
        for name, asset in {**env.scene.articulations, **env.scene.rigid_objects}.items():
            out[name] = {"values": _t(asset.root_view.get_material_properties())}
        return out
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager._model
    out["shape_label"] = list(model.shape_label)
    for key in (
        "shape_material_mu",
        "shape_material_restitution",
        "shape_material_mu_torsional",
        "shape_material_mu_rolling",
    ):
        out[key] = _t(getattr(model, key))
    return out


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    for group in ("policy", "critic"):
        group_cfg = getattr(env_cfg.observations, group, None)
        if group_cfg is not None:
            group_cfg.enable_corruption = False

    gen = torch.Generator().manual_seed(args_cli.seed)
    num_tasks = len(env_cfg.events.reset_from_reset_states.params["reset_types"])
    _FORCED["task_ids"] = torch.arange(args_cli.num_envs) % num_tasks
    _FORCED["state_ids"] = torch.randint(0, 1 << 30, (args_cli.num_envs,), generator=gen)
    omnireset_events.MultiResetManager.__call__ = _forced_call

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    after_startup = {"state": _state(env), "materials": _native_materials(env, args_cli.backend)}
    torch.manual_seed(args_cli.seed)
    env.reset(seed=args_cli.seed)
    action_dim = env.action_manager.total_action_dim
    raw_actions = [torch.zeros(action_dim)] + [
        torch.rand(action_dim, generator=gen) * 2 - 1 for _ in range(args_cli.num_actions - 1)
    ]
    out = {
        "meta": {
            "task": args_cli.task,
            "backend": args_cli.backend,
            "num_envs": env.num_envs,
            "seed": args_cli.seed,
            "physics_cfg": repr(env_cfg.sim.physics),
            "robot_usd": env_cfg.scene.robot.spawn.usd_path,
            "step_dt": env.step_dt,
            "action_terms": env.action_manager.active_terms,
            "action_term_dim": env.action_manager.action_term_dim,
        },
        "forced": {
            "task_ids": _FORCED["task_ids"],
            "state_ids": _FORCED["state_ids"],
            "written": {k: (ids, _cpu_tree(s)) for k, (ids, s) in _FORCED.get("written", {}).items()},
        },
        "after_startup": after_startup,
        "after_reset": _state(env),
        "observations": _observations(env),
        "commands": _commands(env),
        "actions": _actions(env, raw_actions),
        "rewards_terminations": [_rewards_terminations(env, a) for a in raw_actions],
    }
    out["meta"]["joint_ordering"] = env_cfg.scene.robot.joint_ordering
    out["meta"]["body_ordering"] = env_cfg.scene.robot.body_ordering
    if args_cli.step_check:
        _, _, terminated, truncated, _ = env.step(torch.zeros(env.num_envs, action_dim, device=env.device))
        out["after_step"] = _state(env)
        out["after_step"]["terminated"] = _t(terminated)
        out["after_step"]["truncated"] = _t(truncated)
        out["after_step"]["termination_terms"] = {
            n: _t(env.termination_manager.get_term(n)) for n in env.termination_manager.active_terms
        }
    torch.save(out, args_cli.out)
    print(
        f"[mdp_probe] wrote {args_cli.out}: obs groups={list(out['observations'])} "
        f"actions={out['meta']['action_terms']} reward terms={list(out['rewards_terminations'][0]['reward_terms'])}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
