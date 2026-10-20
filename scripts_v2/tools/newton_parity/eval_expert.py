# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint on a backend: per-reset-type success and every abnormal_robot termination.

Episodes use the task's own random resets. Success is read when an episode resets, from the same expression the
reset manager logs (``progress_context`` success).

.. code-block:: bash

    python scripts_v2/tools/newton_parity/eval_expert.py --visualizer none --backend newton \\
        --checkpoint <url or path> --out newton_expert.pt
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint per reset type.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episodes_per_env", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hole_sdf", type=int, default=None, help="Newton SDF max resolution for the hole collision mesh.")
parser.add_argument(
    "--peg_sdf", type=int, default=None, help="Collide the peg as an SDF mesh with this max resolution."
)
parser.add_argument(
    "--log_vel_steps", type=int, default=0, help="Record per-joint max |qd|/limit over envs for the first N steps."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.backend == "newton" and not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import time
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import events as omnireset_events
from uwlab_tasks.utils.hydra import hydra_task_compose

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "reinforcement_learning", "rsl_rl")
)
import cli_args  # noqa: E402  isort: skip

_EPISODES: list = []
_ABNORMAL: list = []
_TRACK: dict = {}
TABLE_TOP, TABLE_X, TABLE_Y = -0.013, (-0.14, 0.94), (-0.68, 0.68)


def _wrap_reset_manager():
    original = omnireset_events.MultiResetManager.__call__

    def call(self, env, env_ids, dataset_dir, reset_types, probs, success=None):
        if env_ids is not None and success is not None:
            lengths = env.episode_length_buf[env_ids]
            done = lengths > 0
            if done.any():
                ok = eval(success)[env_ids]
                through = _TRACK["through"][env_ids] if "through" in _TRACK else torch.zeros_like(ok)
                off = _TRACK["off"][env_ids] if "off" in _TRACK else torch.zeros_like(ok)
                _EPISODES.append(
                    torch.stack(
                        [
                            self.task_id[env_ids][done].float(),
                            ok[done].float(),
                            lengths[done].float(),
                            through[done].float(),
                            off[done].float(),
                            _TRACK["ep"][env_ids][done].float(),
                        ],
                        1,
                    ).cpu()
                )
                _TRACK["ep"][env_ids[done]] += 1
                if "through" in _TRACK:
                    _TRACK["through"][env_ids] = False
                    _TRACK["off"][env_ids] = False
        return original(self, env, env_ids, dataset_dir, reset_types, probs, success)

    omnireset_events.MultiResetManager.__call__ = call


def _wrap_abnormal(term_cfg, joint_names, task_ids_fn):
    original = term_cfg.func

    def abnormal(env, **kwargs):
        out = original(env, **kwargs)
        if out.any():
            robot = env.scene["robot"]
            ratio = robot.data.joint_vel.torch.abs() / (2.0 * robot.data.joint_vel_limits.torch)
            ids = out.nonzero().flatten()
            value, joint = ratio[ids].max(dim=1)
            for i, v, j in zip(ids.tolist(), value.tolist(), joint.tolist()):
                _ABNORMAL.append({
                    "env": i,
                    "task": int(task_ids_fn()[i]),
                    "joint": joint_names[j],
                    "ratio": v,
                    "step": int(env.episode_length_buf[i]),
                })
        return out

    term_cfg.func = abnormal


@hydra_task_compose(args_cli.task, args_cli.agent, hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.hole_sdf or args_cli.peg_sdf:
        from isaaclab_newton.sim.schemas import NewtonSDFCollisionPropertiesCfg

        if args_cli.hole_sdf:
            env_cfg.scene.receptive_object.spawn.collision_props = NewtonSDFCollisionPropertiesCfg(
                sdf_max_resolution=args_cli.hole_sdf
            )
        if args_cli.peg_sdf:
            peg_spawn = env_cfg.scene.insertive_object.spawn
            if hasattr(peg_spawn, "collision_approximation"):
                peg_spawn.collision_approximation = "sdf"
            peg_spawn.collision_props = NewtonSDFCollisionPropertiesCfg(sdf_max_resolution=args_cli.peg_sdf)
    _wrap_reset_manager()
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    reset_term = unwrapped.event_manager.get_term_cfg("reset_from_reset_states").func
    if env_cfg.terminations.abnormal_robot is not None:
        _wrap_abnormal(
            unwrapped.termination_manager.get_term_cfg("abnormal_robot"),
            list(unwrapped.scene["robot"].joint_names),
            lambda: reset_term.task_id,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=unwrapped.device)
    steps = int(args_cli.episodes_per_env * unwrapped.max_episode_length) + 1
    obs = env.get_observations()
    robot = unwrapped.scene["robot"]
    peg = unwrapped.scene["insertive_object"]
    _TRACK["through"] = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
    _TRACK["off"] = torch.zeros_like(_TRACK["through"])
    _TRACK["ep"] = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
    vel_log = []
    t0 = time.time()
    with torch.inference_mode():
        for k in range(steps):
            p = peg.data.root_link_pose_w.torch[:, :3] - unwrapped.scene.env_origins
            inside = (p[:, 0] > TABLE_X[0]) & (p[:, 0] < TABLE_X[1]) & (p[:, 1] > TABLE_Y[0]) & (p[:, 1] < TABLE_Y[1])
            below = p[:, 2] < TABLE_TOP - 0.03
            _TRACK["through"] |= inside & below
            _TRACK["off"] |= ~inside & below
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            if k < args_cli.log_vel_steps:
                ratio = robot.data.joint_vel.torch.abs() / robot.data.joint_vel_limits.torch
                vel_log.append(torch.stack([ratio.max(0).values, ratio.quantile(0.5, dim=0)]).cpu())
    wall = time.time() - t0
    episodes = torch.cat(_EPISODES) if _EPISODES else torch.zeros(0, 6)
    reset_types = env_cfg.events.reset_from_reset_states.params["reset_types"]
    out = {
        "meta": {
            "task": args_cli.task,
            "backend": args_cli.backend,
            "checkpoint": args_cli.checkpoint,
            "num_envs": args_cli.num_envs,
            "steps": steps,
            "wall_s": wall,
            "env_steps_per_s": steps * args_cli.num_envs / wall,
            "reset_types": reset_types,
            "physics_cfg": repr(env_cfg.sim.physics),
        },
        "episodes": episodes,
        "abnormal": _ABNORMAL,
        "vel_log": torch.stack(vel_log) if vel_log else None,
        "joint_names": list(robot.joint_names),
    }
    torch.save(out, args_cli.out)
    print(
        f"[eval_expert] {args_cli.backend}: through-table {int(episodes[:, 3].sum())}, off-table falls"
        f" {int(episodes[:, 4].sum())}"
    )
    print(
        f"[eval_expert] {args_cli.backend}: {len(episodes)} episodes, {len(_ABNORMAL)} abnormal, "
        f"{out['meta']['env_steps_per_s']:.0f} env-steps/s"
    )
    for t, name in enumerate(reset_types):
        sel = episodes[:, 0] == t
        n = int(sel.sum())
        rate = float(episodes[sel, 1].mean()) if n else float("nan")
        first, later = sel & (episodes[:, 5] == 0), sel & (episodes[:, 5] > 0)
        split = " | ".join(
            f"{tag} {float(episodes[m, 1].mean()) if m.any() else float('nan'):.3f} (n={int(m.sum())})"
            for tag, m in (("first ep", first), ("later eps", later))
        )
        print(f"[eval_expert]   {name:34s} success {rate:.3f} over {n} episodes | {split}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
