# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that the peg pose a reset writes is what the success term captures as initial_pos, and what the sim simulates."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reset pose read-back probe.")
parser.add_argument("--task", type=str, default="OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0")
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.backend == "newton" and not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    n = env.num_envs
    peg = env.scene["insertive_object"]
    written = torch.full((n, 3), float("nan"), device=env.device)
    log = []

    for fn_name in ("write_root_state_to_sim", "write_root_pose_to_sim", "write_root_link_pose_to_sim"):
        if hasattr(peg, fn_name):
            orig = getattr(peg, fn_name)

            def wrapped(*a, _orig=orig, _name=fn_name, **kw):
                data = a[0] if a else kw.get("root_state", kw.get("root_pose"))
                ids = kw.get("env_ids", a[1] if len(a) > 1 else None)
                ids = torch.arange(n, device=env.device) if ids is None else torch.as_tensor(ids, device=env.device)
                d = data.torch if hasattr(data, "torch") else data
                written[ids] = d[:, :3].to(env.device)
                log.append(_name)
                return _orig(*a, **kw)

            setattr(peg, fn_name, wrapped)

    term = env.termination_manager.get_term_cfg("success").func
    orig_reset = term.reset
    captured = {}

    def reset(env_ids=None):
        orig_reset(env_ids)
        ids = torch.arange(n, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
        captured["ids"] = ids
        captured["init"] = peg.initial_pos[ids].clone()
        captured["written"] = written[ids].clone()
        captured["data_now"] = peg.data.root_pos_w.torch[ids].clone()

    term.reset = reset
    env.reset()
    actions = torch.zeros(env.action_space.shape, device=env.device)
    actions[:, -1] = -1.0
    sim_step = env.sim.step
    first = {}

    def step(render=True):
        sim_step(render=False)
        if "after1" not in first and "ids" in captured:
            first["after1"] = peg.data.root_pos_w.torch[captured["ids"]].clone()

    env.sim.step = step
    for k in range(int(env.max_episode_length) + 2):
        captured.clear()
        first.clear()
        env.step(actions)
        if "ids" in captured and k > 2:
            break
    ids = captured["ids"]

    def mm(x):
        return [round(float(v), 2) for v in torch.quantile(x * 1000, torch.tensor([0.5, 0.9, 1.0], device=x.device))]

    print(f"[probe] {args_cli.backend}: {len(ids)} reset envs; peg write calls seen: {sorted(set(log))}")
    print(
        "[probe]   |initial_pos - written|  mm q50/q90/max ="
        f" {mm((captured['init'] - captured['written']).norm(dim=1))}"
    )
    print(
        "[probe]   |data at reset - written| mm q50/q90/max ="
        f" {mm((captured['data_now'] - captured['written']).norm(dim=1))}"
    )
    print(
        "[probe]   (the next env.step's first physics substep is not captured here: resets happen at the end of a step)"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
