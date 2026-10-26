# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic success rate of an rsl_rl FlashSAC checkpoint: every completed episode over ``--rounds`` x horizon
steps, per reset type. An env that terminates on a step is auto-reset inside ``env.step``, so its alignment on that step
belongs to the next episode and is not counted."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Success rate of an rsl_rl FlashSAC checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="flashsac_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True, help="Output .json.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--rounds", type=int, default=8)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--stochastic", action="store_true", help="Sample actions (FlashSACActor only) instead of the mean.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json
import math
import os
import time

import gymnasium as gym
import torch

from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
from rsl_rl_flashsac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_rl.flashsac import make_flashsac_env
from uwlab_rl.flashsac.rgb import _PixelPolicy
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.algorithm.expert_buffer_path = ""
    agent_cfg.algorithm.replay_buffer_size = 4 * args_cli.num_envs
    agent_cfg.algorithm.buffer_min_length = args_cli.num_envs
    env = make_flashsac_env(
        gym.make(args_cli.task, cfg=env_cfg), FlashSACVecEnvWrapper, agent_cfg.clip_actions, agent_cfg.action_bound,
        agent_cfg.action_bound_scale,
    )
    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
    actor = runner.alg.get_policy().eval()
    env_units = isinstance(actor, _PixelPolicy)

    assert not (args_cli.stochastic and env_units), "--stochastic is implemented for FlashSACActor policies only"

    def policy(o):
        a = actor(o, stochastic_output=True) if args_cli.stochastic else actor(o)
        return (a - env.action_bias) / env.action_scale if env_units else a

    base = env.unwrapped
    cmd = base.command_manager.get_term("task_command")
    reset_term = base.event_manager.get_term_cfg("reset_from_reset_states").func
    names = [rt.replace("Object", "").replace("EEGrasped", "") for rt in
             base.event_manager.get_term_cfg("reset_from_reset_states").params["reset_types"]]
    horizon = int(base.max_episode_length)
    obs = env.get_observations()
    task_cur = reset_term.task_id.clone()
    succ = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    done_task, done_succ = [], []
    t_start = time.time()
    with torch.no_grad():
        for _ in range(args_cli.rounds * horizon):
            obs, _, dones, _ = env.step(policy(obs))
            term = dones.bool()
            succ |= cmd.orientation_aligned & cmd.position_aligned & ~term
            if term.any():
                done_task.append(task_cur[term].clone())
                done_succ.append(succ[term].clone())
                succ[term] = False
                task_cur[term] = reset_term.task_id[term]
    rate_sps = args_cli.rounds * horizon * base.num_envs / (time.time() - t_start)
    print(f"[eval_success] rollout steps/s={rate_sps:.0f}", flush=True)
    tasks, wins = torch.cat(done_task).cpu(), torch.cat(done_succ).float().cpu()

    def rate(mask):
        n = int(mask.sum())
        p = float(wins[mask].mean()) if n else float("nan")
        return {"success": p, "n": n, "ci95": 1.96 * math.sqrt(p * (1 - p) / n) if n else float("nan")}

    result = {"checkpoint": args_cli.checkpoint, "task": args_cli.task, "hydra": hydra_args, "stochastic": args_cli.stochastic,
              "overall": rate(torch.ones_like(wins, dtype=torch.bool)),
              "per_reset": {names[k]: rate(tasks == k) for k in range(len(names))}}
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    json.dump(result, open(args_cli.out, "w"), indent=1)
    o = result["overall"]
    print(f"[eval_success] overall {o['success']:.3f} +- {o['ci95']:.3f} over {o['n']} episodes", flush=True)
    for k, v in result["per_reset"].items():
        print(f"[eval_success]   {k}: {v['success']:.3f} (n={v['n']})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
