# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Critic diagnostics of FlashSAC-RGB checkpoints: roll out the stochastic policy and compare Q with the discounted return
earned, the temperature / entropy, and Q for policy, expert, noisy and random actions in the same states."""

import argparse
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--agent", default="flashsac_cfg_entry_point")
p.add_argument("--checkpoints", required=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(p)
args_cli, hydra_args = p.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json
import os

import gymnasium as gym
import torch
from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
from rsl_rl_flashsac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_rl.flashsac import make_flashsac_env
from uwlab_rl.flashsac.expert_envs import attach_expert_envs, to_policy_space
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs, env_cfg.seed = args_cli.num_envs, args_cli.seed
    agent_cfg.algorithm.replay_buffer_size = 4 * args_cli.num_envs
    agent_cfg.algorithm.buffer_min_length = args_cli.num_envs
    agent_cfg.algorithm.use_compile = False
    env = make_flashsac_env(gym.make(args_cli.task, cfg=env_cfg), FlashSACVecEnvWrapper, agent_cfg.clip_actions,
                            agent_cfg.action_bound, agent_cfg.action_bound_scale)
    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    alg, base = runner.alg, env.unwrapped
    attach_expert_envs(runner, env, args_cli.task, agent_cfg.device)
    cmd, g, H = base.command_manager.get_term("task_command"), alg.gamma, int(base.max_episode_length)
    results = []
    for ck in args_cli.checkpoints.split(","):
        runner.load(ck, map_location=agent_cfg.device)
        torch.manual_seed(args_cli.seed)
        base.reset(seed=args_cli.seed)
        obs = env.get_observations().to(runner.device)
        rows, rew, logs, succ, done_at = [], [], [], torch.zeros(base.num_envs, dtype=torch.bool, device=base.device), None
        alive = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
        live = []
        q_of = lambda f, a: torch.minimum(*alg.critic.evaluate(f, a, training=False)[0])
        with torch.no_grad():
            for t in range(H):
                f = alg._encode(obs, alg.encoder)
                mean, std = alg.actor.get_mean_and_std(f, training=False)
                a_s, lp = alg._actor_sample(f, training=False)
                a_m = torch.tanh(mean)
                a_e = to_policy_space(alg.expert_policy(obs.select(alg.expert_obs_group)), alg.expert_action_scale, alg.expert_action_bias).float()
                noise = lambda s: (a_m + s * torch.randn_like(a_m)).clamp(-1, 1)
                rows.append(torch.stack([q_of(f, a_s), q_of(f, a_m), q_of(f, a_e), q_of(f, noise(0.05)), q_of(f, noise(0.2)),
                                         q_of(f, torch.empty_like(a_m).uniform_(-1, 1)), lp.reshape(-1), (a_m - a_e).pow(2).mean(1)], 1))
                logs.append(std)
                live.append(alive.clone())
                obs, r, d, _ = env.step(a_s)
                obs = obs.to(runner.device)
                rew.append(r.reshape(-1) * alive)
                succ |= (cmd.orientation_aligned & cmd.position_aligned & alive)
                alive &= ~d.bool().reshape(-1)
                if not bool(alive.any()):
                    break
        X, R, S, L = torch.stack(rows), torch.stack(rew), torch.stack(logs), torch.stack(live)
        G = torch.zeros_like(R)
        run = torch.zeros(base.num_envs, device=R.device)
        for t in range(R.shape[0] - 1, -1, -1):
            run = R[t] + g * run
            G[t] = run
        m = L & (torch.arange(R.shape[0], device=R.device)[:, None] < R.shape[0] - 40)
        c = lambda a, b: float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        q = X[..., 0]
        T10 = 10
        win = X[..., 2] > X[..., 1]
        out = dict(ckpt=os.path.basename(ck), alpha=float(alg.temperature()), success=float(succ.float().mean()),
                   undisc_reward=float(R.sum(0).mean()), reward_min_step=float(R.min()), G_mean=float(G[m].mean()), G_start=float(G[0].mean()),
                   Q_policy_mean=float(q[m].mean()), Q_start=float(q[0].mean()), Q_min=float(q[m].min()), Q_max=float(q[m].max()),
                   corr_Q_G_all=c(q[m], G[m]), corr_Q0_Gstart=c(q[T10], G[T10]), q_at_t10_succ=float(q[T10][succ].mean()), q_at_t10_fail=float(q[T10][~succ].mean()),
                   logp_mean=float(X[..., 6][m].mean()), entropy_cost_per_step=float(alg.temperature()) * float(X[..., 6][m].mean()),
                   std_per_dim=[round(float(v), 4) for v in S[m].mean(0)], policy_expert_mse=float(X[..., 7][m].mean()),
                   dQ_expert=float((X[..., 2] - X[..., 1])[m].mean()), frac_expert_preferred=float(win[m].float().mean()),
                   dQ_noise05=float((X[..., 3] - X[..., 1])[m].mean()), dQ_noise20=float((X[..., 4] - X[..., 1])[m].mean()),
                   dQ_random=float((X[..., 5] - X[..., 1])[m].mean()), frac_random_preferred=float((X[..., 5] > X[..., 1])[m].float().mean()))
        print("[probe] " + json.dumps(out), flush=True)
        results.append(out)
    if args_cli.out:
        json.dump(results, open(args_cli.out, "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
