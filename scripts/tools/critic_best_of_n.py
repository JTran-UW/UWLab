# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Best-of-N probe: roll out FlashSAC-RGB checkpoints acting with one policy sample, or the best / worst / a random one of
N policy samples under the (min of two) vision critic, from the same starting states; success per arm."""

import argparse
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--agent", default="flashsac_cfg_entry_point")
p.add_argument("--checkpoints", required=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--n", type=int, default=16)
p.add_argument("--arms", default="plain,best,worst,random")
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
    cmd, H = base.command_manager.get_term("task_command"), int(base.max_episode_length)
    results = []
    n = args_cli.n
    for ck in args_cli.checkpoints.split(","):
        runner.load(ck, map_location=agent_cfg.device)
        for arm in args_cli.arms.split(","):
            torch.manual_seed(args_cli.seed)
            base.reset(seed=args_cli.seed)
            obs = env.get_observations().to(runner.device)
            B = base.num_envs
            succ, alive, hold = (torch.zeros(B, dtype=torch.bool, device=base.device) for _ in range(3))
            alive[:] = True
            end = torch.zeros(B, dtype=torch.bool, device=base.device)
            rew, spread, rng, gap, act_std = torch.zeros(B, device=base.device), [], [], [], []
            with torch.no_grad():
                for t in range(H):
                    f = alg._encode(obs, alg.encoder)
                    mean, std = alg.actor.get_mean_and_std(f, training=False)
                    # N samples of the deployed policy, one critic value each
                    z = torch.randn(n, *mean.shape, device=mean.device)
                    cand = torch.tanh(mean + std * z)
                    q = torch.minimum(*alg.critic.evaluate(f.repeat(n, 1), cand.reshape(n * B, -1), training=False)[0]).view(n, B)
                    spread.append(q.std(0)[alive].mean()), rng.append((q.max(0)[0] - q.min(0)[0])[alive].mean()), act_std.append(std[alive].mean())
                    pick = {"plain": torch.zeros(B, dtype=torch.long, device=q.device), "best": q.argmax(0), "worst": q.argmin(0),
                            "random": torch.randint(n, (B,), device=q.device)}[arm]
                    a = cand[pick, torch.arange(B, device=q.device)]
                    torch.manual_seed(1000 * args_cli.seed + t)
                    obs, r, d, _ = env.step(a)
                    obs = obs.to(runner.device)
                    d = d.bool().reshape(-1)
                    ok = cmd.orientation_aligned & cmd.position_aligned
                    rew += r.reshape(-1) * alive
                    succ |= ok & alive & ~d
                    end = torch.where(alive & ~d, ok, end)  # aligned at the last live step
                    alive &= ~d
                    if not bool(alive.any()):
                        break
            m = float(succ.float().mean())
            out = dict(ckpt=os.path.basename(ck), arm=arm, n=n, envs=B, success_any=m, se=float((m * (1 - m) / B) ** 0.5), success_end=float(end.float().mean()), reward=float(rew.mean()),
                       q_std_among_samples=float(torch.stack(spread).mean()), q_range_among_samples=float(torch.stack(rng).mean()), policy_std=float(torch.stack(act_std).mean()))
            print("[bon] " + json.dumps(out), flush=True)
            results.append(out)
    if args_cli.out:
        json.dump(results, open(args_cli.out, "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
