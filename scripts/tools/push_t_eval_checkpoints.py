# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic PushT eval of every RSL-RL checkpoint in a run, ManiSkill style (``success_once`` over one full episode
from a synchronized reset, mean action). Writes ``<run_dir>/eval_<num_envs>.csv``."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--train_envs", type=int, required=True, help="Envs of the training run (for the env-step axis).")
parser.add_argument("--every", type=int, default=1, help="Evaluate every k-th checkpoint (the last is always kept).")
parser.add_argument("--seed", type=int, default=123)
AppLauncher.add_app_launcher_args(parser)
# Isaac Lab 3.0 reads ``enable_cameras`` from the launcher args but no longer registers the flag
if not any("--enable_cameras" in a.option_strings for a in parser._actions):
    parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors.")
args = parser.parse_args()
app = AppLauncher(args).app

import csv  # noqa: E402
import glob  # noqa: E402
import gymnasium as gym  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.manager_based.manipulation.push_t.mdp import maniskill  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # score exactly one training episode; a longer env limit keeps the time-out reset out of the scored steps
    horizon = int(round(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))
    env_cfg.episode_length_s *= 2.0
    agent_cfg = load_cfg_from_registry(args.task, args.agent)
    is_flashsac = hasattr(agent_cfg, "action_bound") and hasattr(agent_cfg.algorithm, "buffer_min_length")
    if is_flashsac:  # Holiday-Robot rsl_rl_flashsac runner / wrapper; inference only: tiny buffer, no expert data
        from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
        from rsl_rl_flashsac.runners import OffPolicyRunner as FlashSACRunner

        env = FlashSACVecEnvWrapper(
            gym.make(args.task, cfg=env_cfg),
            action_bound=agent_cfg.action_bound,
            action_bound_scale=agent_cfg.action_bound_scale,
        )
        agent_cfg.algorithm.replay_buffer_size = 8 * args.num_envs
        if hasattr(agent_cfg.algorithm, "expert_buffer_path"):
            agent_cfg.algorithm.expert_buffer_path = ""
        runner = FlashSACRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    else:
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        if agent_cfg.class_name == "OffPolicyRunner":
            agent_cfg.algorithm.replay_buffer_size = 2 * args.num_envs  # inference only
        runner_cls = OffPolicyRunner if agent_cfg.class_name == "OffPolicyRunner" else OnPolicyRunner
        runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    base = env.unwrapped

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "model_*.pt")), key=lambda p: int(re.findall(r"\d+", p)[-1]))
    ckpts = ckpts[:: args.every] + ([ckpts[-1]] if (len(ckpts) - 1) % args.every else [])
    steps_per_it = args.train_envs * agent_cfg.num_steps_per_env
    out = os.path.join(args.run_dir, f"eval_{args.num_envs}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "env_steps", "success_once", "success_at_end", "final_intersection"])
        for ck in ckpts:
            it = int(re.findall(r"\d+", ck)[-1])
            if is_flashsac:
                runner.load(ck, load_cfg={"actor": True, "critic": True, "optimizer": False}, strict=True)
                runner.alg.eval_mode()
                policy = runner.alg.get_policy()
            else:
                runner.load(ck, load_cfg={"actor": True}, strict=True)
                policy = runner.get_inference_policy(device=base.device)
            obs, _ = env.reset()
            once = torch.zeros(args.num_envs, dtype=torch.bool, device=base.device)
            for _ in range(horizon):
                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs))
                succ = maniskill.intersection_success(base)
                once |= succ
            inter = maniskill.intersection_area(base)
            row = [it, (it + 1) * steps_per_it, once.float().mean().item(), succ.float().mean().item(), inter.mean().item()]
            w.writerow(row)
            f.flush()
            print(f"[eval] it {it:5d} steps {row[1] / 1e6:6.2f}M success_once {row[2]:.3f} at_end {row[3]:.3f}")
    print(f"[eval] wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
