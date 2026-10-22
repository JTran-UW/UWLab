# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint and record its rollouts as a FlashSAC expert buffer.

Transitions go through the FlashSAC env wrapper (true final obs on time-outs) and FlashSAC's ``TorchUniformBuffer``
(n-step returns at insertion, raw rewards), using ``n_steps`` / ``gamma`` / obs groups from the task's
``flashsac_cfg_entry_point``. Load the output with ``uwlab_rl.flashsac.FlashSACDemo``.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL checkpoint (model_*.pt).")
parser.add_argument("--expert_agent", type=str, default="rsl_rl_cfg_entry_point", help="Expert's agent cfg entry point.")
parser.add_argument("--fs_agent", type=str, default="flashsac_cfg_entry_point", help="FlashSAC cfg defining the buffer.")
parser.add_argument("--obs_dtype", type=str, default=None, help="Buffer observation dtype, e.g. float16 for images.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_steps", type=int, default=200, help="Vector-env steps to record (num_envs per step).")
parser.add_argument("--eval_only", action="store_true", help="Only report success; record nothing.")
parser.add_argument("--num_episodes", type=int, default=4096, help="Episodes to finish with --eval_only.")
parser.add_argument("--output", type=str, default=None, help="Output .pt (required unless --eval_only).")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
# Isaac Lab 3.0 reads ``enable_cameras`` from the launcher args but no longer registers the flag
if not any("--enable_cameras" in a.option_strings for a in parser._actions):
    parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors.")
args = parser.parse_args()
if not args.eval_only and args.output is None:
    parser.error("--output is required unless --eval_only")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402

from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner  # noqa: E402
from rsl_rl_flashsac.storage import TorchUniformBuffer  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.manager_based.manipulation.push_t.mdp import maniskill  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    fs_cfg = load_cfg_from_registry(args.task, args.fs_agent)
    env = gym.make(args.task, cfg=env_cfg)
    # the FlashSAC action interface: clamp(a, -1, 1) * scale
    env = FlashSACVecEnvWrapper(env, action_bound=fs_cfg.action_bound, action_bound_scale=fs_cfg.action_bound_scale)
    base_env = env.unwrapped
    dev = base_env.device

    expert_cfg = load_cfg_from_registry(args.task, args.expert_agent)
    if expert_cfg.class_name == "OffPolicyRunner":
        # inference only: a token replay buffer (the training one would hold every obs group, images included)
        expert_cfg.algorithm.replay_buffer_size = 2 * args.num_envs
        runner = OffPolicyRunner(env, expert_cfg.to_dict(), log_dir=None, device=dev)
        runner.load(args.checkpoint, load_cfg={"actor": True})
    else:
        expert_cfg = handle_deprecated_rsl_rl_cfg(expert_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(env, expert_cfg.to_dict(), log_dir=None, device=dev)
        runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=dev)

    obs = env.get_observations()
    buffer = None
    if not args.eval_only:
        groups = sorted(set(fs_cfg.obs_groups["actor"]) | set(fs_cfg.obs_groups["critic"]))
        buffer = TorchUniformBuffer(
            obs=obs,
            num_actions=env.num_actions,
            n_step=fs_cfg.algorithm.n_steps,
            gamma=fs_cfg.algorithm.gamma,
            max_length=args.num_envs * args.num_steps,
            min_length=1,
            sample_batch_size=1,
            device=dev,
            obs_storage_dtype=getattr(torch, args.obs_dtype) if args.obs_dtype else None,
            store_groups=groups,
        )

    ep_len = torch.zeros(args.num_envs, device=dev)
    once = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
    first_succ = torch.zeros(args.num_envs, device=dev)
    n_ep = n_succ = 0
    lens_succ = []
    step = 0
    while (step < args.num_steps) if buffer is not None else (n_ep < args.num_episodes):
        with torch.inference_mode():
            actions = policy(obs).clamp(-1.0, 1.0)
        next_obs, rew, dones, extras = env.step(actions)
        # success_once per episode (also for tasks where success does not terminate); read before the reset
        # inside step() would hide it, so use the success term's pre-reset value when it terminates
        success = maniskill.intersection_success(base_env) | base_env.termination_manager.get_term("success")
        ep_len += 1
        done_ids = dones.bool()
        n_ep += int(done_ids.sum())
        first_succ = torch.where(success & ~once, ep_len, first_succ)
        once |= success
        n_succ += int((once & done_ids).sum())
        lens_succ.append(first_succ[once & done_ids].clone())
        ep_len[done_ids] = 0
        once[done_ids] = False

        if buffer is not None:
            time_outs = extras["time_outs"].float().reshape(-1)
            mask = time_outs.bool()
            # the wrapper only provides time_outs_obs after the first reset, as FlashSAC.process_env_step allows
            if "time_outs_obs" in extras:
                true_next = TensorDict(
                    {
                        k: torch.where(mask.view(-1, *[1] * (next_obs[k].ndim - 1)), extras["time_outs_obs"][k], next_obs[k])
                        for k in next_obs.keys()
                    },
                    batch_size=next_obs.batch_size,
                )
            else:
                true_next = next_obs
            dones_f = dones.float().reshape(-1)
            buffer.add(
                {
                    "observation": obs,
                    "action": actions,
                    "reward": rew.reshape(-1),
                    "terminated": (dones_f - time_outs).clamp(min=0.0),
                    "truncated": time_outs,
                    "next_observation": true_next,
                }
            )
        obs = next_obs
        step += 1

    lens = torch.cat(lens_succ)
    success_rate = n_succ / max(n_ep, 1)
    print(
        f"[expert] {args.checkpoint}: success {success_rate:.4f} over {n_ep} episodes;"
        f" first-success step mean {lens.mean().item() if len(lens) else float('nan'):.1f}"
    )

    if buffer is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        buffer.save(args.output)
        data = torch.load(args.output, weights_only=False)
        data["metadata"] = {
            "task": args.task,
            "checkpoint": os.path.abspath(args.checkpoint),
            "n_step": fs_cfg.algorithm.n_steps,
            "gamma": fs_cfg.algorithm.gamma,
            "obs_groups": buffer._store_groups,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "episodes": n_ep,
            "success_rate": success_rate,
        }
        torch.save(data, args.output)
        print(f"[expert] wrote {data['num_in_buffer']} transitions to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
