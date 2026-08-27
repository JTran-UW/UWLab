# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a goal-conditioned RSL-RL policy and report a success rate.

``play.py`` renders but reports no metric, so there is no way to score a GC checkpoint from it.
This runs a fixed number of episodes per env and reports the success rate under the SAME criterion
the run was trained on -- ``GCProgressContext.success``, read off the reward manager, which already
reflects whatever ``require_orientation`` / ``position_threshold`` overrides were used.

Success is counted per env with surplus episodes discarded: an episode scores if the success
condition held at ANY step (``ever_success``), matching how the training-time
``metrics/task_N_success_rate`` is fed. Counting globally instead would bias the rate, because
successful episodes are not the same length as failed ones.

Example::

    python eval_gc.py --task <same task as training> --num_envs 512 --headless \\
        --checkpoint <path>/model_700.pt --episodes_per_env 2 \\
        env.rewards.progress_context.params.require_orientation=false \\
        env.rewards.progress_context.params.position_threshold=0.03
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a goal-conditioned RSL-RL agent.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--episodes_per_env", type=int, default=2, help="Episodes to score per env.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    resume_path = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[eval_gc] loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # The success criterion actually used by the run, including any Hydra overrides.
    ctx = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
    print(
        f"[eval_gc] criterion: require_orientation="
        f"{env.unwrapped.reward_manager.get_term_cfg('progress_context').params.get('require_orientation')}"
        f"  position_threshold="
        f"{env.unwrapped.reward_manager.get_term_cfg('progress_context').params.get('position_threshold')}"
    )

    device = env.unwrapped.device
    n = env.unwrapped.num_envs
    target = args_cli.episodes_per_env

    ever_success = torch.zeros(n, dtype=torch.bool, device=device)
    ep_count = torch.zeros(n, dtype=torch.long, device=device)
    succ_count = torch.zeros(n, dtype=torch.long, device=device)
    term_count = torch.zeros(n, dtype=torch.long, device=device)
    ep_steps = torch.zeros(n, dtype=torch.long, device=device)
    success_steps: list[int] = []

    # Per-reset-path tallies, under the TERMINAL criterion (success at the episode's last step)
    # -- the quantity `metrics/task_N_success_rate` is fed from. task_id is re-sampled during the
    # in-step reset, so the path each episode belongs to is captured at its START.
    reset_term = env.unwrapped.event_manager.get_term_cfg("reset_from_reset_states").func
    num_tasks = reset_term.num_tasks
    task_at_start = reset_term.task_id.clone()
    task_ep = torch.zeros(num_tasks, dtype=torch.long, device=device)
    task_term_succ = torch.zeros(num_tasks, dtype=torch.long, device=device)

    obs = env.get_observations()  # RslRlVecEnvWrapper returns a TensorDict, not a tuple
    while bool((ep_count < target).any()):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            ep_steps += 1
            ever_success |= ctx.success.to(device).bool()

            done_mask = dones.bool()
            if bool(done_mask.any()):
                # only score episodes for envs that still owe us one; surplus is discarded
                counting = done_mask & (ep_count < target)
                terminal = ctx.success.to(device).bool()  # success at the episode's final step
                succ_count += (ever_success & counting).long()
                term_count += (terminal & counting).long()
                for t in range(num_tasks):
                    sel = counting & (task_at_start == t)
                    task_ep[t] += sel.sum()
                    task_term_succ[t] += (terminal & sel).sum()
                task_at_start[done_mask] = reset_term.task_id[done_mask]
                ep_count += counting.long()
                for k in torch.nonzero(ever_success & counting).flatten().tolist():
                    success_steps.append(int(ep_steps[k].item()))
                ever_success[done_mask] = False
                ep_steps[done_mask] = 0

    total = int(ep_count.sum().item())
    succ = int(succ_count.sum().item())
    rate = succ / total if total else 0.0
    stderr = (rate * (1 - rate) / total) ** 0.5 if total else 0.0
    print("\n" + "=" * 62)
    print(f"[eval_gc] episodes scored : {total}  ({n} envs x {target})")
    print(f"[eval_gc] successes       : {succ}")
    print(f"[eval_gc] SUCCESS RATE (ever)    : {rate:.4f}  +/- {stderr:.4f} (1 s.e.)")
    trate = int(term_count.sum().item()) / total if total else 0.0
    tse = (trate * (1 - trate) / total) ** 0.5 if total else 0.0
    print(f"[eval_gc] SUCCESS RATE (terminal): {trate:.4f}  +/- {tse:.4f}  <- compare to wandb")
    for t in range(num_tasks):
        n_t = int(task_ep[t].item())
        r_t = int(task_term_succ[t].item()) / n_t if n_t else float("nan")
        se_t = (r_t * (1 - r_t) / n_t) ** 0.5 if n_t else 0.0
        print(f"[eval_gc]   task_{t} terminal    : {r_t:.4f}  +/- {se_t:.4f}  ({n_t} eps)")
    if success_steps:
        mean_step = sum(success_steps) / len(success_steps)
        print(f"[eval_gc] mean steps to first success (successful eps): {mean_step:.1f}")
    print("=" * 62 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
