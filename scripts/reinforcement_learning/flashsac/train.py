# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train FlashSAC (Holiday-Robot/rsl_rl_flashsac) on a task with a ``flashsac_cfg_entry_point``.

Follows Holiday-Robot/rsl_rl_flashsac ``isaaclab_flashsac/scripts/train.py``; agent cfg overrides go through Hydra
(e.g. ``agent.algorithm.mini_batch_size=1024``).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with FlashSAC.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# FlashSAC's shared hyperparameters assume 1024 envs (two updates per env step)
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument(
    "--agent", type=str, default="flashsac_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and agent.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--lr_decay_iterations", type=int, default=None,
                    help="Cosine LR decay length in iterations (default: max_iterations); the LR then holds learning_rate_end.")
parser.add_argument("--lr_hold_iterations", type=int, default=None,
                    help="Hold learning_rate_peak for this many iterations (replaces the warmup) before the cosine decay.")
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment folder under logs/flashsac.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
parser.add_argument("--logger", type=str, default=None, choices={"tensorboard", "wandb"}, help="Logger to use.")
parser.add_argument("--log_project_name", type=str, default=None, help="W&B project (default: experiment name).")
parser.add_argument("--resume_path", type=str, default=None, help="Checkpoint to resume training from.")
parser.add_argument(
    "--reset_action_scaling", action="store_true", default=False,
    help="With --resume_path: re-apply the env's current action bounds to the loaded actor (finetune with new bounds).",
)
parser.add_argument(
    "--reset_lr_schedule", action="store_true", default=False,
    help="With --resume_path: restart every LR schedule from step 0 (optimizer moments are kept).",
)
parser.add_argument(
    "--freeze_policy", action="store_true", default=False,
    help="Diagnostic: collect rollouts and log metrics without any update (weights and BatchNorm stats stay fixed).",
)
parser.add_argument(
    "--buffer_device", type=str, default=None, help="Replay buffer storage device, e.g. cpu (default: training device)."
)
parser.add_argument(
    "--log_dir_name", type=str, default=None, help="Fixed run dir under the experiment (instead of a timestamp)."
)
AppLauncher.add_app_launcher_args(parser)
# Isaac Lab 3.0 reads ``enable_cameras`` from the launcher args but no longer registers the flag
if not any("--enable_cameras" in a.option_strings for a in parser._actions):
    parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors.")
if not any("--distributed" in a.option_strings for a in parser._actions):
    parser.add_argument("--distributed", action="store_true", default=False, help="One process per GPU (torch.distributed.run).")
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_flashsac.rl_cfg import FlashSACRunnerCfg
from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
from uwlab_rl.flashsac import make_flashsac_env
from uwlab_rl.flashsac.expert_envs import attach_expert_envs
from rsl_rl_flashsac.runners import OffPolicyRunner

from uwlab_tasks.utils.hydra import hydra_task_config

# matmul settings of the official FlashSAC training setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: FlashSACRunnerCfg):
    """Train with FlashSAC."""
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.lr_decay_iterations is not None:
        alg = agent_cfg.algorithm
        alg.learning_rate_decay_steps = (
            args_cli.lr_decay_iterations * getattr(alg, "num_learning_epochs", 1) * getattr(alg, "num_mini_batches", 1)
        )
    if args_cli.lr_hold_iterations is not None:
        alg = agent_cfg.algorithm
        alg.learning_rate_init = alg.learning_rate_peak
        alg.learning_rate_warmup_steps = (
            args_cli.lr_hold_iterations * getattr(alg, "num_learning_epochs", 1) * getattr(alg, "num_mini_batches", 1)
        )
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    if args_cli.buffer_device is not None:
        agent_cfg.algorithm.buffer_device = args_cli.buffer_device
    if args_cli.distributed:
        # one process per GPU (torch.distributed.run): own device, own env / init seed
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.seed = agent_cfg.seed + app_launcher.local_rank
    if args_cli.logger == "wandb":
        agent_cfg.logger = {
            "class_name": "WandbLogWriter",
            "project_name": args_cli.log_project_name or agent_cfg.experiment_name,
        }
    elif args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = agent_cfg.device

    torch.manual_seed(agent_cfg.seed)
    torch.cuda.manual_seed_all(agent_cfg.seed)

    log_root_path = os.path.abspath(os.path.join("logs", "flashsac", agent_cfg.experiment_name))
    if args_cli.log_dir_name:
        log_dir = args_cli.log_dir_name
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = make_flashsac_env(
        env, FlashSACVecEnvWrapper, agent_cfg.clip_actions, agent_cfg.action_bound, agent_cfg.action_bound_scale
    )

    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if args_cli.resume_path is not None:
        resume_path = retrieve_file_path(args_cli.resume_path)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        if args_cli.reset_action_scaling:
            old_scale = runner.alg.actor.action_scale.detach().clone()
            runner.alg.actor.set_action_scaling(env.action_bias.to(runner.alg.actor.action_scale.device),
                                                env.action_scale.to(runner.alg.actor.action_scale.device))
            print(f"[INFO] Actor action scaling reset: {old_scale.tolist()} -> {runner.alg.actor.action_scale.tolist()}")
        if args_cli.reset_lr_schedule:
            for name in ("actor_scheduler", "critic_scheduler", "temperature_scheduler", "encoder_scheduler"):
                sched = getattr(runner.alg, name, None)
                if sched is None:
                    continue
                sched.last_epoch = 0
                sched._last_lr = [base * fn(0) for base, fn in zip(sched.base_lrs, sched.lr_lambdas)]
                for group, lr in zip(sched.optimizer.param_groups, sched._last_lr):
                    group["lr"] = lr
                print(f"[INFO] {name} reset to step 0: lr {sched._last_lr}")
    # expert envs: a state expert drives the reset event's expert_env_every subset (agent.algorithm.expert_policy_checkpoint)
    attach_expert_envs(runner, env, args_cli.task, agent_cfg.device)
    if hasattr(runner.alg, "replay_audit_dir"):
        runner.alg.replay_audit_dir = os.path.join(log_dir, "replay_audit")
        runner.alg._audit_num_envs = env.num_envs

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # random initial episode lengths decorrelate resets across envs (training only), as in the official FlashSAC;
    # ManiSkill-parity cfgs set init_at_random_ep_len=False (all envs reset together)
    # on resume, train only the remaining iterations of the original budget
    if args_cli.freeze_policy:
        runner.alg.update = lambda: {}
    runner.learn(
        num_learning_iterations=max(agent_cfg.max_iterations - runner.current_learning_iteration, 0),
        init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", True),
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
