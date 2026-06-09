# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    help="Unique run identifier (e.g., SLURM_JOB_ID). If provided, uses this as the directory name "
    "instead of a timestamp, enabling automatic resumption when a job is requeued.",
)
parser.add_argument(
    "--resume_path", type=str, default=None,
    help="Direct path to a checkpoint file to resume from (bypasses log directory search).",
)
parser.add_argument(
    "--init_weights", type=str, default=None,
    help="Path to a fixed initial policy/critic weights file. If it exists, the policy is "
    "initialized from it on every run (byte-identical params across seeds/machines/GPUs); "
    "if it does not exist, the freshly-initialized weights are saved there once to mint the "
    "canonical init. Ignored when resuming. Used for diversity ablations that need to hold "
    "network initialization fixed.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Inject UWLab distillation classes into rsl_rl's distillation_runner module so
# the runner's eval(class_name) lookup resolves them.
import rsl_rl.runners.distillation_runner as _distillation_runner_module

from uwlab_rl.rsl_rl.distillation_dagger import DistillationDAgger
from uwlab_rl.rsl_rl.distillation_dagger_weighted import DistillationDAggerWeighted
from uwlab_rl.rsl_rl.distillation_runner_split import DistillationRunnerSplit
from uwlab_rl.rsl_rl.student_teacher_mlp import StudentTeacherMLP
from uwlab_rl.rsl_rl.student_teacher_vision import StudentTeacherVision
from uwlab_rl.rsl_rl.student_teacher_vision_recurrent import StudentTeacherVisionRecurrent

_distillation_runner_module.StudentTeacherVision = StudentTeacherVision
_distillation_runner_module.StudentTeacherVisionRecurrent = StudentTeacherVisionRecurrent
_distillation_runner_module.StudentTeacherMLP = StudentTeacherMLP
_distillation_runner_module.DistillationDAgger = DistillationDAgger
_distillation_runner_module.DistillationDAggerWeighted = DistillationDAggerWeighted
_distillation_runner_module.DistillationRunnerSplit = DistillationRunnerSplit

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

import rsl_rl.runners.on_policy_runner as _runner_module
from uwlab_rl.rsl_rl.actor_critic_encoder import ActorCriticWithEncoder
from uwlab_rl.rsl_rl.on_policy_runner_with_classifier import OnPolicyRunnerWithClassifier
from uwlab_rl.rsl_rl.on_policy_runner_with_success_critic import OnPolicyRunnerWithSuccessCritic
from uwlab_rl.rsl_rl.success_critic_only_runner import SuccessCriticOnlyRunner
_runner_module.ActorCriticWithEncoder = ActorCriticWithEncoder
from uwlab_rl.rsl_rl.actor_critic_depth import ActorCriticDepth
from uwlab_rl.rsl_rl.bc_ppo import BCPPO
from uwlab_rl.rsl_rl.grpo import GRPO
from uwlab_rl.rsl_rl.ppo_pbrs import PPOPBRS
_runner_module.ActorCriticDepth = ActorCriticDepth
_runner_module.BCPPO = BCPPO
_runner_module.GRPO = GRPO
_runner_module.PPOPBRS = PPOPBRS
from uwlab_rl.rsl_rl.actor_critic_rma import ActorCriticRMA
from uwlab_rl.rsl_rl.ppo_rma import PPO_RMA
_runner_module.ActorCriticRMA = ActorCriticRMA
_runner_module.PPO_RMA = PPO_RMA
# Also expose PPO_RMA via rsl_rl.algorithms so cli_args.sanitize_rsl_rl_cfg can
# look it up and strip kwargs the upstream PPO base doesn't accept.
import rsl_rl.algorithms as _rsl_rl_algorithms
_rsl_rl_algorithms.PPO_RMA = PPO_RMA
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def apply_fixed_init_weights(runner, path: str, device: str, is_distributed: bool) -> None:
    """Deterministic, hardware-independent policy/critic initialization.

    Loads a fixed set of network weights from ``path`` so every run starts from
    byte-identical parameters regardless of seed, machine, GPU, or PyTorch RNG.
    If ``path`` does not yet exist, the freshly-constructed (randomly initialized)
    weights are saved there once to mint the canonical init, which can then be
    committed and reused. This is the robust alternative to seeding alone, which
    is not reproducible across hardware or library versions.
    """
    policy = runner.alg.policy
    if os.path.exists(path):
        loaded = torch.load(path, map_location=device)
        state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        policy.load_state_dict(state_dict)
        print(f"[INFO] Loaded fixed initial policy weights from: {path}")
    else:
        # Minting per-rank would give each rank different weights — require a prior
        # single-process run to create the canonical file before distributed launch.
        if is_distributed:
            raise FileNotFoundError(
                f"--init_weights file not found: {path}. In distributed runs the file must be minted "
                f"first by a single-process run (each rank would otherwise mint different weights). "
                f"Run once without --distributed to create it, then re-launch."
            )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cpu_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        torch.save({"model_state_dict": cpu_state}, path)
        print(f"[INFO] Minted canonical initial policy weights to: {path}")


def find_reset_manager(env):
    """Return the MultiResetManager event-term instance for this env, or None.

    Walks the event manager's class-term configs (the term's ``func`` is the live
    instance). The instance exists from env creation even though its sequential
    state is only built on the first reset (lazy init)."""
    base_env = getattr(env, "unwrapped", env)
    event_mgr = getattr(base_env, "event_manager", None)
    if event_mgr is None:
        return None
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import MultiResetManager
    for mode_cfgs in getattr(event_mgr, "_mode_class_term_cfgs", {}).values():
        for term_cfg in mode_cfgs:
            if isinstance(term_cfg.func, MultiResetManager):
                return term_cfg.func
    return None


def install_reset_state_checkpointing(runner, env):
    """Embed the sequential-reset counter inside each checkpoint so requeued runs
    continue the deterministic reset stream instead of restarting from index 0.

    Adds a ``reset_manager_state`` key to the saved ``.pt`` (re-read with original tensor
    devices preserved, so the model/optimizer placement is unchanged). A separate sidecar
    file is avoided on purpose: ``get_checkpoint_path`` matches every filename in the run
    dir and would mis-select it as the checkpoint. No-op unless a MultiResetManager in
    sequential mode is present (random sampling is stateless). Only rank 0 writes
    checkpoints, so only its counter is stored; on resume every rank restores from it —
    exact for single-GPU, a small approximation under --distributed where each rank
    advances its own counter."""
    orig_save = runner.save

    def save_with_reset_state(path, infos=None):
        orig_save(path, infos)
        mgr = find_reset_manager(env)
        state = mgr.get_sequential_state() if mgr is not None else None
        if state is not None:
            ckpt = torch.load(path, weights_only=False)  # default map_location keeps tensor devices
            ckpt["reset_manager_state"] = state
            torch.save(ckpt, path)

    runner.save = save_with_reset_state


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    is_main_process = not args_cli.distributed or app_launcher.global_rank == 0

    if args_cli.run_id:
        # deterministic directory for cluster jobs — enables auto-resume on requeue
        log_dir = args_cli.run_id
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        if os.path.exists(log_dir):
            checkpoint_files = [f for f in os.listdir(log_dir) if f.endswith(".pt")]
            if checkpoint_files:
                if is_main_process:
                    print(f"[INFO] Found existing run with {len(checkpoint_files)} checkpoints. Auto-resuming...")
                agent_cfg.resume = True
                agent_cfg.load_run = os.path.basename(log_dir)
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    if is_main_process:
        print(f"Exact experiment name requested from command line: {os.path.basename(log_dir)}")

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if args_cli.resume_path is not None:
        resume_path = retrieve_file_path(args_cli.resume_path)
        agent_cfg.resume = True
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerWithClassifier":
        runner = OnPolicyRunnerWithClassifier(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerWithSuccessCritic":
        runner = OnPolicyRunnerWithSuccessCritic(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "SuccessCriticOnlyRunner":
        runner = SuccessCriticOnlyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunnerSplit":
        runner = DistillationRunnerSplit(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "BCPPORunner":
        from uwlab_rl.rsl_rl.bc_ppo_runner import BCPPORunner
        runner = BCPPORunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "GRPOGroupedRunner":
        from uwlab_rl.rsl_rl.grpo_grouped_runner import GRPOGroupedRunner
        runner = GRPOGroupedRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerRMA":
        from uwlab_rl.rsl_rl.on_policy_runner_rma import OnPolicyRunnerRMA
        runner = OnPolicyRunnerRMA(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # Persist the sequential-reset counter into each checkpoint (no-op for non-sequential envs).
    install_reset_state_checkpointing(runner, env)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # Resume the deterministic reset stream from where the checkpoint left off.
        reset_mgr = find_reset_manager(env)
        if reset_mgr is not None:
            loaded = torch.load(resume_path, weights_only=False, map_location="cpu")
            if isinstance(loaded, dict) and "reset_manager_state" in loaded:
                reset_mgr.load_sequential_state(loaded["reset_manager_state"])
                print("[INFO]: Restored sequential reset counter from checkpoint.")
    elif args_cli.init_weights:
        # Fixed initial weights: hold the network init constant across runs (resume takes priority).
        apply_fixed_init_weights(runner, args_cli.init_weights, agent_cfg.device, args_cli.distributed)

    # dump the configuration into log-directory (only on main process to avoid duplicates)
    if is_main_process:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # calculate remaining iterations (handles auto-resume from requeue)
    iterations_to_run = agent_cfg.max_iterations
    if agent_cfg.resume and hasattr(runner, "current_learning_iteration"):
        iterations_to_run = agent_cfg.max_iterations - runner.current_learning_iteration
        print(f"[INFO] Resuming from iteration {runner.current_learning_iteration}. Remaining iterations: {iterations_to_run}")
        if iterations_to_run <= 0:
            print(
                f"[INFO] Training already completed (Current: {runner.current_learning_iteration} >="
                f" Max: {agent_cfg.max_iterations}). Exiting."
            )
            env.close()
            return

    # run training
    runner.learn(num_learning_iterations=iterations_to_run, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
