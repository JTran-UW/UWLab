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
    "--resume_path", type=str, default=None,
    help="Direct path to a checkpoint file to resume from (bypasses log directory search).",
)
parser.add_argument(
    "--gc_checkpoint", type=str, default=None, required=False,
    help=(
        "PPO goal-conditioned policy checkpoint that performs the autonomous resets. Required. "
        "Must match the task's `gc` observation group term-for-term."
    ),
)
parser.add_argument(
    "--phase_video", action="store_true", default=False,
    help=(
        "Record an annotated mp4 of the collect -> update -> reset cycle: green banner while the "
        "SAC assembly policy collects, orange while gradient updates run, red while the GC "
        "disassembly policy performs the reset. Also draws the disassembly target keypoints."
    ),
)
parser.add_argument(
    "--phase_video_steps", type=int, default=0,
    help=(
        "Optional cap on recorded env steps for --phase_video. 0 (default) records for the whole "
        "run; the file is finalised on exit, including Ctrl+C."
    ),
)
parser.add_argument(
    "--phase_video_update_frames", type=int, default=30,
    help="Frames to hold on the orange TRAINING banner (updates consume no env steps).",
)
parser.add_argument(
    "--phase_video_path", type=str, default=None,
    help="Output mp4 for --phase_video (default: <log_dir>/phase_video.mp4).",
)
parser.add_argument(
    "--collect_steps", type=int, default=160,
    help="Steps per FastSAC collection episode (replay buffer ON).",
)
parser.add_argument(
    "--reset_steps", type=int, default=160,
    help="Steps the GC policy gets to drive the robot to the sampled reset state (buffer OFF).",
)
parser.add_argument(
    "--replay_buffer_path", type=str, default=None,
    help=(
        "Optional path to an online replay_buffer_*.pt file (saved by save_replay_buffer_interval). "
        "If set, the online replay buffer is loaded into FastSACAgent after the model checkpoint. "
        "agent.buffer_size / num_envs must match the saved buffer's shape."
    ),
)
parser.add_argument(
    "--sgft",
    action="store_true",
    default=False,
    help=(
        "Shaped-guidance finetuning. Freezes the resumed checkpoint's actor and critic into a "
        "source value function V(s) = mean_i Q_i(s, mu(s)) and stores potential-shaped rewards "
        "r_hat = r + gamma*V(s') - V(s) instead of r. V is never updated. Requires --resume_path "
        "(there is nothing to freeze otherwise). Potential-based, so the optimal policy is "
        "unchanged; only credit assignment moves."
    ),
)
parser.add_argument(
    "--sgft_source_ckpt",
    type=str,
    default=None,
    help=(
        "Optional checkpoint to freeze as the SGFT source value function instead of the resumed "
        "weights. Lets the shaping potential come from a different (e.g. earlier or expert) "
        "checkpoint than the one being finetuned. Same format as regular save()/load() "
        "checkpoints. Only meaningful with --sgft or --h_step_backup; with it set, "
        "--resume_path is no longer required for those flags."
    ),
)
parser.add_argument(
    "--h_step_backup",
    action="store_true",
    default=False,
    help=(
        "Bootstrap the n-step critic target off the frozen source value function instead of the "
        "learned target critic: target = sum_k gamma^k r_k + gamma^n * V_source(s_n). V_source is "
        "distributional, so its atom distribution is pushed through the same categorical Bellman "
        "projection. Freezes the resumed checkpoint, so it requires --resume_path. Can be used "
        "with or without --sgft, but enabling both double-counts V_source (warned at startup)."
    ),
)
parser.add_argument(
    "--cpu_replay_buffer",
    action="store_true",
    default=False,
    help=(
        "Hold the online AND expert replay buffers in host RAM instead of on the compute device. "
        "Off by default: GPU residency removes a host->device copy per update. Turn it on when the "
        "buffers do not fit -- note the expert buffer is loaded straight to the buffer's device, so "
        "a 7 GB expert file costs 7 GB of VRAM on top of the online buffer and of PhysX's fixed GPU "
        "arenas (which are sized by config, not by --num_envs, and so cost the same at 1 env as at "
        "4096). Sampled batches are moved to the compute device before every update, so results are "
        "unchanged; only the per-update transfer cost differs."
    ),
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--expert_transitions",
    type=str,
    default=None,
    help=(
        "Path to a .pt file of recorded transitions (from play.py --record_transitions). "
        "If set, FastSACAgent mixes expert transitions into each training batch."
    ),
)
parser.add_argument(
    "--expert_ratio",
    type=float,
    default=0.5,
    help="Fraction of each training batch sampled from the expert replay buffer (default: 0.5).",
)
parser.add_argument(
    "--expert_ratio_anneal_steps",
    type=int,
    default=0,
    help="Linearly anneal expert_ratio to 0 over this many global steps. 0 = no annealing (default).",
)
parser.add_argument(
    "--expert_checkpoint",
    type=str,
    default=None,
    help=(
        "Path to an RSL-RL (PPO) checkpoint .pt file. "
        "If set, loads an OnPolicyRunner from that checkpoint and passes its policy and critic "
        "to FastSACAgent as BC regularization targets."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video or args_cli.phase_video:
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
import re
import torch
import wandb
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.agents.rsl_rl_cfg import Base_PPORunnerCfg
from vecenv_wrapper import HolosomaVecEnvWrapper

from autonomous_reset_agent import AutonomousResetFastSACAgent
from holosoma.config_types.experiment import ExperimentConfig

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
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
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

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
    if args_cli.phase_video:
        # Frame the env being recorded. The default viewer looks at the WORLD origin, but env 0
        # sits at its own origin (env_spacing offset), so the arm and the goal markers land at the
        # edge of frame or outside it entirely.
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        # Head-on down the +x axis at the table. The previous 3/4 view put a scene pole between
        # the camera and the workspace.
        env_cfg.viewer.eye = (1.55, 0.0, 0.55)
        env_cfg.viewer.lookat = (0.40, 0.0, 0.08)

    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if (args_cli.video or args_cli.phase_video) else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if args_cli.resume_path is not None:
        resume_path = retrieve_file_path(args_cli.resume_path)
        agent_cfg.resume = True
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif int(os.getenv("SLURM_RESTART_COUNT", "0")) > 0:
        # Job was preempted and requeued by SLURM — auto-resume from latest checkpoint.
        # Match the most recent run directory for this run_name (or any run if no name set).
        run_dir_pattern = f".*_{re.escape(agent_cfg.run_name)}" if agent_cfg.run_name else ".*"
        try:
            resume_path = get_checkpoint_path(log_root_path, run_dir_pattern, checkpoint="model_.*.pt")
            agent_cfg.resume = True
            print(
                f"[INFO] SLURM requeue detected (SLURM_RESTART_COUNT={os.getenv('SLURM_RESTART_COUNT')}). "
                f"Auto-resuming from: {resume_path}"
            )
        except ValueError:
            print(
                f"[INFO] SLURM requeue detected but no checkpoint found in '{log_root_path}' "
                f"matching run_name='{agent_cfg.run_name}'. Starting fresh."
            )

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
    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load expert policy and critic from an RSL-RL checkpoint if provided
    expert_policy = None
    expert_critic = None
    if args_cli.expert_checkpoint is not None:
        print(f"[INFO] Loading expert checkpoint from: {args_cli.expert_checkpoint}")
        expert_resume_path = retrieve_file_path(args_cli.expert_checkpoint)
        # Temporarily wrap the underlying gym env with RslRlVecEnvWrapper so OnPolicyRunner can
        # read env dimensions — env.env is the gym env before HolosomaVecEnvWrapper was applied.
        expert_rsl_env = RslRlVecEnvWrapper(env.env, clip_actions=agent_cfg.clip_actions)
        expert_ppo_cfg = cli_args.sanitize_rsl_rl_cfg(Base_PPORunnerCfg())
        expert_runner = OnPolicyRunner(expert_rsl_env, expert_ppo_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        expert_runner.load(expert_resume_path)
        expert_policy = expert_runner.get_distribution_fn(device=agent_cfg.device)
        expert_critic = expert_runner.get_inference_critic(device=agent_cfg.device)
        print("[INFO] Expert policy and critic loaded successfully.")

    # initialize wandb if requested (rank-0 only when distributed)
    gpu_global_rank = int(os.getenv("RANK", "0"))
    is_rank_zero = (not args_cli.distributed) or gpu_global_rank == 0
    if agent_cfg.logger == "wandb" and is_rank_zero:
        run_name = args_cli.run_name if args_cli.run_name else f"{os.path.basename(log_dir)}_{args_cli.task}"
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(
            project=agent_cfg.wandb_project,
            name=run_name,
            config={
                "task": args_cli.task,
                "num_envs": env_cfg.scene.num_envs,
                "seed": agent_cfg.seed,
                "max_iterations": agent_cfg.max_iterations,
                "device": agent_cfg.device,
                "expert_transitions": args_cli.expert_transitions,
                "expert_ratio": args_cli.expert_ratio,
                "expert_checkpoint": args_cli.expert_checkpoint,
                "replay_buffer_path": args_cli.replay_buffer_path,
                "resume_path": args_cli.resume_path,
            },
            dir=log_dir,
            sync_tensorboard=True,
        )

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = AutonomousResetFastSACAgent(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device,
            expert_policy=expert_policy,
            expert_critic=None, # expert_critic,
            lambda_bc_policy=1.0,
            lambda_bc_critic=1.0,
            use_cpu_rb=args_cli.cpu_replay_buffer,
        )
        if args_cli.cpu_replay_buffer:
            print("[INFO] Replay buffers (online + expert) held on CPU; batches move to device per update.")
        runner.setup()
        if args_cli.gc_checkpoint is None:
            raise ValueError("--gc_checkpoint is required: it is the policy that performs the resets.")
        runner.configure_autonomous_reset(
            gc_checkpoint=args_cli.gc_checkpoint,
            collect_steps=args_cli.collect_steps,
            reset_steps=args_cli.reset_steps,
        )
        if args_cli.phase_video:
            runner.configure_phase_video(
                path=args_cli.phase_video_path or os.path.join(log_dir, "phase_video.mp4"),
                steps=args_cli.phase_video_steps,
                update_frames=args_cli.phase_video_update_frames,
            )
        # update_interval is meaningless here: updates fire once at the end of each collection
        # episode, so the schedule is (num_updates per collect_steps) by construction.
        if int(getattr(agent_cfg, "update_interval", 1) or 1) != 1:
            print(
                "[WARN]: update_interval is ignored by the autonomous-reset loop; updates run once "
                "per collection episode. Use --collect_steps and agent.num_updates instead."
            )
        runner.expert_ratio = args_cli.expert_ratio
        runner.expert_ratio_anneal_steps = args_cli.expert_ratio_anneal_steps
        runner.attach_checkpoint_metadata(ExperimentConfig(), log_dir)
        if args_cli.expert_transitions is not None:
            runner.load_expert_replay_buffer(args_cli.expert_transitions)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # SGFT: freeze the source V -- from --sgft_source_ckpt if given, else a snapshot of the
    # just-loaded weights. Must run after runner.load() -- before it, the networks are still
    # randomly initialised -- and after the expert buffer is loaded, so enable_sgft() can reshape
    # those stored rewards too.
    if args_cli.sgft or args_cli.h_step_backup:
        _which = " / ".join(f for f, on in (("--sgft", args_cli.sgft),
                                            ("--h_step_backup", args_cli.h_step_backup)) if on)
        if not (agent_cfg.resume or args_cli.resume_path is not None or args_cli.sgft_source_ckpt is not None):
            raise ValueError(
                f"{_which} needs a source policy to freeze; pass --resume_path <checkpoint> "
                "or --sgft_source_ckpt <checkpoint>."
            )
        if not hasattr(runner, "enable_sgft"):
            raise ValueError(f"{_which} is only supported by FastSACAgent, got {type(runner).__name__}.")
        runner.enable_sgft(
            shape_rewards=args_cli.sgft,
            h_step_backup=args_cli.h_step_backup,
            ckpt_path=args_cli.sgft_source_ckpt,
        )

    # Optional: load online replay buffer from a saved snapshot (FastSAC only).
    if args_cli.replay_buffer_path is not None and hasattr(runner, "load_replay_buffer"):
        print(f"[INFO]: Loading online replay buffer from: {args_cli.replay_buffer_path}")
        runner.load_replay_buffer(args_cli.replay_buffer_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn() #num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # finalize wandb before sim closes
    if agent_cfg.logger == "wandb" and is_rank_zero and wandb.run is not None:
        wandb.finish()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
