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
    "--restart_iteration", action="store_true", default=False,
    help="Start the iteration counter at 0 instead of continuing from the checkpoint (default: continue).",
)
parser.add_argument(
    "--resume_path", type=str, default=None,
    help="Direct path to a checkpoint file to resume from (bypasses log directory search).",
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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

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
    class _NanSafeRslRlVecEnvWrapper(RslRlVecEnvWrapper):
        """Zero non-finite rewards/obs: a solver blow-up (Newton) terminates+resets the env via
        abnormal_robot, but the reward of that step is computed from the non-finite state."""

        _t_env = 0.0; _t_n = 0; _t_last = 0.0

        def step(self, actions):
            import time as _time
            torch.cuda.synchronize(); _t0 = _time.perf_counter()
            obs, rew, dones, extras = super().step(actions)
            torch.cuda.synchronize(); self._t_env += _time.perf_counter() - _t0; self._t_n += 1
            if self._t_n % 32 == 0 and int(os.environ.get("RANK", "0")) == 0:
                extra = ""
                try:
                    from isaaclab_newton.physics.newton_manager import NewtonManager as _NM
                    mjd = getattr(_NM._solver, "mjw_data", None)
                    if mjd is not None:
                        ni = mjd.solver_niter.numpy(); cap = int(_NM._solver.mjw_model.opt.iterations)
                        ws = torch.isfinite(torch.as_tensor(mjd.qacc_warmstart.numpy())).all(dim=1)
                        extra = f" | solver niter max={ni.max()} mean={ni.mean():.1f} at_cap={(ni >= cap).sum()}/{ni.size} | non-finite warmstart worlds={int((~ws).sum())}"
                except Exception as exc:
                    extra = f" | solver stats n/a ({type(exc).__name__})"
                ph = " ".join(f"{k}={v/32*1000:.0f}ms" for k, v in self._phase.items())
                print(f"[step-time] mean env.step over last 32 steps: {self._t_env/32*1000:.0f} ms  (dones {int(dones.sum())}) [{ph}]{extra}", flush=True)
                self._t_env = 0.0
                for k in self._phase: self._phase[k] = 0.0
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
            for k in list(obs.keys()):
                v = obs[k]
                if torch.is_tensor(v) and not torch.isfinite(v).all():
                    obs[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return obs, rew, dones, extras

    # step-phase timers (rank 0 prints them with [step-time])
    _uw = env.unwrapped
    _phase = {"physics": 0.0, "action": 0.0, "reset": 0.0, "obs": 0.0, "reward": 0.0}
    def _timed(name, fn):
        import time as _time
        def w(*a, **k):
            torch.cuda.synchronize(); t0 = _time.perf_counter(); r = fn(*a, **k); torch.cuda.synchronize(); _phase[name] += _time.perf_counter() - t0; return r
        return w
    _uw.sim.step = _timed("physics", _uw.sim.step)
    _uw.action_manager.apply_action = _timed("action", _uw.action_manager.apply_action)
    _uw._reset_idx = _timed("reset", _uw._reset_idx)
    _uw.observation_manager.compute = _timed("obs", _uw.observation_manager.compute)
    _uw.reward_manager.compute = _timed("reward", _uw.reward_manager.compute)
    _NanSafeRslRlVecEnvWrapper._phase = _phase

    env = _NanSafeRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(
            resume_path,
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": not args_cli.restart_iteration, "rnd": False},
        )
        print(f"[INFO] resumed at iteration {runner.current_learning_iteration} (wandb x-axis continues from there)")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
