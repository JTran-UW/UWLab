# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""train.py plus a tiled per-env camera recording of the first N seconds of training.

Identical training loop; adds ``--tile_video_out/--tile_video_seconds/--tile_video_fps``.
"""

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
parser.add_argument("--tile_video_out", type=str, default=None, help="Write a tiled per-env mp4 of training to this path.")
parser.add_argument("--tile_video_seconds", type=float, default=600.0)
parser.add_argument("--tile_video_fps", type=int, default=10)
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
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video or args_cli.tile_video_out:
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



class TiledCameraRecorder(gym.Wrapper):
    """Encode one tiled frame (all envs, per-env camera) per env step until the frame budget is spent."""

    RESET_TYPES = [
        "ObjectAnywhereEEAnywhere",
        "ObjectRestingEEGrasped",
        "ObjectAnywhereEEGrasped",
        "ObjectPartiallyAssembledEEGrasped",
    ]

    def __init__(self, env, out_path: str, seconds: float, fps: int):
        super().__init__(env)
        import math
        import subprocess
        import imageio_ffmpeg

        u = env.unwrapped
        self._u = u
        self._cam = u.scene["cam"]
        self._n = u.num_envs
        self._w, self._h = self._cam.cfg.width, self._cam.cfg.height
        self._cols = math.ceil(math.sqrt(self._n))
        self._rows = math.ceil(self._n / self._cols)
        W, H = self._cols * self._w, self._rows * self._h
        origins = u.scene.env_origins
        eyes = origins + torch.tensor([1.05, -0.65, 0.65], device=u.device)
        targets = origins + torch.tensor([0.42, 0.10, 0.05], device=u.device)
        self._cam.set_world_poses_from_view(eyes, targets)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        self._proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
            stdin=subprocess.PIPE,
        )
        self._W, self._H = W, H
        self._budget = int(seconds * fps)
        self._frames = 0
        self._steps = 0
        self._out_path = out_path
        self._reset_term = u.event_manager.get_term_cfg("reset_from_reset_states").func
        self._progress = u.reward_manager.get_term_cfg("progress_context").func
        print(f"[TiledCameraRecorder] {self._n} envs -> {W}x{H} @ {fps} fps, {self._budget} frames -> {out_path}")

    def step(self, action):
        out = self.env.step(action)
        self._steps += 1
        if self._frames < self._budget:
            self._capture()
        elif self._proc is not None:
            self._finish()
        return out

    def _capture(self):
        import numpy as np
        from PIL import Image, ImageDraw

        u = self._u
        self._cam.update(u.step_dt, force_recompute=True)
        rgb = self._cam.data.output["rgb"]
        arr = (rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb))[..., :3].astype(np.uint8)
        task_ids = self._reset_term.task_id
        success = self._progress.success
        ep_len = u.episode_length_buf
        it = self._steps // 32
        canvas = Image.new("RGB", (self._W, self._H))
        for i in range(self._n):
            tile = Image.fromarray(arr[i])
            d = ImageDraw.Draw(tile)
            s = bool(success[i])
            label = (f"env{i}  {self.RESET_TYPES[int(task_ids[i])]}  ep_t={int(ep_len[i])}  "
                     f"iter~{it}  success={s}")
            d.rectangle([0, 0, self._w, 18], fill=(0, 0, 0))
            d.text((4, 3), label, fill=(0, 255, 0) if s else (255, 255, 255))
            canvas.paste(tile, ((i % self._cols) * self._w, (i // self._cols) * self._h))
        self._proc.stdin.write(canvas.tobytes())
        self._frames += 1

    def _finish(self):
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
            print(f"[TiledCameraRecorder] wrote {self._out_path}: {self._frames} frames")

    def close(self):
        self._finish()
        return self.env.close()


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

    if args_cli.tile_video_out:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg

        env_cfg.scene.cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/cam",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)
            ),
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.tile_video_out:
        env = TiledCameraRecorder(env, args_cli.tile_video_out, args_cli.tile_video_seconds, args_cli.tile_video_fps)

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
        runner.load(resume_path)

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
