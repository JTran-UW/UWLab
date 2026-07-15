# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained diffusion policy."""

"""Launch Isaac Sim Simulator first."""

# NOTE: Pre-import numpy and numba BEFORE isaaclab / AppLauncher. Isaac Sim's Kit runtime
# mutates sys.path at startup, which causes any later `import numba` to resolve to
# /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/numba (0.59.x, incompatible with
# numpy 2.x) instead of the conda env's pinned numba 0.64. By importing them here while
# PYTHONPATH ordering still holds, sys.modules caches the correct versions and every
# subsequent `import numba` (e.g. via diffusion_policy.common.sampler when Hydra loads
# TrainMLPImageWorkspace for the policy checkpoint) returns the cached conda module.
import numpy  # noqa: F401
import numba  # noqa: F401

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using diffusion policy for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to diffusion policy checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel.")
parser.add_argument(
    "--num_trajectories",
    type=int,
    default=100,
    help="Number of trajectories to evaluate. If None, run until simulation is stopped.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video of the policy by capturing per-step camera obs (only works on tasks with rgb camera observations or scene rgb sensors).")
parser.add_argument(
    "--record_viewport_video",
    action="store_true",
    default=False,
    help=(
        "Wrap the env with gym.wrappers.RecordVideo to dump viewport pixels to mp4."
        " Independent of the task's observation space — works on proprioceptive-only eval tasks too."
    ),
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Number of env steps to capture when --record_viewport_video is set.",
)
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help=(
        "Absolute path to write the viewport-recording mp4 to. If unset, falls back to the"
        " process's launch cwd captured before Hydra changes it (Hydra's @hydra_task_compose"
        " decorator chdir's into a per-job output dir, which is rarely what the caller wants)."
    ),
)
parser.add_argument(
    "--transformer_mini_batch_size",
    type=int,
    default=64,
    help=(
        "Mini-batch size used by DiffusionPolicyWrapper when serializing transformer inference"
        " across envs. Bounds peak activation memory; too-small values dominate wall time for"
        " large num_envs."
    ),
)
parser.add_argument(
    "--stats_output_path",
    type=str,
    default=None,
    help=(
        "Optional path to a JSON file that will be written with final eval statistics"
        " (episodes, successful_episodes, success_rate, averaged Metrics/* and Episode_Reward/*)."
        " Consumed by run_incontext_exploration.py via incontext_eval_log.IncontextEvalLog."
    ),
)
parser.add_argument(
    "--iteration",
    type=int,
    default=None,
    help="DAgger iteration id to embed in the stats file (ignored unless --stats_output_path is set).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()

# Capture the launch cwd BEFORE Hydra's @hydra_task_compose chdir's into its
# per-job output dir. Used as the viewport-video fallback target when the user
# didn't pass --video_output_dir explicitly.
import os as _os_for_launch_cwd
_LAUNCH_CWD = _os_for_launch_cwd.getcwd()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys

import gymnasium as gym
import numpy as np
import random
import torch
from contextlib import nullcontext
from tqdm import tqdm

import dill
import hydra
import imageio
import isaaclab_tasks  # noqa: F401
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Diffusion policy imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

# Import the Diffusion policy wrapper
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

# Make the repo-root abstraction importable when this script is invoked as
# `python scripts_v2/tools/eval_distilled_policy.py` (sys.path[0] is scripts_v2/tools,
# so the repo root needs to be added explicitly).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from incontext_eval_log import write_eval_stats_file  # noqa: E402


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_policy(ckpt_path: str, device: torch.device, use_ema: bool = False) -> BaseImagePolicy:
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    return policy.eval().to(device)


def _discover_cameras(obs_dict, env):
    """Return (cam_keys, scene_cam_names) for video recording."""
    cam_keys = sorted(k for k in obs_dict["policy"] if "rgb" in k)
    if cam_keys:
        return cam_keys, []
    scene_cam_names = sorted(
        name
        for name, sensor in env.unwrapped.scene._sensors.items()
        if hasattr(sensor, "data") and hasattr(sensor.data, "output") and "rgb" in sensor.data.output
    )
    if scene_cam_names:
        print(f"Using scene cameras for video: {scene_cam_names}")
    return cam_keys, scene_cam_names


def _capture_frame(obs_dict, env, env_idx: int, cam_keys: list, scene_cam_names: list) -> np.ndarray | None:
    """Capture and concatenate camera images for one environment."""
    imgs = []
    if cam_keys:
        for cam in cam_keys:
            img = obs_dict["policy"][cam][env_idx].detach().cpu().permute(1, 2, 0).numpy()
            imgs.append((img * 255).clip(0, 255).astype("uint8"))
    elif scene_cam_names:
        for cam_name in scene_cam_names:
            img = env.unwrapped.scene._sensors[cam_name].data.output["rgb"][env_idx].detach().cpu().numpy()
            if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1]:
                img = img.transpose(1, 2, 0)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype("uint8")
            if img.shape[-1] == 4:
                img = img[..., :3]
            imgs.append(img)
    return np.concatenate(imgs, axis=1) if imgs else None


def _get_progress_context(env):
    """Return the ``progress_context`` reward term object, which exposes a
    per-env ``success`` bool tensor recomputed each step from the analytical
    position+orientation alignment check (the same source ``success_reward``
    reads from). Returns None if the term isn't present (then we fall back to
    the legacy 'success' termination flag, if any).
    """
    rm = getattr(env.unwrapped, "reward_manager", None)
    if rm is None:
        return None
    try:
        return rm.get_term_cfg("progress_context").func
    except (AttributeError, KeyError, ValueError):
        return None


def _collect_metrics(infos: dict, episode_metrics: dict):
    if "log" not in infos:
        return
    for key, value in infos["log"].items():
        if key.startswith("Metrics/") or key.startswith("Episode_Reward/"):
            episode_metrics.setdefault(key, []).append(value)


def _print_results(episodes: int, successful_episodes: int, episode_metrics: dict):
    print("\nFinal Statistics:")
    print(f"Total trajectories evaluated: {episodes}")
    if episodes > 0:
        print(f"Successful trajectories: {successful_episodes}")
        print(f"Success rate: {successful_episodes / episodes * 100:.2f}%")
    else:
        print("Success rate: N/A (no completed episodes)")
    if episode_metrics:
        print("\nAverage Metrics:")
        for metric_name, values in sorted(episode_metrics.items()):
            if values:
                floats = [float(v) if isinstance(v, torch.Tensor) else v for v in values]
                print(f"{metric_name}: {sum(floats) / len(floats):.4f}")


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Run a trained diffusion policy with Isaac Lab environment."""
    _set_seeds(args_cli.seed)

    device = torch.device(args_cli.device if args_cli.device else "cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # Viewport recording — wraps env.render() output, independent of obs space.
    # Use this for tasks (like *State*-StudentEval-v0) that have no rgb obs / sensors.
    if args_cli.record_viewport_video:
        from datetime import datetime as _dt
        video_folder = args_cli.video_output_dir or _LAUNCH_CWD
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"play-{_dt.now().strftime('%Y%m%d-%H%M%S')}",
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print(f"[eval] viewport video recording enabled (length={args_cli.video_length}, folder={video_folder})")

    policy = _load_policy(args_cli.checkpoint, device)
    wrapped_policy = DiffusionPolicyWrapper(
        policy,
        device,
        n_obs_steps=policy.n_obs_steps,
        num_envs=args_cli.num_envs,
        mini_batch_size=args_cli.transformer_mini_batch_size,
        sample_action=False,
    )

    # Persist the AR head's bin spec next to the checkpoint (no-op for continuous heads).
    spec_path = os.path.join(os.path.dirname(args_cli.checkpoint) or ".", "discretize_spec.json")
    saved = wrapped_policy.save_discretize_spec(spec_path)
    if saved is not None:
        print(f"[eval] discrete AR head spec written to {saved}")

    obs_dict, _ = env.reset()
    dones = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
    wrapped_policy.reset((dones > 0).nonzero(as_tuple=False).reshape(-1))

    # Per-episode success is computed analytically from the progress_context
    # reward term's ``success`` buffer (position + orientation alignment).
    # An episode counts as a success if ``success`` was True at *any* step,
    # which matches the training-time semantic without requiring an early-
    # terminating ``success`` term in the eval env.
    progress_ctx = _get_progress_context(env)
    term_names = env.unwrapped.termination_manager._term_names  # type: ignore
    fallback_to_term = progress_ctx is None
    if fallback_to_term and "success" not in term_names:
        raise RuntimeError(
            "Eval env exposes neither a 'progress_context' reward term nor a 'success' "
            "termination term — cannot determine episode success."
        )
    episode_ever_successful = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)

    episodes, steps, successful_episodes = 0, 0, 0
    episode_metrics: dict = {}

    pbar = None
    if args_cli.num_trajectories is not None:
        pbar = tqdm(total=args_cli.num_trajectories, desc="Evaluating trajectories (Success: 0.00%)")

    # Video recording state
    cam_keys, scene_cam_names, env_frames, frames_to_save = [], [], [], []
    if args_cli.save_video:
        cam_keys, scene_cam_names = _discover_cameras(obs_dict, env)
        env_frames = [[] for _ in range(args_cli.num_envs)]

    while simulation_app.is_running():
        if args_cli.num_trajectories is not None and episodes >= args_cli.num_trajectories:
            print(f"\nReached target number of trajectories ({args_cli.num_trajectories}). Stopping evaluation.")
            break

        with torch.inference_mode(), torch.autocast(device_type=device.type) if args_cli.use_amp else nullcontext():
            actions = wrapped_policy.predict_action(obs_dict)

            if args_cli.save_video:
                for i in range(args_cli.num_envs):
                    frame = _capture_frame(obs_dict, env, i, cam_keys, scene_cam_names)
                    if frame is not None:
                        env_frames[i].append(frame)

            step_result = env.step(actions)
            if len(step_result) == 4:
                obs_dict, rewards, dones, infos = step_result
            else:
                obs_dict, rewards, terminated, truncated, infos = step_result
                dones = terminated | truncated

            steps += 1

            # Accumulate per-step success into the per-env "ever-successful" buffer
            # BEFORE handling resets, so the last step of a successful episode counts.
            if progress_ctx is not None and hasattr(progress_ctx, "success"):
                episode_ever_successful |= progress_ctx.success

            if isinstance(dones, torch.Tensor):
                new_ids = (dones > 0).nonzero(as_tuple=False)
                episodes += len(new_ids)
            elif dones:
                new_ids = [0]
                episodes += 1
            else:
                new_ids = []

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                if progress_ctx is not None:
                    successful_episodes += int(episode_ever_successful[reset_ids].sum().item())
                    episode_ever_successful[reset_ids] = False
                else:
                    # Legacy: read 'success' termination flag (one-shot, set the step it triggers).
                    term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]
                    success_idx = term_names.index("success")
                    successful_episodes += int(term_dones[:, success_idx].sum().item())
                wrapped_policy.reset(reset_ids)
                _collect_metrics(infos, episode_metrics)
                steps = 0

                if args_cli.save_video:
                    for i in reset_ids:
                        frames_to_save.extend(env_frames[i])
                        env_frames[i] = []
                    imageio.mimsave("policy_cameras.mp4", frames_to_save, fps=10, codec="libx264")

                if pbar is not None:
                    pbar.update(len(new_ids))
                    rate = (successful_episodes / episodes * 100) if episodes > 0 else 0.0
                    pbar.set_description(f"Evaluating trajectories (Success: {rate:.2f}%)")

    _print_results(episodes, successful_episodes, episode_metrics)
    if args_cli.stats_output_path is not None:
        write_eval_stats_file(
            stats_path=args_cli.stats_output_path,
            episodes=episodes,
            successful_episodes=successful_episodes,
            episode_metrics=episode_metrics,
            iteration=args_cli.iteration,
            checkpoint=args_cli.checkpoint,
            task=args_cli.task,
        )
        print(f"Wrote eval stats to: {args_cli.stats_output_path}")
    if pbar is not None:
        pbar.close()
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
