# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect demonstrations from a trained holosoma FastSAC checkpoint.

The scripts_v2 collector loads a BC/offline expert via ``behavior_cloning_cfg`` and calls
``compute_distribution``. FastSAC checkpoints expose neither, so this variant rebuilds a
``FastSACAgent`` (the only way to restore actor + observation normalizer together) and reads the
Gaussian straight off the actor.

Recording works exactly as in scripts_v2: ``ActionStateRecorderManagerCfg`` dumps the task's whole
``data_collection`` observation group each step, so RGB is present iff the task defines a camera
term in that group. The expert's action distribution is injected into the same dict so it is
persisted alongside the observations without touching the recorder.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import gymnasium as gym
import os
import sys
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect demonstrations from a holosoma FastSAC checkpoint.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.zarr", help="Output dataset path.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a FastSAC model_*.pt checkpoint.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use the mean of the policy distribution instead of sampling.",
)
parser.add_argument(
    "--expert_obs_keys",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Observation group(s) the expert acts from. Defaults to the checkpoint's own "
        "agent_cfg.actor_obs_keys. Override when a state-trained expert drives a data-collection "
        "task whose state groups are named differently."
    ),
)
parser.add_argument(
    "--no_first_step_mask",
    action="store_true",
    default=False,
    help=(
        "Disable zeroing the action on each env's first step after reset. That mask exists because "
        "the first rendered frame may be stale, and the recorded observation is what matters."
    ),
)
parser.add_argument("--agent", type=str, default=None, help="Agent config entry point override.")
# Declared but unused by collection itself: docker/cluster/run_singularity.sh appends --distributed
# to EVERY cluster job. AppLauncher does not register the flag, so without this it survives
# parse_known_args, lands in hydra_args, and Hydra rejects it as a malformed override before the
# script does anything. train.py declares it for the same reason.
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run with multiple GPUs or nodes."
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Hand Hydra ONLY the leftover overrides. hydra_task_config reads sys.argv directly, so leaving this
# script's own flags in place makes Hydra reject them as "unrecognized arguments".
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.datasets import HDF5DatasetFileHandler

from uwlab.utils.datasets import ZarrDatasetFileHandler

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from uwlab_tasks.utils.hydra import hydra_task_config

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent

# This directory's vecenv_wrapper shadows holosoma's on sys.path; import it the same way play.py does.
import pathlib  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vecenv_wrapper import HolosomaVecEnvWrapper  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent or "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Roll out a FastSAC expert and record its demonstrations."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dataset_handler = (
        ZarrDatasetFileHandler if args_cli.dataset_file.endswith(".zarr") else HDF5DatasetFileHandler
    )

    # The recorder dumps obs_buf["data_collection"] verbatim every step; RGB shows up only if the
    # task defines a camera term in that group.
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = dataset_handler

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")
    if not is_fastsac:
        raise ValueError(
            f"Task '{args_cli.task}' resolves to a non-FastSAC agent cfg ({agent_cfg.class_name}). "
            "Use scripts_v2/tools/collect_demos.py for BC/offline experts."
        )

    # Sampling assumes the actor's Gaussian IS the action distribution. With use_tanh the actor
    # returns a PRE-squash mean and the executed action is tanh(mean)*scale+bias, so
    # normal(mean, std) would emit actions from a different space than the policy uses -- plausible
    # looking demos, silently wrong. Refuse rather than write a corrupt dataset.
    if getattr(agent_cfg, "use_tanh", False):
        raise ValueError(
            "This collector requires use_tanh=False: with tanh squashing the actor's mean/log_std "
            "describe the pre-squash Gaussian, not the executed action. Re-export the checkpoint "
            "with use_tanh=False, or extend this script to squash explicitly."
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading FastSAC checkpoint from: {resume_path}")
    runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.setup()
    runner.load(resume_path)

    actor = runner.actor
    actor.eval()
    obs_normalizer = runner.obs_normalizer
    obs_normalizer.eval()
    expert_obs_keys = args_cli.expert_obs_keys or list(agent_cfg.actor_obs_keys)
    print(f"[INFO]: Expert acts from obs groups: {expert_obs_keys}")
    print(f"[Policy] {'Deterministic (mean)' if args_cli.deterministic else 'Stochastic (sampled)'} actions")

    def expert_distribution(obs_td):
        """Gaussian (mean, std) of the frozen expert for the current observation."""
        actor_obs = torch.cat([obs_td[k] for k in expert_obs_keys], dim=-1)
        norm_obs = obs_normalizer(actor_obs, update=False) if runner.obs_normalization else actor_obs
        # forward -> (action, mean, log_std); with use_tanh=False action == mean, so the Gaussian
        # here is directly the action distribution.
        _, mean, log_std = actor(norm_obs)
        return mean, log_std.exp()

    obs_td, _ = env.reset()

    current_recorded_demo_count = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        pbar = tqdm(total=args_cli.num_demos, desc="Recording Demonstrations", unit="demo")

        while True:
            mean, std = expert_distribution(obs_td)
            actions = mean if args_cli.deterministic else torch.normal(mean, std)

            # The recorded first frame after a reset may be stale, so neutralise that step.
            if not args_cli.no_first_step_mask:
                first_step_mask = env.unwrapped.episode_length_buf == 0
                if torch.any(first_step_mask):
                    actions[first_step_mask, :-1] = 0.0
                    actions[first_step_mask, -1] = -1.0  # close gripper

            # The recorder copies obs_buf["data_collection"] wholesale, so injecting here persists
            # the expert distribution alongside the observations without changing the recorder.
            env.unwrapped.obs_buf["data_collection"]["expert_action_mean"] = mean.clone()
            env.unwrapped.obs_buf["data_collection"]["expert_action_std"] = std.clone()

            obs_td = env.step(actions)[0]

            new_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                pbar.update(new_count - current_recorded_demo_count)
                current_recorded_demo_count = new_count

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            if env.unwrapped.sim.is_stopped():
                break

        pbar.close()

    env.close()


if __name__ == "__main__":
    main()  # type: ignore
    simulation_app.close()
