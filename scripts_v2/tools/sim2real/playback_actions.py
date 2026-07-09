# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded action sequence open-loop and record every observation term.

Resets the sim to a known initial state, then steps the env through a fixed list of
actions with no policy in the loop.  Every observation term is recorded per step and
saved to its own array in the output ``.npz``, so a recorded rollout and its sim
replay can be diffed to expose the sim2real gap.

The env (``Ur5eRobotiq2f85PlaybackCfg``) carries the eval dynamics -- explicit
actuator, stiff OSC gains, fixed sysid -- and a single observation term returning
the trailing 6 joint positions (the Robotiq 2F-85 gripper joints).  Add terms to
``PlaybackObservationsCfg`` and they appear as extra keys with no change here.

Two action sources, differing in where the initial state comes from:

  * ``--dataset_file`` (HDF5, preferred) -- an Isaac Lab demo dataset.  Each episode
    carries BOTH its actions and the ``initial_state`` it was recorded from, so the
    replay starts exactly where the demo did.  This is what ``scripts/tools/replay_demos.py``
    does; the difference here is that observations are recorded per term.

  * ``--actions`` (``.npz`` / ``.npy``) + ``--reset_type`` / ``--idx`` -- a bare action
    array with no initial state of its own, so the scene is reset to an OmniReset
    reset state by index (exactly like scene_capture.py).  Only meaningful if that
    reset state is the one the actions were recorded from: replaying an action list
    from a state it never saw is open-loop, so the arm quickly ends up somewhere the
    recording never went and the replay is not a sim2real measurement.  For a ``.npz``,
    the actions are read from ``--action_key`` (default ``action``) and every other
    array is copied through to the output under a ``real/`` prefix.

Actions are (T, 7): 6 Cartesian EE deltas + 1 binary gripper command.

Observations are recorded BEFORE each action is applied, plus once after the last
one, so the arrays have T+1 rows and ``obs[t]`` is the state that produced
``actions[t]`` -- the same convention as the real recordings.

Mirrors the sim2real tooling pattern:
  scripts_v2/tools/sim2real/align_cameras.py    →  align sim camera to real photo
  scripts_v2/tools/sim2real/scene_capture.py    →  align real scene to sim photo
  scripts_v2/tools/sim2real/playback_actions.py →  replay recorded actions in sim

Usage:
    python scripts_v2/tools/sim2real/playback_actions.py --headless \
        --dataset_file datasets/demos.hdf5 --episode 0

    python scripts_v2/tools/sim2real/playback_actions.py --headless \
        --actions /home/yandabao/diffusion_policy/pc_debug/ep2_actions.npz \
        --reset_type ObjectAnywhereEEAnywhere --idx 3
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

parser = argparse.ArgumentParser(description="Open-loop playback of a recorded action sequence.")
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--dataset_file", type=str, help="Isaac Lab HDF5 demo dataset; replays an episode's own actions")
source.add_argument("--actions", type=str, help="Path to a .npz or .npy holding the action sequence")
parser.add_argument("--episode", type=int, default=0, help="Episode index to replay from --dataset_file")
parser.add_argument("--action_key", type=str, default="action", help="Array key holding the (T, 7) actions in a .npz")
parser.add_argument("--max_steps", type=int, default=None, help="Truncate the action sequence to this many steps")
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEAnywhere", help="Reset dataset name (--actions)")
parser.add_argument("--idx", type=int, default=0, help="Index of the reset state to load from the dataset (--actions)")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
    help="Root directory of the OmniReset dataset",
)
parser.add_argument("--output", type=str, default=None, help="Output .npz path (default: playback_<stem>_<tag>.npz)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.playback_cfg import (  # noqa: E402
    Ur5eRobotiq2f85PlaybackCfg,
)


def load_episode(path, episode_index, device):
    """Return (actions (T, 7) tensor, initial_state) for one episode of an HDF5 demo dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file {path} does not exist.")
    handler = HDF5DatasetFileHandler()
    handler.open(path)
    episode_names = list(handler.get_episode_names())
    if not -len(episode_names) <= episode_index < len(episode_names):
        raise IndexError(f"episode {episode_index} out of range ({len(episode_names)} episodes in {path})")

    episode = handler.load_episode(episode_names[episode_index], device)
    actions = episode.data["actions"]
    if actions.ndim != 2:
        raise ValueError(f"episode actions must be (T, action_dim), got {tuple(actions.shape)}")
    return actions, episode.get_initial_state()


def load_actions(path, action_key, max_steps):
    """Return (actions (T, 7) float32, extras {name: array}) from a .npz or .npy."""
    if path.endswith(".npy"):
        actions, extras = np.load(path), {}
    else:
        data = np.load(path)
        if action_key not in data.files:
            raise KeyError(f"'{action_key}' not in {path} (has: {data.files})")
        actions = data[action_key]
        extras = {k: data[k] for k in data.files if k != action_key}

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be (T, action_dim), got {actions.shape}")
    if max_steps is not None and max_steps < len(actions):
        # Truncate the per-step extras alongside the actions; leave anything else alone.
        num_steps = len(actions)
        extras = {k: (v[:max_steps] if v.ndim >= 1 and v.shape[0] == num_steps else v) for k, v in extras.items()}
        actions = actions[:max_steps]
    return actions, extras


def load_reset_state(env, dataset_dir, reset_type, idx):
    """Load a single reset state (batched to 1 env) from the OmniReset dataset.

    Same path as scene_capture.py: the object-pair directory is derived from the
    scene's spawned USDs, exactly as MultiResetManager does.
    """
    pair = task_mdp.compute_pair_dir(
        env.scene["insertive_object"].cfg.spawn.usd_path,
        env.scene["receptive_object"].cfg.spawn.usd_path,
    )
    dataset_file = f"{dataset_dir}/Resets/{pair}/resets_{reset_type}.pt"
    local_path = task_mdp.safe_retrieve_file_path(dataset_file)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Dataset file {dataset_file} could not be accessed or downloaded.")

    dataset = torch.load(local_path, map_location="cpu")
    num_states = len(dataset["initial_state"]["articulation"]["robot"]["joint_position"])
    if not -num_states <= idx < num_states:
        raise IndexError(f"idx {idx} out of range for {reset_type} (has {num_states} states)")

    idx_t = torch.tensor([idx], device=env.device)
    return task_mdp.sample_state_data_set(dataset["initial_state"], idx_t, env.device)


def obs_terms_to_numpy(obs):
    """Flatten the non-concatenated policy group into {term_name: (D,) array} for env 0."""
    return {name: tensor[0].detach().cpu().numpy().copy() for name, tensor in obs["policy"].items()}


def merge_into_scene_state(env, partial_state):
    """Overlay a recorded state onto the live full scene state.

    ``env.reset_to()`` requires every scene asset to be present, but a recorded
    state may only hold a subset (the OmniReset reset dataset carries just the
    robot and the two objects).  Assets it doesn't mention keep their current state.
    """
    state = env.scene.get_state(is_relative=True)
    for category, assets in partial_state.items():
        for asset_name, asset_state in assets.items():
            state[category][asset_name] = asset_state
    return state


def main():
    cfg = Ur5eRobotiq2f85PlaybackCfg()
    cfg.scene.num_envs = 1
    env = gym.make("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Playback-v0", cfg=cfg).unwrapped
    env.reset()

    # ---- resolve the action sequence and the state to replay it from ----
    extras = {}
    if args_cli.dataset_file is not None:
        # The episode carries the initial state it was recorded from: replay starts
        # exactly where the demo did.
        actions_t, initial_state = load_episode(args_cli.dataset_file, args_cli.episode, env.device)
        if args_cli.max_steps is not None:
            actions_t = actions_t[: args_cli.max_steps]
        source, tag = args_cli.dataset_file, f"ep{args_cli.episode}"
    else:
        actions, extras = load_actions(args_cli.actions, args_cli.action_key, args_cli.max_steps)
        actions_t = torch.from_numpy(actions).to(env.device)
        initial_state = load_reset_state(env, args_cli.dataset_dir, args_cli.reset_type, args_cli.idx)
        source, tag = args_cli.actions, f"idx{args_cli.idx}"

    action_dim = env.action_manager.total_action_dim
    if actions_t.shape[1] != action_dim:
        raise ValueError(f"actions have {actions_t.shape[1]} dims but the env expects {action_dim}")

    env_ids = torch.tensor([0], device=env.device)
    obs, _ = env.reset_to(merge_into_scene_state(env, initial_state), env_ids=env_ids, is_relative=True)

    # ---- open-loop playback: one recorded action per env step, no policy ----
    recorded = [obs_terms_to_numpy(obs)]  # obs BEFORE actions[0]
    for t in range(actions_t.shape[0]):
        obs, *_ = env.step(actions_t[t : t + 1])
        recorded.append(obs_terms_to_numpy(obs))

    # ---- save each obs term as its own array ----
    out = {name: np.stack([step[name] for step in recorded]) for name in recorded[0]}
    out["actions"] = actions_t.detach().cpu().numpy()
    for name, value in extras.items():
        out[f"real/{name}"] = value

    stem = os.path.splitext(os.path.basename(source))[0]
    out_path = args_cli.output or f"playback_{stem}_{tag}.npz"
    np.savez(out_path, **out)

    print("\n" + "=" * 72)
    print(f"Playback: {source}  ({tag})")
    print(f"Stepped {actions_t.shape[0]} actions; recorded {len(recorded)} observations (T+1).")
    for name in recorded[0]:
        print(f"  {name:<24}{str(out[name].shape):>16}")
    print(f"Saved to {out_path}")
    print("=" * 72 + "\n")
    # simulation_app.close() tears the process down without flushing Python's
    # buffer, so a piped/redirected stdout would swallow the summary above.
    sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
