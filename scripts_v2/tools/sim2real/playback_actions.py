# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded action sequence open-loop from an OmniReset reset state.

Resets the sim to one specific reset state (object + robot pose) from the OmniReset
dataset -- exactly like scene_capture.py -- then steps the env through a fixed list
of actions with no policy in the loop.  Every observation term is recorded per step
and saved to its own array in the output ``.npz``, so a real-robot rollout and its
sim replay can be diffed to expose the sim2real gap.

The env (``Ur5eRobotiq2f85PlaybackCfg``) carries the eval dynamics -- explicit
actuator, stiff OSC gains, fixed sysid -- and a single observation term returning
the trailing 6 joint positions (the Robotiq 2F-85 gripper joints).  Add terms to
``PlaybackObservationsCfg`` and they appear as extra keys with no change here.

Action sources:
  * ``.npz`` -- e.g. a real-robot episode; the action array is read from
    ``--action_key`` (default ``action``), shape (T, 7) = 6 Cartesian deltas + 1
    binary gripper.  Any other array in the file is copied through to the output
    under a ``real/`` prefix so sim and real sit side by side.
  * ``.npy`` -- a bare (T, 7) array.

Observations are recorded BEFORE each action is applied, plus once after the last
one, so the arrays have T+1 rows and ``obs[t]`` is the state that produced
``actions[t]`` -- the same convention as the real recordings.

Mirrors the sim2real tooling pattern:
  scripts_v2/tools/sim2real/align_cameras.py    →  align sim camera to real photo
  scripts_v2/tools/sim2real/scene_capture.py    →  align real scene to sim photo
  scripts_v2/tools/sim2real/playback_actions.py →  replay real actions in sim

Usage:
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
parser.add_argument("--actions", type=str, required=True, help="Path to a .npz or .npy holding the action sequence")
parser.add_argument("--action_key", type=str, default="action", help="Array key holding the (T, 7) actions in a .npz")
parser.add_argument("--max_steps", type=int, default=None, help="Truncate the action sequence to this many steps")
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEAnywhere", help="Reset dataset name")
parser.add_argument("--idx", type=int, default=0, help="Index of the reset state to load from the dataset")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
    help="Root directory of the OmniReset dataset",
)
parser.add_argument("--output", type=str, default=None, help="Output .npz path (default: playback_<stem>_idx<idx>.npz)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.playback_cfg import (  # noqa: E402
    Ur5eRobotiq2f85PlaybackCfg,
)


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


def main():
    actions, extras = load_actions(args_cli.actions, args_cli.action_key, args_cli.max_steps)

    cfg = Ur5eRobotiq2f85PlaybackCfg()
    cfg.scene.num_envs = 1
    env = gym.make("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Playback-v0", cfg=cfg).unwrapped
    env.reset()

    action_dim = env.action_manager.total_action_dim
    if actions.shape[1] != action_dim:
        raise ValueError(f"actions have {actions.shape[1]} dims but the env expects {action_dim}")

    # Build a full scene state, then overlay the dataset assets (robot + objects).
    # env.reset_to() requires every scene asset present, but the dataset only holds
    # the robot and the two objects -- the rest keep their current (default) state.
    state = env.scene.get_state(is_relative=True)
    reset_state = load_reset_state(env, args_cli.dataset_dir, args_cli.reset_type, args_cli.idx)
    for category, assets in reset_state.items():
        for asset_name, asset_state in assets.items():
            state[category][asset_name] = asset_state

    env_ids = torch.tensor([0], device=env.device)
    obs, _ = env.reset_to(state, env_ids=env_ids, is_relative=True)

    # ---- open-loop playback: one recorded action per env step, no policy ----
    actions_t = torch.from_numpy(actions).to(env.device)
    recorded = [obs_terms_to_numpy(obs)]  # obs BEFORE actions[0]
    for t in range(actions_t.shape[0]):
        obs, *_ = env.step(actions_t[t : t + 1])
        recorded.append(obs_terms_to_numpy(obs))

    # ---- save each obs term as its own array ----
    out = {name: np.stack([step[name] for step in recorded]) for name in recorded[0]}
    out["actions"] = actions
    for name, value in extras.items():
        out[f"real/{name}"] = value

    stem = os.path.splitext(os.path.basename(args_cli.actions))[0]
    out_path = args_cli.output or f"playback_{stem}_idx{args_cli.idx}.npz"
    np.savez(out_path, **out)

    print("\n" + "=" * 72)
    print(f"Playback: {args_cli.actions}  reset={args_cli.reset_type}  idx={args_cli.idx}")
    print(f"Stepped {actions.shape[0]} actions; recorded {len(recorded)} observations (T+1).")
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
