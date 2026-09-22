# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay episodes from a replay buffer as videos (one mp4 per episode).

Modes:
  * ``state`` (default): kinematic playback. Requires a buffer recorded with
    ``play.py --record_transitions --record_scene_state`` (``buffer_tensors['scene_state']``); every step the
    stored env-relative scene state is written into a 1-env sim via ``reset_to`` and a frame is rendered.
    Exact reproduction of what was recorded.
  * ``action``: open-loop physics replay. Resets to ``--reset_state`` (single-state file) and steps the recorded
    actions; reports per-step joint drift vs. the recorded state (if present) so you can judge fidelity.

    python scripts/reinforcement_learning/holosoma/replay_rb_trajectories.py \
        --task <Single-Reset train task> --buffer rbs/<filtered>.pt --episodes 0-9 --output_dir videos/replay --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay replay-buffer episodes as videos.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--buffer", type=str, required=True)
parser.add_argument("--episodes", type=str, default="0-9", help="comma list and/or ranges, e.g. 0-4,10,17; 'all' for every episode")
parser.add_argument("--mode", choices=["state", "action"], default="state")
parser.add_argument("--reset_state", type=str, default="reset_states/single_reset_seed42_r0.pt",
                    help="action mode: one-state reset file (reset-dataset format) every episode starts from")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--speedups", type=int, nargs="+", default=[1, 4])
parser.add_argument("--panel_width", type=int, default=640)
parser.add_argument("--panel_height", type=int, default=480)
parser.add_argument("--cam_offset", type=float, nargs=3, default=[0.9, 0.0, 0.45])
parser.add_argument("--cam_lookat_z", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


def _parse_episodes(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def _unflatten(flat: dict, t: int, device) -> dict:
    """{'articulation/robot/joint_position': (1,T,D), ...} -> nested state dict for step t with batch 1."""
    st: dict = {}
    for key, v in flat.items():
        parts = key.split("/"); d = st
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v[0, t:t + 1].to(device)
    return st


def _label(frame, text, sub=None):
    f = frame.copy(); cv2.rectangle(f, (0, 0), (f.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(f, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(f, sub, (8, f.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return f


@hydra_task_config(args_cli.task, "")
def main(env_cfg, agent_cfg):
    buf = torch.load(args_cli.buffer, map_location="cpu", weights_only=False)
    bt, meta = buf["buffer_tensors"], buf["metadata"]
    assert meta["n_env"] == 1, "expected a single-env (collapsed / filtered) buffer"
    dones = bt["dones"][0]; ends = torch.nonzero(dones).flatten(); starts = torch.cat([torch.tensor([0]), ends[:-1] + 1])
    n_ep = len(ends); episodes = _parse_episodes(args_cli.episodes, n_ep)
    has_state = "scene_state" in bt
    if args_cli.mode == "state" and not has_state:
        raise SystemExit("buffer has no scene_state; re-record with play.py --record_scene_state or use --mode action")
    print(f"[replay] {n_ep} episodes in buffer, replaying {len(episodes)}, mode={args_cli.mode}, scene_state={has_state}")

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.viewer.resolution = (args_cli.panel_width, args_cli.panel_height)
    env_cfg.log_dir = args_cli.output_dir
    os.makedirs(args_cli.output_dir, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped; device = uenv.device
    dt = float(env_cfg.sim.dt * env_cfg.decimation); fps = int(round(1.0 / dt))
    env.reset()
    hole = uenv.scene["receptive_object"].data.root_pos_w[0].cpu().numpy()
    lookat = hole + np.array([0.0, 0.0, args_cli.cam_lookat_z])
    uenv.sim.set_camera_view((lookat + np.array(args_cli.cam_offset)).tolist(), lookat.tolist())

    reset_state = None
    if args_cli.mode == "action":
        ds = torch.load(args_cli.reset_state, map_location="cpu", weights_only=False)["initial_state"]

        def _first(x):
            return {k: _first(v) for k, v in x.items()} if isinstance(x, dict) else x[0].unsqueeze(0).to(device)
        reset_state = _first(ds)

    joint_key = "articulation/robot/joint_position"
    summary = []
    for ei in episodes:
        s, e = int(starts[ei]), int(ends[ei]); T = e - s + 1
        frames, drift = [], []
        if args_cli.mode == "state":
            uenv.reset_to(_unflatten(bt["scene_state"], s, device), None, is_relative=True)
            for _ in range(2):  # renderer warm-up: the first capture after a reset comes back black
                uenv.sim.render(); uenv.render()
            for t in range(s, e + 1):
                uenv.reset_to(_unflatten(bt["scene_state"], t, device), None, is_relative=True)
                uenv.sim.render()
                frames.append(_label(uenv.render(), f"episode {ei}  step {t - s + 1}/{T}", "kinematic playback of recorded scene state"))
        else:
            uenv.reset_to(reset_state, None, is_relative=True)
            for _ in range(2):
                uenv.sim.render(); uenv.render()
            for t in range(s, e + 1):
                if has_state:
                    live = uenv.scene["robot"].data.joint_pos[0].cpu(); rec = bt["scene_state"][joint_key][0, t]
                    drift.append(float((live - rec).abs().max()))
                sub = f"open-loop action replay | joint drift {drift[-1]:.3f} rad" if drift else "open-loop action replay"
                frames.append(_label(uenv.render(), f"episode {ei}  step {t - s + 1}/{T}", sub))
                a = bt["actions"][0, t:t + 1].to(device)
                env.step(a)
        for sp in args_cli.speedups:
            path = os.path.join(args_cli.output_dir, f"episode_{ei:04d}{'' if sp == 1 else f'_{sp}x'}.mp4")
            imageio.mimwrite(path, frames, fps=fps * sp, codec="libx264", quality=8, macro_block_size=1)
        rec = {"episode": ei, "steps": T, "reward_sum": float(bt["rewards"][0, s:e + 1].sum()),
               "ends_with_trunc": bool(bt["truncations"][0, e])}
        if drift:
            rec["joint_drift_max_rad"] = max(drift); rec["joint_drift_final_rad"] = drift[-1]
        summary.append(rec); print(f"[episode {ei}] {T} steps" + (f", action-replay joint drift max {max(drift):.3f} rad" if drift else ""))
    with open(os.path.join(args_cli.output_dir, "replay_summary.json"), "w") as f:
        json.dump({"buffer": os.path.abspath(args_cli.buffer), "mode": args_cli.mode, "episodes": summary}, f, indent=1)
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
