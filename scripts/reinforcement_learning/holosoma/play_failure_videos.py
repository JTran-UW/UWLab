# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay the failures of a play.py eval and record one clip per failure.

Phase 1 (``play.py --eval_episodes_per_env N --save_failure_states f.pt``) saves the full scene
start state of every counted failed episode. This script loads them into a 1-env instance of the
same task, ``env.reset_to``s each state (asserted to match), rolls out the same checkpoint and
writes ``clips/f<idx>_env<id>_<orig-outcome>.mp4`` plus a padded grid montage and ``results.json``.

Actions are sampled when ``--stochastic`` and the reset-mode DR terms (peg mass, materials, gains,
sysid scales) re-draw inside ``reset_to``, so a replay can succeed where the eval failed. Each
failure is therefore replayed up to ``--max_replays`` times, stopping at the first attempt that
fails again; only that attempt is written as a clip (labelled with the original outcome, the replay
outcome and the attempt number). Failures that succeed on every attempt get no clip and are listed
as unreproduced in ``results.json`` and the summary.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Replay eval failures and record clips.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--failure_states", type=str, required=True, help=".pt written by play.py --save_failure_states")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--max_clips", type=int, default=None, help="Replay only the first N failures.")
parser.add_argument("--max_replays", type=int, default=5,
                    help="Replay each failure state up to this many times, stopping at the first attempt that fails again.")
parser.add_argument(
    "--exclude_outcomes", type=str, nargs="*", default=["abnormal_robot"],
    help="Original failure buckets to skip (names from the eval breakdown: timeout, abnormal_robot, other). "
         "Default skips abnormal_robot; pass with no values to replay everything.",
)
parser.add_argument("--panel_width", type=int, default=640)
parser.add_argument("--panel_height", type=int, default=480)
parser.add_argument("--montage_cols", type=int, default=6, help="Columns in the grid montage; 0 disables it.")
parser.add_argument("--montage_scale", type=float, default=0.5, help="Per-panel scale in the montage.")
parser.add_argument("--cam_offset", type=float, nargs=3, default=[0.9, 0.0, 0.45],
                    help="Camera eye offset (x,y,z) from the look-at point, which is the peghole.")
parser.add_argument("--cam_lookat_z", type=float, default=0.05, help="Look-at height above the peghole root.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # offscreen viewport render for rgb_array
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
from tensordict import TensorDict

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from vecenv_wrapper import HolosomaVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


def _state_slice(state, i: int, device):
    if isinstance(state, dict):
        return {k: _state_slice(v, i, device) for k, v in state.items()}
    return state[i : i + 1].to(device)


def _max_state_diff(a, b) -> float:
    if isinstance(a, dict):
        return max((_max_state_diff(a[k], b[k]) for k in a), default=0.0)
    return float((a.cpu().float() - b.cpu().float()).abs().max())


def _draw_label(frame: np.ndarray, top: str, bottom: str | None, dim: bool, success: bool = False) -> np.ndarray:
    f = frame.copy()
    if dim:
        f = (f * 0.45).astype(np.uint8)
    bar_h = 36
    cv2.rectangle(f, (0, 0), (f.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(f, top, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if success:
        txt = "SUCCESS"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x, y = f.shape[1] - tw - 16, bar_h + th + 14
        cv2.rectangle(f, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 110, 0), -1)
        cv2.putText(f, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 255, 90), 3, cv2.LINE_AA)
    if bottom:
        (tw, th), _ = cv2.getTextSize(bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x, y = (f.shape[1] - tw) // 2, f.shape[0] // 2
        cv2.rectangle(f, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
        color = (80, 220, 80) if "success" in bottom else (80, 80, 255)
        cv2.putText(f, bottom, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return f


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.viewer.eye = (3.5, 0.0, 1.3)
    env_cfg.viewer.resolution = (args_cli.panel_width, args_cli.panel_height)
    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    data = torch.load(args_cli.failure_states, weights_only=False, map_location="cpu")
    N_total = int(data["num_failures"])
    if N_total == 0:
        print("[INFO] no failures recorded in", args_cli.failure_states)
        return
    outcome_names = data["outcome_names"]
    excluded = set(args_cli.exclude_outcomes or [])
    unknown = excluded - set(outcome_names.values())
    if unknown:
        raise ValueError(f"--exclude_outcomes {sorted(unknown)} not in {sorted(outcome_names.values())}")
    keep = [i for i in range(N_total) if outcome_names[int(data["outcome"][i])] not in excluded]
    n_excluded = N_total - len(keep)
    if args_cli.max_clips:
        keep = keep[: args_cli.max_clips]
    N = len(keep)
    if N == 0:
        print(f"[INFO] nothing to replay: {N_total} failures, {n_excluded} excluded ({sorted(excluded)})")
        return
    if data.get("task") and data["task"] != args_cli.task:
        print(f"[WARN] states were recorded on task {data['task']!r}, replaying on {args_cli.task!r}")
    if data.get("checkpoint") and os.path.basename(data["checkpoint"]) != os.path.basename(resume_path):
        print(f"[WARN] states were recorded with {os.path.basename(data['checkpoint'])}, replaying {os.path.basename(resume_path)}")
    print(f"[INFO] replaying {N}/{N_total} failures from {args_cli.failure_states}"
          f" ({n_excluded} excluded: {sorted(excluded) or 'none'})")

    os.makedirs(args_cli.output_dir, exist_ok=True)
    clips_dir = os.path.join(args_cli.output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    try:
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import describe_assembly_assets

        print(describe_assembly_assets(env_cfg))
    except Exception as exc:  # noqa: BLE001
        print(f"[assembly] could not describe peg/hole assets: {exc}")

    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")
    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions) if is_fastsac else RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped
    device = uenv.device
    fps = int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)))

    if is_fastsac:
        runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.setup()
        runner.load(resume_path)
        actor_obs_keys = agent_cfg.actor_obs_keys
        if args_cli.stochastic:
            actor = runner.actor.to(device).eval()
            obs_norm = runner.obs_normalizer.to(device).eval()

            def _policy(obs):
                x = obs["actor_obs"]
                nx = obs_norm(x, update=False) if runner.obs_normalization else x
                return actor.explore(nx, deterministic=False)
        else:
            _policy = runner.get_inference_policy(device=device)

        def act(obs):
            return _policy({"actor_obs": torch.cat([obs[k] for k in actor_obs_keys], dim=1)})
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        act = runner.get_inference_policy_stochastic(device=device) if args_cli.stochastic else runner.get_inference_policy(device=device)
    print(f"[INFO] policy: {'stochastic' if args_cli.stochastic else 'deterministic'}, fps={fps}")

    context_term = uenv.reward_manager.get_term_cfg("progress_context").func
    term_mgr = uenv.termination_manager
    has_abnormal = "abnormal_robot" in term_mgr.active_terms

    def run_episode(obs):
        frames = [uenv.render()]
        ever_success, first_success, outcome, step = False, None, "timeout", 0
        while True:
            with torch.inference_mode():
                a = act(obs)
            obs, _, dones, _ = env.step(a.clone())
            step += 1
            if not ever_success and bool((context_term.orientation_aligned & context_term.position_aligned)[0]):
                ever_success, first_success = True, step
            if bool(dones[0]):
                if has_abnormal and bool(term_mgr.get_term("abnormal_robot")[0]):
                    outcome = "abnormal"
                break
            frames.append(uenv.render())
        if ever_success:
            outcome = "success"
        return frames, {"steps": step, "success": ever_success, "first_success_step": first_success, "outcome": outcome}

    # One ordinary reset to initialise buffers, then aim the camera at the peghole.
    env.reset()
    hole = uenv.scene["receptive_object"].data.root_pos_w[0].cpu().numpy()
    lookat = hole + np.array([0.0, 0.0, args_cli.cam_lookat_z])
    eye = lookat + np.array(args_cli.cam_offset)
    uenv.sim.set_camera_view(eye.tolist(), lookat.tolist())
    print(f"[INFO] camera eye={eye.round(3).tolist()} lookat={lookat.round(3).tolist()}")

    results = []
    for n, i in enumerate(keep):
        state = _state_slice(data["states"], i, device)
        uenv.reset_to(state, None, is_relative=True)
        diff = _max_state_diff(uenv.scene.get_state(is_relative=True), state)
        assert diff < 1e-4, f"start state replay mismatch for failure {i}: max diff {diff}"
        obs = TensorDict(uenv.obs_buf, batch_size=[1]) if is_fastsac else env.get_observations()

        orig = outcome_names[int(data["outcome"][i])]
        env_id = int(data["env_id"][i])
        attempts = []
        for attempt in range(1, args_cli.max_replays + 1):
            if attempt > 1:
                uenv.reset_to(state, None, is_relative=True)
                obs = TensorDict(uenv.obs_buf, batch_size=[1]) if is_fastsac else env.get_observations()
            frames, info = run_episode(obs)
            if n == 0 and attempt == 1:
                imageio.imwrite(os.path.join(args_cli.output_dir, "camera_preview.png"), frames[0])
            attempts.append(info["outcome"])
            if not info["success"]:
                break
        info.update({
            "index": i,
            "env_id": env_id,
            "orig_outcome": orig,
            "orig_duration": int(data["duration"][i]),
            "init_peg_xyz": data["init_peg_xyz"][i].tolist(),
            "attempts": attempts,
            "reproduced": not info["success"],
            "clip": None,
        })
        if info["reproduced"]:
            clip = os.path.join(clips_dir, f"f{i:03d}_env{env_id:04d}_{orig}_try{len(attempts)}.mp4")
            imageio.mimwrite(clip, frames, fps=fps, codec="libx264", quality=9, macro_block_size=1)
            info["clip"] = clip
        results.append(info)
        print(f"[{n + 1}/{N}] env {env_id}: orig {orig} ({info['orig_duration']} steps) -> "
              f"{'reproduced as ' + info['outcome'] + f' on attempt {len(attempts)}' if info['reproduced'] else f'NOT reproduced in {len(attempts)} attempts'}"
              f" ({info['steps']} steps)")

    with open(os.path.join(args_cli.output_dir, "results.json"), "w") as f:
        json.dump({"fps": fps, "task": args_cli.task, "checkpoint": resume_path,
                   "stochastic": args_cli.stochastic, "num_failures_total": N_total,
                   "excluded_outcomes": sorted(excluded), "num_excluded": n_excluded,
                   "results": results}, f, indent=2)
    env.close()

    repro = [r for r in results if r["reproduced"]]
    n_repro = len(repro)
    n_abn = sum(r["outcome"] == "abnormal" for r in repro)
    n_first = sum(len(r["attempts"]) == 1 for r in repro)
    print(f"[SUMMARY] replayed {N} of {N_total} failures ({n_excluded} excluded), up to {args_cli.max_replays} attempts each: "
          f"reproduced {n_repro} ({n_repro / N:.1%}; timeout {n_repro - n_abn}, abnormal {n_abn}; "
          f"{n_first} on the first attempt), not reproduced {N - n_repro}")

    # ---- montage: grid of the reproduced failures, each padded (frozen, dimmed, tagged) to the longest ----
    cols = args_cli.montage_cols
    if cols <= 0 or n_repro == 0:
        return
    results = repro
    N = n_repro
    W = int(args_cli.panel_width * args_cli.montage_scale)
    H = int(args_cli.panel_height * args_cli.montage_scale)
    clips = [[np.asarray(x) for x in imageio.mimread(r["clip"], memtest=False)] for r in results]
    L = max(len(c) for c in clips)
    rows = (N + cols - 1) // cols
    out_path = os.path.join(args_cli.output_dir, "failures_montage.mp4")
    out_path_4x = os.path.join(args_cli.output_dir, "failures_montage_4x.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=9, macro_block_size=1)
    writer4x = imageio.get_writer(out_path_4x, fps=fps * 4, codec="libx264", quality=9, macro_block_size=1)
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    for t in range(L):
        panels = []
        for r, fr in zip(results, clips):
            live = t < len(fr)
            frame = cv2.resize(fr[t] if live else fr[-1], (W, H))
            top = f"f{r['index']} env{r['env_id']} orig:{r['orig_outcome']} try{len(r['attempts'])}"
            tag = None if live else f"done: {r['outcome']}"
            fs = r["first_success_step"]
            panels.append(_draw_label(frame, top, tag, dim=not live, success=bool(r["success"]) and fs is not None and t >= fs))
        panels += [blank] * (rows * cols - N)
        grid = np.concatenate([np.concatenate(panels[r * cols:(r + 1) * cols], axis=1) for r in range(rows)], axis=0)
        writer.append_data(grid)
        writer4x.append_data(grid)
    writer.close()
    writer4x.close()
    print(f"[INFO] wrote {out_path} and {out_path_4x}")


if __name__ == "__main__":
    main()
    simulation_app.close()
