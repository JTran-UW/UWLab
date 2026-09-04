# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Side-by-side video of TWO policies (FastSAC + PPO) replayed from identical recorded reset states.

Reset states come from a torch file {resets: {idx: {state, ...}}} (scene.get_state(is_relative=True)
slices). For each chosen reset both policies are run from that exact state (one stochastic draw each);
panels are padded per reset to the longer episode (frozen, dimmed, outcome tag) so both advance to the
next reset together. Episodes end on the task's own terminations (time_out / abnormal_robot).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Paired FastSAC-vs-PPO replay video.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--fastsac_checkpoint", type=str, required=True)
parser.add_argument("--ppo_checkpoint", type=str, required=True)
parser.add_argument("--ppo_agent_task", type=str, required=True, help="task whose rsl_rl_cfg_entry_point is the PPO runner cfg")
parser.add_argument("--states", type=str, required=True)
parser.add_argument("--resets", type=str, required=True, help='"idx:rate,idx:rate,..." (rate shown in the left title)')
parser.add_argument("--mode_label", type=str, default="timeout")
parser.add_argument("--left_label", type=str, default="FastSAC 250k seed0")
parser.add_argument("--right_label", type=str, default="PPO seed0")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--panel_width", type=int, default=640)
parser.add_argument("--panel_height", type=int, default=480)
parser.add_argument("--cam_offset", type=float, nargs=3, default=[0.9, 0.0, 0.45])
parser.add_argument("--cam_lookat_z", type=float, default=0.05)
cli_args.add_rsl_rl_args(parser)
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
from tensordict import TensorDict

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from vecenv_wrapper import HolosomaVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


def _to_dev(state, device):
    if isinstance(state, dict):
        return {k: _to_dev(v, device) for k, v in state.items()}
    return state.to(device)


def _draw_label(frame, top, bottom, dim, success=False):
    f = frame.copy()
    if dim:
        f = (f * 0.45).astype(np.uint8)
    bar_h = 36
    cv2.rectangle(f, (0, 0), (f.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(f, top, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.viewer.eye = (3.5, 0.0, 1.3)
    env_cfg.viewer.resolution = (args_cli.panel_width, args_cli.panel_height)
    env_cfg.log_dir = args_cli.output_dir
    os.makedirs(args_cli.output_dir, exist_ok=True)
    clips_dir = os.path.join(args_cli.output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    try:
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import describe_assembly_assets
        print(describe_assembly_assets(env_cfg))
    except Exception as exc:  # noqa: BLE001
        print(f"[assembly] could not describe assets: {exc}")
    h_env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped
    device = uenv.device
    fps = int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)))

    # --- FastSAC (stochastic) ---
    runner = FastSACAgent(h_env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.setup()
    runner.load(retrieve_file_path(args_cli.fastsac_checkpoint))
    actor_obs_keys = agent_cfg.actor_obs_keys
    _actor = runner.actor.to(device).eval()
    _norm = runner.obs_normalizer.to(device).eval()

    def fastsac_policy(obs: TensorDict) -> torch.Tensor:
        x = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
        nx = _norm(x, update=False) if runner.obs_normalization else x
        return _actor.explore(nx, deterministic=False)

    # --- PPO (stochastic) ---
    ppo_cfg = load_cfg_from_registry(args_cli.ppo_agent_task, "rsl_rl_cfg_entry_point")
    ppo_cfg = cli_args.sanitize_rsl_rl_cfg(ppo_cfg)
    ppo_cfg.device = str(device)
    r_env = RslRlVecEnvWrapper(env, clip_actions=getattr(ppo_cfg, "clip_actions", None))
    ppo_runner = OnPolicyRunner(r_env, ppo_cfg.to_dict(), log_dir=None, device=str(device))
    ppo_path = retrieve_file_path(args_cli.ppo_checkpoint)
    try:
        ppo_runner.load(ppo_path)
    except RuntimeError as e:
        loaded = torch.load(ppo_path, weights_only=False, map_location=str(device))
        ckpt_sd = loaded["model_state_dict"]
        model_sd = ppo_runner.alg.policy.state_dict()
        mismatched = [k for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape != v.shape]
        if any(not k.startswith(("critic.", "critic_obs_normalizer.")) for k in mismatched):
            raise e
        ppo_runner.alg.policy.load_state_dict({k: v for k, v in ckpt_sd.items() if k not in mismatched}, strict=False)
        print(f"[WARN] PPO strict load failed; skipped critic-side mismatches: {mismatched}")
    ppo_policy = ppo_runner.get_inference_policy_stochastic(device=str(device))
    print("[INFO] both policies loaded (stochastic)")

    context_term = uenv.reward_manager.get_term_cfg("progress_context").func
    term_mgr = uenv.termination_manager

    def run_episode(obs: TensorDict, policy, is_fastsac: bool):
        frames = [uenv.render()]
        ever_success, first_success, outcome, step = False, None, "timeout", 0
        while True:
            with torch.inference_mode():
                a = policy(obs) if is_fastsac else policy(obs)
            obs, _, dones, _ = h_env.step(a.clone().float())
            step += 1
            if bool((context_term.orientation_aligned & context_term.position_aligned)[0]) and not ever_success:
                ever_success, first_success = True, step
            if bool(dones[0]):
                if bool(term_mgr.get_term("abnormal_robot")[0]):
                    outcome = "abnormal"
                break
            frames.append(uenv.render())
        return frames, {"steps": step, "success": ever_success, "first_success_step": first_success, "outcome": outcome}

    states = torch.load(args_cli.states, map_location="cpu", weights_only=False)
    resets = [(int(p.split(":")[0]), float(p.split(":")[1])) for p in args_cli.resets.split(",")]
    policies = [(args_cli.left_label, fastsac_policy, True), (args_cli.right_label, ppo_policy, False)]
    results = []
    camera_set = False
    for k, (ridx, rate) in enumerate(resets):
        st = _to_dev(states["resets"][ridx]["state"], device)
        entry = {"reset": ridx, "fastsac_rate": rate, "panels": []}
        for label, pol, is_fs in policies:
            uenv.reset_to(st, None, is_relative=bool(states.get("is_relative", True)))
            if not camera_set:
                hole = uenv.scene["receptive_object"].data.root_pos_w[0].cpu().numpy()
                lookat = hole + np.array([0.0, 0.0, args_cli.cam_lookat_z])
                eye = lookat + np.array(args_cli.cam_offset)
                uenv.sim.set_camera_view(eye.tolist(), lookat.tolist())
                camera_set = True
            obs = TensorDict(uenv.obs_buf, batch_size=[1])
            frames, info = run_episode(obs, pol, is_fs)
            clip = os.path.join(clips_dir, f"reset{ridx:03d}_{'fastsac' if is_fs else 'ppo'}.mp4")
            imageio.mimwrite(clip, frames, fps=fps, codec="libx264", quality=9, macro_block_size=1)
            info.update({"label": label, "clip": clip})
            entry["panels"].append(info)
            print(f"[reset {ridx} | {label}] steps={info['steps']} outcome={info['outcome']} success={info['success']}")
        results.append(entry)

    with open(os.path.join(args_cli.output_dir, "results.json"), "w") as f:
        json.dump({"fps": fps, "mode": args_cli.mode_label, "results": results}, f, indent=2)
    env.close()

    W, H = args_cli.panel_width, args_cli.panel_height
    name = f"failures_{args_cli.mode_label}_side_by_side_fastsac_vs_ppo"
    out, out4 = os.path.join(args_cli.output_dir, name + ".mp4"), os.path.join(args_cli.output_dir, name + "_4x.mp4")
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=9, macro_block_size=1)
    w4 = imageio.get_writer(out4, fps=fps * 4, codec="libx264", quality=9, macro_block_size=1)
    for entry in results:
        per = [(p, [np.asarray(x) for x in imageio.mimread(p["clip"], memtest=False)]) for p in entry["panels"]]
        L = max(len(fr) for _, fr in per)
        for t in range(L):
            panels = []
            for p, fr in per:
                live = t < len(fr)
                frame = fr[t] if live else fr[-1]
                if frame.shape[:2] != (H, W):
                    frame = cv2.resize(frame, (W, H))
                top = f"{p['label']} | reset {entry['reset']}"
                if p["label"] == args_cli.left_label:
                    top += f" | FastSAC {args_cli.mode_label} rate {entry['fastsac_rate']:.2f}"
                tag = None if live else "done: " + ("success" if p["success"] else p["outcome"])
                fs = p["first_success_step"]
                succ = bool(p["success"]) and fs is not None and t >= fs
                panels.append(_draw_label(frame, top, tag, dim=not live, success=succ))
            row = np.concatenate(panels, axis=1)
            w.append_data(row); w4.append_data(row)
    w.close(); w4.close()
    print(f"[INFO] wrote {out} and {out4}")
    for entry in results:
        print(f"reset {entry['reset']:3d} (FastSAC {args_cli.mode_label} rate {entry['fastsac_rate']:.2f}): "
              + " | ".join(f"{p['label']}: {'SUCCESS' if p['success'] else p['outcome']}" for p in entry["panels"]))


if __name__ == "__main__":
    main()
    simulation_app.close()
