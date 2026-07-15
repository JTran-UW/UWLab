# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval a DAgger depth student policy and record per-env videos showing both
depth cameras side-by-side as the policy executes. Use to inspect failure modes.

Usage:
    ./_isaaclab/IsaacLab/isaaclab.sh -p scripts/eval_depth_dagger_video.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-4Canonical-Lean-v0 \\
        --resume_path /path/to/model_70000.pt \\
        --num_envs 4 --num_steps 320 \\
        --out_dir /tmp/dagger_eval \\
        env.scene.insertive_object=peg env.scene.receptive_object=peghole
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# cli_args lives in scripts/reinforcement_learning/rsl_rl/ — add to sys.path
# BEFORE import.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "reinforcement_learning", "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Eval depth DAgger policy with per-env depth videos.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=320, help="Total env steps to record.")
parser.add_argument("--resume_path", type=str, required=True, help="DAgger checkpoint path (model_*.pt).")
parser.add_argument("--out_dir", type=str, default="/tmp/dagger_eval")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
parser.add_argument("--depth_clip_max_m", type=float, default=2.0, help="Clip depth at this distance (m) before normalization.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import imageio.v2 as imageio
from rsl_rl.runners import DistillationRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

# Inject UWLab classes (same pattern as play.py).
import rsl_rl.runners.distillation_runner as _distillation_runner_module
from uwlab_rl.rsl_rl.distillation_dagger import DistillationDAgger
from uwlab_rl.rsl_rl.distillation_dagger_weighted import DistillationDAggerWeighted
from uwlab_rl.rsl_rl.distillation_runner_split import DistillationRunnerSplit
from uwlab_rl.rsl_rl.student_teacher_mlp import StudentTeacherMLP
from uwlab_rl.rsl_rl.student_teacher_vision import StudentTeacherVision
from uwlab_rl.rsl_rl.student_teacher_vision_recurrent import StudentTeacherVisionRecurrent

_distillation_runner_module.StudentTeacherVision = StudentTeacherVision
_distillation_runner_module.StudentTeacherVisionRecurrent = StudentTeacherVisionRecurrent
_distillation_runner_module.StudentTeacherMLP = StudentTeacherMLP
_distillation_runner_module.DistillationDAgger = DistillationDAgger
_distillation_runner_module.DistillationDAggerWeighted = DistillationDAggerWeighted
_distillation_runner_module.DistillationRunnerSplit = DistillationRunnerSplit

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

from uwlab_tasks.utils.hydra import hydra_task_config


def _depth_to_uint8(d: np.ndarray, hi: float) -> np.ndarray:
    """Map depth (m) ∈ [0, hi] to uint8. Inf/NaN become 255 (far)."""
    x = np.where(np.isfinite(d), d, hi)
    x = np.clip(x, 0.0, hi)
    x = (x / hi) * 255.0
    return x.astype(np.uint8)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = args_cli.out_dir
    os.makedirs(args_cli.out_dir, exist_ok=True)

    # Disable success terminations so the simulator continues running past the
    # success moment — lets the video show the seated/inserted state for some
    # frames before the (delayed) reset triggers. Episodes still end on
    # ``time_out`` / ``abnormal_robot``. We trigger a manual reset N steps after
    # the first success-step per env (see ``_post_success_steps`` below).
    env_cfg.terminations.success = None
    env_cfg.terminations.early_success = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.resume_path)
    print(f"[eval] Loading DAgger checkpoint: {resume_path}")
    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy_nn = runner.alg.policy
    # MATCH TRAINING-TIME EVAL CONDITIONS (avoids BN train/eval mode shift):
    # 1. keep model in train mode → BN uses batch stats just like training did
    # 2. sample actions (policy.act()) instead of using deterministic mean
    #    (policy.act_inference()) — DAgger trained with std=0.1 sampling
    policy_nn.to(env.unwrapped.device)
    policy_nn.train()
    @torch.no_grad()
    def policy(obs):
        return policy_nn.act(obs).detach()

    # Per-env, per-camera frame buffers + per-env episode tracking.
    cam_keys = []
    obs0 = env.get_observations()
    for k in ("side_depth", "wrist_depth"):
        if k in obs0.keys():
            cam_keys.append(k)
    print(f"[eval] depth cameras detected: {cam_keys}")
    if not cam_keys:
        raise RuntimeError("No depth camera obs groups found in this env (looked for side_depth, wrist_depth).")

    # frames[env_idx] = list of HxW concatenated uint8 arrays
    frames = [[] for _ in range(args_cli.num_envs)]
    # annotations[env_idx] = list of (step, success_bool, done_bool)
    annotations = [[] for _ in range(args_cli.num_envs)]
    cur_ep_len = torch.zeros(args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device)
    cur_ep_id = torch.zeros(args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device)
    ep_outcomes = [[] for _ in range(args_cli.num_envs)]  # per-env list of "S"/"F"

    # Post-success delayed-reset state (success terms disabled, see env_cfg).
    POST_SUCCESS_RESET_STEPS = 10
    # -1 = no success yet this episode; else number of steps since first success.
    post_succ = np.full(args_cli.num_envs, -1, dtype=np.int64)
    # whether the (current) episode ever crossed the success threshold.
    ep_had_success = np.zeros(args_cli.num_envs, dtype=bool)

    obs = env.get_observations()
    progress_term = None
    try:
        progress_term = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
    except Exception:
        pass

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            # Render the obs the policy is about to act on.
            # IsaacLab auto-resets done envs *inside* env.step() before returning
            # obs, so rendering after the step would show the next reset state
            # instead of the success/terminal frame. By rendering pre-step, frame
            # N shows the state the policy saw at step N.
            cam_arrays = []
            for k in cam_keys:
                d = obs[k]  # [N, 1, H, W] or [N, H, W]
                if d.ndim == 4:
                    d = d.squeeze(1)
                cam_arrays.append(d.detach().cpu().float().numpy())
            for env_idx in range(args_cli.num_envs):
                stacked = [_depth_to_uint8(c[env_idx], args_cli.depth_clip_max_m) for c in cam_arrays]
                concat = np.concatenate(stacked, axis=1)  # H x (W*ncam)
                # convert grayscale -> RGB so video encodes properly
                rgb = np.stack([concat] * 3, axis=-1)
                frames[env_idx].append(rgb)

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # Recurrent state reset
            policy_nn.reset(dones)

            cur_ep_len += 1

            # Track terminations + success
            done_np = dones.detach().cpu().numpy().astype(bool)
            if progress_term is not None:
                try:
                    succ_np = progress_term.success.detach().cpu().numpy().astype(bool)
                except Exception:
                    succ_np = np.zeros(args_cli.num_envs, dtype=bool)
            else:
                succ_np = np.zeros(args_cli.num_envs, dtype=bool)

            # Post-success delayed reset: count POST_SUCCESS_RESET_STEPS frames
            # after the first success step, then trigger a manual reset for that
            # env. Lets the video show the seated/inserted state for ~10 frames.
            ep_had_success |= succ_np
            first_success = succ_np & (post_succ < 0)
            post_succ[first_success] = 0
            in_count = post_succ >= 0
            post_succ[in_count] += 1
            manual_reset_ids = np.nonzero(post_succ >= POST_SUCCESS_RESET_STEPS)[0]
            if len(manual_reset_ids) > 0:
                ids_t = torch.as_tensor(manual_reset_ids, dtype=torch.long, device=env.unwrapped.device)
                env.unwrapped._reset_idx(ids_t)
                # mark as done for accounting + record the outcome below
                done_np[manual_reset_ids] = True
                # progress_term.success buffer is post-reset all-zero now;
                # use ep_had_success to label outcome.
                policy_nn.reset(torch.as_tensor(done_np, device=env.unwrapped.device))

            for env_idx in range(args_cli.num_envs):
                if done_np[env_idx]:
                    outcome = "S" if (ep_had_success[env_idx] or succ_np[env_idx]) else "F"
                    ep_outcomes[env_idx].append(
                        f"ep{int(cur_ep_id[env_idx])}@{step}_{outcome}_len{int(cur_ep_len[env_idx])}"
                    )
                    cur_ep_id[env_idx] += 1
                    cur_ep_len[env_idx] = 0
                    post_succ[env_idx] = -1
                    ep_had_success[env_idx] = False

            if step % 20 == 0:
                print(f"[eval] step {step}/{args_cli.num_steps}")

    print("[eval] writing videos…")
    for env_idx in range(args_cli.num_envs):
        out_path = os.path.join(args_cli.out_dir, f"env{env_idx}_depth.mp4")
        # imageio needs T, H, W, C uint8
        arr = np.stack(frames[env_idx], axis=0)
        imageio.mimwrite(out_path, arr, fps=args_cli.fps, codec="libx264")
        print(f"  env {env_idx}: {out_path} ({arr.shape[0]} frames, {arr.shape[1]}x{arr.shape[2]})")
        # Episode summary
        if ep_outcomes[env_idx]:
            print(f"    episodes: {' '.join(ep_outcomes[env_idx])}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
