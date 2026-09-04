# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a per-env eval with a per-env camera and keep only the clips of episodes that FAILED.

Single pass, no replay: a head-on ``TiledCamera`` (eye at env-local ``--cam_eye``, looking at each
env's peghole -- the same view as the gap videos) is added to every env, frames are buffered per env while its episode runs, and when the episode ends the
buffer is written to ``clips/env<id>_ep<k>_<outcome>.mp4`` if it failed (timeout / abnormal_robot /
other, minus ``--exclude_outcomes``) or discarded if it succeeded. Runs until every env has
completed ``--episodes_per_env`` episodes; surplus episodes are neither counted nor recorded, so
the reported success rate is length-unbiased (same gate as ``play.py --eval_episodes_per_env``).

Memory: frames stay in CPU RAM until an episode ends, worst case
num_envs x max_episode_steps x H x W x 3 bytes (128 envs x 160 steps x 320x240 ~ 4.7 GB).
Lower ``--num_envs`` or ``--width/--height`` if that is too much.

``first_episode_termination`` is disabled in-script (it would file the first post-reset episode
of every env as an "other" failure).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record per-env clips of failed eval episodes.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes_per_env", type=int, default=2)
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument(
    "--exclude_outcomes", type=str, nargs="*", default=["abnormal_robot"],
    help="Failure buckets NOT to record (timeout, abnormal_robot, other). Default skips abnormal_robot; "
         "pass with no values to record every failure.",
)
parser.add_argument("--record_successes", action="store_true", default=False,
                    help="Also write clips of successful episodes (default: failures only).")
parser.add_argument(
    "--abnormal_from_state", action="store_true", default=False,
    help=(
        "Classify abnormal episodes by evaluating the abnormal-robot condition every step (via the "
        "abnormal_robot REWARD term's resolved func) instead of the termination term. Use with "
        "env.terminations.abnormal_robot=null so abnormal episodes run to the horizon; any "
        "ever-abnormal episode counts as an abnormal failure even if it later succeeds, and such "
        "'recovered' episodes are tallied separately."
    ),
)
parser.add_argument(
    "--policy_autocast", type=str, choices=["off", "bf16", "fp16"], default="off",
    help=(
        "Run obs normalization + actor forward under torch.autocast with this dtype, mirroring the "
        "FastSAC training rollout (fast_sac_agent._maybe_amp); actions are cast back to fp32 for "
        "env.step exactly as training does. Default off = fp32 eval."
    ),
)
parser.add_argument("--max_clips", type=int, default=None, help="Stop writing clips after this many (eval still runs to completion).")
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--cam_eye", type=float, nargs=3, default=[1.28, 0.0, 0.50],
                    help="Camera eye position in env-local coordinates (fixed; the peghole moves per reset and a "
                         "hole-relative eye can end up inside the mount frame).")
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.45, 0.0, 0.05],
                    help="Look-at point in env-local coordinates (nominal peghole position; the hole moves by ~0.1 m per reset).")
parser.add_argument("--focal_length", type=float, default=18.15, help="Camera focal length (mm); 18.15 matches the Isaac viewport default.")
parser.add_argument("--montage_cols", type=int, default=0, help="Columns in the failure grid montage; 0 (default) disables it.")
parser.add_argument("--sequential_speedups", type=int, nargs="*", default=[1, 4],
                    help="Compose the recorded clips back to back at these playback multipliers "
                         "(via compose_failure_videos.py); pass with no values to skip.")
parser.add_argument("--montage_scale", type=float, default=1.0, help="Per-panel scale in the montage.")
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
import tqdm
from tensordict import TensorDict

import isaaclab.sim as sim_utils
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from vecenv_wrapper import HolosomaVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

OUTCOME_SUCCESS, OUTCOME_TIMEOUT, OUTCOME_ABNORMAL, OUTCOME_OTHER = 0, 1, 2, 3
OUTCOME_NAME = {OUTCOME_SUCCESS: "success", OUTCOME_TIMEOUT: "timeout", OUTCOME_ABNORMAL: "abnormal_robot", OUTCOME_OTHER: "other"}


def _draw_label(frame: np.ndarray, top: str, bottom: str | None, dim: bool) -> np.ndarray:
    f = frame.copy()
    if dim:
        f = (f * 0.45).astype(np.uint8)
    bar_h = 24
    cv2.rectangle(f, (0, 0), (f.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(f, top, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if bottom:
        (tw, th), _ = cv2.getTextSize(bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x, y = (f.shape[1] - tw) // 2, f.shape[0] // 2
        cv2.rectangle(f, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.putText(f, bottom, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
    return f


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    # Per-env head-on camera: a static env-local offset (eye -> target), baked as an OpenGL-convention
    # quaternion. Poses set at runtime via set_world_poses_from_view did not reach the tiled render.
    eye = torch.tensor([args_cli.cam_eye], dtype=torch.float32)
    target = torch.tensor([args_cli.cam_target], dtype=torch.float32)
    cam_rot = quat_from_matrix(create_rotation_matrix_from_view(eye, target, "Z", device="cpu"))[0].tolist()
    env_cfg.scene.failure_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/failure_camera",
        update_period=0,
        height=args_cli.height,
        width=args_cli.width,
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(args_cli.cam_eye), rot=tuple(cam_rot), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=args_cli.focal_length),
    )
    print(f"[INFO] camera: eye={args_cli.cam_eye} target={args_cli.cam_target} (env-local), quat(opengl)={[round(q, 4) for q in cam_rot]}")
    env_cfg.sim.render_interval = env_cfg.decimation  # render once per env step, not per physics substep
    env_cfg.num_rerenders_on_reset = 1
    if getattr(env_cfg.terminations, "first_episode_termination", None) is not None:
        env_cfg.terminations.first_episode_termination = None
        print("[INFO] disabled first_episode_termination for the eval")

    excluded = set(args_cli.exclude_outcomes or [])
    unknown = excluded - set(OUTCOME_NAME.values())
    if unknown:
        raise ValueError(f"--exclude_outcomes {sorted(unknown)} not in {sorted(OUTCOME_NAME.values())}")

    os.makedirs(args_cli.output_dir, exist_ok=True)
    clips_dir = os.path.join(args_cli.output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import describe_assembly_assets

        print(describe_assembly_assets(env_cfg))
    except Exception as exc:  # noqa: BLE001
        print(f"[assembly] could not describe peg/hole assets: {exc}")

    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")
    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions) if is_fastsac else RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped
    device = uenv.device
    N = uenv.num_envs
    fps = int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)))

    if is_fastsac:
        runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.setup()
        runner.load(resume_path)
        actor_obs_keys = agent_cfg.actor_obs_keys
        if args_cli.stochastic:
            actor = runner.actor.to(device).eval()
            obs_norm = runner.obs_normalizer.to(device).eval()

            import contextlib

            def _amp():
                if args_cli.policy_autocast == "off":
                    return contextlib.nullcontext()
                dt = torch.bfloat16 if args_cli.policy_autocast == "bf16" else torch.float16
                return torch.autocast(device_type="cuda", dtype=dt)

            def _policy(obs):
                x = obs["actor_obs"]
                with _amp():
                    nx = obs_norm(x, update=False) if runner.obs_normalization else x
                    a = actor.explore(nx, deterministic=False)
                return a.float()

            if args_cli.policy_autocast != "off":
                # One-time sanity: |actor mean under autocast - under fp32| on the current obs batch.
                def _amp_sanity(obs):
                    x = obs["actor_obs"]
                    with torch.no_grad():
                        nx32 = obs_norm(x, update=False) if runner.obs_normalization else x
                        m32 = actor(nx32)[1]
                        with _amp():
                            nxa = obs_norm(x, update=False) if runner.obs_normalization else x
                            ma = actor(nxa)[1]
                    d = (ma.float() - m32).abs()
                    print(f"[autocast-sanity] mean|dmean|={d.mean():.5f}  max={d.max():.5f}  "
                          f"(fp32 mean|a|={m32.abs().mean():.3f}) over {x.shape[0]} obs")
                _record_amp_sanity = _amp_sanity
            else:
                _record_amp_sanity = None
        else:
            _policy = runner.get_inference_policy(device=device)
            _record_amp_sanity = None

        def act(obs):
            x = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
            global _amp_sanity_done
            if _record_amp_sanity is not None and not globals().get("_amp_sanity_done"):
                globals()["_amp_sanity_done"] = True
                _record_amp_sanity({"actor_obs": x})
            if args_cli.abnormal_from_state:
                # With abnormal_robot termination disabled, diverged envs can emit NaN/inf obs;
                # sanitize so the actor keeps producing finite actions and the episode gets filmed.
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return _policy({"actor_obs": x})
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        act = runner.get_inference_policy_stochastic(device=device) if args_cli.stochastic else runner.get_inference_policy(device=device)
    print(f"[INFO] policy: {'stochastic' if args_cli.stochastic else 'deterministic'}, fps={fps}, envs={N}, "
          f"episodes/env={args_cli.episodes_per_env}, excluded={sorted(excluded) or 'none'}")

    cam = uenv.scene.sensors["failure_camera"]
    context_term = uenv.reward_manager.get_term_cfg("progress_context").func
    term_mgr = uenv.termination_manager
    has_timeout = "time_out" in term_mgr.active_terms
    has_abnormal = "abnormal_robot" in term_mgr.active_terms

    def grab_frames() -> np.ndarray:
        rgb = cam.data.output["rgb"]
        return rgb[..., :3].to(torch.uint8).cpu().numpy()

    obs = env.get_observations()
    # Make sure the camera holds the initial state before the first frame is taken.
    uenv.sim.render()
    cam.update(0.0, force_recompute=True)
    preview = grab_frames()[0]
    imageio.imwrite(os.path.join(args_cli.output_dir, "camera_preview.png"), preview)
    if float(preview.mean()) < 1.0:
        print("[WARN] camera preview is black -- check --cam_eye (camera may be inside geometry)")

    E = args_cli.episodes_per_env
    ep_count = torch.zeros(N, dtype=torch.long, device=device)
    ever_success = torch.zeros(N, dtype=torch.bool, device=device)
    ever_abnormal = torch.zeros(N, dtype=torch.bool, device=device)
    n_recovered = 0
    abn_state_cfg = uenv.reward_manager.get_term_cfg("abnormal_robot") if args_cli.abnormal_from_state else None
    ep_steps = torch.zeros(N, dtype=torch.long, device=device)
    zeros_b = torch.zeros(N, dtype=torch.bool, device=device)
    frames: list[list[np.ndarray]] = [[] for _ in range(N)]
    episodes: list[dict] = []
    n_clips = 0
    tally = {k: 0 for k in OUTCOME_NAME.values()}

    pbar = tqdm.tqdm(total=E * N, desc="Eval episodes", unit="ep")
    while bool((ep_count < E).any()):
        # Frame of the pre-step state s_t for every env that is still recording.
        rgb = grab_frames()
        recording = (ep_count < E).cpu().numpy()
        for i in np.nonzero(recording)[0]:
            frames[i].append(rgb[i])

        with torch.inference_mode():
            a = act(obs)
            obs, _, dones, _ = env.step(a)
            ever_success |= context_term.orientation_aligned & context_term.position_aligned
            if abn_state_cfg is not None:
                ever_abnormal |= abn_state_cfg.func(uenv, **abn_state_cfg.params).bool()
        ep_steps += 1
        done_mask = dones.bool()
        counting = done_mask & (ep_count < E)
        successes = ever_success & counting
        failed = counting & ~successes
        timed_out = term_mgr.get_term("time_out").bool() if has_timeout else zeros_b
        abnormal = term_mgr.get_term("abnormal_robot").bool() if has_abnormal else zeros_b
        outcome = torch.full((N,), OUTCOME_OTHER, dtype=torch.long, device=device)
        outcome[failed & timed_out & ~abnormal] = OUTCOME_TIMEOUT
        outcome[failed & abnormal] = OUTCOME_ABNORMAL
        outcome[successes] = OUTCOME_SUCCESS
        if args_cli.abnormal_from_state:
            # Ever-abnormal wins over everything, including a later success ("recovered").
            n_recovered += int((counting & ever_abnormal & ever_success).sum())
            outcome[counting & ever_abnormal] = OUTCOME_ABNORMAL

        for i in torch.nonzero(counting, as_tuple=False).flatten().tolist():
            name = OUTCOME_NAME[int(outcome[i])]
            tally[name] += 1
            rec = {"env_id": i, "episode": int(ep_count[i]), "outcome": name, "steps": int(ep_steps[i]), "clip": None}
            wanted = (name == "success" and args_cli.record_successes) or (name != "success" and name not in excluded)
            if wanted and (args_cli.max_clips is None or n_clips < args_cli.max_clips):
                clip = os.path.join(clips_dir, f"env{i:04d}_ep{int(ep_count[i])}_{name}.mp4")
                imageio.mimwrite(clip, frames[i], fps=fps, codec="libx264", quality=9, macro_block_size=1)
                rec["clip"] = clip
                n_clips += 1
            episodes.append(rec)
            frames[i] = []
        # Envs that finished an uncounted (surplus) episode also drop their buffer.
        for i in torch.nonzero(done_mask & ~counting, as_tuple=False).flatten().tolist():
            frames[i] = []
        ep_count += counting.long()
        pbar.update(int(counting.sum().item()))
        ever_success[done_mask] = False
        ever_abnormal[done_mask] = False
        ep_steps[done_mask] = 0
    pbar.close()

    total = sum(tally.values())
    n_succ = tally["success"]
    print(f"\n[EVAL] Episodes: {total}  |  Successes: {n_succ}  |  Success rate: {n_succ / total:.2%}")
    print(f"[EVAL] Failure breakdown ({total - n_succ} failures):")
    if args_cli.abnormal_from_state:
        print(f"[EVAL]   recovered-abnormal (ever-abnormal but inserted): {n_recovered}")
    for name in ("timeout", "abnormal_robot", "other"):
        print(f"[EVAL]   {name:15s}: {tally[name]:6d}  ({tally[name] / total:.2%} of episodes)" + ("  [excluded from clips]" if name in excluded else ""))
    print(f"[EVAL] wrote {n_clips} {'episode' if args_cli.record_successes else 'failure'} clips to {clips_dir}")

    with open(os.path.join(args_cli.output_dir, "results.json"), "w") as f:
        json.dump({"fps": fps, "task": args_cli.task, "checkpoint": resume_path, "stochastic": args_cli.stochastic,
                   "num_envs": N, "episodes_per_env": E, "tally": tally, "recovered_abnormal": n_recovered,
                   "abnormal_from_state": args_cli.abnormal_from_state, "excluded_outcomes": sorted(excluded),
                   "episodes": episodes}, f, indent=2)
    env.close()

    # ---- sequential composition: all recorded clips back to back (one file per speedup) ----
    if n_clips > 0 and args_cli.sequential_speedups:
        import subprocess
        composer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compose_failure_videos.py")
        cmd = [sys.executable, composer, args_cli.output_dir, "--speedups", *map(str, args_cli.sequential_speedups)]
        print("[INFO] composing sequential video:", " ".join(cmd))
        subprocess.run(cmd, check=False)

    # ---- montage: grid of the recorded failure clips, padded (frozen, dimmed, tagged) to the longest ----
    cols = args_cli.montage_cols
    recorded = [r for r in episodes if r["clip"]]
    if cols <= 0 or not recorded:
        return
    W = int(args_cli.width * args_cli.montage_scale)
    H = int(args_cli.height * args_cli.montage_scale)
    clips = [[np.asarray(x) for x in imageio.mimread(r["clip"], memtest=False)] for r in recorded]
    L = max(len(c) for c in clips)
    M = len(recorded)
    rows = (M + cols - 1) // cols
    out_path = os.path.join(args_cli.output_dir, "failures_montage.mp4")
    out_path_4x = os.path.join(args_cli.output_dir, "failures_montage_4x.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=9, macro_block_size=1)
    writer4x = imageio.get_writer(out_path_4x, fps=fps * 4, codec="libx264", quality=9, macro_block_size=1)
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    for t in range(L):
        panels = []
        for r, fr in zip(recorded, clips):
            live = t < len(fr)
            frame = fr[t] if live else fr[-1]
            frame = cv2.resize(frame, (W, H)) if frame.shape[:2] != (H, W) else frame
            panels.append(_draw_label(frame, f"env{r['env_id']} ep{r['episode']} {r['outcome']}", None if live else f"done: {r['outcome']}", dim=not live))
        panels += [blank] * (rows * cols - M)
        grid = np.concatenate([np.concatenate(panels[k * cols:(k + 1) * cols], axis=1) for k in range(rows)], axis=0)
        writer.append_data(grid)
        writer4x.append_data(grid)
    writer.close()
    writer4x.close()
    print(f"[INFO] wrote {out_path} and {out_path_4x}")


if __name__ == "__main__":
    main()
    simulation_app.close()
