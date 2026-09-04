# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Side-by-side peg-mass-gap video: the same policy, the same start states, one panel per peg mass.

Pass 0 ("no gap", training DR) samples N start states through the task's reset manager and records
the full scene state of each. Every mass condition then replays those exact states via
``env.reset_to`` and pins the peg mass, so all panels start every episode from identical s_k.
Episodes end on the task's own terminations (time_out / abnormal_robot; no success termination).
Composition pads each panel to the longest episode for that s_k (frozen, dimmed last frame with an
outcome tag) so every panel advances to s_{k+1} at the same time.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Peg-mass-gap side-by-side eval video.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--sweep", type=str, default="peg_mass", choices=["peg_mass", "peg_noise", "peg_bias", "friction"])
parser.add_argument("--masses", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
parser.add_argument("--noise_stds", type=float, nargs="+", default=[0.001, 0.01])
parser.add_argument("--biases", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
parser.add_argument("--frictions", type=str, nargs="+", default=["2.2/0.7/1.3", "2.5/0.8/1.5", "3.0/1.0/1.8", "3.5/1.2/2.2"],
                    help="peg/socket/robot friction triples")
parser.add_argument("--num_start_states", type=int, default=10)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--panel_width", type=int, default=640)
parser.add_argument("--panel_height", type=int, default=480)
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
from isaaclab.envs.mdp.events import randomize_rigid_body_mass, randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from vecenv_wrapper import HolosomaVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def _to_cpu(state):
    if isinstance(state, dict):
        return {k: _to_cpu(v) for k, v in state.items()}
    return state.detach().clone().cpu()


def _to_dev(state, device):
    if isinstance(state, dict):
        return {k: _to_dev(v, device) for k, v in state.items()}
    return state.to(device)


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
    cv2.putText(f, top, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
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
        cv2.putText(f, bottom, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 80) if "success" in bottom else (80, 80, 255), 2, cv2.LINE_AA)
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

    os.makedirs(args_cli.output_dir, exist_ok=True)
    clips_dir = os.path.join(args_cli.output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    try:
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import describe_assembly_assets

        print(describe_assembly_assets(env_cfg))
    except Exception as exc:  # noqa: BLE001
        print(f"[assembly] could not describe peg/hole assets: {exc}")
    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped
    device = uenv.device
    fps = int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)))

    runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.setup()
    runner.load(resume_path)
    actor_obs_keys = agent_cfg.actor_obs_keys
    if args_cli.stochastic:
        actor = runner.actor.to(device).eval()
        obs_norm = runner.obs_normalizer.to(device).eval()

        def policy(obs):
            x = obs["actor_obs"]
            nx = obs_norm(x, update=False) if runner.obs_normalization else x
            return actor.explore(nx, deterministic=False)
    else:
        policy = runner.get_inference_policy(device=device)
    print(f"[INFO] policy: {'stochastic' if args_cli.stochastic else 'deterministic'}, fps={fps}")

    insertive = uenv.scene["insertive_object"]
    context_term = uenv.reward_manager.get_term_cfg("progress_context").func
    term_mgr = uenv.termination_manager

    def peg_mass() -> float:
        return float(insertive.root_physx_view.get_masses().flatten()[0])

    def make_mass_pin(mass: float):
        params = {
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "mass_distribution_params": (mass, mass),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        }
        term = randomize_rigid_body_mass(EventTermCfg(func=randomize_rigid_body_mass, mode="reset", params=params), uenv)
        return lambda: term(uenv, None, **params)

    def make_friction_pin(values: dict[str, float]):
        terms = []
        for asset, v in values.items():
            params = {
                "asset_cfg": SceneEntityCfg(asset),
                "static_friction_range": (v, v),
                "dynamic_friction_range": (v, v),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "make_consistent": True,
            }
            t = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, mode="reset", params=params), uenv)
            terms.append((t, params))
        def _apply():
            for t, pr in terms:
                t(uenv, None, **pr)
            for asset in values:
                mat = uenv.scene[asset].root_physx_view.get_material_properties()
                print(f"[friction readback] {asset}: static {mat[..., 0].min():.3f}-{mat[..., 0].max():.3f}"
                      f" dynamic {mat[..., 1].min():.3f}-{mat[..., 1].max():.3f}")
        return _apply

    def run_episode(obs: TensorDict):
        frames = [uenv.render()]
        ever_success = False
        first_success = None
        outcome = "timeout"
        step = 0
        while True:
            with torch.inference_mode():
                actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
                a = policy({"actor_obs": actor_obs})
            obs, _, dones, _ = env.step(a.clone())
            step += 1
            if bool((context_term.orientation_aligned & context_term.position_aligned)[0]) and not ever_success:
                ever_success, first_success = True, step
            if bool(dones[0]):
                if bool(term_mgr.get_term("abnormal_robot")[0]):
                    outcome = "abnormal"
                break
            frames.append(uenv.render())
        return frames, {"steps": step, "success": ever_success, "first_success_step": first_success, "outcome": outcome}

    om = uenv.observation_manager
    peg_terms = [
        om._group_obs_term_cfgs["policy"][om._group_obs_term_names["policy"].index(n)]
        for n in ("insertive_asset_pose", "insertive_asset_in_receptive_asset_frame")
    ]

    def set_obs_gap(noise_std: float = 0.0, bias: float = 0.0):
        for t in peg_terms:
            t.params["pos_noise_std"] = float(noise_std)
            t.params["pos_bias"] = [float(bias)] * 3

    # each condition: (label, pre_reset_fn, post_reset_fn)
    noop = lambda: None  # noqa: E731
    sweep = args_cli.sweep
    if sweep == "peg_mass":
        conditions = [("no gap (peg mass DR)", set_obs_gap, noop)] + [
            (f"peg mass {m:g} kg", set_obs_gap, make_mass_pin(m)) for m in args_cli.masses
        ]
    elif sweep == "peg_noise":
        conditions = [("no gap (clean obs)", set_obs_gap, noop)] + [
            (f"peg obs noise std {v:g} m", (lambda v=v: set_obs_gap(noise_std=v)), noop) for v in args_cli.noise_stds
        ]
    elif sweep == "peg_bias":
        conditions = [("no gap (clean obs)", set_obs_gap, noop)] + [
            (f"peg obs bias {b:g} m", (lambda b=b: set_obs_gap(bias=b)), noop) for b in args_cli.biases
        ]
    else:
        conditions = [("no gap (friction DR)", set_obs_gap, noop)]
        for trip in args_cli.frictions:
            pf, sf, rf = (float(x) for x in trip.split("/"))
            conditions.append((
                f"friction peg {pf:g} / socket {sf:g} / robot {rf:g}",
                set_obs_gap,
                make_friction_pin({"insertive_object": pf, "receptive_object": sf, "robot": rf}),
            ))
    print("[INFO] conditions:", [c[0] for c in conditions])

    N = args_cli.num_start_states
    start_states = []
    results = {}

    for ci, (label, pre_reset, post_reset) in enumerate(conditions):
        slug = _slug(label)
        os.makedirs(os.path.join(clips_dir, slug), exist_ok=True)
        results[label] = []
        for k in range(N):
            pre_reset()
            if ci == 0:
                obs, _ = env.reset()
                start_states.append(_to_cpu(uenv.scene.get_state(is_relative=True)))
                if k == 0:
                    hole = uenv.scene["receptive_object"].data.root_pos_w[0].cpu().numpy()
                    lookat = hole + np.array([0.0, 0.0, args_cli.cam_lookat_z])
                    eye = lookat + np.array(args_cli.cam_offset)
                    uenv.sim.set_camera_view(eye.tolist(), lookat.tolist())
                    print(f"[INFO] camera eye={eye.round(3).tolist()} lookat={lookat.round(3).tolist()}")
                    imageio.imwrite(os.path.join(args_cli.output_dir, "camera_preview.png"), uenv.render())
            else:
                uenv.reset_to(_to_dev(start_states[k], device), None, is_relative=True)
                post_reset()
                diff = _max_state_diff(uenv.scene.get_state(is_relative=True), start_states[k])
                assert diff < 1e-4, f"start state replay mismatch for s_{k}: max diff {diff}"
                obs = TensorDict(uenv.obs_buf, batch_size=[1])
            m_actual = peg_mass()
            frames, info = run_episode(obs)
            info.update({"start_state": k, "peg_mass": m_actual})
            clip = os.path.join(clips_dir, slug, f"s{k:02d}.mp4")
            imageio.mimwrite(clip, frames, fps=fps, codec="libx264", quality=9, macro_block_size=1)
            info["clip"] = clip
            results[label].append(info)
            print(f"[{label}] s_{k}: mass={m_actual:.3f} kg steps={info['steps']} outcome={info['outcome']} success={info['success']}")

    with open(os.path.join(args_cli.output_dir, "results.json"), "w") as f:
        json.dump({"fps": fps, "conditions": [c[0] for c in conditions], "results": results}, f, indent=2)
    env.close()

    # ---- compose ----
    W, H = args_cli.panel_width, args_cli.panel_height
    out_path = os.path.join(args_cli.output_dir, f"{sweep}_gap_side_by_side.mp4")
    out_path_4x = os.path.join(args_cli.output_dir, f"{sweep}_gap_side_by_side_4x.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=9, macro_block_size=1)
    writer4x = imageio.get_writer(out_path_4x, fps=fps * 4, codec="libx264", quality=9, macro_block_size=1)
    for k in range(N):
        per_cond = []
        for label, _, _ in conditions:
            info = results[label][k]
            fr = [np.asarray(x) for x in imageio.mimread(info["clip"], memtest=False)]
            per_cond.append((label, info, fr))
        L = max(len(fr) for _, _, fr in per_cond)
        for t in range(L):
            panels = []
            for label, info, fr in per_cond:
                live = t < len(fr)
                frame = fr[t] if live else fr[-1]
                frame = cv2.resize(frame, (W, H)) if frame.shape[:2] != (H, W) else frame
                top = f"{label}  |  s_{k}" + (f"  |  m={info['peg_mass']:.2f} kg" if sweep == "peg_mass" else "")
                if live:
                    tag = None
                else:
                    tag = "success" if info["success"] else ("abnormal" if info["outcome"] == "abnormal" else "timeout")
                    tag = f"done: {tag}"
                fs = info["first_success_step"]
                succeeded = bool(info["success"]) and fs is not None and t >= fs
                panels.append(_draw_label(frame, top, tag, dim=not live, success=succeeded))
            row = np.concatenate(panels, axis=1)
            writer.append_data(row)
            writer4x.append_data(row)
    writer.close()
    writer4x.close()
    print(f"[INFO] wrote {out_path} and {out_path_4x}")

    for label, _, _ in conditions:
        r = results[label]
        print(f"{label:26s} success {sum(x['success'] for x in r)}/{N}  abnormal {sum(x['outcome']=='abnormal' for x in r)}  mean steps {np.mean([x['steps'] for x in r]):.1f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
