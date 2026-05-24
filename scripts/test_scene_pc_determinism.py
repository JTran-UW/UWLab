"""Probe ScenePointCloud determinism across GPU0 / GPU1 / CPU for peg+peghole.

Builds the gravity env (ScenePC FullDR teacher cfg) on the requested device,
runs through __init__ (which samples the canonical 512 points), prints the
points_hash diagnostic + a bytes-level SHA256 of the full selected_local
tensor.

Usage:
    PYTHONPATH=... python scripts/test_scene_pc_determinism.py --sim_device cuda:0
    PYTHONPATH=... python scripts/test_scene_pc_determinism.py --sim_device cuda:1
    PYTHONPATH=... python scripts/test_scene_pc_determinism.py --sim_device cpu

Compare the printed hashes across runs.
"""

from __future__ import annotations

import argparse
import hashlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--sim_device", default="cuda:0", help="cuda:0 / cuda:1 / cpu")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument(
    "--task",
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-FullDR-v0",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
# Force device override BEFORE AppLauncher launches (it reads args.device)
args.device = args.sim_device
app_launcher = AppLauncher(args)
app = app_launcher.app

# ---- post-launch imports ----------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uwlab_tasks  # noqa: F401,E402  (registers tasks)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main():
    # Build cfg with peg variant pinned
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.sim.device = args.sim_device
    if hasattr(env_cfg, "variants"):
        v = env_cfg.variants
        if "scene.insertive_object" in v and "peg" in v["scene.insertive_object"]:
            env_cfg.scene.insertive_object = v["scene.insertive_object"]["peg"]
        if "scene.receptive_object" in v and "peghole" in v["scene.receptive_object"]:
            env_cfg.scene.receptive_object = v["scene.receptive_object"]["peghole"]

    print(f"[probe] building env on {args.sim_device} (num_envs={args.num_envs})")
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # Find the ScenePointCloud term instance
    om = env.observation_manager
    scene_pc_term = None
    for group_name, term_names in om.active_terms.items():
        for term_name in term_names:
            # term_name is just the field name (e.g. "scene_pc")
            term_cfg_list = om._group_obs_term_cfgs[group_name]
            for term_cfg in term_cfg_list:
                func = term_cfg.func
                if hasattr(func, "__class__") and "ScenePointCloud" in func.__class__.__name__:
                    scene_pc_term = func
                    print(f"[probe] found ScenePointCloud in group={group_name}")
                    break

    if scene_pc_term is None:
        # Fallback: search term func mapping
        for group_name in om._group_obs_term_cfgs:
            for term_cfg in om._group_obs_term_cfgs[group_name]:
                if "ScenePointCloud" in str(type(term_cfg.func)):
                    scene_pc_term = term_cfg.func
                    print(f"[probe] found ScenePointCloud via fallback in group={group_name}")
                    break

    if scene_pc_term is None:
        print("[probe] ERROR: ScenePointCloud term not found in observation manager")
        # Dump what we have
        for group_name, term_names in om.active_terms.items():
            print(f"  group={group_name}: {term_names}")
        return

    sel = scene_pc_term.selected_local_task  # (T, P, 3)
    print(f"[probe] selected_local_task shape={tuple(sel.shape)} dtype={sel.dtype} device={sel.device}")

    # Move to CPU for hashing so we get bit-identical bytes across devices when
    # the tensor values are identical
    sel_cpu = sel.detach().to("cpu").contiguous()
    raw = sel_cpu.float().numpy().tobytes()
    h = hashlib.sha256(raw).hexdigest()
    print(f"[RESULT] device={args.sim_device}  shape={tuple(sel.shape)}  sha256={h}")

    # Per-source slices for diagnostics
    src = scene_pc_term.selected_source_type_task[0]
    n_robot = int((src == scene_pc_term._SRC_ROBOT).sum().item())
    n_ins = int((src == scene_pc_term._SRC_INSERTIVE).sum().item())
    n_rec = int((src == scene_pc_term._SRC_RECEPTIVE).sum().item())
    print(f"[RESULT] device={args.sim_device}  counts: robot={n_robot} ins={n_ins} rec={n_rec}")

    # First 5 points (a quick visual sanity diff if hashes differ)
    print(f"[RESULT] device={args.sim_device}  first5={sel_cpu[0, :5].tolist()}")


if __name__ == "__main__":
    main()
    app.close()
