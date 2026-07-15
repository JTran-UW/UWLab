# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect (sim2real point cloud -> expert action) demonstrations.

Rolls out the JIT state/ScenePC expert (``teachers/patrick_jit_expert.pt``) in the
data-collection env (eval-after-Stage-2 + calibrated/DR sim2real point cloud) and
records, per step, every term of the ``data_collect`` observation group paired with
the expert's action. Successful episodes are written to an HDF5 file (robomimic-style
layout).

The env exposes:
  * ``teacher``      -- the expert's input (proprio 25 + clean ScenePointCloud 512 = 1561d)
  * ``data_collect`` -- the student observation group we record. With
    ``concatenate_terms=False`` it is a dict of per-term tensors (e.g. the
    sim2real-augmented ``scene_pc`` cloud, ``joint_pos``, ``end_effector_pose``).
    Each term is saved as its own dataset under ``obs/`` -- whatever terms the
    cfg declares are recorded, no script change needed.

Usage::

    ./_isaaclab/IsaacLab/isaaclab.sh -p scripts/tools/sim2real/collect_pc_demos.py \\
        --num_envs 64 --num_demos 200 --out demos/pc_demos.hdf5 --headless

Single task chosen via the usual hydra overrides, e.g. peg (default) or leg:
    env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect sim2real PC <-> expert-action demos.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-v0")
parser.add_argument("--expert", type=str, default="teachers/patrick_jit_expert.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_demos", type=int, default=200, help="Number of demos to save, then stop.")
parser.add_argument("--out", type=str, default="demos/pc_demos.hdf5")
parser.add_argument("--obs_group", type=str, default="data_collect",
                    help="Observation group whose terms are recorded (one dataset per term).")
parser.add_argument("--keep_failures", action="store_true", help="Also save unsuccessful episodes.")
parser.add_argument("--log_every", type=int, default=20, help="Log expert success stats every N finished episodes.")
parser.add_argument("--max_iters", type=int, default=100000, help="Safety cap on env steps.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--compress", action="store_true", help="gzip-compress HDF5 datasets.")
parser.add_argument("--no_save", action="store_true",
                    help="Benchmark mode: run the full collection loop but skip writing HDF5 (no disk).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os
import sys

import gymnasium as gym
import h5py
import numpy as np
import torch
from tqdm import tqdm

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

# PC observation signature builder (lives with the BC training code).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "imitation_learning", "point_cloud"))
from pc_signature import signature_from_obs_group  # noqa: E402

# Terms whose flat output is a point cloud; reshaped to (T, num_points, 3) on save.
# Everything else is stored as its natural (T, feature_dim) per-step vector.
_PC_TERMS = {"scene_pc"}


def main():
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # PC viz off for collection.
    group_cfg = getattr(env_cfg.observations, args_cli.obs_group)
    if hasattr(group_cfg, "scene_pc") and "visualize" in group_cfg.scene_pc.params:
        group_cfg.scene_pc.params["visualize"] = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    device = env.unwrapped.device
    num_envs = args_cli.num_envs

    expert = torch.jit.load(args_cli.expert, map_location=device).eval()
    for p in expert.parameters():
        p.requires_grad_(False)
    print(f"[collect] loaded expert: {args_cli.expert}")

    # per-env episode-final success (survives reset; only the counter is cleared)
    success_ref = env.unwrapped.reward_manager.get_term_cfg("progress_context").func.success

    # -- Trivial-episode filter --
    # Episodes that RESET already at the goal (e.g. the ObjectPartiallyAssembledEEGrasped path)
    # terminate as soon as the success term's consecutive-success streak fills, yielding
    # ~(num_consecutive_successes + 1)-step "demos" with no task content (~14% of clean_seg_100k
    # was a length-3 spike). Read the streak length off the env's success termination and skip
    # any demo at or under that length. Falls back to 0 (no filtering) if the term/param is absent.
    min_demo_len = 0
    try:
        succ_cfg = env.unwrapped.termination_manager.get_term_cfg("success")
        min_demo_len = int(succ_cfg.params.get("num_consecutive_successes", 0)) + 1
    except (KeyError, ValueError, AttributeError):
        pass
    print(f"[collect] trivial-episode filter: dropping demos with length <= {min_demo_len} "
          f"(from success term num_consecutive_successes)")

    pbar = tqdm(total=args_cli.num_demos, desc="demos", unit="demo", smoothing=0.1)
    obs, _ = env.reset()

    # Discover the recorded terms from the live obs (concatenate_terms=False -> dict).
    group_obs = obs[args_cli.obs_group]
    if not isinstance(group_obs, dict):
        raise RuntimeError(
            f"obs group '{args_cli.obs_group}' is not a per-term dict; set concatenate_terms=False "
            f"on its ObsGroup cfg so each term is recorded separately."
        )
    obs_keys = list(group_obs.keys())
    print(f"[collect] recording obs group '{args_cli.obs_group}' terms: {obs_keys}")

    # -- Per-prim labeling discovery --
    # A PER-PRIM PC obs term (OccludedScenePointCloud with per_prim=True) exposes ``prim_names``
    # (the id->name table) on its instance and appends a per-point prim id as the FINAL channel
    # of scene_pc. When present, we peel that last channel into a separate ``scene_pc_prim_id``
    # dataset and record the name table, so training can select prims + zero-pad. Absent ->
    # legacy flat scene_pc (no prim id).
    pc_term_name = next((k for k in obs_keys if k in _PC_TERMS), None)
    prim_names = None
    if pc_term_name is not None:
        om = env.unwrapped.observation_manager
        for name, tcfg in zip(
            om._group_obs_term_names[args_cli.obs_group], om._group_obs_term_cfgs[args_cli.obs_group]
        ):
            if name == pc_term_name:
                prim_names = getattr(tcfg.func, "prim_names", None)
                break
    per_prim = prim_names is not None

    # Point-cloud channel count: 3 (xyz) or 4 (xyz + per-point segmentation label). Inferred
    # from the live flat dim / num_points. For per-prim the live term has ONE extra trailing
    # channel (the prim id), which we peel off -> base pc_point_dim is one less.
    pc_point_dim = 3
    pc_num_points = group_cfg.scene_pc.params.get("num_points") if hasattr(group_cfg, "scene_pc") else None
    if pc_num_points:
        for k in obs_keys:
            if k in _PC_TERMS:
                total_ch = int(group_obs[k].shape[1] // pc_num_points)
                pc_point_dim = total_ch - 1 if per_prim else total_ch
                break
    if per_prim:
        print(f"[collect] PER-PRIM cloud: {pc_num_points} points, {pc_point_dim} ch + prim-id channel "
              f"-> obs/{pc_term_name} (xyz) + obs/{pc_term_name}_prim_id (int)")
        print(f"[collect]   prims ({len(prim_names)}): {prim_names}")
    else:
        print(f"[collect] point-cloud channels: {pc_point_dim} (num_points={pc_num_points})")

    # -- Expert std discovery (for DEXTRAH inverse-variance BC weighting) --
    # If the expert exposes its action-distribution std we record it per step, so the BC trainer can
    # weight supervision by (1/sigma)^2. Two export styles are supported: a JIT whose forward returns
    # (mean, std), or one with a `compute_distribution(obs) -> (mean, std)` method (the extended
    # rsl_rl exporter). Neither present -> mean-only dataset (plain MSE downstream). Probed once here.
    has_dist_method = hasattr(expert, "compute_distribution")

    def expert_forward(teacher_obs):
        """Return (mean, std_or_None) for the expert on the teacher obs, whichever export style."""
        out = expert.compute_distribution(teacher_obs) if has_dist_method else expert(teacher_obs)
        if isinstance(out, (tuple, list)):
            return out[0], (out[1] if len(out) > 1 else None)
        return out, None

    with torch.no_grad():
        _, _probe_std = expert_forward(obs["teacher"])
    has_std = _probe_std is not None
    print(f"[collect] expert reports std: {has_std}"
          + ("" if has_std else " -- mean-only dataset; downstream inverse-variance weighting "
             "(action_var_weighting) will be UNAVAILABLE. Re-export the expert with a std head to enable it."))

    h5 = None
    data_grp = None
    if not args_cli.no_save:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
        h5 = h5py.File(args_cli.out, "w")
        data_grp = h5.create_group("data")
        data_grp.attrs["env_name"] = args_cli.task
        data_grp.attrs["obs_group"] = args_cli.obs_group
        data_grp.attrs["obs_keys"] = obs_keys
        data_grp.attrs["pc_point_dim"] = pc_point_dim
        data_grp.attrs["has_expert_std"] = has_std
        # PC observation signature (classes / per-class budget / frame / proprio layout),
        # extracted from the live obs group cfg. Travels dataset -> ckpt hparams -> JIT meta
        # so the real-eval harness can configure its perception pipeline (pc_signature.py).
        pc_sig = signature_from_obs_group(group_cfg)
        if pc_sig is not None:
            data_grp.attrs["pc_signature"] = json.dumps(pc_sig)
            print(f"[collect] pc_signature: {json.dumps(pc_sig)}")
        if per_prim:
            # Per-prim: the trainer reads the id->name table to map each point's prim id
            # (obs/<pc_term>_prim_id) to a name, then selects via pc_parts + zero-pads.
            data_grp.attrs["pc_term"] = pc_term_name
            data_grp.attrs["pc_prim_names"] = prim_names
    else:
        print("[collect] --no_save: benchmark mode, HDF5 will NOT be written.", flush=True)

    # per-env rolling buffers: one list-of-steps per recorded obs term, plus actions (and, when the
    # expert reports it, the per-step expert action std for inverse-variance BC weighting).
    buf_obs = {k: [[] for _ in range(num_envs)] for k in obs_keys}
    buf_act = [[] for _ in range(num_envs)]
    buf_std = [[] for _ in range(num_envs)]

    n_saved = 0
    n_attempted = 0
    n_success = 0  # expert eval: episodes where success was True at episode end
    n_trivial = 0  # successful episodes dropped by the trivial-episode length filter
    comp = "gzip" if args_cli.compress else None

    def save_demo(ep_obs, ep_act, success, ep_std=None):
        nonlocal n_saved
        if args_cli.no_save:  # benchmark mode: count it, write nothing
            n_saved += 1
            return
        g = data_grp.create_group(f"demo_{n_saved}")
        g.attrs["num_samples"] = len(ep_act)
        g.attrs["success"] = bool(success)
        g.create_dataset("actions", data=np.stack(ep_act), compression=comp)
        if ep_std is not None:  # per-step expert action std (raw action units) for BC var-weighting
            g.create_dataset("expert_action_std", data=np.stack(ep_std), compression=comp)
        og = g.create_group("obs")
        for k in obs_keys:
            arr = np.stack(ep_obs[k])  # (T, term_dim)
            if k in _PC_TERMS:
                if per_prim:
                    # Live term is (T, N, pc_point_dim + 1): peel the trailing prim-id channel
                    # into its own int dataset; keep the xyz(+seg) cloud under the term name.
                    arr = arr.reshape(arr.shape[0], -1, pc_point_dim + 1)
                    og.create_dataset(k, data=arr[:, :, :pc_point_dim], compression=comp)
                    og.create_dataset(f"{k}_prim_id", data=arr[:, :, pc_point_dim].astype(np.int16),
                                      compression=comp)
                    continue
                arr = arr.reshape(arr.shape[0], -1, pc_point_dim)  # (T, num_points, pc_point_dim)
            og.create_dataset(k, data=arr, compression=comp)
        n_saved += 1

    it = 0
    while n_saved < args_cli.num_demos and it < args_cli.max_iters:
        with torch.no_grad():
            action, std = expert_forward(obs["teacher"])  # (N, 7) expert mean (+ std if reported)

        # buffer the (obs, action[, std]) tuple BEFORE stepping
        obs_np = {k: obs[args_cli.obs_group][k].detach().cpu().numpy() for k in obs_keys}
        act_np = action.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy() if has_std else None
        for e in range(num_envs):
            for k in obs_keys:
                buf_obs[k][e].append(obs_np[k][e])
            buf_act[e].append(act_np[e])
            if has_std:
                buf_std[e].append(std_np[e])

        obs, _, terminated, truncated, _ = env.step(action)
        done = (terminated | truncated).detach().cpu().numpy()
        succ = success_ref.detach().cpu().numpy()  # episode-final success (pre-reset)

        saved_before = n_saved
        for e in range(num_envs):
            if done[e]:
                n_attempted += 1
                n_success += int(bool(succ[e]))  # expert eval metric
                if (succ[e] or args_cli.keep_failures) and len(buf_act[e]) > 0:
                    if len(buf_act[e]) <= min_demo_len:  # already-at-goal episode: no task content
                        n_trivial += 1
                    else:
                        save_demo({k: buf_obs[k][e] for k in obs_keys}, buf_act[e], succ[e],
                                  ep_std=(buf_std[e] if has_std else None))
                for k in obs_keys:
                    buf_obs[k][e] = []
                buf_act[e] = []
                buf_std[e] = []
                if n_saved >= args_cli.num_demos:
                    break
        if n_saved > saved_before:
            pbar.update(n_saved - saved_before)
            pbar.set_postfix(succ=f"{n_success / max(n_attempted, 1) * 100:.0f}%", step=it)
        elif it % 25 == 0:
            pbar.set_postfix(step=it, episodes=n_attempted)  # keeps elapsed time ticking pre-first-batch
            pbar.refresh()
        it += 1

    pbar.close()
    expert_sr = n_success / max(n_attempted, 1) * 100
    if not args_cli.no_save:
        data_grp.attrs["total"] = n_saved
        data_grp.attrs["expert_success_rate"] = expert_sr
        data_grp.attrs["episodes_attempted"] = n_attempted
        data_grp.attrs["min_demo_len_filter"] = min_demo_len
        data_grp.attrs["trivial_episodes_filtered"] = n_trivial
        h5.close()
        print(f"[collect] DONE: wrote {n_saved} demos to {args_cli.out}", flush=True)
    else:
        print(f"[collect] DONE (no_save): {n_saved} demos completed over {it} env steps", flush=True)
    print(f"[collect] EXPERT EVAL: success {n_success}/{n_attempted} episodes "
          f"({expert_sr:.1f}%)", flush=True)
    print(f"[collect] TRIVIAL FILTER: dropped {n_trivial} successful episodes with length <= "
          f"{min_demo_len}", flush=True)
    if expert_sr < 50.0:
        print(f"[collect] WARNING: expert success rate {expert_sr:.1f}% is low -- check the "
              f"expert / env / action scale before trusting these demos.", flush=True)


if __name__ == "__main__":
    main()
    # Isaac Sim's simulation_app.close() can hang on shutdown; the HDF5 is already
    # flushed, so exit hard for a clean, prompt return.
    simulation_app.close()
    os._exit(0)
