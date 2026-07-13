# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect a DAgger round: roll out a BC STUDENT, label the visited states with the EXPERT.

Classic dataset-aggregation DAgger (Ross et al.). Unlike ``collect_pc_demos.py`` (which rolls out
the expert), here the **student drives the env** so we visit the states the student actually
reaches -- including the off-distribution ones a one-shot BC policy drifts into -- and we record
each visited ``data_collect`` observation paired with the **expert's** action as the correction
label. Aggregating this with the original demos and retraining tests whether the BC ceiling is a
state-coverage / compounding-error problem (DAgger closes it) or an action-multimodality problem
(it doesn't).

  * env steps with the STUDENT action (a BC PointNet loaded via ``bc_utils``)
  * each step records ``data_collect`` obs + the EXPERT (teacher JIT) action as the label
  * ``--beta`` optionally mixes in expert control per step (beta=0 = pure student rollout, default)
  * ALL episodes are saved by default (the OOD/failure states are the point of DAgger)

Run in the uw-lab-base container, e.g.::

    /isaac-sim/python.sh scripts/tools/sim2real/collect_dagger_pc.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-Clean-v0 \\
        --student /tmp/student.ckpt --expert /tmp/teacher_jit.pt \\
        --num_envs 1024 --num_demos 20000 --out scripts/_collect_out/dagger_round1.hdf5 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect a DAgger round (student rollout, expert labels).")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-Clean-v0")
parser.add_argument("--student", type=str, required=True, help="BC student Lightning .ckpt (drives the env).")
parser.add_argument("--expert", type=str, required=True, help="Teacher JIT (labels the visited states).")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_demos", type=int, default=20000, help="Number of episodes to save, then stop.")
parser.add_argument("--out", type=str, default="scripts/_collect_out/dagger_round.hdf5")
parser.add_argument("--obs_group", type=str, default="data_collect",
                    help="Observation group recorded (one dataset per term) AND fed to the student.")
parser.add_argument("--beta", type=float, default=0.0,
                    help="DAgger mixing: per-step prob of stepping with the EXPERT instead of the student "
                         "(0 = pure student rollout).")
parser.add_argument("--success_only", action="store_true",
                    help="Save only successful episodes (default: save ALL -- DAgger wants the OOD states).")
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--max_iters", type=int, default=100000, help="Safety cap on env steps.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--compress", action="store_true", help="gzip-compress HDF5 datasets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym
import h5py
import numpy as np
import torch
from tqdm import tqdm

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

# bc_utils (student loader + obs->action) lives in the PC imitation-learning dir.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "imitation_learning", "point_cloud"))
import bc_utils  # noqa: E402
from pc_signature import signature_from_obs_group  # noqa: E402

# Terms whose flat output is a point cloud; reshaped to (T, num_points, ch) on save.
_PC_TERMS = {"scene_pc"}


def main():
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    group_cfg = getattr(env_cfg.observations, args_cli.obs_group)
    if hasattr(group_cfg, "scene_pc") and "visualize" in group_cfg.scene_pc.params:
        group_cfg.scene_pc.params["visualize"] = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    device = env.unwrapped.device
    num_envs = args_cli.num_envs

    # Expert (teacher) JIT: labels each visited state.
    expert = torch.jit.load(args_cli.expert, map_location=device).eval()
    for p in expert.parameters():
        p.requires_grad_(False)
    # Student: a BC PointNet that drives the env (loaded via the eval helper).
    student = bc_utils.load_bc_pointnet(args_cli.student, device)
    print(f"[dagger] expert (labels): {args_cli.expert}")
    print(f"[dagger] student (drives, beta={args_cli.beta}): {args_cli.student} "
          f"arch={student['hp'].get('architecture')} proprio_dim={student['proprio_dim']} point_dim={student['point_dim']}")

    success_ref = env.unwrapped.reward_manager.get_term_cfg("progress_context").func.success

    pbar = tqdm(total=args_cli.num_demos, desc="dagger demos", unit="demo", smoothing=0.1)
    obs, _ = env.reset()

    group_obs = obs[args_cli.obs_group]
    if not isinstance(group_obs, dict):
        raise RuntimeError(f"obs group '{args_cli.obs_group}' must be a per-term dict (concatenate_terms=False).")
    obs_keys = list(group_obs.keys())
    print(f"[dagger] recording obs group '{args_cli.obs_group}' terms: {obs_keys}")

    # Per-prim labeling discovery (mirrors collect_pc_demos.py); clean env -> not per-prim.
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

    pc_point_dim = 3
    pc_num_points = group_cfg.scene_pc.params.get("num_points") if hasattr(group_cfg, "scene_pc") else None
    if pc_num_points:
        for k in obs_keys:
            if k in _PC_TERMS:
                total_ch = int(group_obs[k].shape[1] // pc_num_points)
                pc_point_dim = total_ch - 1 if per_prim else total_ch
                break
    print(f"[dagger] point-cloud channels: {pc_point_dim} (num_points={pc_num_points}, per_prim={per_prim})")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
    h5 = h5py.File(args_cli.out, "w")
    data_grp = h5.create_group("data")
    data_grp.attrs["env_name"] = args_cli.task
    data_grp.attrs["obs_group"] = args_cli.obs_group
    data_grp.attrs["obs_keys"] = obs_keys
    data_grp.attrs["pc_point_dim"] = pc_point_dim
    data_grp.attrs["dagger"] = True
    data_grp.attrs["student_ckpt"] = args_cli.student
    data_grp.attrs["expert_jit"] = args_cli.expert
    data_grp.attrs["beta"] = float(args_cli.beta)
    # PC observation signature (classes / budgets / frame / proprio) -> ckpt -> JIT meta.
    pc_sig = signature_from_obs_group(group_cfg)
    if pc_sig is not None:
        import json
        data_grp.attrs["pc_signature"] = json.dumps(pc_sig)
        print(f"[dagger] pc_signature: {json.dumps(pc_sig)}")
    if per_prim:
        data_grp.attrs["pc_term"] = pc_term_name
        data_grp.attrs["pc_prim_names"] = prim_names

    buf_obs = {k: [[] for _ in range(num_envs)] for k in obs_keys}
    buf_act = [[] for _ in range(num_envs)]

    n_saved = 0
    n_attempted = 0
    n_success = 0  # STUDENT success (how the student does on its own rollout)
    comp = "gzip" if args_cli.compress else None

    def save_demo(ep_obs, ep_act, success):
        nonlocal n_saved
        g = data_grp.create_group(f"demo_{n_saved}")
        g.attrs["num_samples"] = len(ep_act)
        g.attrs["success"] = bool(success)
        g.create_dataset("actions", data=np.stack(ep_act), compression=comp)
        og = g.create_group("obs")
        for k in obs_keys:
            arr = np.stack(ep_obs[k])
            if k in _PC_TERMS:
                if per_prim:
                    arr = arr.reshape(arr.shape[0], -1, pc_point_dim + 1)
                    og.create_dataset(k, data=arr[:, :, :pc_point_dim], compression=comp)
                    og.create_dataset(f"{k}_prim_id", data=arr[:, :, pc_point_dim].astype(np.int16), compression=comp)
                    continue
                arr = arr.reshape(arr.shape[0], -1, pc_point_dim)
            og.create_dataset(k, data=arr, compression=comp)
        n_saved += 1

    it = 0
    while n_saved < args_cli.num_demos and it < args_cli.max_iters:
        with torch.no_grad():
            t_out = expert(obs["teacher"])
            teacher_action = t_out[0] if isinstance(t_out, (tuple, list)) else t_out   # (N, 7) LABEL
            student_action = bc_utils.bc_actions(student, obs, args_cli.obs_group)       # (N, 7) CONTROL

        # Record (visited obs, EXPERT label) BEFORE stepping.
        obs_np = {k: obs[args_cli.obs_group][k].detach().cpu().numpy() for k in obs_keys}
        lbl_np = teacher_action.detach().cpu().numpy()
        for e in range(num_envs):
            for k in obs_keys:
                buf_obs[k][e].append(obs_np[k][e])
            buf_act[e].append(lbl_np[e])

        # Step with the student (DAgger beta: per-step chance of expert control instead).
        step_action = student_action
        if args_cli.beta > 0.0:
            use_expert = (torch.rand(num_envs, 1, device=device) < args_cli.beta).float()
            step_action = use_expert * teacher_action + (1.0 - use_expert) * student_action

        obs, _, terminated, truncated, _ = env.step(step_action)
        # Clear the student's rolling (state, action) history for terminated envs (no-op unless the
        # student is a history-conditioned BC PointNet) so a new episode never attends to the old one.
        bc_utils.bc_reset(student, terminated | truncated)
        done = (terminated | truncated).detach().cpu().numpy()
        succ = success_ref.detach().cpu().numpy()

        saved_before = n_saved
        for e in range(num_envs):
            if done[e]:
                n_attempted += 1
                n_success += int(bool(succ[e]))
                if (succ[e] or not args_cli.success_only) and len(buf_act[e]) > 0:
                    save_demo({k: buf_obs[k][e] for k in obs_keys}, buf_act[e], succ[e])
                for k in obs_keys:
                    buf_obs[k][e] = []
                buf_act[e] = []
                if n_saved >= args_cli.num_demos:
                    break
        if n_saved > saved_before:
            pbar.update(n_saved - saved_before)
            pbar.set_postfix(student_sr=f"{n_success / max(n_attempted, 1) * 100:.0f}%", step=it)
        elif it % 25 == 0:
            pbar.set_postfix(step=it, episodes=n_attempted)
            pbar.refresh()
        it += 1

    pbar.close()
    student_sr = n_success / max(n_attempted, 1) * 100
    data_grp.attrs["total"] = n_saved
    data_grp.attrs["student_success_rate"] = student_sr
    data_grp.attrs["episodes_attempted"] = n_attempted
    # Reuse the expert_success_rate key the trainer prints; here it is the STUDENT's rollout SR.
    data_grp.attrs["expert_success_rate"] = student_sr
    h5.close()
    print(f"[dagger] DONE: wrote {n_saved} demos to {args_cli.out}", flush=True)
    print(f"[dagger] STUDENT ROLLOUT SR: {n_success}/{n_attempted} ({student_sr:.1f}%) "
          f"-- labels are the EXPERT's actions at these states.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
