# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Film reset-state data collection: the same actions as record_reset_states.py, with each attempt's outcome.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/record_collection.py --visualizer none --backend newton \\
        --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --out_dir clips/col_newton
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Film reset-state data collection.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--frame_stride", type=int, default=6, help="Keep every Nth physics substep.")
parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480])
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.backend == "newton" and not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import numpy as np
import os
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors import TiledCameraCfg

from uwlab_tasks.utils.hydra import hydra_task_compose


def _wrap_success(env, last_fail):
    """Record which check_reset_state_success sub-checks fail at each env step."""
    term_cfg = env.termination_manager.get_term_cfg("success")
    term = term_cfg.func
    original = term.__call__

    def call(env_, **kwargs):
        out = original(env_, **kwargs)
        robot = term.robot_asset
        fails = {"unstable": term.stability_counter < term.consecutive_stability_steps}
        ee_quat = robot.data.body_link_quat_w.torch[:, term.ee_body_idx]
        approach = torch.tensor(term.gripper_approach_direction, device=env.device, dtype=torch.float32).expand(
            env.num_envs, -1
        )
        fails["gripper not pointing down"] = math_utils.quat_apply(ee_quat, approach)[:, 2] >= -0.5
        for asset in term.assets_to_check:
            is_robot = asset is robot
            pos = asset.data.body_link_pos_w.torch[:, term.ee_body_idx] if is_robot else asset.data.root_pos_w.torch
            limit = term.max_robot_pos_deviation if is_robot else term.max_object_pos_deviation
            name = "ee" if is_robot else ("peg" if asset is term.object_assets[0] else "hole")
            fails[f"{name} moved > {limit * 1000:.0f} mm"] = (pos - asset.initial_pos).norm(dim=1) > limit
            fails[f"{name} below z"] = pos[:, 2] < term.pos_z_threshold
        ids = torch.arange(env.num_envs, device=env.device)
        for label, analyzer in zip(
            ("robot-peg collision", "robot-hole collision", "peg-hole collision"), term.collision_analyzers
        ):
            fails[label] = ~analyzer(env_, ids)
        if term.assembly_success_prob is not None:
            ins_pos, ins_quat = term.insertive_asset_offset.apply(term.insertive_asset)
            rec_pos, rec_quat = term.receptive_asset_offset.apply(term.receptive_asset)
            rel_pos, rel_quat = math_utils.subtract_frame_transforms(rec_pos, rec_quat, ins_pos, ins_quat)
            ex, ey, _ = math_utils.euler_xyz_from_quat(rel_quat)
            assembled = (rel_pos.norm(dim=1) < term.assembly_pos_threshold) & (
                math_utils.wrap_to_pi(ex).abs() + math_utils.wrap_to_pi(ey).abs() < term.assembly_ori_threshold
            )
            fails["wanted assembled, is not"] = term.require_assembly_success & ~assembled
            fails["wanted NOT assembled, is"] = ~term.require_assembly_success & assembled
        stacked = {k: v.cpu() for k, v in fails.items()}
        for e in range(env.num_envs):
            last_fail[e] = [k for k, v in stacked.items() if bool(v[e])]
        return out

    call.reset = term.reset
    term_cfg.func = call


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    width, height = args_cli.resolution
    env_cfg.scene.viz_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/viz_camera",
        update_period=0,
        width=width,
        height=height,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.46613815, 0.25072644, 0.47107948, 0.70564552),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    camera = env.scene["viz_camera"]
    n = env.num_envs
    torch.manual_seed(args_cli.seed)
    env.reset(seed=args_cli.seed)

    grasped_task = "ObjectAnywhereEEGrasped" in args_cli.task or "ObjectRestingEEGrasped" in args_cli.task
    actions = torch.zeros(env.action_space.shape, device=env.device)
    actions[:, -1] = -1.0 if grasped_task else torch.randint(0, 2, (n,), device=env.device).float() * 2 - 1

    frames, attempt_of_frame = [], []
    attempt = torch.zeros(n, dtype=torch.long)
    outcomes = [[] for _ in range(n)]
    counter = {"substep": 0}
    sim_step = env.sim.step

    def grab():
        env.sim.forward()
        env.sim.render()
        camera.update(env.physics_dt, force_recompute=True)
        rgb = camera.data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        frames.append(rgb[..., :3].detach().cpu().numpy().astype(np.uint8))
        attempt_of_frame.append(attempt.clone())

    def step(render=True):
        sim_step(render=False)
        counter["substep"] += 1
        if counter["substep"] % args_cli.frame_stride == 0:
            grab()

    env.sim.step = step
    terms = env.termination_manager.active_terms
    last_fail = [[] for _ in range(n)]
    if "success" in terms:
        _wrap_success(env, last_fail)
    steps = int(round(args_cli.duration / env.step_dt))
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(actions)
        done = (terminated | truncated).cpu()
        if done.any():
            fired = {t: env.termination_manager.get_term(t).cpu() for t in terms}
            for e in done.nonzero().flatten().tolist():
                reason = [t for t in terms if bool(fired[t][e])]
                if "success" in reason:
                    outcomes[e].append("KEPT")
                elif reason == ["time_out"] and last_fail[e]:
                    outcomes[e].append("REJECTED (" + ", ".join(last_fail[e]) + ")")
                else:
                    outcomes[e].append("REJECTED (" + ",".join(reason) + ")")
                attempt[e] += 1
            if not grasped_task:
                idx = done.nonzero().flatten().to(env.device)
                actions[idx, -1] = torch.randint(0, 2, (len(idx),), device=env.device).float() * 2 - 1
    os.makedirs(args_cli.out_dir, exist_ok=True)
    video = np.stack(frames)
    attempts = torch.stack(attempt_of_frame)
    for e in range(n):
        np.savez_compressed(os.path.join(args_cli.out_dir, f"env{e}.npz"), frames=video[:, e])
    summary = {
        "task": args_cli.task,
        "backend": args_cli.backend,
        "fps_sim": 1.0 / (env.physics_dt * args_cli.frame_stride),
        "attempt_of_frame": attempts.tolist(),
        "outcomes": outcomes,
    }
    with open(os.path.join(args_cli.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f)
    for e in range(n):
        kept = sum(o == "KEPT" for o in outcomes[e])
        print(
            f"[record_collection] {args_cli.backend} env {e}: {len(outcomes[e])} attempts, {kept} kept: {outcomes[e]}"
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
