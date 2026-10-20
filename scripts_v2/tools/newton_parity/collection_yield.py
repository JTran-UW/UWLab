# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless reset-state collection yield: same actions as record_reset_states.py, per-attempt outcome and failed checks.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/collection_yield.py --visualizer none --backend newton \\
        --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --out yield_newton.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure reset-state collection yield.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--rounds", type=float, default=5.0, help="Episodes per env.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mass_only", action="store_true", help="Print object masses and exit.")
parser.add_argument(
    "--trace", action="store_true", help="Store per-step peg/ee displacement and assembly error per attempt."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.backend == "newton" and not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

from uwlab_tasks.utils.hydra import hydra_task_compose


def _wrap_success(env, last_fail, last_want, trace=None):
    """Record which check_reset_state_success sub-checks fail at each env step."""
    term_cfg = env.termination_manager.get_term_cfg("success")
    term = term_cfg.func
    original = term.__call__

    def call(env_, **kwargs):
        out = original(env_, **kwargs)
        robot = term.robot_asset
        fails = {"unstable": term.stability_counter < term.consecutive_stability_steps}
        peg_ = term.object_assets[0]
        fails["now: robot sum|qd| >= 5"] = robot.data.joint_vel.torch.abs().sum(dim=1) >= 5.0
        fails["now: gripper sum|qd| >= 5"] = robot.data.joint_vel.torch[:, 6:].abs().sum(dim=1) >= 5.0
        fails["now: peg lin vel >= 0.1"] = peg_.data.body_lin_vel_w.torch.abs().sum(dim=2).sum(dim=1) >= 0.1
        fails["now: peg ang vel >= 1"] = peg_.data.body_ang_vel_w.torch.abs().sum(dim=2).sum(dim=1) >= 1.0
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
            if trace is not None:
                peg = term.object_assets[0]
                ee = robot.data.body_link_pos_w.torch[:, term.ee_body_idx]
                trace.append(
                    torch.stack(
                        [
                            (peg.data.root_pos_w.torch - peg.initial_pos).norm(dim=1),
                            (ee - robot.initial_pos).norm(dim=1),
                            rel_pos.norm(dim=1),
                            math_utils.wrap_to_pi(ex).abs() + math_utils.wrap_to_pi(ey).abs(),
                        ],
                        1,
                    )
                )
        last_fail.clear()
        last_fail.update({k: v.clone() for k, v in fails.items()})
        if term.assembly_success_prob is not None:
            last_want[:] = term.require_assembly_success
        return out

    call.reset = term.reset
    term_cfg.func = call


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    n = env.num_envs
    for key in ("insertive_object", "receptive_object"):
        obj = env.scene[key]
        m = obj.data.body_mass.torch if hasattr(obj.data.body_mass, "torch") else obj.data.body_mass
        print(f"[collection_yield] {key} mass readback (env 0..2): {m[:3].flatten().tolist()}")
    if args_cli.mass_only:
        env.close()
        return
    torch.manual_seed(args_cli.seed)
    env.reset(seed=args_cli.seed)
    grasped_task = "ObjectAnywhereEEGrasped" in args_cli.task or "ObjectRestingEEGrasped" in args_cli.task
    actions = torch.zeros(env.action_space.shape, device=env.device)
    actions[:, -1] = -1.0 if grasped_task else torch.randint(0, 2, (n,), device=env.device).float() * 2 - 1
    terms = env.termination_manager.active_terms
    last_fail, last_want = {}, torch.zeros(n, dtype=torch.bool, device=env.device)
    trace_steps = [] if args_cli.trace else None
    _wrap_success(env, last_fail, last_want, trace_steps)
    rows, fail_rows, traces = [], [], []
    hist = torch.zeros(n, int(env.max_episode_length) + 2, 4, device=env.device)
    ar, t = torch.arange(n, device=env.device), torch.zeros(n, dtype=torch.long, device=env.device)
    episode = torch.zeros(n, dtype=torch.long, device=env.device)
    steps = int(args_cli.rounds * env.max_episode_length) + 1
    for _ in range(steps):
        gripper = actions[:, -1].clone()
        _, _, terminated, truncated, _ = env.step(actions)
        t += 1
        done = terminated | truncated
        if trace_steps:
            hist[ar, t.clamp(max=hist.shape[1] - 1)] = trace_steps[-1]
            trace_steps.clear()
        if done.any():
            ids = done.nonzero().flatten()
            fired = torch.stack([env.termination_manager.get_term(name)[ids] for name in terms], 1)
            fails = torch.stack([last_fail[k][ids] for k in last_fail], 1)
            rows.append(torch.stack([ids.float(), episode[ids].float(), gripper[ids], last_want[ids].float()], 1).cpu())
            fail_rows.append(torch.cat([fired, fails], 1).cpu())
            episode[ids] += 1
            if args_cli.trace:
                traces.append(hist[ids].cpu())
                hist[ids] = 0
            t[ids] = 0
            if not grasped_task:
                actions[ids, -1] = torch.randint(0, 2, (len(ids),), device=env.device).float() * 2 - 1
    out = {
        "task": args_cli.task,
        "backend": args_cli.backend,
        "overrides": remaining_args,
        "cols": ["env", "episode", "gripper", "want_assembled"],
        "flag_cols": list(terms) + list(last_fail),
        "rows": torch.cat(rows),
        "flags": torch.cat(fail_rows),
        "trace": torch.cat(traces) if traces else None,
        "trace_cols": ["peg_disp", "ee_disp", "asm_pos_err", "asm_rot_err"],
        "step_dt": env.step_dt,
    }
    torch.save(out, args_cli.out)
    kept = out["flags"][:, list(terms).index("success")].float()
    print(f"[collection_yield] {args_cli.backend}: {len(kept)} attempts, yield {float(kept.mean()):.4f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
