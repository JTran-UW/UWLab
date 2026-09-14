# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What fraction of each reset type registers success immediately after reset?

A 2.x run using the same four reset types (wandb mpwpb818, 4096 envs) reports
task_3_success_rate ~0.99 from iteration 10 onward, while every 3.0 run sits at
0.59-0.73. Task 3 is ObjectPartiallyAssembledEEGrasped -- near-success by
construction -- so that gap exists at reset, before any learning, and cannot be a
learning-speed difference. 2.x's task_0 is likewise never zero (0.28-0.55) while
3.0's is identically 0.000.

This resets from each reset type in turn and evaluates the success predicate the
task itself uses (progress_context's `success`) over the first few steps, which
isolates "the states are wrong" from "the success test is wrong" from "the policy
cannot act".
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--settle_steps", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

RESET_TYPES = [
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
]

print("\n" + "=" * 88)
print("SUCCESS-AT-RESET AUDIT (3.0)")
print("=" * 88)
print(f"  {'reset type':>36} {'success@reset':>14} {'after settle':>14} {'peg_z mean':>12}")

rows = []
for rt in RESET_TYPES:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    cfg.events.reset_from_reset_states.params["reset_types"] = [rt]
    cfg.events.reset_from_reset_states.params["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()

    ctx = env.reward_manager.get_term_cfg("progress_context").func
    peg = env.scene.rigid_objects["insertive_object"]

    s0 = ctx.success.float().mean().item()
    z0 = peg.data.root_pos_w[:, 2].mean().item()
    act = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
    act[:, -1] = 1.0  # hold the gripper closed while settling
    for _ in range(args_cli.settle_steps):
        env.step(act)
    s1 = ctx.success.float().mean().item()

    print(f"  {rt:>36} {s0*100:>13.1f}% {s1*100:>13.1f}% {z0:>12.4f}")
    rows.append((rt, s0, s1))
    env.close()
    del env

print("-" * 88)
print("  2.x reference (wandb mpwpb818, same 4 reset types, 4096 envs, iteration 10):")
print("     task_0 0.547   task_1 0.601   task_2 0.263   task_3 0.997")
print("  Those are rolling success rates over recent resets, so task_3 ~0.99 implies")
print("  PartiallyAssembled states are essentially always successful in 2.x.")
t3 = [r for r in rows if r[0] == "ObjectPartiallyAssembledEEGrasped"]
if t3 and t3[0][1] < 0.9:
    print(f"\n  => 3.0 PartiallyAssembled registers success only {t3[0][1]*100:.1f}% at reset")
    print("     (2.x behaves as ~99%). The reset states or the success predicate differ.")
print("=" * 88 + "\n", flush=True)
app.close()
