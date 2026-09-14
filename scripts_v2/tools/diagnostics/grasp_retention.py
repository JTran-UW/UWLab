# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does an already-established grasp hold in IsaacLab 3.0?

Against the correct 2.x reference (wandb mh4o92uy, 64k envs) the two versions are
identical through iteration 150. From ~250 onward 2.x reaches end-of-episode
success 0.93; 3.0 flatlines at 0.45 with task_0 pinned at exactly 0.000 and task_1
capped at ~0.14, while task_2 (0.87) and task_3 (0.78) train normally.

task_1 and task_2 BOTH start with the peg grasped. The only difference is that
task_1's peg rests on the table. So this tracks the peg relative to the finger
midpoint under a closed-gripper, zero-arm-delta command: if the grasp slips for
the resting case but holds for the airborne one, the failure is grasp retention
under contact load rather than anything in the RL.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=40)
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

TYPES = ["ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"]

print("\n" + "=" * 92)
print("GRASP RETENTION -- gripper commanded CLOSED, zero arm delta")
print("=" * 92)
print(f"  {'reset type':>36} {'|peg-fingers| t=0':>18} {'t=end':>10} {'drift':>9} {'held<1cm':>10}")

for rt in TYPES:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    cfg.events.reset_from_reset_states.params["reset_types"] = [rt]
    cfg.events.reset_from_reset_states.params["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene.articulations["robot"]
    peg = env.scene.rigid_objects["insertive_object"]
    pads = [i for i, b in enumerate(robot.body_names) if "inner_finger" in b and "knuckle" not in b]

    def rel():
        c = robot.data.body_pos_w.torch[:, pads].mean(dim=1)
        return (peg.data.root_pos_w.torch - c).norm(dim=-1)

    r0 = rel().clone()
    act = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
    act[:, -1] = 1.0  # close (BinaryJointPositionAction: >0 -> close_command_expr = 0.785)
    for _ in range(args_cli.steps):
        env.step(act)
    r1 = rel()
    drift = (r1 - r0).abs()
    print(f"  {rt:>36} {r0.mean().item():>18.4f} {r1.mean().item():>10.4f} "
          f"{drift.mean().item():>9.4f} {(drift < 0.01).float().mean().item()*100:>9.1f}%")
    env.close()
    del env

print("-" * 92)
print("  task_1 = ObjectRestingEEGrasped (caps at 0.14)   task_2 = ObjectAnywhereEEGrasped (reaches 0.87)")
print("  Both start grasped; task_1's peg additionally rests on the table.")
print("=" * 92 + "\n", flush=True)
app.close()
