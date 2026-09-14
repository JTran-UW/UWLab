# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Why does ObjectAnywhereEEGrasped still drop the peg ~60% of the time?

After the reset-state quaternion fix, ObjectResting and ObjectPartiallyAssembled
retain the peg 100% of the time, but the airborne case still drops it in ~60% of
envs (falling only to the work surface now, not through it).

This splits the envs into HELD vs DROPPED after a fixed hold and compares their
initial conditions -- how far the peg starts from the finger midpoint, how closed
the gripper is, and how the peg is oriented in the gripper frame -- so the
difference is measured rather than guessed at.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--reset_type", type=str, default="ObjectAnywhereEEGrasped")
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

cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
cfg.events.reset_from_reset_states.params["reset_types"] = [args_cli.reset_type]
cfg.events.reset_from_reset_states.params["probs"] = [1.0]
env = gym.make(args_cli.task, cfg=cfg).unwrapped
env.reset()

robot = env.scene.articulations["robot"]
peg = env.scene.rigid_objects["insertive_object"]
pads = [i for i, b in enumerate(robot.body_names) if "inner_finger" in b and "knuckle" not in b]
fj = robot.joint_names.index("finger_joint")
org = env.scene.env_origins

fm0 = robot.data.body_pos_w.torch[:, pads].mean(dim=1)
peg0 = peg.data.root_pos_w.torch
d0 = (peg0 - fm0).norm(dim=-1)                     # peg offset from finger midpoint
gap0 = (robot.data.body_pos_w.torch[:, pads[0]] - robot.data.body_pos_w.torch[:, pads[1]]).norm(dim=-1)
q0 = robot.data.joint_pos.torch[:, fj].clone()     # how closed the gripper starts
z0 = (peg0 - org)[:, 2].clone()

act = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
act[:, -1] = 1.0  # hold closed
for _ in range(args_cli.steps):
    env.step(act)

z1 = (peg.data.root_pos_w.torch - org)[:, 2]
dropped = (z0 - z1) > 0.10
held = ~dropped

print("\n" + "=" * 84)
print(f"WHAT SEPARATES HELD FROM DROPPED  ({args_cli.reset_type}, n={args_cli.num_envs})")
print("=" * 84)
print(f"  dropped: {dropped.float().mean().item()*100:.1f}%   held: {held.float().mean().item()*100:.1f}%\n")
print(f"  {'metric':>34} {'HELD':>14} {'DROPPED':>14}")


def row(name, v, fmt="{:.4f}"):
    h = fmt.format(v[held].mean().item()) if held.any() else "  -  "
    d = fmt.format(v[dropped].mean().item()) if dropped.any() else "  -  "
    print(f"  {name:>34} {h:>14} {d:>14}")


row("|peg - fingermid| at reset (m)", d0)
row("finger_joint at reset (rad)", q0)
row("finger origin gap at reset (m)", gap0)
row("peg z at reset (m)", z0)
print()
print("  peg is 0.030 m across; close_command_expr = 0.785 rad")
print("  A held grasp should start with the peg close to the finger midpoint and")
print("  the gripper already partly closed on it.")
print("=" * 84 + "\n", flush=True)
app.close()
