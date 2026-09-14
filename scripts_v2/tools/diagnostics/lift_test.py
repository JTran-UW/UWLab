# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can the policy lift the peg off the table in IsaacLab 3.0?

Task success rates split on exactly one thing: whether the object starts in
contact with the table. ObjectAnywhereEEGrasped (airborne, grasped) reaches 0.78;
ObjectRestingEEGrasped (grasped but resting on the table) is stuck at 0.13; and
ObjectAnywhereEEAnywhere (must acquire the grasp) is 0.00. That points at
breaking contact with the table rather than at the policy.

This resets from a chosen reset type, commands a constant upward OSC delta with
the gripper closed, and reports how far the peg actually rises. A grasp that
cannot lift shows up here immediately and independently of any learned policy.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--reset_type", type=str, default="ObjectRestingEEGrasped")
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
# use only the reset type under test so the measurement is not averaged over others
cfg.events.reset_from_reset_states.params["reset_types"] = [args_cli.reset_type]
cfg.events.reset_from_reset_states.params["probs"] = [1.0]
env = gym.make(args_cli.task, cfg=cfg).unwrapped
env.reset()

peg = env.scene.rigid_objects["insertive_object"]
robot = env.scene.articulations["robot"]
pads = [i for i, b in enumerate(robot.body_names) if "inner_finger" in b and "knuckle" not in b]

z0 = peg.data.root_pos_w[:, 2].clone()
ee0 = robot.data.body_pos_w[:, pads].mean(dim=1)
# peg position relative to the finger midpoint: if the grasp holds, this stays constant
rel0 = (peg.data.root_pos_w - ee0).norm(dim=-1)

act = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
act[:, 2] = 1.0    # +z Cartesian delta (lift)
act[:, -1] = 1.0   # gripper CLOSE (BinaryJointPositionAction: >0 -> close_command_expr)

print("\n" + "=" * 76)
print(f"LIFT TEST -- reset_type = {args_cli.reset_type}")
print("=" * 76)
print(f"  envs={args_cli.num_envs}  commanding +z delta with gripper closed for {args_cli.steps} steps")
print(f"  {'step':>6} {'peg dz (m)':>13} {'ee dz (m)':>13} {'|peg-ee| drift':>16} {'held frac':>11}")
for step in range(args_cli.steps + 1):
    if step % max(1, args_cli.steps // 8) == 0:
        ee = robot.data.body_pos_w[:, pads].mean(dim=1)
        rel = (peg.data.root_pos_w - ee).norm(dim=-1)
        dz = (peg.data.root_pos_w[:, 2] - z0)
        held = ((rel - rel0).abs() < 0.02).float().mean()
        print(f"  {step:>6} {dz.mean().item():>13.4f} {(ee[:,2]-ee0[:,2]).mean().item():>13.4f} "
              f"{(rel-rel0).abs().mean().item():>16.4f} {held.item():>11.3f}")
    if step < args_cli.steps:
        env.step(act)

ee = robot.data.body_pos_w[:, pads].mean(dim=1)
rel = (peg.data.root_pos_w - ee).norm(dim=-1)
peg_dz = (peg.data.root_pos_w[:, 2] - z0)
ee_dz = ee[:, 2] - ee0[:, 2]
held = ((rel - rel0).abs() < 0.02)
print("-" * 76)
print(f"  end-effector rose : {ee_dz.mean().item():+.4f} m")
print(f"  peg rose          : {peg_dz.mean().item():+.4f} m")
print(f"  peg still held    : {held.float().mean().item()*100:.1f}% of envs")
print(f"  peg followed EE   : {(peg_dz > 0.5*ee_dz).float().mean().item()*100:.1f}% of envs")
if ee_dz.mean() < 0.01:
    print("\n  INCONCLUSIVE: the end-effector itself barely moved; OSC command may be too small.")
elif held.float().mean() > 0.8 and peg_dz.mean() > 0.5 * ee_dz.mean():
    print("\n  RESULT: grasp lifts the peg off the table normally. Contact is NOT the blocker.")
else:
    print("\n  RESULT: the EE rises but the peg does NOT follow -- the grasp slips or the peg")
    print("          is stuck to the table. This would explain t1/t0 failing while t2 works.")
print("=" * 76 + "\n", flush=True)
app.close()
