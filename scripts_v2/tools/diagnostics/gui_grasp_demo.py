# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a single EE-grasped reset state with the gripper held closed.

`visualize_reset_states.py` reloads a new state every 0.1 s and its gripper logic
deliberately *opens* a closed gripper (to cycle through poses), so an object never
has time to fall there -- which is why the resets look correct in that tool.

This holds ONE state and commands the gripper closed, so what you see is whether
the grasp survives. Peg height is printed each second alongside the drop from its
reset position; the camera is parked on the gripper.

Run with a visualizer (Isaac Lab 3.0 is headless unless one is requested):

    python scripts_v2/tools/diagnostics/gui_grasp_demo.py --viz kit \
        --reset_type ObjectAnywhereEEGrasped \
        env.scene.insertive_object=peg env.scene.receptive_object=peghole \
        env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--reset_type",
    type=str,
    default="ObjectAnywhereEEGrasped",
    help="ObjectAnywhereEEGrasped | ObjectRestingEEGrasped | ObjectPartiallyAssembledEEGrasped",
)
parser.add_argument("--hold_seconds", type=float, default=6.0, help="seconds to hold before reloading a new state")
parser.add_argument("--open_gripper", action="store_true", help="command OPEN instead of closed (control condition)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
# Pass the full namespace: AppLauncher needs --viz to open a window, since 3.0
# defaults to headless unless a visualizer is explicitly requested.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
cfg.events.reset_from_reset_states.params["reset_types"] = [args_cli.reset_type]
cfg.events.reset_from_reset_states.params["probs"] = [1.0]
env = gym.make(args_cli.task, cfg=cfg).unwrapped

robot = env.scene.articulations["robot"]
peg = env.scene.rigid_objects["insertive_object"]
pads = [i for i, b in enumerate(robot.body_names) if "inner_finger" in b and "knuckle" not in b]

grip = -1.0 if args_cli.open_gripper else 1.0
print("\n" + "=" * 78)
print(f"HOLDING ONE STATE: {args_cli.reset_type}   gripper = {'OPEN' if args_cli.open_gripper else 'CLOSED'}")
print("=" * 78)
print("  The gripper is commanded and held; the state is NOT reloaded for")
print(f"  {args_cli.hold_seconds:.0f} s. Watch whether the peg stays in the fingers.")
print("=" * 78 + "\n", flush=True)

try:
    while simulation_app.is_running():
        env.reset()
        # park the camera on the gripper so the grasp fills the view
        fm = robot.data.body_pos_w.torch[0, pads].mean(dim=0).tolist()
        env.sim.set_camera_view(
            eye=(fm[0] + 0.45, fm[1] - 0.45, fm[2] + 0.30), target=(fm[0], fm[1], fm[2])
        )
        z0 = peg.data.root_pos_w.torch[0, 2].item()
        act = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
        act[:, -1] = grip

        t0 = time.time()
        last = 0.0
        while time.time() - t0 < args_cli.hold_seconds and simulation_app.is_running():
            env.step(act)
            el = time.time() - t0
            if el - last >= 1.0:
                last = el
                z = peg.data.root_pos_w.torch[0, 2].item()
                c = robot.data.body_pos_w.torch[0, pads].mean(dim=0)
                d = (peg.data.root_pos_w.torch[0] - c).norm().item()
                print(f"   t={el:4.1f}s   peg_z={z:+.4f}   drop={z0 - z:+.4f} m   |peg-fingers|={d:.4f} m",
                      flush=True)
        print("   --- reloading a fresh state ---\n", flush=True)
except KeyboardInterrupt:
    pass
finally:
    env.close()
    simulation_app.close()
