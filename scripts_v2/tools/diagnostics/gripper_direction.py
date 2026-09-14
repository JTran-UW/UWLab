# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Which way does finger_joint actually move the Robotiq 2F-85?

ROBOTIQ_GRIPPER_BINARY_ACTIONS declares:
    open_command_expr  = {"finger_joint": 0.0}
    close_command_expr = {"finger_joint": 0.785398}

If that is backwards for this USD, then commanding "close" opens the hand, the
recorded EE-grasped states (which saturate at 0.785 because the recorder
commanded it) were captured with the gripper OPEN, and airborne grasps drop while
resting/seated ones survive on support alone -- exactly the observed pattern.

Sweeps finger_joint across its range with NO object present and reports the
separation of the two inner_finger bodies. The 2F-85 opens to ~85 mm, so whichever
end of the range lands near 0.085 m is OPEN.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--settle", type=int, default=90, help="sim steps to settle at each commanded value")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets.robots.ur5e_robotiq_gripper import (  # noqa: E402
    IMPLICIT_UR5E_ROBOTIQ_2F85,
    UR5E_EFFORT_LIMITS,
    UR5E_VELOCITY_LIMITS,
)

ROBOT = IMPLICIT_UR5E_ROBOTIQ_2F85.copy()  # type: ignore[attr-defined]
ROBOT.actuators = dict(ROBOT.actuators)
ROBOT.actuators["arm"] = ImplicitActuatorCfg(
    joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
    stiffness=4000.0,
    damping=200.0,
    effort_limit_sim=UR5E_EFFORT_LIMITS,
    velocity_limit_sim=UR5E_VELOCITY_LIMITS,
)


@configclass
class Cfg(InteractiveSceneCfg):
    robot = ROBOT.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
scene = InteractiveScene(Cfg(num_envs=1, env_spacing=3.0))
sim.reset()
robot: Articulation = scene["robot"]
fj = robot.joint_names.index("finger_joint")
pads = [i for i, b in enumerate(robot.body_names) if "inner_finger" in b and "knuckle" not in b]
lo, hi = robot.data.joint_pos_limits[0, fj].tolist()

print("\n" + "=" * 74)
print("FINGER_JOINT DIRECTION (no object present)")
print("=" * 74)
print(f"  finger_joint limits: [{lo:.4f}, {hi:.4f}] rad")
print(f"  pad bodies: {[robot.body_names[i] for i in pads]}")
print(f"\n  {'finger_joint cmd':>18} {'reached':>10} {'pad separation':>16}")

target = robot.data.default_joint_pos.clone()
for q in (0.0, 0.2, 0.4, 0.6, 0.785398, hi):
    target[:, fj] = q
    for _ in range(args_cli.settle):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(1 / 120)
    sep = (robot.data.body_pos_w.torch[0, pads[0]] - robot.data.body_pos_w.torch[0, pads[1]]).norm().item()
    print(f"  {q:>18.4f} {robot.data.joint_pos.torch[0, fj].item():>10.4f} {sep:>16.4f}")

print("\n  Robotiq 2F-85 opens to ~0.085 m. Whichever end approaches that is OPEN.")
print("  Config says open=0.0, close=0.785398.")
print("=" * 74 + "\n", flush=True)
app.close()
