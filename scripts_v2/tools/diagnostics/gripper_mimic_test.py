# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check whether the Robotiq 2F-85 linkage still closes under IsaacLab 3.0.

Only ``finger_joint`` is actuated; the other five gripper joints are passive and
driven by USD mimic constraints. Every 3.0 run logs PhysX rejecting all four of
them ("needs a finite limit set to be used by the mimic joint feature"), plus a
negative mass on both outer knuckles. If those constraints are inactive the
linkage is uncoupled, which would let pre-grasped tasks train normally while
making grasp *acquisition* impossible -- exactly the observed pattern, where
task 0 sits at 0.000 while tasks 1-3 learn.

This drives ``finger_joint`` open->closed and records every gripper joint. A
working linkage moves the passive joints together with the driven one; a broken
one leaves them at their initial value (or drifting under gravity alone).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=180, help="sim steps over the close sweep")
parser.add_argument("--close_target", type=float, default=0.7, help="finger_joint target, rad")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85  # noqa: E402

GRIPPER_JOINTS = [
    "finger_joint",  # the only actuated one
    "right_outer_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]


@configclass
class GripperSceneCfg(InteractiveSceneCfg):
    ground = sim_utils.GroundPlaneCfg().replace()  # type: ignore[attr-defined]
    robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    scene_cfg = GripperSceneCfg(num_envs=1, env_spacing=2.0)
    # ground plane is optional here; drop it to keep the scene minimal
    scene_cfg.ground = None  # type: ignore[assignment]
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    names = robot.joint_names
    idx = {n: names.index(n) for n in GRIPPER_JOINTS if n in names}
    missing = [n for n in GRIPPER_JOINTS if n not in idx]

    print("\n" + "=" * 78)
    print("ROBOTIQ 2F-85 MIMIC LINKAGE TEST")
    print("=" * 78)
    print(f"  articulation joints : {len(names)}")
    print(f"  gripper joints found: {len(idx)}/{len(GRIPPER_JOINTS)}")
    if missing:
        print(f"  MISSING             : {missing}")
    drive = idx["finger_joint"]
    passives = [n for n in GRIPPER_JOINTS if n != "finger_joint" and n in idx]

    lim = robot.data.joint_pos_limits[0]
    print("\n  joint limits (rad):")
    for n in GRIPPER_JOINTS:
        if n in idx:
            lo, hi = lim[idx[n]].tolist()
            infinite = "  <<< NON-FINITE" if abs(lo) > 1e5 or abs(hi) > 1e5 or lo == hi else ""
            print(f"    {n:34s} [{lo:+.4f}, {hi:+.4f}]{infinite}")

    target = robot.data.default_joint_pos.clone()
    print(f"\n  sweeping finger_joint 0 -> {args_cli.close_target} rad over {args_cli.steps} steps\n")
    print(f"  {'step':>6} " + "".join(f"{n.replace('_joint','')[:15]:>16}" for n in GRIPPER_JOINTS))

    for step in range(args_cli.steps):
        frac = min(1.0, step / max(1, args_cli.steps * 0.6))
        target[:, drive] = args_cli.close_target * frac
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(1 / 120)
        if step % max(1, args_cli.steps // 9) == 0 or step == args_cli.steps - 1:
            q = robot.data.joint_pos[0]
            print(f"  {step:>6} " + "".join(f"{q[idx[n]].item():>16.5f}" for n in GRIPPER_JOINTS))

    q = robot.data.joint_pos[0]
    moved = {n: abs(q[idx[n]].item()) for n in passives}
    driven = abs(q[drive].item())
    print("\n" + "-" * 78)
    print(f"  driven finger_joint final = {driven:.5f} rad (target {args_cli.close_target})")
    print("  passive joint final magnitudes:")
    for n, v in moved.items():
        print(f"    {n:34s} {v:.5f}")
    coupled = sum(1 for v in moved.values() if v > 0.02)
    print(f"\n  passive joints that moved (>0.02 rad): {coupled}/{len(passives)}")
    if driven < 0.05:
        print("  VERDICT: the DRIVEN joint never moved -- actuation problem, not mimic.")
    elif coupled == 0:
        print("  VERDICT: LINKAGE BROKEN. finger_joint moves alone; mimic constraints inactive.")
        print("           The gripper cannot close on an object -> grasp acquisition impossible.")
    elif coupled < len(passives):
        print("  VERDICT: PARTIALLY COUPLED -- some mimic constraints active, some not.")
    else:
        print("  VERDICT: linkage intact; all passive joints follow. Mimic errors are benign.")
    print("=" * 78 + "\n", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
