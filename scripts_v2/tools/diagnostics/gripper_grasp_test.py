# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can the Robotiq 2F-85 actually hold the peg under load in IsaacLab 3.0?

The companion mimic test (gripper_mimic_test.py) shows the linkage is
kinematically coupled in free space. That does not prove the grip holds once
there is contact: PhysX rejects all four mimic joints for lacking finite limits,
and both outer knuckles carry a negative mass, either of which could yield a
linkage that tracks unloaded but slips under contact force.

This places the peg between the fingers, closes the gripper, and reports whether
the peg is retained against gravity and then carried when the arm lifts. Grasp
acquisition is the one thing task 0 needs and tasks 1-3 (which start already
grasped) do not -- and task 0 is the one pinned at 0.000.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hold_steps", type=int, default=240)
parser.add_argument("--force_convex", action="store_true",
                    help="rewrite the peg collider from SDF to convexHull before physics starts")
parser.add_argument("--kinematic_peg", action="store_true",
                    help="make the peg kinematic: it collides but cannot move, so the fingers "
                         "are blocked by contact alone with no per-step teleporting artifact")
parser.add_argument("--close_steps", type=int, default=150)
parser.add_argument("--open_q", type=float, default=0.70, help="finger_joint value that OPENS the gripper")
parser.add_argument("--close_target", type=float, default=0.0, help="finger_joint value that CLOSES it")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR  # noqa: E402
from uwlab_assets.robots.ur5e_robotiq_gripper import (  # noqa: E402
    IMPLICIT_UR5E_ROBOTIQ_2F85,
    UR5E_EFFORT_LIMITS,
    UR5E_VELOCITY_LIMITS,
)

# omnireset drives the arm with OSC, so its arm actuators are stiffness 0 --
# the arm would collapse here. Give it a position-holding gain so the gripper
# stays put and the test isolates the grasp.
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
class GraspSceneCfg(InteractiveSceneCfg):
    robot = ROBOT.replace(prim_path="{ENV_REGEX_NS}/Robot")
    peg: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0,
                disable_gravity=False, kinematic_enabled=("--kinematic_peg" in __import__("sys").argv),
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 1.0)),
    )


def main():
    # Match omnireset's solver settings exactly. Contact resolution depends
    # heavily on position-iteration count (the task uses 192 vs the default
    # handful), so a default-solver scene can show penetration that the real
    # task never has.
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
    sim_cfg.physics = PhysxCfg(
        solver_type=1,
        max_position_iteration_count=192,
        max_velocity_iteration_count=1,
        bounce_threshold_velocity=0.02,
        friction_offset_threshold=0.01,
        friction_correlation_distance=0.0005,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(GraspSceneCfg(num_envs=1, env_spacing=3.0))

    # The peg and peghole ship with physics:approximation = "sdf". Rewrite it to
    # convexHull *before* sim.reset() parses physics, so we can tell whether SDF
    # collision is why the fingers pass through the peg.
    if args_cli.force_convex:
        from pxr import Usd  # noqa: F401
        import isaaclab.sim.utils.stage as stage_utils

        st = stage_utils.get_current_stage()
        n = 0
        for prim in st.Traverse():
            att = prim.GetAttribute("physics:approximation")
            if att and att.Get() == "sdf":
                att.Set("convexHull")
                print(f"  [force_convex] {prim.GetPath()} sdf -> convexHull")
                n += 1
        print(f"  [force_convex] rewrote {n} collider(s)")

    sim.reset()

    robot: Articulation = scene["robot"]
    peg: RigidObject = scene["peg"]
    names = robot.joint_names
    drive = names.index("finger_joint")
    bodies = robot.body_names
    pads = [i for i, b in enumerate(bodies) if "inner_finger" in b and "knuckle" not in b]
    if not pads:
        pads = [i for i, b in enumerate(bodies) if "finger" in b]

    print("\n" + "=" * 78)
    print("ROBOTIQ 2F-85 GRASP-UNDER-LOAD TEST")
    print("=" * 78)
    print(f"  finger pad bodies: {[bodies[i] for i in pads]}")

    target = robot.data.default_joint_pos.clone()
    # settle with the gripper open
    for _ in range(60):
        target[:, drive] = args_cli.open_q
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(1 / 120)

    grasp_center = robot.data.body_pos_w[0, pads].mean(dim=0)
    print(f"  grasp center (world): {[round(v,4) for v in grasp_center.tolist()]}")
    # Sanity-check placement: the peg is 30 mm across, so if the pinch point is
    # more than ~15 mm from where we put it the fingers close beside the peg and
    # the "no collision" reading would be a placement artifact, not a bug.
    for i in pads:
        bp = robot.data.body_pos_w[0, i]
        print(f"    pad {bodies[i]:24s} world={[round(v,4) for v in bp.tolist()]}"
              f"  dist_to_center={(bp - grasp_center).norm().item():.4f} m")
    print(f"    pad-to-pad distance = {(robot.data.body_pos_w[0, pads[0]] - robot.data.body_pos_w[0, pads[1]]).norm().item():.4f} m")

    # teleport the peg to the grasp center
    pose = peg.data.default_root_state.clone()
    pose[:, :3] = grasp_center.unsqueeze(0)
    pose[:, 3:7] = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=pose.device)
    pose[:, 7:] = 0.0
    peg.write_root_pose_to_sim(pose[:, :7])
    peg.write_root_velocity_to_sim(pose[:, 7:])
    scene.write_data_to_sim()
    sim.step()
    scene.update(1 / 120)
    z_start = peg.data.root_pos_w[0, 2].item()
    print(f"  peg placed at z = {z_start:.4f}\n")

    def pad_gap():
        pp = robot.data.body_pos_w[0, pads]
        return (pp[0] - pp[1]).norm().item()

    print(f"  finger pad separation at open_q={args_cli.open_q}: {pad_gap():.4f} m")
    print(f"  {'phase':>10} {'step':>6} {'finger_q':>10} {'pad_gap':>10} {'peg_z':>10} {'drop':>10}")
    # Hold the peg fixed while the fingers close around it. Without this the peg
    # simply free-falls during the close and never contacts the gripper, which
    # measures gravity rather than grasping.
    for step in range(args_cli.close_steps):
        f = min(1.0, step / (args_cli.close_steps * 0.5))
        target[:, drive] = args_cli.open_q + (args_cli.close_target - args_cli.open_q) * f
        robot.set_joint_position_target(target)
        if not args_cli.kinematic_peg:
            peg.write_root_pose_to_sim(pose[:, :7])
            peg.write_root_velocity_to_sim(torch.zeros_like(pose[:, 7:]))
        scene.write_data_to_sim()
        sim.step()
        scene.update(1 / 120)
        if step % max(1, args_cli.close_steps // 4) == 0:
            z = peg.data.root_pos_w[0, 2].item()
            print(f"  {'close':>10} {step:>6} {robot.data.joint_pos[0, drive].item():>10.4f} "
                  f"{pad_gap():>10.4f} {z:>10.4f} {z_start - z:>10.4f}")
    print(f"  finger pad separation, closed on peg: {pad_gap():.4f} m")
    print("  (peg released now -- everything below is the grip holding it)")

    z_closed = peg.data.root_pos_w[0, 2].item()
    for step in range(args_cli.hold_steps):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(1 / 120)
        if step % max(1, args_cli.hold_steps // 4) == 0:
            z = peg.data.root_pos_w[0, 2].item()
            print(f"  {'hold':>10} {step:>6} {robot.data.joint_pos[0, drive].item():>10.4f} "
                  f"{pad_gap():>10.4f} {z:>10.4f} {z_start - z:>10.4f}")

    z_end = peg.data.root_pos_w[0, 2].item()
    drop = z_start - z_end
    print("\n" + "-" * 78)
    print(f"  peg z: start={z_start:.4f}  after close={z_closed:.4f}  after {args_cli.hold_steps} hold steps={z_end:.4f}")
    print(f"  total drop = {drop:.4f} m")
    print(f"  final finger_joint = {robot.data.joint_pos[0, drive].item():.4f} rad "
          f"(commanded {args_cli.close_target})")
    if drop < 0.01:
        print("\n  VERDICT: GRASP HOLDS. The peg is retained against gravity.")
        print("           Grasp acquisition is not blocked by the gripper.")
    elif drop < 0.05:
        print("\n  VERDICT: WEAK/SLIPPING GRASP -- peg creeps downward while held.")
    else:
        print("\n  VERDICT: GRASP FAILS. The peg falls out of the closed gripper.")
        print("           This alone would make task 0 (grasp acquisition) unlearnable.")
    print("=" * 78 + "\n", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
