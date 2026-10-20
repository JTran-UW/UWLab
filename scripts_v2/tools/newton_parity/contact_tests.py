# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted contact tests with known answers, one case per env (robot parked, materials and peg mass fixed).

Cases:
    press_table / press_plate: peg resting on the table / UR5 support plate under a downward force.
    drop_table: peg dropped from a height onto the table.
    ramp: horizontal force ramp on the resting peg (static friction onset).
    slide: constant horizontal force above the slip threshold (sliding friction).
    insert: aligned peg pushed down into the hole at a lateral offset.

.. code-block:: bash

    python scripts_v2/tools/newton_parity/contact_tests.py --visualizer none --backend newton --out newton_contact.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted contact tests.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--backend", type=str, choices=["physx", "newton"], required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--duration", type=float, default=3.0)
parser.add_argument("--mu_static", type=float, default=0.6)
parser.add_argument("--mu_dynamic", type=float, default=0.4)
parser.add_argument("--peg_mass", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.backend == "newton" and not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from uwlab_tasks.utils.hydra import hydra_task_compose

# env frame: table top x in [-0.14, 0.94], y in [-0.68, 0.68]; the support plate sits under the robot base at the origin
TABLE_TOP = -0.013
PEG_HALF = (0.015, 0.015, 0.030)
HOLE_BOTTOM = -0.0177
HOLE_TOP = 0.02294
HOLE_FLOOR = -0.01516
TABLE_XY, HOLE_XY, PARK_XY = (0.65, 0.35), (0.55, -0.35), (0.85, 0.55)
SLIDE_XY = (0.80, 0.60)
GROUND_TOP = (
    -0.868
)  # floor height in the env frame (lying peg rests at -0.853 on PhysX)  # slides run along -y: ~1.2 m of table before the far edge
LYING = (0.7071068, 0.0, 0.0, 0.7071068)  # 90 deg about x (xyzw): peg on its side, 15 mm half height


def make_cases(m: float, g: float, mu_s: float):
    cases = []
    for f in (0.0, 2.0, 10.0, 50.0, 200.0):
        cases.append({"name": f"press_table_{f:g}N", "kind": "press", "xy": TABLE_XY, "surface": TABLE_TOP, "fz": -f})
    for f in (0.0, 2.0, 10.0, 50.0, 200.0):
        cases.append({
            "name": f"press_lying_{f:g}N",
            "kind": "press",
            "xy": TABLE_XY,
            "surface": TABLE_TOP,
            "fz": -f,
            "quat": LYING,
            "half_z": PEG_HALF[1],
        })
    for h in (0.02, 0.10, 0.30):
        cases.append(
            {"name": f"drop_table_{h * 100:g}cm", "kind": "drop", "xy": TABLE_XY, "surface": TABLE_TOP, "height": h}
        )
    slip = mu_s * m * g
    for rate in (0.5 * slip, 2.0 * slip):
        cases.append(
            {"name": f"tip_ramp_{rate:.2f}Nps", "kind": "ramp", "xy": TABLE_XY, "surface": TABLE_TOP, "rate": rate}
        )
        cases.append({
            "name": f"slip_ramp_{rate:.2f}Nps",
            "kind": "ramp",
            "xy": SLIDE_XY,
            "surface": TABLE_TOP,
            "rate": rate,
            "quat": LYING,
            "half_z": PEG_HALF[1],
            "axis": 1,
            "sign": -1.0,
        })
    for fx in (1.5 * slip, 2.0 * slip):
        cases.append({
            "name": f"slide_lying_{fx:.2f}N",
            "kind": "slide",
            "xy": SLIDE_XY,
            "surface": TABLE_TOP,
            "fx": fx,
            "quat": LYING,
            "half_z": PEG_HALF[1],
            "axis": 1,
            "sign": -1.0,
        })
    for vx in (0.0, 1.0, 3.0, 6.0):
        cases.append({
            "name": f"ground_drop_v{vx:g}",
            "kind": "ground",
            "xy": (1.20, 0.0),
            "surface": GROUND_TOP,
            "v0": (vx, 0.0, 0.0),
            "w0": (0.0, 0.0, 0.0),
            "z0": 0.0,
        })
    cases.append({
        "name": "ground_drop_tumble",
        "kind": "ground",
        "xy": (1.20, 0.30),
        "surface": GROUND_TOP,
        "v0": (2.0, 0.0, 0.0),
        "w0": (8.0, 5.0, 0.0),
        "z0": 0.0,
    })
    for dx in (0.0, 0.5, 0.9, 1.5, 2.0, 3.0, 5.0):
        cases.append({
            "name": f"insert_dx{dx:g}mm",
            "kind": "insert",
            "xy": HOLE_XY,
            "surface": TABLE_TOP,
            "dx": dx * 1e-3,
            "fz": -5.0,
            "seated_z": TABLE_TOP - HOLE_BOTTOM + HOLE_FLOOR + PEG_HALF[2],
        })
    return cases


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    g = 9.81
    cases = make_cases(args_cli.peg_mass, g, args_cli.mu_static)
    n = len(cases)
    env_cfg.scene.num_envs = n
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.terminations.abnormal_robot = None
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.enable_corruption = False
    mu = {"static_friction_range": (args_cli.mu_static,) * 2, "dynamic_friction_range": (args_cli.mu_dynamic,) * 2}
    for term in ("insertive_object_material", "receptive_object_material", "table_material"):
        getattr(env_cfg.events, term).params.update(mu)
    env_cfg.events.randomize_insertive_object_mass.params["mass_distribution_params"] = (args_cli.peg_mass,) * 2
    for term in ("randomize_receptive_object_mass", "randomize_table_mass", "randomize_robot_mass"):
        getattr(env_cfg.events, term).params["mass_distribution_params"] = (1.0, 1.0)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot, peg, hole = env.scene["robot"], env.scene["insertive_object"], env.scene["receptive_object"]
    origins = env.scene.env_origins
    dev = env.device
    env.reset(seed=args_cli.seed)

    # park the robot: arm swung away from the test area
    q = robot.data.default_joint_pos.torch.clone()
    q[:, robot.joint_names.index("shoulder_pan_joint")] += 2.5
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)

    peg_pose = torch.zeros(n, 7, device=dev)
    peg_pose[:, 6] = 1.0
    hole_pose = torch.zeros(n, 7, device=dev)
    hole_pose[:, 6] = 1.0
    for i, c in enumerate(cases):
        x, y = c["xy"]
        if c["kind"] == "ground":
            peg_pose[i, :3] = torch.tensor([x, y, c["z0"]])
            hole_pose[i, :3] = torch.tensor([*PARK_XY, TABLE_TOP - HOLE_BOTTOM])
        elif c["kind"] == "insert":
            peg_pose[i, :3] = torch.tensor(
                [x + c["dx"], y, HOLE_TOP - HOLE_BOTTOM + c["surface"] + 0.002 + PEG_HALF[2]]
            )
            hole_pose[i, :3] = torch.tensor([x, y, c["surface"] - HOLE_BOTTOM])
        else:
            z = c["surface"] + c.get("half_z", PEG_HALF[2]) + (c.get("height", 0.0) or 0.0005)
            peg_pose[i, :3] = torch.tensor([x, y, z])
            if "quat" in c:
                peg_pose[i, 3:7] = torch.tensor(c["quat"])
            hole_pose[i, :3] = torch.tensor([*PARK_XY, TABLE_TOP - HOLE_BOTTOM])
    peg_pose[:, :3] += origins
    hole_pose[:, :3] += origins
    peg.write_root_pose_to_sim(peg_pose)
    vel = torch.zeros(n, 6, device=dev)
    for i, c in enumerate(cases):
        if c["kind"] == "ground":
            vel[i, :3] = torch.tensor(c["v0"])
            vel[i, 3:] = torch.tensor(c["w0"])
    peg.write_root_velocity_to_sim(vel)
    hole.write_root_pose_to_sim(hole_pose)
    hole.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)

    phys_dt = env.physics_dt
    t = {"now": 0.0}
    log = []

    def force_at(time):
        f = torch.zeros(n, 1, 3, device=dev)
        for i, c in enumerate(cases):
            settle = time >= 0.3
            if c["kind"] in ("press", "insert") and settle:
                f[i, 0, 2] = c["fz"]
            elif c["kind"] == "ramp" and settle:
                f[i, 0, c.get("axis", 0)] = c.get("sign", 1.0) * c["rate"] * (time - 0.3)
            elif c["kind"] == "slide" and settle:
                f[i, 0, c.get("axis", 0)] = c.get("sign", 1.0) * c["fx"]
        return f

    sim_step, scene_update = env.sim.step, env.scene.update

    def step(render=True):
        peg.permanent_wrench_composer.set_forces_and_torques_index(forces=force_at(t["now"]), is_global=True)
        peg.write_data_to_sim()
        sim_step(render=False)
        t["now"] += phys_dt

    def update(dt):
        scene_update(dt)
        p = peg.data.root_link_pose_w.torch
        v = peg.data.root_lin_vel_w.torch
        log.append(
            torch.cat(
                [torch.full((n, 1), t["now"], device=dev), p[:, :3] - origins, p[:, 3:7], v, force_at(t["now"])[:, 0]],
                1,
            ).cpu()
        )

    env.sim.step, env.scene.update = step, update
    action = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    steps = int(round(args_cli.duration / env.step_dt))
    for _ in range(steps):
        env.step(action)
    out = {
        "meta": {
            "backend": args_cli.backend,
            "physics_cfg": repr(env_cfg.sim.physics),
            "physics_dt": phys_dt,
            "mu_static": args_cli.mu_static,
            "mu_dynamic": args_cli.mu_dynamic,
            "peg_mass": args_cli.peg_mass,
            "g": g,
            "peg_half": PEG_HALF,
            "table_top": TABLE_TOP,
            "columns": ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "fx", "fy", "fz"],
        },
        "cases": cases,
        "log": torch.stack(log, 1),
        "peg_mass_readback": peg.data.body_mass.torch[:, 0].cpu(),
    }
    if args_cli.backend == "newton":
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        model, solver = NewtonManager._model, NewtonManager._solver
        d = solver.mjw_data
        nacon = int(d.nacon.numpy()[0]) if hasattr(d, "nacon") else None
        contact = d.contact
        k = nacon or 0
        out["newton"] = {
            "shape_label": list(model.shape_label),
            "shape_material_mu": model.shape_material_mu.numpy().tolist(),
            "geom_friction_world0": wp.to_torch(solver.mjw_model.geom_friction)[0].cpu().tolist(),
            "nacon": nacon,
            "contact_friction": wp.to_torch(contact.friction)[:k].cpu().tolist(),
            "contact_worldid": wp.to_torch(contact.worldid)[:k].cpu().tolist(),
            "contact_geom": wp.to_torch(contact.geom)[:k].cpu().tolist(),
            "contact_dim": wp.to_torch(contact.dim)[:k].cpu().tolist(),
        }
    torch.save(out, args_cli.out)
    print(f"[contact_tests] wrote {args_cli.out}: {n} cases, {len(log)} samples")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
