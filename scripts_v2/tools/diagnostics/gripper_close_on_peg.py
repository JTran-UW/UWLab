# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Close the gripper-only Robotiq 2F-85 asset on a peg held between its pads.

The peg is kinematic so it cannot be pushed away: if finger/peg contact works
the drive stalls at the peg width; if the fingers pass through, finger_joint
reaches its closed value. Renders one image per phase (``--enable_cameras``).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="/tmp/jtran_close_on_peg")
parser.add_argument("--peg_x", type=float, default=0.16, help="peg center along link +x (approach) from base link")
parser.add_argument("--open_q", type=float, default=0.80)
parser.add_argument("--close_q", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--peg_mass", type=float, default=0.001)
parser.add_argument("--dynamic_peg", action="store_true", help="peg is a free rigid body (default kinematic)")
parser.add_argument("--peg_usd", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR  # noqa: E402
from uwlab_assets.robots.ur5e_robotiq_gripper import ROBOTIQ_2F85  # noqa: E402

PEG_USD = args_cli.peg_usd or f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"
CAM = sim_utils.PinholeCameraCfg(focal_length=35.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0))


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PEG_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=not args_cli.dynamic_peg, disable_gravity=True,
                solver_position_iteration_count=4, solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.peg_mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    cam_a = CameraCfg(prim_path="{ENV_REGEX_NS}/cam_a", update_period=0, height=480, width=640, data_types=["rgb"], spawn=CAM)
    cam_b = CameraCfg(prim_path="{ENV_REGEX_NS}/cam_b", update_period=0, height=480, width=640, data_types=["rgb"], spawn=CAM)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
    sim_cfg.physics = PhysxCfg(
        solver_type=1, max_position_iteration_count=192, max_velocity_iteration_count=1,
        bounce_threshold_velocity=0.02, friction_offset_threshold=0.01, friction_correlation_distance=0.0005,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    os.makedirs(args_cli.out, exist_ok=True)
    dev = sim.device

    robot = scene["robot"]
    peg = scene["peg"]
    names = list(robot.joint_names)
    fi = names.index("finger_joint")
    bn = list(robot.body_names)
    pads = [bn.index("left_inner_finger"), bn.index("right_inner_finger")]

    # gripper: link frame = world (identity), base at origin, so approach = world +x, pads open along world +y? (measured below)
    root = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=dev)
    robot.write_root_pose_to_sim(root)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    # open the gripper by driving it (mimic-consistent), starting from the USD default (all zeros)
    tgt = robot.data.default_joint_pos.torch.clone()
    tgt[:, fi] = args_cli.open_q
    for _ in range(120):
        robot.set_joint_position_target(tgt)
        scene.write_data_to_sim(); sim.step(); scene.update(1 / 120)
    q = robot.data.joint_pos.torch[0]
    print("after open:", " ".join(f"{n}={q[i].item():+.3f}" for i, n in enumerate(names)))
    bp = robot.data.body_pos_w.torch[0]
    base = bp[bn.index("robotiq_base_link")]
    for i in pads:
        print(f"   {bn[i]:20s} rel base = {[round(v,4) for v in (bp[i]-base).tolist()]}")
    pad_axis = (bp[pads[0]] - bp[pads[1]])
    print(f"   pad-pad vector = {[round(v,4) for v in pad_axis.tolist()]}  |gap|={pad_axis.norm().item():.4f}")

    # peg between the pads: along approach at peg_x, centered on the pad midpoint laterally
    mid = 0.5 * (bp[pads[0]] + bp[pads[1]])
    peg_pos = torch.tensor([[base[0].item() + args_cli.peg_x, mid[1].item(), mid[2].item()]], device=dev)
    peg_pose = torch.cat([peg_pos, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)], dim=-1)
    peg.write_root_pose_to_sim(peg_pose)
    peg.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    for _ in range(5):
        robot.set_joint_position_target(tgt)
        scene.write_data_to_sim(); sim.step(); scene.update(1 / 120)
    print(f"peg placed at {[round(v,4) for v in peg.data.root_pos_w.torch[0].tolist()]}  (base at {[round(v,4) for v in base.tolist()]})")

    def snap(tag):
        eyes = torch.tensor([[base[0].item() + 0.12, base[1].item() - 0.45, base[2].item() + 0.0],
                             [base[0].item() + 0.12, base[1].item() + 0.0, base[2].item() + 0.45]], device=dev)
        tgts = torch.tensor([[base[0].item() + 0.12, base[1].item(), base[2].item()]] * 2, device=dev)
        scene["cam_a"].set_world_poses_from_view(eyes[0:1], tgts[0:1])
        scene["cam_b"].set_world_poses_from_view(eyes[1:2], tgts[1:2])
        for _ in range(12):
            sim.render()
            scene["cam_a"].update(0.0, force_recompute=True)
            scene["cam_b"].update(0.0, force_recompute=True)
        for cam in ("cam_a", "cam_b"):
            rgb = scene[cam].data.output["rgb"][0]
            arr = (rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb))[..., :3].astype(np.uint8)
            fn = os.path.join(args_cli.out, f"{tag}_{cam}.png")
            Image.fromarray(arr).save(fn)
        print("   wrote", tag)

    snap("open")
    # close by driving the target down over the first half, then hold
    print(f"{'step':>5} {'finger_q':>9} {'q_tgt':>6} {'pad_gap':>8} {'peg_pos':>28} {'|jv|max':>8}")
    for step in range(args_cli.steps):
        f = min(1.0, step / (args_cli.steps * 0.5))
        tgt[:, fi] = args_cli.open_q + (args_cli.close_q - args_cli.open_q) * f
        robot.set_joint_position_target(tgt)
        scene.write_data_to_sim(); sim.step(); scene.update(1 / 120)
        if step % 20 == 0 or step == args_cli.steps - 1:
            bp = robot.data.body_pos_w.torch[0]
            gap = (bp[pads[0]] - bp[pads[1]]).norm().item()
            print(f"{step:>5} {robot.data.joint_pos.torch[0, fi].item():>9.4f} {tgt[0, fi].item():>6.3f} {gap:>8.4f} "
                  f"{str([round(v,4) for v in peg.data.root_pos_w.torch[0].tolist()]):>28} {robot.data.joint_vel.torch.abs().max().item():>8.2f}")
    q = robot.data.joint_pos.torch[0]
    print("after close:", " ".join(f"{n}={q[i].item():+.3f}" for i, n in enumerate(names)))
    snap("closed")
    simulation_app.close()


if __name__ == "__main__":
    main()
