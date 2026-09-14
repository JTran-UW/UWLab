# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the gripper-only Robotiq 2F-85 asset at several finger_joint values.

Uses the same articulation the grasp sampler loads, with a peg placed at the
metadata pinch point. Writes one PNG per joint value so the open/closed
direction can be checked by eye instead of inferred from body origins.
Run with ``--headless --enable_cameras``.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--qs", type=str, default="0.0,0.4,0.785")
parser.add_argument("--out", type=str, default="/tmp/jtran_gripper_render")
parser.add_argument("--no_peg", action="store_true")
parser.add_argument("--settle", type=int, default=60)
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR  # noqa: E402
from uwlab_assets.robots.ur5e_robotiq_gripper import ROBOTIQ_2F85  # noqa: E402

FINGER_OFFSET = 0.1345  # metadata: pinch point along gripper +x from robotiq_base_link


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.5)
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(FINGER_OFFSET, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    # look along -y at the gripper (approach axis +x runs left->right in the image)
    cam_side = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_side",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.07, -0.45, 0.5), rot=(0.5, 0.5, 0.5, 0.5), convention="ros"),
    )
    # look down -z from above
    cam_top = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_top",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.07, 0.0, 0.95), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    if args_cli.no_peg:
        cfg.peg = None  # type: ignore[assignment]
    scene = InteractiveScene(cfg)
    sim.reset()
    os.makedirs(args_cli.out, exist_ok=True)

    robot = scene["robot"]
    names = list(robot.joint_names)
    fi = names.index("finger_joint")
    bn = list(robot.body_names)
    print("joints:", names)
    for q in [float(v) for v in args_cli.qs.split(",")]:
        pos = robot.data.default_joint_pos.torch.clone()
        pos[:, fi] = q
        robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
        robot.set_joint_position_target(pos)
        for _ in range(args_cli.settle):
            robot.set_joint_position_target(pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(1 / 120)
        eye_t = torch.tensor([[0.07, -0.45, 0.5], [0.07, 0.0, 0.95]], device=sim.device)
        tgt_t = torch.tensor([[0.07, 0.0, 0.5], [0.07, 0.0, 0.5]], device=sim.device)
        scene["cam_side"].set_world_poses_from_view(eye_t[0:1], tgt_t[0:1])
        scene["cam_top"].set_world_poses_from_view(eye_t[1:2], tgt_t[1:2])
        for _ in range(12):
            sim.render()
            scene["cam_side"].update(0.0, force_recompute=True)
            scene["cam_top"].update(0.0, force_recompute=True)
        jp = robot.data.joint_pos.torch[0]
        print(f"q_cmd={q:.3f} -> joints " + " ".join(f"{n}={jp[i].item():+.3f}" for i, n in enumerate(names)))
        bp = robot.data.body_pos_w.torch[0]
        base = bp[bn.index("robotiq_base_link")]
        for b in ("left_inner_finger", "right_inner_finger", "left_outer_finger", "right_outer_finger"):
            r = bp[bn.index(b)] - base
            print(f"    {b:20s} rel base = ({r[0].item():+.4f}, {r[1].item():+.4f}, {r[2].item():+.4f})")
        for cam in ("cam_side", "cam_top"):
            rgb = scene[cam].data.output["rgb"][0]
            arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
            arr = arr[..., :3].astype(np.uint8)
            fn = os.path.join(args_cli.out, f"{cam}_q{q:.3f}{'_nopeg' if args_cli.no_peg else ''}.png")
            Image.fromarray(arr).save(fn)
            print("    wrote", fn)
    simulation_app.close()


if __name__ == "__main__":
    main()
