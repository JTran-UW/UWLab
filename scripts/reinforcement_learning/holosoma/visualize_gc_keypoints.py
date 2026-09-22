# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualise the keypoint objective: current-peg keypoints vs goal-peg keypoints.

Each 2 s episode the env resets via GoalConditionedMultiResetManager. This draws the axis
keypoints under BOTH the peg's current pose (orange) and its sampled goal pose (blue), and prints
the per-env max/mean keypoint distance -- the quantity `success_mode="keypoint"` thresholds on.

Keypoints lie along the peg's local z axis, not at bounding-box corners: a peg is a solid of
revolution, so spin about its own axis does not affect insertability. Axis keypoints are
spin-invariant while still capturing position and tilt.

Hydra-style overrides are supported, e.g.::

    python visualize_gc_keypoints.py --num_envs 4 \\
        env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize GC keypoints on the current and goal peg.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--num_keypoints", type=int, default=4, help="Keypoints along the peg axis.")
parser.add_argument(
    "--keypoint_extent",
    type=float,
    default=-1.0,
    help="Half-length along the peg's z axis, metres. <=0 reads `bottom_offset` from the peg metadata.",
)
parser.add_argument("--print_every", type=int, default=30, help="Steps between distance printouts.")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GCMRM-Visualization-v0",
    help="Name of the task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.rewards import (
    axis_keypoints_local,
    transform_keypoints,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import read_metadata_from_usd_directory
from uwlab_tasks.utils.hydra import hydra_task_config


def make_point_marker(prim_path: str, color: tuple[float, float, float], radius: float = 0.006):
    """A single-prototype sphere instancer; one instance is drawn per keypoint."""
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "p": sim_utils.SphereCfg(
                    radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                )
            },
        )
    )


@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device

    insertive = unwrapped.scene["insertive_object"]

    extent = args_cli.keypoint_extent
    if extent <= 0:
        try:
            meta = read_metadata_from_usd_directory(insertive.cfg.spawn.usd_path)
            extent = abs(float(meta["bottom_offset"]["pos"][2]))
            print(f"[keypoints] extent from metadata bottom_offset: {extent:.4f} m")
        except Exception as exc:  # noqa: BLE001
            extent = 0.03
            print(f"[keypoints] metadata unavailable ({exc}); falling back to extent {extent:.4f} m")
    kp_local = axis_keypoints_local(extent, args_cli.num_keypoints, device)
    print(f"[keypoints] {args_cli.num_keypoints} points along local z, span [{-extent:.4f}, {extent:.4f}] m")
    print("[keypoints] ORANGE = current peg    BLUE = goal peg")

    cur_marker = make_point_marker("/Visuals/KeypointsCurrent", (1.0, 0.45, 0.0))
    goal_marker = make_point_marker("/Visuals/KeypointsGoal", (0.0, 0.45, 1.0))

    reset_term = unwrapped.event_manager.get_term_cfg("reset_from_reset_states").func

    env.reset()
    actions = torch.zeros((unwrapped.num_envs, unwrapped.action_manager.total_action_dim), device=device)
    step = 0
    while simulation_app.is_running():
        origins = unwrapped.scene.env_origins
        goal_pose = reset_term.goal_state["rigid_object"]["insertive_object"]["root_pose"]

        cur_kp = transform_keypoints(insertive.data.root_pos_w, insertive.data.root_quat_w, kp_local)
        goal_kp = transform_keypoints(goal_pose[:, :3] + origins, goal_pose[:, 3:7], kp_local)

        cur_marker.visualize(translations=cur_kp.reshape(-1, 3))
        goal_marker.visualize(translations=goal_kp.reshape(-1, 3))

        if step % args_cli.print_every == 0:
            d = torch.norm(cur_kp - goal_kp, dim=-1)  # [N, K]
            mx, mn = d.max(dim=1).values, d.mean(dim=1)
            per_env = "  ".join(f"env{i}: max={mx[i]:.3f} mean={mn[i]:.3f}" for i in range(min(6, len(mx))))
            print(f"[step {step:5d}] keypoint distance (m)   {per_env}", flush=True)

        env.step(actions)
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
