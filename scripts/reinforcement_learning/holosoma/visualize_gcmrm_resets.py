# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the GCMRM visualization task with no policy.

Each 2 s episode the env resets via GoalConditionedMultiResetManager; a coordinate frame marker
is drawn at each env's sampled insertive-object goal pose. Zero actions are applied throughout.

Hydra-style overrides are supported, e.g.:
    python visualize_gcmrm_resets.py --num_envs 4 \
        env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize GCMRM resets with a goal frame marker.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GCMRM-Visualization-v0",
    help="Name of the task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import make_gc_goal_marker, visualize_gc_goal
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    # magenta sphere = insertive object goal, cyan sphere = EE goal
    goal_marker = make_gc_goal_marker("/Visuals/InsertiveObjectGoal", (1.0, 0.0, 1.0))
    ee_goal_marker = make_gc_goal_marker("/Visuals/EEGoal", (0.0, 1.0, 1.0))

    # the GoalConditionedMultiResetManager instance
    reset_term = unwrapped.event_manager.get_term_cfg("reset_from_reset_states").func

    env.reset()
    actions = torch.zeros(
        (unwrapped.num_envs, unwrapped.action_manager.total_action_dim), device=unwrapped.device
    )
    while simulation_app.is_running():
        # goal poses are env-relative; shift by env origins for world-frame markers
        goal_pose = reset_term.goal_state["rigid_object"]["insertive_object"]["root_pose"]
        visualize_gc_goal(goal_marker, goal_pose[:, :3] + unwrapped.scene.env_origins, goal_pose[:, 3:7])
        ee_goal_pose = reset_term.goal_state["articulation"]["robot"]["ee_pose"]
        visualize_gc_goal(ee_goal_marker, ee_goal_pose[:, :3] + unwrapped.scene.env_origins, ee_goal_pose[:, 3:7])
        env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
