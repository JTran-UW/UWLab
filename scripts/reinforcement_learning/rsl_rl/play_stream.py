# Copyright (c) 2024-2025, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the arm simulation and print the states that the server would send to the client.
Use --headless to run without the UI (via AppLauncher). Does not modify play.py.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Run arm simulation and print viewer state (for streaming). Use --headless for headless."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import torch
import gymnasium as gym
import os

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# Column order matching joint_log CSV and client viewer
# Robotiq 2F-85 knuckle and finger joints (left/right inner/outer); 15th joint duplicated for client
STATE_KEYS = [
    "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3", "robotiq_gripper",
    "robotiq_left_inner_knuckle", "robotiq_left_inner_finger", "robotiq_left_outer_knuckle",
    "robotiq_left_outer_finger", "robotiq_right_inner_knuckle", "robotiq_right_inner_finger",
    "robotiq_right_outer_knuckle", "robotiq_right_outer_finger",
    "ins_x", "ins_y", "ins_z", "ins_xr", "ins_yr", "ins_zr",
    "rec_x", "rec_y", "rec_z", "rec_xr", "rec_yr", "rec_zr",
]


def get_state_dict(env):
    """Build the state dict from observation manager (after full step). Same as play.py CSV row."""
    joint_pos = env.unwrapped.observation_manager._latest_group_obs["joint_pos"].cpu().numpy().flatten().tolist()
    if len(joint_pos) == 14:
        joint_pos = list(joint_pos) + [joint_pos[-1]]
    insertive_pos = env.unwrapped.scene.rigid_objects["insertive_object"].data.root_link_pos_w.cpu().numpy().flatten().tolist()
    insertive_rot = [elem.item() for elem in euler_xyz_from_quat(env.unwrapped.scene.rigid_objects["insertive_object"].data.root_link_quat_w)]
    receptive_pos = env.unwrapped.scene.rigid_objects["receptive_object"].data.root_link_pos_w.cpu().numpy().flatten().tolist()
    receptive_rot = [elem.item() for elem in euler_xyz_from_quat(env.unwrapped.scene.rigid_objects["receptive_object"].data.root_link_quat_w)]
    values = joint_pos + insertive_pos + insertive_rot + receptive_pos + receptive_rot
    values = (values + [0.0] * len(STATE_KEYS))[:len(STATE_KEYS)]
    return dict(zip(STATE_KEYS, values))


def get_state_dict_from_scene(env):
    """Build the state dict by reading directly from scene (robot + rigid objects). Use after each physics step."""
    scene = env.unwrapped.scene
    robot = scene["robot"]
    joint_pos = robot.data.joint_pos.cpu().numpy().flatten().tolist()
    if len(joint_pos) == 14:
        joint_pos = list(joint_pos) + [joint_pos[-1]]
    insertive_pos = scene.rigid_objects["insertive_object"].data.root_link_pos_w.cpu().numpy().flatten().tolist()
    insertive_rot = [elem.item() for elem in euler_xyz_from_quat(scene.rigid_objects["insertive_object"].data.root_link_quat_w)]
    receptive_pos = scene.rigid_objects["receptive_object"].data.root_link_pos_w.cpu().numpy().flatten().tolist()
    receptive_rot = [elem.item() for elem in euler_xyz_from_quat(scene.rigid_objects["receptive_object"].data.root_link_quat_w)]
    values = joint_pos + insertive_pos + insertive_rot + receptive_pos + receptive_rot
    values = (values + [0.0] * len(STATE_KEYS))[:len(STATE_KEYS)]
    return dict(zip(STATE_KEYS, values))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    # Capture state every physics step (decimation steps per env.step()) by wrapping scene.update
    _original_scene_update = env.unwrapped.scene.update

    def _scene_update_with_state_logging(dt):
        _original_scene_update(dt)
        state = get_state_dict_from_scene(env)
        print(state)

    env.unwrapped.scene.update = _scene_update_with_state_logging

    # Print initial state (after reset)
    print(get_state_dict_from_scene(env))

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
