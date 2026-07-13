# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--num_steps", type=int, default=None, help="Maximum number of policy steps (safety cap; prefer --num_episodes for unbiased eval).")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help="Stop after this many episodes have terminated. Unbiased — every counted episode runs to "
    "its natural success/time_out, so longer (failing) episodes are not under-sampled the way "
    "they are when truncating on --num_steps.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--bc_checkpoint",
    type=str,
    default=None,
    help="Path to a PointNet BC Lightning checkpoint (from "
    "scripts/imitation_learning/point_cloud/train_point_net.py). When set, the BC PointNet is rolled "
    "out instead of an RSL-RL policy — the runner is skipped and obs come from --bc_obs_group.",
)
parser.add_argument(
    "--bc_obs_group",
    type=str,
    default="data_collect",
    help="Observation group feeding the BC PointNet: a per-term dict with the point cloud (scene_pc) "
    "and the proprio terms (concatenated in declaration order, matching training).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from uwlab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx, export_vision_student_as_jit

# Inject UWLab distillation classes into rsl_rl's distillation_runner module so
# the runner's eval(class_name) lookup resolves them at checkpoint load time.
import rsl_rl.runners.distillation_runner as _distillation_runner_module

from uwlab_rl.rsl_rl.distillation_dagger import DistillationDAgger
from uwlab_rl.rsl_rl.distillation_dagger_weighted import DistillationDAggerWeighted
from uwlab_rl.rsl_rl.distillation_runner_split import DistillationRunnerSplit
from uwlab_rl.rsl_rl.student_teacher_history_pointcloud import StudentTeacherHistoryPointCloud
from uwlab_rl.rsl_rl.student_teacher_mlp import StudentTeacherMLP
from uwlab_rl.rsl_rl.student_teacher_pointcloud import StudentTeacherPointCloud
from uwlab_rl.rsl_rl.student_teacher_vision import StudentTeacherVision
from uwlab_rl.rsl_rl.student_teacher_vision_recurrent import StudentTeacherVisionRecurrent

_distillation_runner_module.StudentTeacherVision = StudentTeacherVision
_distillation_runner_module.StudentTeacherVisionRecurrent = StudentTeacherVisionRecurrent
_distillation_runner_module.StudentTeacherMLP = StudentTeacherMLP
_distillation_runner_module.StudentTeacherPointCloud = StudentTeacherPointCloud
_distillation_runner_module.StudentTeacherHistoryPointCloud = StudentTeacherHistoryPointCloud
_distillation_runner_module.DistillationDAgger = DistillationDAgger
_distillation_runner_module.DistillationDAggerWeighted = DistillationDAggerWeighted
_distillation_runner_module.DistillationRunnerSplit = DistillationRunnerSplit

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

import rsl_rl.runners.on_policy_runner as _runner_module
from uwlab_rl.rsl_rl.actor_critic_encoder import ActorCriticWithEncoder
_runner_module.ActorCriticWithEncoder = ActorCriticWithEncoder
from uwlab_rl.rsl_rl.actor_critic_depth import ActorCriticDepth
from uwlab_rl.rsl_rl.bc_ppo import BCPPO
from uwlab_rl.rsl_rl.grpo import GRPO
from uwlab_rl.rsl_rl.ppo_pbrs import PPOPBRS
_runner_module.ActorCriticDepth = ActorCriticDepth
_runner_module.BCPPO = BCPPO
_runner_module.GRPO = GRPO
_runner_module.PPOPBRS = PPOPBRS
from uwlab_rl.rsl_rl.actor_critic_rma import ActorCriticRMA
from uwlab_rl.rsl_rl.ppo_rma import PPO_RMA
_runner_module.ActorCriticRMA = ActorCriticRMA
_runner_module.PPO_RMA = PPO_RMA
import rsl_rl.algorithms as _rsl_rl_algorithms
_rsl_rl_algorithms.PPO_RMA = PPO_RMA

from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _is_jit_checkpoint(path: str) -> bool:
    """Return True if *path* is a TorchScript (JIT) file, False if it is a runner checkpoint."""
    try:
        torch.jit.load(path, map_location="cpu")
        return True
    except RuntimeError:
        return False


def _import_bc_utils():
    """Lazily import the BC PointNet helpers (live in scripts/imitation_learning/point_cloud)."""
    pc_bc_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "imitation_learning", "point_cloud"
    )
    sys.path.insert(0, pc_bc_dir)
    from bc_utils import bc_actions, bc_reset, load_bc_pointnet  # noqa: E402

    return load_bc_pointnet, bc_actions, bc_reset


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    bc_mode = args_cli.bc_checkpoint is not None
    if bc_mode:
        # BC PointNet eval: the checkpoint is a Lightning .ckpt, not an RSL-RL run.
        resume_path = os.path.abspath(args_cli.bc_checkpoint)
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # BC PointNet eval bypasses the runner / JIT paths entirely.
    bc = bc_actions = bc_reset = None
    if bc_mode:
        load_bc_pointnet, bc_actions, bc_reset = _import_bc_utils()
        bc = load_bc_pointnet(resume_path, agent_cfg.device)
    # Auto-detect whether the checkpoint is a TorchScript (JIT) file or a runner checkpoint.
    is_jit = False if bc_mode else _is_jit_checkpoint(resume_path)

    if bc_mode:
        print(f"[INFO]: Detected BC PointNet checkpoint — rolling out PointNet on obs group "
              f"'{args_cli.bc_obs_group}'. hp={bc['hp']}")
    elif is_jit:
        print("[INFO]: Detected TorchScript checkpoint — loading with torch.jit.load (skipping runner).")
        policy_nn = torch.jit.load(resume_path, map_location=agent_cfg.device)
        policy_nn.eval()
        # JIT policy forward takes a flat obs tensor; policy() alias used in the loop below.
        policy = policy_nn
    else:
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunnerSplit":
            runner = DistillationRunnerSplit(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "BCPPORunner":
            from uwlab_rl.rsl_rl.bc_ppo_runner import BCPPORunner
            runner = BCPPORunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "GRPOGroupedRunner":
            from uwlab_rl.rsl_rl.grpo_grouped_runner import GRPOGroupedRunner
            runner = GRPOGroupedRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "OnPolicyRunnerRMA":
            from uwlab_rl.rsl_rl.on_policy_runner_rma import OnPolicyRunnerRMA
            runner = OnPolicyRunnerRMA(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        try:
            runner.load(resume_path)
        except ValueError as _e:
            if "optimizer" not in str(_e).lower():
                raise
            # Optimizer param-group mismatch: checkpoint was saved with a different
            # policy architecture (e.g. std-head added/removed between train and play).
            # For inference we only need model weights, so skip the optimizer.
            print(f"[WARN] Optimizer state mismatch — loading model weights only: {_e}")
            _ckpt = torch.load(resume_path, map_location=agent_cfg.device)
            runner.alg.policy.load_state_dict(_ckpt["model_state_dict"])
        except RuntimeError as _e:
            if "size mismatch" not in str(_e):
                raise
            # Critic obs size mismatch (e.g. checkpoint trained with different critic obs dim).
            # strict=False still errors on size mismatches, so drop critic keys entirely.
            print(f"[WARN] State dict size mismatch — dropping critic keys and loading actor only: {_e}")
            _ckpt = torch.load(resume_path, map_location=agent_cfg.device)
            _sd = {k: v for k, v in _ckpt["model_state_dict"].items() if not k.startswith("critic")}
            runner.alg.policy.load_state_dict(_sd, strict=False)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # extract the neural network module
        # we do this in a try-except to maintain backwards compatibility.
        try:
            # version 2.3 onwards
            policy_nn = runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        # StudentTeacherVision needs a custom multi-input exporter (proprio + depth images);
        # the default actor-MLP exporter would silently drop the encoder. Detect by attr.
        if isinstance(policy_nn, StudentTeacherVision):
            export_vision_student_as_jit(policy_nn, path=export_model_dir, filename="depth_policy.pt")
        elif isinstance(policy_nn, StudentTeacherPointCloud):
            # PointNet student forward takes (points, proprio) — the flat-actor exporter
            # would trace garbage (or crash). No PointNet exporter exists yet; skip.
            print("[INFO]: Skipping JIT/ONNX export for StudentTeacherPointCloud (no exporter).")
        else:
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # Episode-success tracking (matches UWLab-ICL play.py). A done with positive
    # reward indicates a success-triggered termination; pure time-outs and
    # abnormal-state terminations land at ~0 reward and are counted as failures.
    num_episodes = 0
    num_successes = 0
    finger_pose_history: list = []
    # DEBUG: collect actions for env 0
    action_history: list = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # BC PointNet reads its own obs group (PC + proprio); JIT policy takes a flat
            # tensor directly; runner inference policy takes the obs dict.
            if bc_mode:
                actions = bc_actions(bc, obs, args_cli.bc_obs_group)
            elif is_jit:
                mean, std = policy(obs)
                actions = mean
            else:
                actions = policy(obs)
            # DEBUG: save actions for env 0
            action_history.append(actions[0].cpu().numpy())
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            if dones.any():
                num_episodes += dones.sum().item()
                num_successes += torch.logical_and(rewards > 0.1, dones).sum().item()
                print(f"Success rate: {num_successes / num_episodes:.2%}")
                print(f"Number of episodes: {num_episodes}")
                print(f"Number of successes: {num_successes}")
            # reset recurrent states for episodes that have terminated.
            # JIT reset() takes no arguments and resets all hidden states. A feed-forward BC PointNet
            # has no state to reset; a history-conditioned BC PointNet keeps a rolling (state, action)
            # buffer, so clear it per-env on done (no-op for the feed-forward policies).
            if bc_mode:
                bc_reset(bc, dones)
            elif not is_jit:
                policy_nn.reset(dones)

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        # Episode-count termination is unbiased: every counted episode ran to
        # its natural end (success-trigger or time_out). Truncating on steps
        # instead would systematically over-sample fast episodes (which are
        # mostly successes) and bias the reported success rate upward.
        if args_cli.num_episodes is not None and num_episodes >= args_cli.num_episodes:
            break
        # --num_steps stays as a safety cap (e.g. you want "stop at 100 episodes
        # OR 10000 steps, whichever first") so a stuck rollout can't hang the eval.
        if args_cli.num_steps is not None and timestep >= args_cli.num_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print("=" * 50)
    print(f"Number of episodes: {num_episodes}")
    print(f"Number of successes: {num_successes}")
    if num_episodes:
        print(f"Success rate: {num_successes / num_episodes:.2%}")

    if finger_pose_history:
        import matplotlib.pyplot as plt
        import numpy as np

        data = np.array(finger_pose_history)  # (T, num_finger_joints)
        fig, ax = plt.subplots(figsize=(10, 4))
        for j in range(data.shape[1]):
            ax.plot(data[:, j], label=f"finger_{j}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Joint position (rad)")
        ax.set_title("Finger joint positions over time (sim)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(log_dir, "finger_joint_positions.png")
        npy_path  = os.path.join(log_dir, "finger_joint_positions.npy")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        np.save(npy_path, data)
        print(f"[INFO] Finger joint position plot saved to: {plot_path}")
        print(f"[INFO] Finger joint position data saved to: {npy_path}")

    # DEBUG: save action history for env 0
    if action_history:
        import numpy as np
        action_npy_path = os.path.join(log_dir, "debug_actions_env0.npy")
        np.save(action_npy_path, np.array(action_history))
        print(f"[DEBUG] Actions for env 0 saved to: {action_npy_path}  shape={np.array(action_history).shape}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
