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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--record_transitions",
    type=int,
    default=None,
    help=(
        "If set, record this many environment steps into a replay buffer and save it, then exit. "
        "Total transitions saved = num_envs * record_transitions. OnPolicyRunner checkpoints only."
    ),
)
parser.add_argument(
    "--transitions_output",
    type=str,
    default=None,
    help="Path to save the recorded replay buffer. Defaults to <log_dir>/play_transitions.pt.",
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
import tqdm
from tensordict import TensorDict

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from uwlab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx
from vecenv_wrapper import HolosomaVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def record_transitions_to_replay_buffer(
    env,
    policy,
    actor_obs_keys: list[str],
    critic_obs_keys: list[str],
    num_steps: int,
    output_path: str,
    task_name: str,
    device: str,
    gamma: float = 0.99,
) -> None:
    """Run a policy for ``num_steps`` and save transitions in SimpleReplayBuffer format.

    Works with both ``RslRlVecEnvWrapper`` (PPO) and ``HolosomaVecEnvWrapper`` (FastSAC).
    The policy callable should accept whatever ``env.get_observations()`` returns
    (TensorDict) and return an actions tensor.

    Transitions are stored so the output can be reloaded for off-policy training
    (e.g. warm-starting FastSAC from a PPO expert).
    """
    n_env = env.num_envs
    n_act = env.num_actions

    # Initial obs
    obs_td = env.get_observations().to(device)
    actor_obs = torch.cat([obs_td[g] for g in actor_obs_keys], dim=-1)
    critic_obs = torch.cat([obs_td[g] for g in critic_obs_keys], dim=-1)
    n_obs = actor_obs.shape[-1]
    n_critic_obs = critic_obs.shape[-1]

    rb = SimpleReplayBuffer(
        n_env=n_env,
        buffer_size=num_steps,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        n_steps=1,
        gamma=gamma,
        device=device,
    )

    pbar = tqdm.tqdm(total=num_steps, desc="Recording transitions")
    for i in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs_td)
            # 1% per-env chance to replace with uniform random action in [-1, 1]
            # random_mask = torch.rand(n_env, 1, device=actions.device) < 0.01
            # random_actions = torch.rand_like(actions) * 2.0 - 1.0
            # actions = torch.where(random_mask, random_actions, actions)
            # actions = torch.ones_like(actions)

            next_obs_td, rewards, dones, extras = env.step(actions.to(env.device))
            next_obs_td = next_obs_td.to(device)
            rewards = rewards.to(device=device, dtype=torch.float)
            dones = dones.to(device=device, dtype=torch.long)
            truncations = extras.get(
                "time_outs", torch.zeros(n_env, dtype=torch.bool, device=device)
            )
            truncations = truncations.to(device).long()

            next_actor_obs = torch.cat([next_obs_td[g] for g in actor_obs_keys], dim=-1)
            next_critic_obs = torch.cat([next_obs_td[g] for g in critic_obs_keys], dim=-1)

            transition = TensorDict(
                {
                    "observations": actor_obs,
                    "actions": actions.to(device=device, dtype=torch.float),
                    "next": {
                        "observations": next_actor_obs,
                        "rewards": rewards,
                        "truncations": truncations,
                        "dones": dones,
                    },
                },
                batch_size=(n_env,),
                device=device,
            )
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = next_critic_obs
            rb.extend(transition)

            obs_td = next_obs_td
            actor_obs = next_actor_obs
            critic_obs = next_critic_obs
        pbar.update(1)
    pbar.close()

    payload = {
        "buffer_tensors": {
            "observations": rb.observations.detach().cpu(),
            "actions": rb.actions.detach().cpu(),
            "rewards": rb.rewards.detach().cpu(),
            "dones": rb.dones.detach().cpu(),
            "truncations": rb.truncations.detach().cpu(),
            "next_observations": rb.next_observations.detach().cpu(),
            "critic_observations": rb.critic_observations.detach().cpu(),
            "next_critic_observations": rb.next_critic_observations.detach().cpu(),
            "ptr": rb.ptr,
        },
        "metadata": {
            "n_env": n_env,
            "buffer_size": num_steps,
            "n_obs": n_obs,
            "n_act": n_act,
            "n_critic_obs": n_critic_obs,
            "actor_obs_keys": actor_obs_keys,
            "critic_obs_keys": critic_obs_keys,
            "task": task_name,
            "total_transitions": n_env * num_steps,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(payload, output_path)
    print(f"[INFO] Saved {n_env * num_steps} transitions to: {output_path}")


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
    if args_cli.use_pretrained_checkpoint:
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

    # determine runner type and wrap env accordingly
    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")

    if is_fastsac:
        env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create runner and load checkpoint
    if is_fastsac:
        runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.setup()
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        actor_obs_keys = agent_cfg.actor_obs_keys
        critic_obs_keys = agent_cfg.critic_obs_keys

        # Optional: record transitions into a replay buffer and exit
        if args_cli.record_transitions is not None and args_cli.record_transitions > 0:
            # FastSAC policy expects {"actor_obs": tensor}; wrap for compatibility
            def fastsac_sample_policy(obs_td):
                a_obs = torch.cat([obs_td[k] for k in actor_obs_keys], dim=-1)
                return policy({"actor_obs": a_obs})

            output_path = args_cli.transitions_output or os.path.join(log_dir, "play_transitions.pt")
            record_transitions_to_replay_buffer(
                env=env,
                policy=fastsac_sample_policy,
                actor_obs_keys=actor_obs_keys,
                critic_obs_keys=critic_obs_keys,
                num_steps=args_cli.record_transitions,
                output_path=output_path,
                task_name=args_cli.task,
                device=agent_cfg.device,
                gamma=getattr(agent_cfg.algorithm, "gamma", 0.99) if hasattr(agent_cfg, "algorithm") else 0.99,
            )
            env.close()
            return
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs_groups = runner.cfg["obs_groups"]
        ppo_actor_keys = list(obs_groups["policy"])
        ppo_critic_keys = list(obs_groups.get("critic", ppo_actor_keys))

        # Optional: record transitions into a replay buffer and exit
        if args_cli.record_transitions is not None and args_cli.record_transitions > 0:
            # policy = runner.get_inference_policy_sample(device=env.unwrapped.device)
            output_path = args_cli.transitions_output or os.path.join(log_dir, "play_transitions.pt")
            record_transitions_to_replay_buffer(
                env=env,
                policy=policy,
                actor_obs_keys=ppo_actor_keys,
                critic_obs_keys=ppo_critic_keys,
                num_steps=args_cli.record_transitions,
                output_path=output_path,
                task_name=args_cli.task,
                device=agent_cfg.device,
                gamma=float(runner.alg_cfg.get("gamma", 0.99)),
            )
            env.close()
            return
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if is_fastsac:
                # FastSACAgent policy expects {"actor_obs": tensor}
                actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
                actions = policy({"actor_obs": actor_obs})
                obs, _, dones, _ = env.step(actions)
            else:
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
