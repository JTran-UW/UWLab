# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare a FastSAC distributional critic against empirical Monte Carlo return distributions.

Per outer iteration:
  1. env.reset(), then broadcast env_0's physics state to all envs so they share s0.
  2. Sample actions a_i from the actor (stochastic) at s0. Record Q distribution at (s0, a_i)
     — averaged across ensemble critics and across envs (since s0 is shared, this is the
     policy's expected Q-distribution at s0 marginalized over the actor's action noise).
  3. Take one step with those actions, then keep sampling stochastically until every env
     has terminated at least once.
  4. Compute per-env discounted MC returns, bin into a histogram with the same support as
     the Q atoms, softmax-normalize the counts.
  5. Overlay the two distributions on a single plot; save to ``q_vs_mc.png``.

Repeats until the sim app is stopped (Ctrl-C).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Compare FastSAC critic distribution to Monte Carlo returns.")
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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for Monte Carlo return computation.")
parser.add_argument("--output_path", type=str, default="q_vs_mc.png", help="Output path for the comparison plot.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import numpy as np
import torch

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from vecenv_wrapper import HolosomaVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _broadcast_env0_state_to_all(env_wrapper) -> None:
    """After ``env.reset()``, overwrite every env's articulation + rigid-object physics state
    with env_0's state, offset by each env's origin so the local pose is identical everywhere.

    This forces s0 to be shared across all envs while leaving the reset event manager's
    bookkeeping (task index, success monitor state) untouched.
    """
    unwrapped = env_wrapper.unwrapped  # ManagerBasedRLEnv
    scene = unwrapped.scene
    env_origins = scene.env_origins  # [n, 3]
    origin_0 = env_origins[0]
    n = scene.num_envs
    device = unwrapped.device
    all_env_ids = torch.arange(n, device=device)

    # Articulations
    for _, articulation in scene._articulations.items():
        pos_w0 = articulation.data.root_pos_w[0]           # [3]
        quat_w0 = articulation.data.root_quat_w[0]         # [4]
        linvel_w0 = articulation.data.root_lin_vel_w[0]    # [3]
        angvel_w0 = articulation.data.root_ang_vel_w[0]    # [3]
        # Position: env_0 local pose + per-env origin
        local_pos = pos_w0 - origin_0                      # [3]
        pos_all = (local_pos.unsqueeze(0) + env_origins).contiguous()   # [n, 3]
        quat_all = quat_w0.unsqueeze(0).expand(n, -1).contiguous()      # [n, 4]
        pose_all = torch.cat([pos_all, quat_all], dim=-1).contiguous()  # [n, 7]
        vel_all = torch.cat([linvel_w0, angvel_w0]).unsqueeze(0).expand(n, -1).contiguous()  # [n, 6]
        joint_pos_all = articulation.data.joint_pos[0:1].expand(n, -1).contiguous()
        joint_vel_all = articulation.data.joint_vel[0:1].expand(n, -1).contiguous()

        articulation.write_root_pose_to_sim(pose_all, env_ids=all_env_ids)
        articulation.write_root_velocity_to_sim(vel_all, env_ids=all_env_ids)
        articulation.write_joint_state_to_sim(joint_pos_all, joint_vel_all, env_ids=all_env_ids)
        # Align PD targets with the new joint state to avoid the controller snapping back.
        articulation.set_joint_position_target(joint_pos_all, env_ids=all_env_ids)
        articulation.set_joint_velocity_target(joint_vel_all, env_ids=all_env_ids)

    # Rigid objects
    for _, rigid_obj in scene._rigid_objects.items():
        pos_w0 = rigid_obj.data.root_pos_w[0]
        quat_w0 = rigid_obj.data.root_quat_w[0]
        linvel_w0 = rigid_obj.data.root_lin_vel_w[0]
        angvel_w0 = rigid_obj.data.root_ang_vel_w[0]
        local_pos = pos_w0 - origin_0
        pos_all = (local_pos.unsqueeze(0) + env_origins).contiguous()
        quat_all = quat_w0.unsqueeze(0).expand(n, -1).contiguous()
        pose_all = torch.cat([pos_all, quat_all], dim=-1).contiguous()
        vel_all = torch.cat([linvel_w0, angvel_w0]).unsqueeze(0).expand(n, -1).contiguous()

        rigid_obj.write_root_pose_to_sim(pose_all, env_ids=all_env_ids)
        rigid_obj.write_root_velocity_to_sim(vel_all, env_ids=all_env_ids)

    scene.write_data_to_sim()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Compare FastSAC critic distribution to MC returns."""
    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve checkpoint path
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create runner and load checkpoint
    runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.setup()
    runner.load(resume_path)

    device = env.unwrapped.device

    # Access the raw pieces so we can (a) stochastically sample from the actor and
    # (b) read the categorical Q-distribution rather than a scalar Q value.
    actor = runner.actor.to(device)
    qnet = runner.qnet.to(device)
    obs_normalizer = runner.obs_normalizer.to(device)
    critic_obs_normalizer = runner.critic_obs_normalizer.to(device)
    actor.eval()
    qnet.eval()
    obs_normalizer.eval()
    critic_obs_normalizer.eval()

    actor_obs_keys = agent_cfg.actor_obs_keys
    critic_obs_keys = agent_cfg.critic_obs_keys

    # Distributional Q-network support (categorical over atoms in [v_min, v_max])
    v_min = float(qnet.v_min)
    v_max = float(qnet.v_max)
    num_atoms = int(qnet.num_atoms)
    q_support = qnet.q_support.detach().cpu().numpy()  # [num_atoms]

    edges = np.linspace(v_min, v_max, num_atoms + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = (v_max - v_min) / num_atoms

    GAMMA = args_cli.gamma
    obs_normalization = runner.obs_normalization

    import matplotlib.pyplot as plt

    n_env = env.num_envs
    print(f"[INFO] Running with num_envs={n_env}, γ={GAMMA}, v_min={v_min}, v_max={v_max}, num_atoms={num_atoms}")

    iteration = 0
    while simulation_app.is_running():
        # ---- 1. Reset every env, then force all envs to share env_0's state ----
        obs, _ = env.reset()
        _broadcast_env0_state_to_all(env)
        obs = env.get_observations()

        # ---- 2. Stochastic action at s0 + record Q distribution ----
        with torch.inference_mode():
            actor_obs_0 = torch.cat([obs[k] for k in actor_obs_keys], dim=-1)
            critic_obs_0 = torch.cat([obs[k] for k in critic_obs_keys], dim=-1)

            norm_actor_obs_0 = obs_normalizer(actor_obs_0, update=False) if obs_normalization else actor_obs_0
            norm_critic_obs_0 = (
                critic_obs_normalizer(critic_obs_0, update=False) if obs_normalization else critic_obs_0
            )

            # get_actions_and_log_probs samples via rsample() → stochastic policy noise
            actions_0, _ = actor.get_actions_and_log_probs(norm_actor_obs_0)

            # Raw distributional logits: [num_critics, batch, num_atoms]
            q_logits = qnet(norm_critic_obs_0, actions_0)
            q_probs = torch.softmax(q_logits, dim=-1)                # per-critic PMF
            q_probs_over_batch = q_probs.mean(dim=0)                 # avg over critic ensemble → [batch, num_atoms]
            q_dist = q_probs_over_batch.mean(dim=0).cpu().numpy()    # avg over envs (shared s0) → [num_atoms]

        # ---- 3a. First rollout step (contributes r_0 at γ^0=1) ----
        new_obs, rew, dones, _extras = env.step(actions_0)
        active_mask = ~dones.bool()                                            # envs that haven't terminated yet
        returns = rew.clone()                                                  # r_0
        discount = torch.full((n_env,), GAMMA, device=device, dtype=rew.dtype) # γ^1 for the next step
        obs = new_obs

        # ---- 3b. Continue stochastic rollout until every env has terminated ----
        while active_mask.any():
            with torch.inference_mode():
                actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=-1)
                norm_actor_obs = obs_normalizer(actor_obs, update=False) if obs_normalization else actor_obs
                actions, _ = actor.get_actions_and_log_probs(norm_actor_obs)
            new_obs, rew, dones, _extras = env.step(actions)
            # Add γ^t * r_t only for envs that were still active going into this step
            returns = returns + torch.where(active_mask, discount * rew, torch.zeros_like(rew))
            discount = discount * GAMMA
            active_mask = active_mask & (~dones.bool())
            obs = new_obs

        mc_returns = returns.detach().cpu().numpy()  # [n_env]

        # ---- 4. Bin MC returns onto the Q support, softmax-normalize ----
        counts, _ = np.histogram(mc_returns, bins=edges)
        x = counts.astype(np.float64)
        x -= x.max()  # numerical stability
        exp_x = np.exp(x)
        mc_pmf = exp_x / exp_x.sum()

        # ---- 5. Overlay both distributions ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(centers, q_dist, width=bin_width * 0.9, alpha=0.55, label="Q distribution (critic)", color="tab:blue")
        ax.bar(
            centers, mc_pmf, width=bin_width * 0.9, alpha=0.55,
            label="MC returns (softmax-normed histogram)", color="tab:orange",
        )
        q_exp = float(np.sum(q_dist * q_support))
        mc_mean = float(mc_returns.mean())
        mc_std = float(mc_returns.std())
        ax.set_xlabel("Return")
        ax.set_ylabel("Probability")
        ax.set_title(
            f"Q vs MC returns — iter {iteration} | "
            f"E[Q]={q_exp:.3f}  MC mean={mc_mean:.3f}  MC std={mc_std:.3f}  n={n_env}"
        )
        ax.legend()
        ax.set_xlim(v_min, v_max)
        fig.tight_layout()
        fig.savefig(args_cli.output_path, dpi=100)
        plt.show()
        plt.close(fig)

        print(
            f"[iter {iteration:04d}] MC mean={mc_mean:.4f}  MC std={mc_std:.4f}  "
            f"E[Q]={q_exp:.4f}  → {args_cli.output_path}"
        )
        iteration += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
