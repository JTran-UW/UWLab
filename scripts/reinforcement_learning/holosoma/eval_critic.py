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
parser.add_argument(
    "--peg_xlim", type=float, nargs=2, default=[-1.0, 2.0], metavar=("XMIN", "XMAX"),
    help="Fixed x-axis bounds (env-local meters) for the peg x-y trajectory plot.",
)
parser.add_argument(
    "--peg_ylim", type=float, nargs=2, default=[-1.0, 2.0], metavar=("YMIN", "YMAX"),
    help="Fixed y-axis bounds (env-local meters) for the peg x-y trajectory plot.",
)
parser.add_argument(
    "--video", action="store_true", default=False,
    help="Record a video of the rollout (viewport camera) to the plots dir each iteration.",
)
parser.add_argument(
    "--video_envs", type=int, nargs="+", default=[0], metavar="ENV_ID",
    help=(
        "Which env indices to record when --video is set (default: [0]). One composite video "
        "is saved per env, named iter{iter}_env{idx}.mp4. The viewport camera is re-pointed at "
        "each requested env and re-rendered every step (no extra physics stepping)."
    ),
)
parser.add_argument(
    "--ppo_checkpoint", type=str, default=None,
    help=(
        "Path to a PPO/OnPolicyRunner checkpoint. If set, pi_ppo(s)'s per-dim action Gaussian "
        "(mean/std, no tanh squash) is overlaid on the action-density row of the --video composite."
    ),
)
parser.add_argument(
    "--ppo_task", type=str, default=None,
    help=(
        "Gym task ID whose rsl_rl_cfg_entry_point is the PPO agent config used to build the "
        "checkpoint's ActorCritic (obs groups, network sizes). Defaults to --task."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# enable cameras so env.render() returns frames for the video
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
import numpy as np
import torch

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from vecenv_wrapper import HolosomaVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _broadcast_env0_state_to_all(env_wrapper) -> None:
    """Set every env to env_0's configuration via the idiomatic ``reset_to`` path.

    Reads ``scene.get_state(is_relative=True)`` (env-origin-relative poses), replaces every env's
    entry with env_0's, then calls ``env.reset_to(state, is_relative=True)`` which re-adds each
    env's origin. Unlike raw ``write_*_to_sim``, ``reset_to`` also resets manager state, forwards
    the sim, and rebuilds the observation history — so ``s0`` is fully consistent.
    """
    unwrapped = env_wrapper.unwrapped  # ManagerBasedRLEnv
    n = unwrapped.scene.num_envs
    state = unwrapped.scene.get_state(is_relative=True)

    def broadcast_env0(node):
        if isinstance(node, dict):
            return {k: broadcast_env0(v) for k, v in node.items()}
        return node[0:1].expand(n, *node.shape[1:]).contiguous()  # env 0 → all envs

    unwrapped.reset_to(broadcast_env0(state), env_ids=None, is_relative=True)


# ---------------------------------------------------------------------------
# Plot rendering (formerly eval_critic_plots.py)
# ---------------------------------------------------------------------------
import os  # noqa: E402

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render_group(d: dict, output_path: str, peg_xy_path: str, q_traj_path: str | None,
                 peg_xlim, peg_ylim, ckpt_name: str) -> None:
    """Render Q-vs-MC (``output_path``), peg x-y trajectories (``peg_xy_path``) and, if
    ``q_traj_path`` is given, the leader env's Q vs return-to-go chart, for one reset group.

    ``d`` keys (numpy): q_dists/qt_dists [C, atoms], q_support/centers [atoms], bin_width, v_min,
    v_max, alpha, soft_mc/reward_mc/entropy_mc [n], mc_pmf [atoms], term_counts [3], term_type [n],
    peg_traj [T, n, 2], peghole_xy [2], q_lead/g_lead/act_lead [T], q_ppo_lead [T] or None,
    reset_idx, n_g, lead.
    """
    q_dists, qt_dists, q_support, centers = d["q_dists"], d["qt_dists"], d["q_support"], d["centers"]
    soft_mc, reward_mc, entropy_mc = d["soft_mc"], d["reward_mc"], d["entropy_mc"]
    term_counts = [int(c) for c in d["term_counts"]]
    n_g, reset_idx, lead = int(d["n_g"]), int(d["reset_idx"]), int(d["lead"])
    v_min, v_max, alpha, bin_width = float(d["v_min"]), float(d["v_max"]), float(d["alpha"]), float(d["bin_width"])
    peg_traj, peghole_xy = d["peg_traj"], d["peghole_xy"]

    # ---- Q vs total soft MC return (top), reward / entropy / terminations (bottom) ----
    num_critics = q_dists.shape[0]
    fig = plt.figure(figsize=(15, 8))
    ax_main = fig.add_subplot(2, 1, 1)
    ax_rew = fig.add_subplot(2, 3, 4)
    ax_ent = fig.add_subplot(2, 3, 5)
    ax_term = fig.add_subplot(2, 3, 6)
    ax_main.bar(centers, d["mc_pmf"], width=bin_width * 0.9, alpha=0.35,
                label="Soft MC return (reward + entropy)", color="tab:orange")
    online_colors = ["tab:blue", "tab:cyan", "tab:green", "tab:olive"]
    target_colors = ["tab:red", "tab:pink", "tab:purple", "tab:brown"]
    for c in range(num_critics):
        q_exp_c = float(np.sum(q_dists[c] * q_support))
        ax_main.plot(centers, q_dists[c], color=online_colors[c % len(online_colors)],
                     lw=1.8, label=f"Q online[{c}]  E={q_exp_c:.2f}")
        qt_exp_c = float(np.sum(qt_dists[c] * q_support))
        ax_main.plot(centers, qt_dists[c], color=target_colors[c % len(target_colors)],
                     lw=1.5, ls="--", label=f"Bellman target y[{c}] (s1,π(a1))  E={qt_exp_c:.2f}")
    q_exp_mean = float(np.mean([np.sum(q_dists[c] * q_support) for c in range(num_critics)]))
    ax_main.axvline(soft_mc.mean(), color="tab:orange", ls=":", lw=2, label=f"MC mean={soft_mc.mean():.2f}")
    ax_main.axvline(q_exp_mean, color="tab:blue", ls=":", lw=2, label=f"E[Q] mean={q_exp_mean:.2f}")
    ax_main.set_xlabel("Return")
    ax_main.set_ylabel("Probability")
    ax_main.set_title(
        f"Q (online + target, per critic) vs soft MC return — iter {reset_idx} | "
        f"soft MC mean={soft_mc.mean():.3f}  n={n_g}  (α={alpha:.4f})"
    )
    ax_main.legend(fontsize=8)
    ax_main.set_xlim(v_min, v_max)
    ax_rew.hist(reward_mc, bins=40, color="tab:green", alpha=0.8)
    ax_rew.set_title(f"Reward component  mean={reward_mc.mean():.3f}  std={reward_mc.std():.3f}")
    ax_rew.set_xlabel("Discounted reward return")
    ax_rew.set_ylabel("Count")
    ax_ent.hist(entropy_mc, bins=40, color="tab:purple", alpha=0.8)
    ax_ent.set_title(f"Entropy bonus  mean={entropy_mc.mean():.3f}  std={entropy_mc.std():.3f}")
    ax_ent.set_xlabel("Discounted −α·logπ return")
    ax_ent.set_ylabel("Count")
    term_labels = ["abnormal", "failure", "success"]
    ax_term.bar(term_labels, term_counts, color=["tab:red", "tab:gray", "tab:green"])
    for i, c in enumerate(term_counts):
        ax_term.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    ax_term.set_title(f"Terminations  (success {term_counts[2]}/{n_g} = {term_counts[2] / n_g:.1%})")
    ax_term.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    # ---- Peg x-y trajectories: one line per env on a shared workspace canvas ----
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    max_lines = 256
    env_ids_plot = np.arange(n_g) if n_g <= max_lines else np.linspace(0, n_g - 1, max_lines).astype(int)
    term_np = d["term_type"]
    succ_ids = [e for e in env_ids_plot if term_np[e] == 2]
    fail_ids = [e for e in env_ids_plot if term_np[e] != 2]
    for ids, colour in ((fail_ids, "tab:red"), (succ_ids, "tab:green")):
        for e in ids:
            ax2.plot(peg_traj[:, e, 0], peg_traj[:, e, 1], color=colour, lw=0.5, alpha=0.35)
    ax2.plot([], [], color="tab:green", lw=1.5, label=f"success ({len(succ_ids)})")
    ax2.plot([], [], color="tab:red", lw=1.5, label=f"failure ({len(fail_ids)})")
    end_xy = []
    for e in env_ids_plot:
        valid = np.flatnonzero(~np.isnan(peg_traj[:, e, 0]))
        if valid.size:
            end_xy.append(peg_traj[valid[-1], e])
    if end_xy:
        end_xy = np.asarray(end_xy)
        ax2.scatter(end_xy[:, 0], end_xy[:, 1], color="gold", s=18, zorder=6,
                    edgecolors="black", linewidths=0.3, label="end")
    ax2.scatter(peghole_xy[0], peghole_xy[1], marker="*", color="black", s=250, zorder=8, label="peghole")
    ax2.scatter(peg_traj[0, 0, 0], peg_traj[0, 0, 1], color="tab:blue", s=60, zorder=9,
                edgecolors="white", linewidths=0.5, label="start (s0)")
    ax2.set_xlim(peg_xlim)
    ax2.set_ylim(peg_ylim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (env-local, m)")
    ax2.set_ylabel("y (env-local, m)")
    ax2.set_title(
        f"{ckpt_name}\n"
        f"success {term_counts[2]}/{n_g} = {term_counts[2] / n_g:.1%}"
        f"  —  iter {reset_idx}  ({len(env_ids_plot)}/{n_g} envs plotted, T={peg_traj.shape[0]})",
        fontsize=10,
    )
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(peg_xy_path, dpi=100)
    plt.close(fig2)

    if q_traj_path is None:
        return
    # ---- Leader env: critic Q vs soft return-to-go over the trajectory ----
    q0_plot, g0_plot = d["q_lead"].copy(), d["g_lead"].copy()
    inact0 = ~d["act_lead"].astype(bool)
    q0_plot[inact0] = np.nan
    g0_plot[inact0] = np.nan
    ts = np.arange(q0_plot.shape[0])
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    ax3.plot(ts, q0_plot, color="tab:blue", lw=2.0, label="Q(s_t, a_t) (critic)")
    ax3.plot(ts, g0_plot, color="tab:orange", lw=2.0, ls="--", label="soft return-to-go G_t (MC)")
    q_ppo_lead = d.get("q_ppo_lead")
    if q_ppo_lead is not None and np.ndim(q_ppo_lead) > 0:
        q_ppo0_plot = np.array(q_ppo_lead, dtype=float)
        q_ppo0_plot[inact0] = np.nan
        ax3.plot(ts, q_ppo0_plot, color="tab:green", lw=2.0, ls=":", label="Q(s_t, a*_ppo)")
    ax3.set_xlabel("timestep t")
    ax3.set_ylabel("value")
    ax3.set_ylim(0, v_max)
    ax3.set_title(f"env {lead}: critic Q vs MC return-to-go — iter {reset_idx}  (MC={soft_mc[0]:.2f})")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(q_traj_path, dpi=100)
    plt.close(fig3)


def save_reset_scatter(reset_records: list[dict], out_path: str, title_prefix: str = "") -> None:
    """3D scatter of each reset's initial peg xyz colored by its success rate (RdYlGn, 0..1)."""
    xyz = np.array([r["init_peg_xyz"] for r in reset_records])
    rates = np.array([r["success_rate"] for r in reset_records])
    hole = np.array(reset_records[0]["peghole_xyz"])
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rates, cmap="RdYlGn", vmin=0.0, vmax=1.0, s=22, alpha=0.85)
    ax.scatter(hole[0], hole[1], hole[2], c="blue", s=200, marker="*", label="peghole (env 0)", zorder=10)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=f"success rate over {reset_records[0]['n_env']} trajectories")
    ax.set_xlabel("x (m, env-local)")
    ax.set_ylabel("y (m, env-local)")
    ax.set_zlabel("z (m, env-local)")
    ax.set_title(
        f"{title_prefix}Per-reset success rate  ({len(reset_records)} resets, mean {rates.mean():.1%}, "
        f"min {rates.min():.0%}, max {rates.max():.0%})"
    )
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    # Save figures into a plots/ subdir of the checkpoint's log directory, named after the
    # checkpoint and iteration (per-iteration filenames so runs don't overwrite each other).
    ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]
    plots_dir = os.path.join(log_dir, "plots", ckpt_name)
    os.makedirs(plots_dir, exist_ok=True)

    # Point the viewport camera at env 0's robot so the video is centered on it (task default is
    # origin_type="world", which sits at the world origin — far from env 0's ~±11m location).
    if args_cli.video and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args_cli.video_envs[0]
        env_cfg.viewer.eye = (2.0, 0.0, 0.75)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    try:
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import describe_assembly_assets

        print(describe_assembly_assets(env_cfg))
    except Exception as exc:  # noqa: BLE001
        print(f"[assembly] could not describe peg/hole assets: {exc}")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")
    env = (HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions) if is_fastsac
           else RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions))

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create runner and load checkpoint
    device = env.unwrapped.device

    if not is_fastsac:
        # PPO / OnPolicyRunner checkpoint: per-reset outcome machinery only; critic diagnostics
        # are zeroed (no distributional Q to compare).
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        _ppo_act = runner.get_inference_policy_stochastic(device=device)
        actor = qnet = qnet_target = obs_normalizer = critic_obs_normalizer = None
        ppo_policy = None
        obs_normalization = False
        print("[eval_critic] OnPolicyRunner checkpoint: critic diagnostics disabled (stochastic inference policy)")
    else:
        _ppo_act = None
        # Eval never samples from the online replay buffer; keep setup() from allocating the full
        # buffer_size x num_envs training buffer (OOM at thousands of envs).
        agent_cfg.buffer_size = 1
        # Eval never steps optimizers; skipping their restore also tolerates checkpoints saved
        # under a different AMP setting (empty GradScaler state fails load_state_dict).
        agent_cfg.reset_optimizers = True
        runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.setup()
        runner.load(resume_path)

        device = env.unwrapped.device

        # Optional PPO expert overlay: pi_ppo(s)'s per-dim Gaussian, plotted alongside the FastSAC
        # actor's action density in the --video composite. Needs its own registered agent config
        # (obs groups, network sizes) to build a matching ActorCritic before loading the checkpoint.
        ppo_policy = None
        if args_cli.ppo_checkpoint:
            ppo_task_name = (args_cli.ppo_task or args_cli.task).split(":")[-1]
            ppo_agent_cfg = load_cfg_from_registry(ppo_task_name, "rsl_rl_cfg_entry_point")
            ppo_agent_cfg = cli_args.sanitize_rsl_rl_cfg(ppo_agent_cfg)
            ppo_resume_path = retrieve_file_path(args_cli.ppo_checkpoint)
            print(f"[INFO]: Loading PPO expert checkpoint from: {ppo_resume_path}")
            ppo_runner = OnPolicyRunner(env, ppo_agent_cfg.to_dict(), log_dir=None, device=device)
            ppo_runner.load(ppo_resume_path)
            ppo_policy = ppo_runner.alg.policy
            ppo_policy.eval()

        # Access the raw pieces so we can (a) stochastically sample from the actor and
        # (b) read the categorical Q-distribution rather than a scalar Q value.
        actor = runner.actor.to(device)
        qnet = runner.qnet.to(device)
        qnet_target = runner.qnet_target.to(device)
        obs_normalizer = runner.obs_normalizer.to(device)
        critic_obs_normalizer = runner.critic_obs_normalizer.to(device)
        actor.eval()
        qnet.eval()
        qnet_target.eval()
        obs_normalizer.eval()
        critic_obs_normalizer.eval()

    actor_obs_keys = agent_cfg.actor_obs_keys if is_fastsac else ["policy"]
    critic_obs_keys = agent_cfg.critic_obs_keys if is_fastsac else ["policy"]

    # Distributional Q-network support (categorical over atoms in [v_min, v_max])
    if is_fastsac:
        v_min = float(qnet.v_min)
        v_max = float(qnet.v_max)
        num_atoms = int(qnet.num_atoms)
        q_support = qnet.q_support.detach().cpu().numpy()  # [num_atoms]
    else:
        v_min, v_max, num_atoms = 0.0, 1.0, 51
        q_support = np.linspace(v_min, v_max, num_atoms)

    edges = np.linspace(v_min, v_max, num_atoms + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = (v_max - v_min) / num_atoms

    GAMMA = args_cli.gamma
    obs_normalization = runner.obs_normalization if is_fastsac else False
    # SAC temperature: the critic predicts SOFT returns (reward plus discounted −α·logπ entropy
    # bonus on future actions), so the MC return must include the same entropy bonus to be comparable.
    alpha = float(runner.log_alpha.exp().detach()) if is_fastsac else 0.0

    import matplotlib.pyplot as plt

    # Per-dim action density (exact, no sampling). If the actor uses tanh squashing, it emits a
    # pre-tanh Normal(mean, std) that gets squashed to [-1,1] then scaled/biased to real actuator
    # units (Actor.get_actions_and_log_probs's Jacobian correction, fast_sac.py) — invert that
    # change-of-variables. Otherwise (this checkpoint: use_tanh=False) the action IS the raw
    # Normal(mean, std), no transform.
    n_act = env.num_actions
    use_tanh = bool(getattr(actor, "use_tanh", True)) if is_fastsac else False
    action_scale = actor.action_scale.detach().cpu().numpy() if is_fastsac else np.ones(n_act)
    action_bias = actor.action_bias.detach().cpu().numpy() if is_fastsac else np.zeros(n_act)
    x_grid_fixed = np.linspace(-50, 50, 800)  # fixed real-unit grid, used directly in the non-tanh case
    _Z_EPS = 1e-4
    z_grid = np.linspace(-1 + _Z_EPS, 1 - _Z_EPS, 200)  # normalized grid, mapped per-dim (tanh case)
    u_grid = np.arctanh(z_grid)
    log_jacobian_z = np.log(1 - z_grid**2)  # log|dz/du| term, shared across dims (scale folded in below)

    def action_dim_grid(d: int) -> np.ndarray:
        # Real-unit x-axis grid for action dim d.
        if not use_tanh:
            return x_grid_fixed
        return action_bias[d] + action_scale[d] * z_grid

    def tanh_normal_pdf(mean_d: float, std_d: float, d: int) -> np.ndarray:
        if not use_tanh:
            # Plain Gaussian: the actor's output IS the action, no squashing.
            return np.exp(-0.5 * ((x_grid_fixed - mean_d) / std_d) ** 2) / (std_d * np.sqrt(2 * np.pi))
        # p_X(x) = p_U(u) / (scale * (1 - z^2)), u = atanh(z), x = bias + scale*z.
        log_pu = -0.5 * ((u_grid - mean_d) / std_d) ** 2 - np.log(std_d) - 0.5 * np.log(2 * np.pi)
        log_px = log_pu - np.log(action_scale[d] + 1e-8) - log_jacobian_z
        return np.exp(log_px)

    def normal_pdf(mean_d: float, std_d: float) -> np.ndarray:
        # PPO overlay: rsl_rl's ActorCritic does not tanh-squash its Gaussian output, so this is
        # a plain Normal(mean, std) over the same fixed x_grid_fixed used for the non-tanh case.
        return np.exp(-0.5 * ((x_grid_fixed - mean_d) / std_d) ** 2) / (std_d * np.sqrt(2 * np.pi))

    n_env = env.num_envs
    print(f"[INFO] Running with num_envs={n_env}, γ={GAMMA}, v_min={v_min}, v_max={v_max}, num_atoms={num_atoms}")

    scene = env.unwrapped.scene

    def peg_local_xy():
        # Peg (insertive_object) x-y in each env's local frame (env-origin subtracted), so all
        # envs overlay on a shared workspace canvas. Returns [n_env, 2] numpy.
        p = scene["insertive_object"].data.root_pos_w[:, :2]
        return (p - scene.env_origins[:, :2]).detach().cpu().numpy()

    def peg_local_xyz():
        # Peg x-y-z in each env's local frame; recorded alongside the x-y history for spread analyses.
        p = scene["insertive_object"].data.root_pos_w[:, :3]
        return (p - scene.env_origins[:, :3]).detach().cpu().numpy()

    def per_env_q(obs_dict, actions):
        # Scalar critic value Q(s,a) per env: expected value of the categorical dist, averaged
        # over the critic ensemble. Returns [n_env].
        if qnet is None:
            return torch.zeros(n_env, device=device)
        co = torch.cat([obs_dict[k] for k in critic_obs_keys], dim=-1)
        nco = critic_obs_normalizer(co, update=False) if obs_normalization else co
        return qnet.get_value(torch.softmax(qnet(nco, actions), dim=-1)).mean(dim=0)

    term_mgr = env.unwrapped.termination_manager
    _term_names = list(term_mgr.active_terms)
    _has_success_term = "success" in _term_names
    _has_abnormal_term = "abnormal_robot" in _term_names
    print(f"[eval_critic] termination terms: {_term_names}")

    # Tasks without a `success` termination (the *-Sparse-*-v0 family) never end an episode on
    # success, so every episode times out and a termination-only rule files them all as failures.
    # play.py handles this by reading ProgressContext's instantaneous success flag each step and
    # counting an episode as a success if ANY step satisfied it. Do the same here.
    # ProgressContext.success is defined as `orientation_aligned & position_aligned`, which is the
    # exact expression play.py accumulates -- and is weaker than the `success` TERMINATION term,
    # which additionally requires the continuous-success counter to reach its threshold.
    try:
        _progress_ctx = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
    except Exception as exc:  # noqa: BLE001 - task simply may not define the term
        _progress_ctx = None
        print(f"[eval_critic] no `progress_context` reward term ({exc}); ever-success unavailable.")
    _has_ever_success = _progress_ctx is not None and hasattr(_progress_ctx, "success")
    if _has_ever_success:
        print("[eval_critic] success = ProgressContext.success ever true during the episode "
              "(matches play.py); timeouts are NOT automatically failures.")
    if not _has_success_term and not _has_ever_success:
        print("[eval_critic] WARNING: no 'success' termination term and no ProgressContext; "
              "nothing will be counted as a success.")

    def accumulate_success(ever_success):
        """OR this step's instantaneous ProgressContext success flag into the per-env episode flag.

        Must be called after EVERY ``env.step`` -- including for envs that terminate on this step,
        whose success would otherwise be lost when the env auto-resets.
        """
        if not _has_ever_success:
            return
        ever_success |= _progress_ctx.success.to(device).bool()

    def classify_terminations(term_type, dones, extras, was_active, ever_success):
        """Bucket envs terminating for the first time this step.

        0 = abnormal robot, 1 = failure, 2 = success.

        Per-term flags come off the TerminationManager rather than being inferred from
        ``time_outs``. An episode counts as a success if EITHER the `success` termination fired
        (tasks that terminate on success) OR the ProgressContext success condition was met at any
        point during the episode (tasks that do not). Without the second clause a task with no
        success termination scores 0% by construction, since every episode necessarily times out.
        """
        newly = was_active & dones.bool()
        if not bool(newly.any()):
            return
        zeros = torch.zeros(n_env, dtype=torch.bool, device=device)
        succ = term_mgr.get_term("success").to(device).bool() if _has_success_term else zeros
        succ = succ | ever_success
        abnormal = term_mgr.get_term("abnormal_robot").to(device).bool() if _has_abnormal_term else zeros
        # Default any new termination to failure, then let the specific causes override. Success is
        # applied last so it wins if it coincides with a timeout or an abnormal flag.
        term_type[newly] = 1
        term_type[newly & abnormal] = 0
        term_type[newly & succ] = 2

    base_seed = args_cli.seed if args_cli.seed is not None else 42

    import json

    plot_shared = {"q_support": q_support, "centers": centers, "bin_width": bin_width,
                   "v_min": v_min, "v_max": v_max, "alpha": alpha}

    # Per-reset summary (initial peg xyz, peghole xyz, outcome counts), rewritten to JSON every
    # iteration so a partial run is still usable; feeds the success-rate scatter after the loop.
    reset_records = []
    reset_records_path = os.path.join(plots_dir, "reset_success_rates.json")

    iteration = 0
    while simulation_app.is_running():
        # ---- 1. Reset every env, then force all envs to share env_0's state ----
        # Re-seed per iteration so s0 is reproducible across checkpoints: iteration i always draws
        # the same reset state for a given --seed, independent of rollout history. Makes Q-vs-MC
        # comparisons between checkpoints apples-to-apples (identical start states per iteration).
        seed_i = base_seed + iteration
        torch.manual_seed(seed_i)
        torch.cuda.manual_seed_all(seed_i)
        obs, _ = env.reset()
        _broadcast_env0_state_to_all(env)
        obs = env.get_observations()

        # Diagnostic: env_0's insertive-object world position at s0 (should be consistent across
        # iterations; a mismatch on iter 0 confirms the unsettled-spawn transient).
        ins_pos0 = env.unwrapped.scene["insertive_object"].data.root_pos_w[0].detach().cpu().numpy()
        print(f"[iter {iteration}] env0 insertive_object pos @ s0: {ins_pos0}")
        init_peg_xyz_all = (
            env.unwrapped.scene["insertive_object"].data.root_pos_w - env.unwrapped.scene.env_origins
        ).detach().cpu().numpy()
        peghole_xyz_all = (
            env.unwrapped.scene["receptive_object"].data.root_pos_w - env.unwrapped.scene.env_origins
        ).detach().cpu().numpy()
        video_envs = args_cli.video_envs
        video_paths = {e: os.path.join(plots_dir, f"iter{iteration:04d}_env{e}.mp4") for e in video_envs}

        # Per-env video frames. The viewport camera is re-pointed at each requested env and
        # re-rendered (no physics stepping) so one rollout yields videos for every requested env.
        vcc = env.unwrapped.viewport_camera_controller if args_cli.video else None

        def capture_video_frames(frames_dict, alive_dict):
            for e in video_envs:
                if not alive_dict[e]:
                    continue
                vcc.set_view_env_index(e)
                frames_dict[e].append(env.unwrapped.render())

        frames = {e: [] for e in video_envs} if args_cli.video else None
        env_alive = {e: True for e in video_envs} if args_cli.video else None
        if args_cli.video:
            capture_video_frames(frames, env_alive)

        # Record peg x-y trajectory per env; terminated envs get NaN so their line stops (no
        # jump to the auto-reset position). Peghole (receptive_object) is shared across envs (broadcast).
        peg_xy_hist = [peg_local_xy()]  # s0 (all envs active, shared start)
        peg_xyz_hist = [peg_local_xyz()]

        # Per-step trajectory buffers for the Q-vs-return-to-go plot: critic value Q(s_t,a_t),
        # reward r_t and entropy bonus (both masked to envs active going into the step), and the
        # active mask. Aligned so index t is "state s_t, action a_t taken there".
        q_traj, rew_traj, ent_traj, act_traj = [], [], [], []
        # Q(s_t, a*_t) where a* is the PPO expert's deterministic action — same critic, alternate
        # action, so we can see how the critic values the expert's choice at each visited state.
        q_ppo_traj = []
        # Per-step actor distribution params (pre-tanh mean/log_std) and the realized action, sliced
        # to the requested video envs only — feeds the per-dim action-density row in the video.
        mean_traj, logstd_traj, action_traj = [], [], []
        # Per-step PPO expert distribution params (plain mean/std, no tanh), sliced to video envs.
        ppo_mean_traj, ppo_std_traj = [], []
        if args_cli.video:
            video_idx = torch.tensor(video_envs, device=device, dtype=torch.long)
        peghole_xy = (
            scene["receptive_object"].data.root_pos_w[0, :2] - scene.env_origins[0, :2]
        ).detach().cpu().numpy()

        # ---- 2. Stochastic action at s0 + record Q distribution ----
        with torch.inference_mode():
            if not is_fastsac:
                actions_0 = _ppo_act(obs)
                q_probs = torch.zeros(1, n_env, num_atoms, device=device)
                q0_scalar = torch.zeros(n_env, device=device)
                mean_0 = log_std_0 = None
            else:
                actor_obs_0 = torch.cat([obs[k] for k in actor_obs_keys], dim=-1)
                critic_obs_0 = torch.cat([obs[k] for k in critic_obs_keys], dim=-1)

                norm_actor_obs_0 = (
                    obs_normalizer(actor_obs_0, update=False) if obs_normalization else actor_obs_0
                )
                norm_critic_obs_0 = (
                    critic_obs_normalizer(critic_obs_0, update=False) if obs_normalization else critic_obs_0
                )

                # Deterministic initial action: the policy mean (tanh-squashed), no sampling noise.
                actions_0, mean_0, log_std_0 = actor(norm_actor_obs_0)
                actions_0 = actions_0.float()
                if args_cli.video:
                    mean_traj.append(mean_0[video_idx].detach())
                    logstd_traj.append(log_std_0[video_idx].detach())
                    action_traj.append(actions_0[video_idx].detach())
                    if ppo_policy is not None:
                        ppo_policy.act(obs)  # populates ppo_policy.action_mean / .action_std
                        ppo_mean_traj.append(ppo_policy.action_mean[video_idx].detach())
                        ppo_std_traj.append(ppo_policy.action_std[video_idx].detach())
                if ppo_policy is not None:
                    a_star_0 = ppo_policy.act_inference(obs)
                    q_ppo_traj.append(per_env_q(obs, a_star_0))

                # Raw distributional logits at (s0, a0): [num_critics, batch, num_atoms]. Keep each
                # critic separate (no ensemble averaging); average only over envs (shared s0).
                q_logits = qnet(norm_critic_obs_0, actions_0)
                q_probs = torch.softmax(q_logits, dim=-1)                                # [num_critics, n_env, num_atoms]
                q0_scalar = qnet.get_value(torch.softmax(q_logits, dim=-1)).mean(dim=0)  # [n_env] Q(s0,a0)

        # Termination bucket per env: -1 not-yet, 0 abnormal, 1 failure, 2 success.
        term_type = torch.full((n_env,), -1, device=device, dtype=torch.long)
        all_active = torch.ones(n_env, dtype=torch.bool, device=device)
        # Sticky per-episode success flag. Each iteration is exactly one episode per env (all envs
        # are reset above and rolled out until every one terminates), so this is cleared per
        # iteration rather than per done.
        ever_success = torch.zeros(n_env, dtype=torch.bool, device=device)

        # ---- 3a. First rollout step (contributes r_0 at γ^0=1; NO entropy bonus on a_0,
        #         matching the soft target which only bonuses future actions a_{t>=1}) ----
        q_traj.append(q0_scalar)                                                # Q(s_0, a_0)
        new_obs, rew, dones, extras = env.step(actions_0)
        accumulate_success(ever_success)

        if not is_fastsac:
            qt_probs = torch.zeros(1, n_env, num_atoms, device=device)
        else:
            # ---- Bootstrapped Bellman target y = r_0 + γ(1-d)[Qtarget(s1,a1) − α·logπ(a1|s1)],
            #      matching FastSACAgent._update_main exactly (fast_sac_agent.py) — NOT the target
            #      network evaluated at (s0, a0), which is what a naive Q(s0,a0) comparison would give.
            with torch.inference_mode():
                actor_obs_1 = torch.cat([new_obs[k] for k in actor_obs_keys], dim=-1)
                critic_obs_1 = torch.cat([new_obs[k] for k in critic_obs_keys], dim=-1)
                norm_actor_obs_1 = (
                    obs_normalizer(actor_obs_1, update=False) if obs_normalization else actor_obs_1
                )
                norm_critic_obs_1 = (
                    critic_obs_normalizer(critic_obs_1, update=False) if obs_normalization else critic_obs_1
                )
                next_actions_1, next_log_probs_1 = actor.get_actions_and_log_probs(norm_actor_obs_1)
                next_actions_1 = next_actions_1.float()
                next_log_probs_1 = next_log_probs_1.float()
                bootstrap_0 = (~dones.bool()).float()
                discount_0 = torch.full((n_env,), GAMMA, device=device, dtype=rew.dtype)
                target_reward_arg = rew - discount_0 * bootstrap_0 * alpha * next_log_probs_1
                target_dist = qnet_target.projection(
                    norm_critic_obs_1, next_actions_1, target_reward_arg, bootstrap_0, discount_0
                )
                qt_probs = target_dist  # [num_critics, n_env, num_atoms]

        classify_terminations(term_type, dones, extras, all_active, ever_success)
        active_mask = ~dones.bool()                                             # envs not yet terminated
        reward_returns = rew.clone()                                           # Σ γ^t r_t, seeded with r_0
        entropy_returns = torch.zeros(n_env, device=device, dtype=rew.dtype)   # Σ_{t>=1} γ^t (−α logπ(a_t|s_t))
        discount = torch.full((n_env,), GAMMA, device=device, dtype=rew.dtype) # γ^1 for the next step
        rew_traj.append(rew.clone())                                           # r_0 (all envs active)
        ent_traj.append(torch.zeros(n_env, device=device, dtype=rew.dtype))    # a_0 deterministic → no entropy
        act_traj.append(all_active.clone())
        obs = new_obs
        xy = peg_local_xy(); xy[(~active_mask).detach().cpu().numpy()] = np.nan
        peg_xy_hist.append(xy)
        peg_xyz_hist.append(np.where(np.isnan(xy[:, :1]), np.nan, peg_local_xyz()))
        if args_cli.video:
            capture_video_frames(frames, env_alive)
            for e in video_envs:
                env_alive[e] = env_alive[e] and not bool(dones[e])

        # ---- 3b. Continue stochastic rollout until every env has terminated ----
        while active_mask.any():
            print(f"{torch.sum(active_mask)} active envs remaining")
            with torch.inference_mode():
                if not is_fastsac:
                    actions = _ppo_act(obs)
                    log_probs = torch.zeros(n_env, device=device)
                    q_t = torch.zeros(n_env, device=device)
                    actor_obs = None
                else:
                    actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=-1)
                if is_fastsac:
                    norm_actor_obs = (
                        obs_normalizer(actor_obs, update=False) if obs_normalization else actor_obs
                    )
                    actions, log_probs = actor.get_actions_and_log_probs(norm_actor_obs)
                    actions = actions.float()
                    log_probs = log_probs.float()
                    q_t = per_env_q(obs, actions)                                   # Q(s_t, a_t)
                if args_cli.video:
                    _, mean_t, log_std_t = actor(norm_actor_obs)
                    mean_traj.append(mean_t[video_idx].detach())
                    logstd_traj.append(log_std_t[video_idx].detach())
                    if ppo_policy is not None:
                        ppo_policy.act(obs)
                        ppo_mean_traj.append(ppo_policy.action_mean[video_idx].detach())
                        ppo_std_traj.append(ppo_policy.action_std[video_idx].detach())
                    action_traj.append(actions[video_idx].detach())
                if ppo_policy is not None:
                    a_star_t = ppo_policy.act_inference(obs)
                    q_ppo_traj.append(per_env_q(obs, a_star_t))
            was_active = active_mask.clone()
            new_obs, rew, dones, extras = env.step(actions)
            accumulate_success(ever_success)
            classify_terminations(term_type, dones, extras, was_active, ever_success)
            # Accumulate reward and entropy bonus separately, both discounted by γ^t and gated to
            # envs still active going into this step. Entropy bonus is −α·logπ(a_t|s_t).
            print(f"Average reward: {torch.mean(rew[active_mask])}")
            entropy_bonus = -alpha * log_probs
            reward_returns = reward_returns + torch.where(active_mask, discount * rew, torch.zeros_like(rew))
            entropy_returns = entropy_returns + torch.where(
                active_mask, discount * entropy_bonus, torch.zeros_like(entropy_bonus)
            )
            # Per-step trajectory records: mask reward/entropy to envs active going into the step.
            zeros = torch.zeros_like(rew)
            q_traj.append(q_t)
            rew_traj.append(torch.where(was_active, rew, zeros))
            ent_traj.append(torch.where(was_active, entropy_bonus, zeros))
            act_traj.append(was_active.clone())
            discount = discount * GAMMA
            active_mask = active_mask & (~dones.bool())
            obs = new_obs
            xy = peg_local_xy(); xy[(~active_mask).detach().cpu().numpy()] = np.nan
            peg_xy_hist.append(xy)
            peg_xyz_hist.append(np.where(np.isnan(xy[:, :1]), np.nan, peg_local_xyz()))
            if args_cli.video:
                capture_video_frames(frames, env_alive)
                for e in video_envs:
                    env_alive[e] = env_alive[e] and not bool(dones[e])

        reward_mc = reward_returns.detach().cpu().numpy()   # [n_env]  discounted reward return
        entropy_mc = entropy_returns.detach().cpu().numpy()  # [n_env]  discounted entropy-bonus return
        soft_mc = reward_mc + entropy_mc                     # [n_env]  total soft return (comparable to Q)

        # ---- 7. Q(s_t,a_t) vs soft return-to-go G_t over the trajectory, a few representative envs ----
        q_arr = torch.stack(q_traj)      # [T, n_env]  critic value at each step
        rew_arr = torch.stack(rew_traj)  # [T, n_env]  reward (masked to active)
        ent_arr = torch.stack(ent_traj)  # [T, n_env]  entropy bonus (masked to active)
        act_arr = torch.stack(act_traj)  # [T, n_env]  active going into step t
        T = q_arr.shape[0]
        # Soft return-to-go: G[t] = r[t] + γ·(ent[t+1] + G[t+1]) — entropy bonus on future actions
        # only, matching the soft Q. rew/ent are already 0 for inactive steps, so this stays correct
        # past each env's termination.
        G = torch.zeros_like(rew_arr)
        running = torch.zeros(n_env, device=device)
        for t in range(T - 1, -1, -1):
            ent_next = ent_arr[t + 1] if t + 1 < T else torch.zeros(n_env, device=device)
            running = rew_arr[t] + GAMMA * (ent_next + running)
            G[t] = running
        q_np = q_arr.cpu().numpy()
        G_np = G.cpu().numpy()
        inactive = ~act_arr.cpu().numpy()
        q_np[inactive] = np.nan  # stop each env's line at its termination
        G_np[inactive] = np.nan

        # Env 0's per-step critic value and return-to-go (unmasked; env 0 is active over its episode).
        ts = np.arange(T)
        q0 = q_arr[:, 0].cpu().numpy()
        g0 = G[:, 0].cpu().numpy()
        if ppo_policy is not None:
            q_ppo_arr = torch.stack(q_ppo_traj)  # [T, n_env]  Q(s_t, a*_ppo) — same critic, expert action
            q_ppo0 = q_ppo_arr[:, 0].cpu().numpy()

        peg_traj = np.stack(peg_xy_hist, axis=0)  # [T, n_env, 2], NaN after each env terminates
        peg_traj_xyz = np.stack(peg_xyz_hist, axis=0)  # [T, n_env, 3], same NaN masking

        # ---- Per-reset analysis: all envs share s0, so bins / plots / records cover every env. ----
        lead = 0
        reset_idx = iteration
        n_g = n_env
        output_path = os.path.join(plots_dir, f"iter{reset_idx:04d}_q_vs_mc.png")
        peg_xy_path = os.path.join(plots_dir, f"iter{reset_idx:04d}_peg_xy.png")
        q_traj_path = os.path.join(plots_dir, f"iter{reset_idx:04d}_q_over_traj.png")
        term_type_g = term_type
        soft_mc_g, reward_mc_g, entropy_mc_g = soft_mc, reward_mc, entropy_mc
        q_dists_g = q_probs.mean(dim=1).cpu().numpy()    # [num_critics, num_atoms]
        qt_dists_g = qt_probs.mean(dim=1).cpu().numpy()  # [num_critics, num_atoms]
        peg_traj_g = peg_traj
        peg_traj_xyz_g = peg_traj_xyz
        init_peg_xyz = init_peg_xyz_all[0]
        peghole_xyz = peghole_xyz_all[0]
        peghole_xy = peghole_xyz[:2]

        # ---- 4. Bin total soft MC return onto the Q support as a proper normalized histogram
        #         (PMF summing to 1), clipping to the support so all envs are counted. ----
        counts, _ = np.histogram(np.clip(soft_mc_g, v_min, v_max), bins=edges)
        total = counts.sum()
        mc_pmf = counts / total if total > 0 else counts.astype(np.float64)

        # Termination breakdown: abnormal robot / timeout-no-success / timeout-success.
        term_counts = [int((term_type_g == k).sum()) for k in (0, 1, 2)]
        reset_records.append({
            "iteration": reset_idx,
            "seed": seed_i,
            "init_peg_xyz": [float(v) for v in init_peg_xyz],
            "peghole_xyz": [float(v) for v in peghole_xyz],
            "n_env": int(n_g),
            "abnormal": term_counts[0],
            "failure": term_counts[1],
            "success": term_counts[2],
            "success_rate": term_counts[2] / n_g,
        })
        with open(reset_records_path, "w") as f:
            json.dump(reset_records, f, indent=1)

        gd = {
            "q_dists": q_dists_g, "qt_dists": qt_dists_g,
            "soft_mc": soft_mc_g, "reward_mc": reward_mc_g, "entropy_mc": entropy_mc_g, "mc_pmf": mc_pmf,
            "term_counts": np.array(term_counts), "term_type": term_type_g.detach().cpu().numpy(),
            "peg_traj": peg_traj_g, "peg_traj_xyz": peg_traj_xyz_g, "peghole_xy": peghole_xy,
            "q_lead": q_arr[:, lead].cpu().numpy(), "g_lead": G[:, lead].cpu().numpy(),
            "act_lead": act_arr[:, lead].cpu().numpy(),
            "q_ppo_lead": q_ppo_arr[:, lead].cpu().numpy() if ppo_policy is not None else None,
            "reset_idx": reset_idx, "n_g": n_g, "lead": lead,
        }
        render_group({**gd, **plot_shared}, output_path, peg_xy_path,
                     None if args_cli.video else q_traj_path,
                     args_cli.peg_xlim, args_cli.peg_ylim, ckpt_name)

        term_counts = [int((term_type == k).sum()) for k in (0, 1, 2)]
        if args_cli.video:
            # ---- 8. Composite video per requested env: render (top) + progressively-revealed
            #         Q-vs-return-to-go chart (middle) + per-dim action-density row (bottom) ----
            import io

            import cv2
            import imageio.v2 as imageio

            # [T, n_video_envs, n_act] actor distribution params + realized action, per requested env.
            mean_arr = torch.stack(mean_traj).cpu().numpy()
            std_arr = np.exp(torch.stack(logstd_traj).cpu().numpy())
            action_arr = torch.stack(action_traj).cpu().numpy()
            x_grids = [action_dim_grid(d) for d in range(n_act)]
            if ppo_policy is not None:
                ppo_mean_arr = torch.stack(ppo_mean_traj).cpu().numpy()  # [T, n_video_envs, n_act]
                ppo_std_arr = torch.stack(ppo_std_traj).cpu().numpy()

            fps = int(env.unwrapped.metadata.get("render_fps", 30))
            for e_pos, e in enumerate(video_envs):
                e_frames = frames[e]
                if len(e_frames) <= 1:
                    continue
                qe = q_arr[:, e].cpu().numpy()
                ge = G[:, e].cpu().numpy()
                mean_e = mean_arr[:, e_pos, :]
                std_e = std_arr[:, e_pos, :]
                action_e = action_arr[:, e_pos, :]
                if ppo_policy is not None:
                    ppo_mean_e = ppo_mean_arr[:, e_pos, :]
                    ppo_std_e = ppo_std_arr[:, e_pos, :]

                sim_h, sim_w = e_frames[0].shape[:2]
                chart_h = sim_h // 2
                dist_row_h = 60  # px per stacked action-dim row
                dist_h = dist_row_h * n_act
                dpi = 100

                fig3 = plt.figure(figsize=(sim_w / dpi, chart_h / dpi), dpi=dpi)
                ax3 = fig3.add_subplot(111)
                ax3.set_xlim(0, max(1, T - 1))
                ax3.set_ylim(0, v_max)
                ax3.set_xlabel("timestep t")
                ax3.set_ylabel("value")
                line_q, = ax3.plot([], [], color="tab:blue", lw=2.0, label="Q(s_t, a_t)")
                line_g, = ax3.plot([], [], color="tab:orange", lw=2.0, ls="--", label="return-to-go G_t")
                dot_q, = ax3.plot([], [], "o", color="tab:blue", ms=5)
                dot_g, = ax3.plot([], [], "o", color="tab:orange", ms=5)
                if ppo_policy is not None:
                    q_ppo_e = q_ppo_arr[:, e].cpu().numpy()
                    line_qppo, = ax3.plot([], [], color="tab:green", lw=2.0, ls=":", label="Q(s_t, a*_ppo)")
                    dot_qppo, = ax3.plot([], [], "o", color="tab:green", ms=5)
                ax3.set_title(f"env {e}")
                ax3.legend(loc="upper right", fontsize=8)
                fig3.tight_layout()

                # Stacked per-dim action-density plots (one row per action dim): tanh-Normal PDF
                # over real action units, with a vertical marker at the action actually taken.
                fig4 = plt.figure(figsize=(sim_w / dpi, dist_h / dpi), dpi=dpi)
                axes4 = np.atleast_1d(fig4.subplots(n_act, 1, sharex=True))
                pdf_lines, action_markers = [], []
                ppo_pdf_lines = []
                for d, axd in enumerate(axes4):
                    line, = axd.plot(
                        x_grids[d], np.zeros_like(x_grids[d]), color="tab:blue", lw=1.2,
                        label="pi_fastsac" if d == 0 and ppo_policy is not None else None,
                    )
                    marker = axd.axvline(x_grids[d][len(x_grids[d]) // 2], color="k", ls=":", lw=1.0)
                    if ppo_policy is not None:
                        ppo_line, = axd.plot(
                            x_grid_fixed, np.zeros_like(x_grid_fixed), color="tab:green", lw=1.2, ls="--",
                            label="pi_ppo" if d == 0 else None,
                        )
                        ppo_pdf_lines.append(ppo_line)
                    axd.set_xlim(-50, 50)
                    axd.set_ylim(0, 1)
                    axd.set_ylabel(f"a[{d}]", fontsize=7)
                    axd.tick_params(labelsize=6)
                    pdf_lines.append(line)
                    action_markers.append(marker)
                axes4[-1].set_xlabel("action", fontsize=7)
                if ppo_policy is not None:
                    axes4[0].legend(fontsize=6, loc="upper right")
                fig4.tight_layout()

                composite = []
                for i in range(len(e_frames)):
                    end = min(i + 1, T)
                    line_q.set_data(ts[:end], qe[:end])
                    line_g.set_data(ts[:end], ge[:end])
                    dot_q.set_data([end - 1], [qe[end - 1]])
                    dot_g.set_data([end - 1], [ge[end - 1]])
                    if ppo_policy is not None:
                        line_qppo.set_data(ts[:end], q_ppo_e[:end])
                        dot_qppo.set_data([end - 1], [q_ppo_e[end - 1]])
                    buf = io.BytesIO()
                    fig3.savefig(buf, format="png", dpi=dpi)
                    buf.seek(0)
                    chart = imageio.imread(buf)[..., :3]  # [h, w, 3], backend-independent
                    chart = cv2.resize(chart, (sim_w, chart_h))

                    t_i = min(i, mean_e.shape[0] - 1)
                    for d, axd in enumerate(axes4):
                        pdf_y = tanh_normal_pdf(mean_e[t_i, d], std_e[t_i, d], d)
                        pdf_lines[d].set_data(x_grids[d], pdf_y)
                        if ppo_policy is not None:
                            ppo_pdf_y = normal_pdf(ppo_mean_e[t_i, d], ppo_std_e[t_i, d])
                            ppo_pdf_lines[d].set_data(x_grid_fixed, ppo_pdf_y)
                        action_markers[d].set_xdata([action_e[t_i, d], action_e[t_i, d]])
                    buf2 = io.BytesIO()
                    fig4.savefig(buf2, format="png", dpi=dpi)
                    buf2.seek(0)
                    dist_img = imageio.imread(buf2)[..., :3]
                    dist_img = cv2.resize(dist_img, (sim_w, dist_h))

                    composite.append(np.vstack([e_frames[i], chart, dist_img]))
                plt.close(fig3)
                plt.close(fig4)

                imageio.mimsave(video_paths[e], composite, fps=fps)
                print(f"[iter {iteration:04d}] saved composite env {e} video ({len(composite)} frames) → {video_paths[e]}")

        q_exp_online = float(np.mean([np.sum(q_probs[c].mean(dim=0).cpu().numpy() * q_support) for c in range(q_probs.shape[0])]))
        video_summary = ", ".join(str(p) for p in video_paths.values()) if args_cli.video else q_traj_path
        print(
            f"[iter {iteration:04d}] soft MC mean={soft_mc.mean():.4f}  reward mean={reward_mc.mean():.4f}  "
            f"entropy mean={entropy_mc.mean():.4f}  E[Q online mean]={q_exp_online:.4f}  "
            f"term[abnormal/fail/success]={term_counts}  → {output_path}, {peg_xy_path}, "
            f"{video_summary}"
        )
        iteration += 1


    if reset_records:
        out_path = os.path.join(plots_dir, "reset_success_rate_xyz.png")
        save_reset_scatter(reset_records, out_path)
        print(f"[eval_critic] Saved per-reset success-rate scatter to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
