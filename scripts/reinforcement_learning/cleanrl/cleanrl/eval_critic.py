# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace a CleanRL/FlashSAC critic along an expert trajectory.

A PPO expert checkpoint drives the episode; at every visited state we record two values and plot
them against each other over the trajectory:

  * ``Q(s_t, a_t)``  -- the CleanRL FlashSAC critic's estimate of the expert's own action, read as
    the expectation of its C51 categorical distribution over the atom support.
  * ``G_t``          -- the realized discounted return-to-go, computed backwards from actual rewards.

The point is to see *where* along a good trajectory the learned critic disagrees with reality: a
critic that is flat, saturated, or mistimed relative to G_t shows up immediately here in a way that
an aggregate success rate does not.

Two things to keep in mind when reading the plot:

  * The FlashSAC critic is a **soft** Q -- it predicts reward plus a discounted -alpha*log(pi)
    entropy bonus over its *own* policy's future actions, whereas G_t here is a plain reward return
    under the *expert's* actions. A roughly constant positive offset of order alpha * E[sum
    gamma^t H] is therefore expected and is not by itself a bug. The script prints alpha so the
    magnitude can be judged; see ``holosoma/eval_critic.py`` for the entropy-corrected variant.
  * Episodes cut short by the time limit yield a truncated G_t near the end of the trace (the tail
    of the return is simply missing), so the last few steps of a timed-out episode read low.

The env must expose the expert's state ``policy``/``critic`` groups *and* the group the CleanRL
critic was trained on, which is why this defaults to a DataCollection task rather than a Play one:
the Train/Play variants drop the expert's state groups. In the asymmetric grayscale task the
CleanRL critic's group is called ``critic_no_priv`` (there, ``critic`` is the privileged group the
expert itself was built against), hence ``--critic_obs_key``.

Example:

    python eval_critic.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-DataCollection-FastRender-v0 \
        --checkpoint /path/to/ppo/model_1500.pt \
        --cleanrl_checkpoint checkpoints/MRQ-GS-Asym-NU5-BS1024-L40S/model_final.pt \
        --num_envs 8 --enable_cameras --headless

Add ``--video`` to also write an mp4 of the rollout per ``--video_envs`` entry, named after
``--output_path`` (``critic_vs_returns_env0.mp4``). Each recorded env costs one extra viewport
render per step, and every video ends at its own env's termination.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Trace a CleanRL/FlashSAC critic along a PPO expert trajectory.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-DataCollection-FastRender-v0",
    help="Name of the task. Must expose the expert's state groups and the CleanRL critic's group.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--cleanrl_checkpoint",
    type=str,
    required=True,
    help="model_*.pt written by one of the mrq_/flashsac_ CleanRL scripts; supplies the critic under test.",
)
parser.add_argument(
    "--critic_obs_key",
    type=str,
    default="critic_no_priv",
    help=(
        "Observation group feeding the CleanRL critic. This is whatever the training task exposed as "
        "'critic'; on the asymmetric collection task that group is named 'critic_no_priv'."
    ),
)
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for the return-to-go.")
parser.add_argument(
    "--max_steps", type=int, default=2000, help="Safety cap on rollout length if some env never terminates."
)
parser.add_argument(
    "--plot_envs", type=int, nargs="+", default=[0], metavar="ENV_ID", help="Env indices to draw one panel each for."
)
parser.add_argument("--output_path", type=str, default="critic_vs_returns.png", help="Output path for the plot.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Also save an mp4 of the rollout per --video_envs entry."
)
parser.add_argument(
    "--video_envs",
    type=int,
    nargs="+",
    default=[0],
    metavar="ENV_ID",
    help=(
        "Which env indices to record when --video is set. The viewport camera is re-pointed at each "
        "one and re-rendered every step, so each extra env costs another render per step."
    ),
)
parser.add_argument(
    "--video_fps", type=float, default=None, help="Video frame rate. Defaults to the env's control rate (1/step_dt)."
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# env.render() only returns frames when the rendering pipeline is up.
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import numpy as np
import torch

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

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cleanrl_utils.utils import EmpiricalNormalization

# Network definitions only; the module keeps its training loop behind __main__.
import mrq_flashsac_continuous_action_vision_asymmetric as mrq

# PLACEHOLDER: Extension template (do not remove this comment)


def load_cleanrl_critic(ckpt, device):
    """Rebuild the FlashSAC C51 critic and its obs normalizer from a CleanRL checkpoint.

    Widths come from the saved tensors rather than the training args so this also accepts
    checkpoints written before a hyperparameter was added. ``v_min``/``v_max`` are the one thing
    that cannot be recovered -- ``q_support`` is a plain attribute, not a registered buffer, so it
    never enters the state dict -- and fall back to the class defaults.
    """
    qsd = ckpt["qf1"]
    # UnitLinear wraps an inner nn.Linear, hence the ".linear." in these keys.
    action_dim = ckpt["actor"]["fc_mu.linear.weight"].shape[0]
    critic_obs_dim = qsd["norm1.weight"].shape[0] - action_dim  # norm1 spans [critic_obs | action]
    hidden_dim = qsd["linear1.linear.weight"].shape[0]
    num_atoms = qsd["linear2.linear.weight"].shape[0]

    qf1 = mrq.FlashSACQNetwork(
        critic_obs_dim, action_dim, hidden_dim=hidden_dim, num_atoms=num_atoms,
        num_blocks=mrq.infer_num_blocks(qsd, default=ckpt["args"].get("critic_num_blocks", 2)),
    ).to(device)
    qf1.load_state_dict(qsd)
    qf1.eval()

    critic_norm = None
    if ckpt["args"].get("obs_normalization", False):
        critic_norm = EmpiricalNormalization(shape=(critic_obs_dim,), device=device)
        critic_norm.load_state_dict(ckpt["critic_obs_normalizer"])
        critic_norm.eval()

    return qf1, critic_norm, critic_obs_dim, action_dim


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out the PPO expert and trace the CleanRL critic against realized returns."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Point the viewport at the robot of the first recorded env; the task default sits at the world
    # origin, metres away from any individual env.
    if args_cli.video and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args_cli.video_envs[0]
        env_cfg.viewer.eye = (2.0, 0.0, 0.75)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    print(f"[INFO]: Loading PPO expert checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(resume_path)
    ppo_policy = ppo_runner.alg.policy
    ppo_policy.eval()

    print(f"[INFO]: Loading CleanRL critic from: {args_cli.cleanrl_checkpoint}")
    ckpt = torch.load(args_cli.cleanrl_checkpoint, map_location=device, weights_only=False)
    qf1, critic_norm, critic_obs_dim, action_dim = load_cleanrl_critic(ckpt, device)
    print(
        f"[critic] trained on '{ckpt['args'].get('env_id')}' at global_step={ckpt.get('global_step')} | "
        f"obs {critic_obs_dim}-d, {action_dim} actions, {qf1.num_atoms} atoms on "
        f"[{qf1.v_min}, {qf1.v_max}]"
    )
    if "log_alpha" in ckpt:
        print(
            f"[critic] SAC alpha = {float(ckpt['log_alpha'].exp()):.5f} -- this critic is a SOFT Q, so it "
            "should sit ABOVE the plain reward return-to-go by roughly alpha * E[sum gamma^t H]."
        )

    n_env = env.unwrapped.num_envs
    GAMMA = args_cli.gamma

    obs, _ = env.reset()
    if args_cli.critic_obs_key not in obs.keys():
        raise KeyError(
            f"Task '{args_cli.task}' has no observation group '{args_cli.critic_obs_key}'. "
            f"Available: {sorted(obs.keys())}. Pass --critic_obs_key with the group matching the one "
            f"the CleanRL run trained its critic on."
        )
    got = int(np.prod(obs[args_cli.critic_obs_key].shape[1:]))
    if got != critic_obs_dim:
        raise ValueError(
            f"Observation group '{args_cli.critic_obs_key}' is {got}-d but the critic expects "
            f"{critic_obs_dim}-d. Wrong group or wrong task."
        )

    def critic_q(obs_dict, actions):
        """Scalar Q(s, a): expectation of the C51 categorical over the atom support. [n_env]"""
        co = obs_dict[args_cli.critic_obs_key].reshape(n_env, -1)
        if critic_norm is not None:
            co = critic_norm(co, update=False)
        probs = torch.softmax(qf1(co, actions, training=False), dim=1)
        return (probs * qf1.q_support.to(probs.device)).sum(dim=1)

    # Per-step traces, index t == "state s_t, action a_t taken there". Rewards are gated by
    # `active` so an env that has already terminated contributes nothing to its return-to-go.
    q_traj, rew_traj, act_traj = [], [], []
    active = torch.ones(n_env, dtype=torch.bool, device=device)

    video_envs = [e for e in args_cli.video_envs if 0 <= e < n_env] if args_cli.video else []
    vcc = getattr(env.unwrapped, "viewport_camera_controller", None) if args_cli.video else None
    if args_cli.video and vcc is None:
        # Isaac only builds the controller at PARTIAL_RENDERING or above; without it render() has
        # no viewport to re-point. Say so now rather than dying mid-rollout after a long startup.
        print("[video] WARNING: no viewport camera controller (rendering is off) -- skipping video.")
        video_envs = []
    frames = {e: [] for e in video_envs}

    def capture_frames():
        """Re-point the viewport at each recorded env and grab a frame -- no extra physics stepping.

        Envs that have already terminated are skipped so each video ends at its own episode
        boundary instead of running on through the auto-reset.
        """
        for e in video_envs:
            if not bool(active[e]):
                continue
            vcc.set_view_env_index(e)
            frames[e].append(env.unwrapped.render())

    capture_frames()  # s0, before the first action

    step = 0
    while bool(active.any()) and step < args_cli.max_steps:
        with torch.inference_mode():
            actions = ppo_policy.act_inference(obs)
            q_t = critic_q(obs, actions)

        q_traj.append(q_t)
        act_traj.append(active.clone())

        obs, rew, dones, extras = env.step(actions)
        rew_traj.append(torch.where(active, rew.reshape(-1), torch.zeros_like(rew.reshape(-1))))

        # Freeze each env at its first termination; auto-reset means it would otherwise start a
        # fresh episode mid-trace and corrupt both the return-to-go and the plot.
        active = active & ~dones.bool().reshape(-1)
        step += 1

        if video_envs:
            capture_frames()

    T = len(q_traj)
    q_arr = torch.stack(q_traj)      # [T, n_env]
    rew_arr = torch.stack(rew_traj)  # [T, n_env]
    act_arr = torch.stack(act_traj)  # [T, n_env]

    # Return-to-go: G[t] = r[t] + gamma * G[t+1]. Rewards past termination are already zero.
    G = torch.zeros_like(rew_arr)
    running = torch.zeros(n_env, device=device)
    for t in range(T - 1, -1, -1):
        running = rew_arr[t] + GAMMA * running
        G[t] = running

    inactive = ~act_arr.cpu().numpy()
    q_np, g_np = q_arr.cpu().numpy(), G.cpu().numpy()
    for arr in (q_np, g_np):
        arr[inactive] = np.nan  # stop each env's line at its own termination

    ep_len = act_arr.sum(dim=0).cpu().numpy()
    print(f"[rollout] T={T} steps | episode lengths per env: {ep_len.tolist()}")

    if video_envs:
        import imageio

        # Default to the control rate so the video plays back in real time.
        fps = args_cli.video_fps
        if fps is None:
            step_dt = getattr(env.unwrapped, "step_dt", None)
            fps = (1.0 / step_dt) if step_dt else 30.0
        stem = os.path.splitext(args_cli.output_path)[0]
        for e in video_envs:
            if not frames[e]:
                print(f"[video] env {e}: no frames captured, skipping.")
                continue
            path = f"{stem}_env{e}.mp4"
            imageio.mimsave(path, frames[e], fps=fps)
            print(f"[video] env {e}: {len(frames[e])} frames @ {fps:.1f} fps -> {path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_envs = [e for e in args_cli.plot_envs if 0 <= e < n_env]
    fig, axes = plt.subplots(len(plot_envs), 1, figsize=(11, 4.2 * len(plot_envs)), squeeze=False)
    ts = np.arange(T)
    for ax, e in zip(axes[:, 0], plot_envs):
        ax.plot(ts, q_np[:, e], color="tab:blue", lw=2.0, label="Q(s_t, a_t)  CleanRL/FlashSAC critic")
        ax.plot(ts, g_np[:, e], color="tab:orange", lw=2.0, ls="--", label="G_t  realized discounted return-to-go")
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_xlabel("timestep t")
        ax.set_ylabel("value")
        ax.set_title(f"env {e}  (episode length {int(ep_len[e])}, G_0 = {g_np[0, e]:.3f}, Q_0 = {q_np[0, e]:.3f})")
        ax.legend(fontsize=9)
    fig.suptitle(
        f"CleanRL critic vs realized returns under the PPO expert  (gamma={GAMMA})", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(args_cli.output_path, dpi=110)
    plt.close(fig)

    # Aggregate fit over every env-step where the return-to-go is complete (i.e. the env had
    # already terminated by the end of the trace, so nothing is missing from its tail).
    # Use the post-loop mask, not act_arr[-1]: act_arr[t] records who was active *going into* step
    # t, so an env that terminated on the final step still shows True there and would be wrongly
    # treated as having an incomplete tail.
    complete = ~active.cpu().numpy()  # envs that actually terminated within the rollout
    if complete.any():
        m = np.isfinite(q_np) & np.isfinite(g_np)
        m &= complete[None, :]
        n_steps = int(m.sum())
        # Offset and correlation are worth reading separately: a large mean error with a high
        # correlation means the critic tracks the trajectory but sits on a different scale, whereas
        # low correlation is the case that actually indicts the network.
        err = q_np[m] - g_np[m]
        corr = np.corrcoef(q_np[m], g_np[m])[0, 1] if n_steps > 1 else float("nan")
        print(
            f"[fit] over {n_steps} steps from {int(complete.sum())} completed episodes: "
            f"mean(Q - G) = {err.mean():+.4f}, MAE = {np.abs(err).mean():.4f}, "
            f"corr(Q, G) = {corr:+.3f}"
        )
    else:
        print("[fit] no env terminated within --max_steps; skipping aggregate fit stats.")

    print(f"[done] wrote {args_cli.output_path}")

    # Detach the viewport from the robot before teardown. "asset_root" tracking installs a
    # per-frame callback that reads env.scene, which env.close() deletes -- leaving it attached
    # raises AttributeError from the callback on the way out, after the run has already finished.
    if vcc is not None:
        vcc.update_view_to_world()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
