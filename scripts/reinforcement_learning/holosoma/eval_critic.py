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

parser.add_argument(
    "--group_size", type=int, default=None,
    help=(
        "Envs per reset group. Each iteration resets all envs, then broadcasts every group leader's "
        "(env g*group_size) state to its group, so one rollout evaluates num_envs/group_size distinct "
        "resets. Default: all envs share env_0's state (one reset per iteration)."
    ),
)
parser.add_argument(
    "--distributed", action="store_true", default=False,
    help="Accepted for cluster launcher compatibility (run_singularity.sh appends it); single-process eval ignores it.",
)
parser.add_argument(
    "--sweep", type=str, default=None,
    help=(
        "In-process sweep of one event-term param, e.g. "
        "env.events.dynamics_gap.params.peg_mass=0.5,0.6,0.7 : each value runs the full set of "
        "iterations (same seeds, so identical resets per config) into plots/<ckpt>/<param>_<value>/, "
        "then the per-config success-rate scatters are concatenated side by side."
    ),
)
parser.add_argument(
    "--sweep_level", type=str, action="append", default=None, metavar="LABEL|KEY=VAL|KEY=VAL...",
    help=(
        "General in-process sweep: one level per flag, 'LABEL|key=value|key=value...'. Keys are "
        "env.events.<term>.params.<p> or env.observations.<group>.<term>.params.<p>; values are "
        "floats or [a,b,c] lists. A level with no assignments is the in-distribution reference. "
        "Params touched by earlier levels are restored to their defaults before each level."
    ),
)
parser.add_argument(
    "--defer_plots", action="store_true", default=False,
    help=(
        "Skip the per-reset figures during the run; dump each rollout's arrays to rollout####.npz "
        "instead and render later with render_eval_critic_plots.py. JSON + final scatter still written."
    ),
)
parser.add_argument(
    "--policy_autocast", type=str, choices=["off", "bf16", "fp16"], default="off",
    help="Run actor-obs normalization + actor forward under torch.autocast (mirrors FastSAC training "
         "rollout); actions are cast back to fp32. Default off = fp32.",
)
parser.add_argument(
    "--deterministic", action="store_true", default=False,
    help="Roll out with the policy mean (actor.explore(deterministic=True)) instead of sampling; "
         "entropy bonuses are zero so soft-MC == reward-MC.",
)
parser.add_argument(
    "--num_iterations", type=int, default=None,
    help="Stop after this many reset iterations (default: run until the sim app is stopped).",
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


def _broadcast_group_states(env_wrapper, group_size: int) -> None:
    """Set every env to its group leader's configuration (env ``g*group_size`` for group ``g``).

    ``group_size == num_envs`` reduces to broadcasting env_0 to all envs.
    """
    unwrapped = env_wrapper.unwrapped
    n = unwrapped.scene.num_envs
    idx = (torch.arange(n, device=unwrapped.device) // group_size) * group_size
    state = unwrapped.scene.get_state(is_relative=True)

    def gather_leaders(node):
        if isinstance(node, dict):
            return {k: gather_leaders(v) for k, v in node.items()}
        return node[idx.to(node.device)].contiguous()

    unwrapped.reset_to(gather_leaders(state), env_ids=None, is_relative=True)


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
        _ppo_act = (runner.get_inference_policy(device=device) if args_cli.deterministic
                    else runner.get_inference_policy_stochastic(device=device))
        actor = qnet = qnet_target = obs_normalizer = critic_obs_normalizer = None
        ppo_policy = None
        obs_normalization = False
        print("[eval_critic] OnPolicyRunner checkpoint: critic diagnostics disabled "
              f"({'deterministic' if args_cli.deterministic else 'stochastic'} inference policy)")
    else:
        _ppo_act = None
        # Eval never samples from the online replay buffer; keep setup() from allocating the full
        # buffer_size x num_envs training buffer (OOM at thousands of envs).
        agent_cfg.buffer_size = 1
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

    import contextlib

    def _amp():
        if args_cli.policy_autocast == "off":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda",
                              dtype=torch.bfloat16 if args_cli.policy_autocast == "bf16" else torch.float16)

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
    group_size = args_cli.group_size or n_env
    if n_env % group_size != 0:
        raise ValueError(f"--num_envs ({n_env}) must be a multiple of --group_size ({group_size})")
    n_groups = n_env // group_size
    print(f"[eval_critic] {n_groups} reset group(s) of {group_size} envs per iteration")
    print(f"[INFO] Running with num_envs={n_env}, γ={GAMMA}, v_min={v_min}, v_max={v_max}, num_atoms={num_atoms}")

    # Warm up physics before measuring: the very first reset after app launch leaves assets in
    # their raw spawn pose (not yet physics-settled). _broadcast_env0_state_to_all would copy that
    # unsettled env_0 pose to every env, contaminating the first episode's rewards. Step the policy
    # briefly and discard, so the first *measured* reset produces a settled env_0 state.
    # warm_obs, _ = env.reset()
    # for _ in range(30):
    #     with torch.inference_mode():
    #         wa = torch.cat([warm_obs[k] for k in actor_obs_keys], dim=-1)
    #         wa = obs_normalizer(wa, update=False) if obs_normalization else wa
    #         wa, _ = actor.get_actions_and_log_probs(wa)
    #     warm_obs, _, _, _ = env.step(wa)

    scene = env.unwrapped.scene

    def peg_local_xy():
        # Peg (insertive_object) x-y in each env's local frame (env-origin subtracted), so all
        # envs overlay on a shared workspace canvas. Returns [n_env, 2] numpy.
        p = scene["insertive_object"].data.root_pos_w[:, :2]
        return (p - scene.env_origins[:, :2]).detach().cpu().numpy()

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
    import time
    from eval_critic_plots import concat_labeled, render_group, save_reset_scatter

    plot_shared = {"q_support": q_support, "centers": centers, "bin_width": bin_width,
                   "v_min": v_min, "v_max": v_max, "alpha": alpha}

    # In-process sweeps: one Isaac instance, event/observation term params mutated between configs.
    # Both managers re-read term_cfg.params on every call, so a new value applies from the next
    # reset/step; the per-iteration seeding below makes every config see the same reset states.
    def _term_cfg(key: str):
        parts = key.split(".")
        if len(parts) == 5 and parts[:2] == ["env", "events"] and parts[3] == "params":
            return env.unwrapped.event_manager.get_term_cfg(parts[2])
        if len(parts) == 6 and parts[:2] == ["env", "observations"] and parts[4] == "params":
            om = env.unwrapped.observation_manager
            return om._group_obs_term_cfgs[parts[2]][om.active_terms[parts[2]].index(parts[3])]
        raise ValueError(f"unsupported sweep key {key!r}: expected env.events.<term>.params.<p> or "
                         "env.observations.<group>.<term>.params.<p>")

    def _rebuild_term(key: str):
        # Class-based terms (ManagerTermBase, e.g. apply_dynamics_gap) read cfg.params in __init__
        # and ignore later mutations, so re-instantiate them; plain functions get **params per call.
        from isaaclab.managers import ManagerTermBase

        cfg = _term_cfg(key)
        if isinstance(cfg.func, ManagerTermBase):
            cfg.func = type(cfg.func)(cfg, env.unwrapped)

    def _param_target(key: str):
        parts = key.split(".")
        if len(parts) == 5 and parts[:2] == ["env", "events"] and parts[3] == "params":
            return env.unwrapped.event_manager.get_term_cfg(parts[2]).params, parts[4]
        if len(parts) == 6 and parts[:2] == ["env", "observations"] and parts[4] == "params":
            om = env.unwrapped.observation_manager
            names = om.active_terms[parts[2]]
            return om._group_obs_term_cfgs[parts[2]][names.index(parts[3])].params, parts[5]
        raise ValueError(f"unsupported sweep key {key!r}: expected env.events.<term>.params.<p> or "
                         "env.observations.<group>.<term>.params.<p>")

    def _parse_value(text: str):
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            return [float(v) for v in text[1:-1].split(",") if v.strip()]
        return float(text)

    sweep_param = None
    levels: list[tuple[str | None, list[tuple[str, object]]]] = [(None, [])]
    if args_cli.sweep and args_cli.sweep_level:
        raise ValueError("use either --sweep or --sweep_level, not both")
    if args_cli.sweep:
        key, _, vals = args_cli.sweep.partition("=")
        if not vals:
            raise ValueError(f"--sweep expects <key>=v1,v2,... ; got {args_cli.sweep!r}")
        _param_target(key)  # validates the key / term now
        sweep_param = key.split(".")[-1]
        levels = [(f"{sweep_param.upper()} {float(v):g}", [(key, float(v))]) for v in vals.split(",")]
    elif args_cli.sweep_level:
        levels = []
        for spec in args_cli.sweep_level:
            label, *assigns = spec.split("|")
            pairs = []
            for a in assigns:
                k, _, v = a.partition("=")
                _param_target(k)
                pairs.append((k, _parse_value(v)))
            levels.append((label.strip(), pairs))
        sweep_param = "level"
    if len(levels) > 1 or levels[0][0] is not None:
        print(f"[eval_critic] sweeping {len(levels)} level(s) in-process: {[lv[0] for lv in levels]}")

    base_plots_dir = plots_dir
    scatter_paths, scatter_labels = [], []
    param_defaults: dict[str, object] = {}
    for level_label, assignments in levels:
        cfg_tag = ""
        plots_dir = base_plots_dir
        if level_label is not None:
            # restore everything touched so far, then apply this level's assignments
            for key, default in param_defaults.items():
                params, name = _param_target(key)
                params[name] = default
            for key, value in assignments:
                params, name = _param_target(key)
                param_defaults.setdefault(key, params[name])
                params[name] = value
            for key in {k for k in param_defaults}:
                _rebuild_term(key)
            cfg_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in level_label).strip("_")
            plots_dir = os.path.join(base_plots_dir, cfg_tag)
            os.makedirs(plots_dir, exist_ok=True)
            print(f"[eval_critic] === config {level_label!r} {dict(assignments)}: outputs -> {plots_dir}")
            with open(os.path.join(plots_dir, "level.json"), "w") as f:
                json.dump({"label": level_label, "assignments": assignments, "group_size": group_size}, f, indent=1)

        # Per-reset summary (initial peg xyz, peghole xyz, outcome counts), rewritten to JSON every
        # iteration so a partial run is still usable; feeds the success-rate scatter after the loop.
        reset_records = []
        reset_records_path = os.path.join(plots_dir, "reset_success_rates.json")

        iteration = 0
        while simulation_app.is_running() and (args_cli.num_iterations is None or iteration < args_cli.num_iterations):
            t_iter0 = time.time()
            # ---- 1. Reset every env, then force all envs to share env_0's state ----
            # Re-seed per iteration so s0 is reproducible across checkpoints: iteration i always draws
            # the same reset state for a given --seed, independent of rollout history. Makes Q-vs-MC
            # comparisons between checkpoints apples-to-apples (identical start states per iteration).
            seed_i = base_seed + iteration
            torch.manual_seed(seed_i)
            torch.cuda.manual_seed_all(seed_i)
            obs, _ = env.reset()
            _broadcast_group_states(env, group_size)
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

                    with _amp():
                        norm_actor_obs_0 = (
                            obs_normalizer(actor_obs_0, update=False) if obs_normalization else actor_obs_0
                        )
                    norm_critic_obs_0 = (
                        critic_obs_normalizer(critic_obs_0, update=False) if obs_normalization else critic_obs_0
                    )

                    # Deterministic initial action: the policy mean (tanh-squashed), no sampling noise.
                    with _amp():
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
                    with _amp():
                        norm_actor_obs_1 = (
                            obs_normalizer(actor_obs_1, update=False) if obs_normalization else actor_obs_1
                        )
                    norm_critic_obs_1 = (
                        critic_obs_normalizer(critic_obs_1, update=False) if obs_normalization else critic_obs_1
                    )
                    with _amp():
                        if args_cli.deterministic:
                            next_actions_1 = actor.explore(norm_actor_obs_1, deterministic=True)
                            next_log_probs_1 = torch.zeros(n_env, device=device)
                        else:
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
                        with _amp():
                            norm_actor_obs = (
                                obs_normalizer(actor_obs, update=False) if obs_normalization else actor_obs
                            )
                            if args_cli.deterministic:
                                actions = actor.explore(norm_actor_obs, deterministic=True)
                                log_probs = torch.zeros(n_env, device=device)
                            else:
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
                if args_cli.video:
                    capture_video_frames(frames, env_alive)
                    for e in video_envs:
                        env_alive[e] = env_alive[e] and not bool(dones[e])

            t_rollout = time.time() - t_iter0
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

            t_plots0 = time.time()
            # ---- Per-reset analysis: each group shares one s0, so bins / plots / records are per group. ----
            rollout_dump = {}
            for g in range(n_groups):
                lead = g * group_size
                sl = slice(lead, lead + group_size)
                reset_idx = iteration * n_groups + g
                n_g = group_size
                output_path = os.path.join(plots_dir, f"iter{reset_idx:04d}_q_vs_mc.png")
                peg_xy_path = os.path.join(plots_dir, f"iter{reset_idx:04d}_peg_xy.png")
                q_traj_path = os.path.join(plots_dir, f"iter{reset_idx:04d}_q_over_traj.png")
                term_type_g = term_type[sl]
                soft_mc_g, reward_mc_g, entropy_mc_g = soft_mc[sl], reward_mc[sl], entropy_mc[sl]
                q_dists_g = q_probs[:, sl].mean(dim=1).cpu().numpy()    # [num_critics, num_atoms]
                qt_dists_g = qt_probs[:, sl].mean(dim=1).cpu().numpy()  # [num_critics, num_atoms]
                peg_traj_g = peg_traj[:, sl]
                init_peg_xyz = init_peg_xyz_all[lead]
                peghole_xyz = peghole_xyz_all[lead]
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
                    "rollout": iteration,
                    "group": g,
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
                    "peg_traj": peg_traj_g, "peghole_xy": peghole_xy,
                    "q_lead": q_arr[:, lead].cpu().numpy(), "g_lead": G[:, lead].cpu().numpy(),
                    "act_lead": act_arr[:, lead].cpu().numpy(),
                    "q_ppo_lead": q_ppo_arr[:, lead].cpu().numpy() if ppo_policy is not None else None,
                    "reset_idx": reset_idx, "n_g": n_g, "lead": lead,
                }
                if args_cli.defer_plots:
                    rollout_dump.update({f"g{g}_{k}": v for k, v in gd.items() if v is not None})
                else:
                    render_group({**gd, **plot_shared}, output_path, peg_xy_path,
                                 None if args_cli.video else q_traj_path,
                                 args_cli.peg_xlim, args_cli.peg_ylim, ckpt_name)

            if args_cli.defer_plots:
                rollout_dump.update({"ckpt_name": ckpt_name, "n_groups": n_groups, **plot_shared})
                np.savez_compressed(os.path.join(plots_dir, f"rollout{iteration:04d}.npz"), **rollout_dump)
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

            t_plots_video = time.time() - t_plots0
            print(f"[iter {iteration:04d}] timing: rollout {t_rollout:.1f}s  plots+video {t_plots_video:.1f}s  "
                  f"total {time.time() - t_iter0:.1f}s")
            q_exp_online = float(np.mean([np.sum(q_probs[c].mean(dim=0).cpu().numpy() * q_support) for c in range(q_probs.shape[0])]))
            video_summary = ", ".join(str(p) for p in video_paths.values()) if args_cli.video else q_traj_path
            print(
                f"[iter {iteration:04d}] soft MC mean={soft_mc.mean():.4f}  reward mean={reward_mc.mean():.4f}  "
                f"entropy mean={entropy_mc.mean():.4f}  E[Q online mean]={q_exp_online:.4f}  "
                f"term[abnormal/fail/success]={term_counts} over {n_groups} reset group(s)  → {output_path}, {peg_xy_path}, "
                f"{video_summary}"
            )
            iteration += 1


        if reset_records:
            out_path = os.path.join(plots_dir, "reset_success_rate_xyz.png")
            save_reset_scatter(reset_records, out_path)
            print(f"[eval_critic] Saved per-reset success-rate scatter to: {out_path}")
            if level_label is not None:
                rates = np.array([r["success_rate"] for r in reset_records])
                scatter_paths.append(out_path)
                scatter_labels.append(f"{level_label} — mean {rates.mean():.1%}  ({int((rates < 0.5).sum())} resets <50%)")

    if len(scatter_paths) > 1:
        out_path = os.path.join(base_plots_dir, f"sweep_{sweep_param}_reset_success_rate_xyz.png")
        concat_labeled(scatter_paths, scatter_labels, out_path)
        print(f"[eval_critic] Saved sweep concat to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
