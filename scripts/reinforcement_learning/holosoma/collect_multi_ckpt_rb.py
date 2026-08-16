# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fill one replay buffer from many checkpoints, splitting the transitions evenly.

Motivation: prefilling from a single policy (or relying on ``learning_starts``) gives a buffer whose
data is all drawn from one narrow state distribution. Rolling out a *ladder* of checkpoints -- early,
middle and late training -- gives the diverse, genuinely off-policy buffer that off-policy learning
benefits from.

Given N checkpoints and a total transition budget T, each checkpoint contributes T/N transitions.
Isaac Sim is started once and the env is reused across every checkpoint; only the network weights are
swapped between segments.

Output is the FastSAC *online* replay-buffer format (top-level keys, ``n_env=1``, env dim collapsed
into time), i.e. what ``agent.load_replay_buffer_path`` reads. Use ``--num_envs`` purely as a
collection-speed knob: 10e6 transitions at 4096 envs is ~2.4k steps, at 1 env it is 10e6 steps.

Example::

    python collect_multi_ckpt_rb.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-...-v0 \
        --num_envs 4096 --headless \
        --checkpoint_dir logs/rsl_rl/<run> --num_checkpoints 10 \
        --total_transitions 10000000 \
        --output rbs/multi_ckpt_rb.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect a replay buffer from a ladder of checkpoints.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Envs to collect with (speed knob; output is n_env=1).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--checkpoints",
    type=str,
    nargs="+",
    default=None,
    help="Explicit list of checkpoint .pt files, in the order they should be rolled out.",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help="Directory of model_*.pt to use instead of --checkpoints. Sorted by step; see --num_checkpoints.",
)
parser.add_argument(
    "--num_checkpoints",
    type=int,
    default=None,
    help=(
        "With --checkpoint_dir, evenly subsample this many checkpoints across the run (always "
        "including the first and last). Omit to use every checkpoint in the directory."
    ),
)
parser.add_argument(
    "--total_transitions",
    type=int,
    required=True,
    help="Total transitions across all checkpoints. Each checkpoint contributes total/num_checkpoints.",
)
parser.add_argument("--output", type=str, required=True, help="Path to write the replay buffer .pt.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help=(
        "Use the policy mean instead of sampling. Default is stochastic (matches the training "
        "rollout, fast_sac_agent sets policy = actor.explore) which gives wider state coverage."
    ),
)
parser.add_argument(
    "--no_reset_between_checkpoints",
    action="store_true",
    default=False,
    help=(
        "Keep stepping across a checkpoint switch instead of resetting. By default the env is reset "
        "when weights change, so no episode is half-driven by two different policies."
    ),
)
parser.add_argument(
    "--record_n_steps",
    "--num_steps",
    dest="record_n_steps",
    type=int,
    default=1,
    help="n-step horizon to tag the buffer with. Transitions are always stored as single steps.",
)
parser.add_argument(
    "--record_actor_obs_keys",
    type=str,
    nargs="+",
    default=None,
    help="Env observation group(s) to store as actor observations. Defaults to the policy's groups.",
)
parser.add_argument(
    "--record_critic_obs_keys",
    type=str,
    nargs="+",
    default=None,
    help="Env observation group(s) to store as critic observations. Defaults to the policy's groups.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import glob
import os
import torch
import tqdm
from tensordict import TensorDict

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from vecenv_wrapper import HolosomaVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

TENSOR_SPECS = [
    ("observations", "n_obs", torch.float),
    ("actions", "n_act", torch.float),
    ("rewards", None, torch.float),
    ("dones", None, torch.long),
    ("truncations", None, torch.long),
    ("next_observations", "n_obs", torch.float),
    ("critic_observations", "n_critic_obs", torch.float),
    ("next_critic_observations", "n_critic_obs", torch.float),
]


def _flatten_obs_groups(obs_td, keys: list[str]) -> torch.Tensor:
    """Concatenate observation groups into the flat (n_env, D) layout the agent expects.

    Mirrors ``play.py``: each group is flattened before concatenation so image groups (rank > 2)
    do not get concatenated along width.
    """
    return torch.cat([obs_td[k].reshape(obs_td[k].shape[0], -1) for k in keys], dim=-1)


def _stamp_segment_truncations(buf: dict[str, torch.Tensor], seg_end_idx: torch.Tensor) -> None:
    """Flag the last transition of each contiguous segment as truncated.

    Collapsing (n_env, T) into a flat time axis puts unrelated trajectories next to each other. The
    n-step sampler walks ``(i + offset) % buffer_size`` and only stops at a done or truncation, so
    without a flag it would sum rewards across a seam and bootstrap off a state from another env's
    trajectory. ``SimpleReplayBuffer.sample`` protects its own circular-wrap seam the same way
    (``truncations[pos-1] = ~dones[pos-1]``); this is that trick applied to every seam we create.

    Only transitions that are not already genuine terminals are flagged, so real episode ends keep
    their done semantics.
    """
    already_done = buf["dones"][0, seg_end_idx] > 0
    buf["truncations"][0, seg_end_idx] = torch.where(
        already_done,
        buf["truncations"][0, seg_end_idx],
        torch.ones_like(buf["truncations"][0, seg_end_idx]),
    )


def _validate_no_unflagged_seams(buf: dict[str, torch.Tensor], seg_len: int, n_steps: int) -> None:
    """Assert every segment boundary terminates an n-step walk.

    The invariant: at each segment end the sampler must see a done or truncation, otherwise an
    n-step window starting up to ``n_steps - 1`` earlier would read across into another trajectory.
    """
    total = buf["dones"].shape[1]
    seg_end_idx = torch.arange(seg_len - 1, total, seg_len)
    stops = (buf["dones"][0, seg_end_idx] > 0) | (buf["truncations"][0, seg_end_idx] > 0)
    bad = int((~stops).sum())
    if bad:
        raise AssertionError(f"{bad}/{len(seg_end_idx)} segment boundaries lack a done/truncation flag")
    print(
        f"[INFO] seam check OK: {len(seg_end_idx)} segment boundaries (every {seg_len} transitions) "
        f"all terminate the n-step walk (n_steps={n_steps})"
    )


def _resolve_checkpoints() -> list[str]:
    """Build the ordered checkpoint list from --checkpoints or --checkpoint_dir."""
    if args_cli.checkpoints:
        ckpts = [retrieve_file_path(p) for p in args_cli.checkpoints]
    elif args_cli.checkpoint_dir:
        ckpts = sorted(glob.glob(os.path.join(args_cli.checkpoint_dir, "model_*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"No model_*.pt found in {args_cli.checkpoint_dir}")
        n = args_cli.num_checkpoints
        if n is not None and n < len(ckpts):
            # Evenly spaced across the run, endpoints included, so the ladder spans early -> late.
            idx = torch.linspace(0, len(ckpts) - 1, n).round().long().tolist()
            ckpts = [ckpts[i] for i in sorted(set(idx))]
    else:
        raise ValueError("Pass either --checkpoints or --checkpoint_dir")
    return ckpts


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoints = _resolve_checkpoints()
    n_ckpt = len(checkpoints)
    device = agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")
    if not is_fastsac:
        raise ValueError("collect_multi_ckpt_rb.py supports FastSAC checkpoints only.")
    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    n_env = env.num_envs
    n_act = env.num_actions

    # Budget: floor to whole steps so every checkpoint contributes exactly the same count.
    per_ckpt = args_cli.total_transitions // n_ckpt
    steps_per_ckpt = per_ckpt // n_env
    if steps_per_ckpt < 1:
        raise ValueError(
            f"--total_transitions {args_cli.total_transitions} over {n_ckpt} checkpoints at {n_env} envs "
            f"gives <1 step per checkpoint. Lower --num_envs or raise --total_transitions."
        )
    total = steps_per_ckpt * n_env * n_ckpt

    # use_cpu_rb: the agent builds its own training replay buffer in setup(), which this script never
    # touches. Keeping it off the GPU leaves that memory for PhysX and lets collection share a card
    # with a running training job.
    runner = FastSACAgent(env, agent_cfg, log_dir=None, device=device, use_cpu_rb=True)
    runner.setup()
    actor_obs_keys = args_cli.record_actor_obs_keys or agent_cfg.actor_obs_keys
    critic_obs_keys = args_cli.record_critic_obs_keys or agent_cfg.critic_obs_keys

    obs_td = env.get_observations().to(device)
    n_obs = _flatten_obs_groups(obs_td, actor_obs_keys).shape[-1]
    n_critic_obs = _flatten_obs_groups(obs_td, critic_obs_keys).shape[-1]
    dims = {"n_obs": n_obs, "n_act": n_act, "n_critic_obs": n_critic_obs}

    # Accumulate on CPU: at ~3.5 KB/transition a 10e6 buffer is ~35 GB, far past GPU memory.
    buf: dict[str, torch.Tensor] = {}
    for key, dim_name, dtype in TENSOR_SPECS:
        shape = (1, total) if dim_name is None else (1, total, dims[dim_name])
        buf[key] = torch.zeros(shape, dtype=dtype, device="cpu")
    est_gb = sum(t.numel() * t.element_size() for t in buf.values()) / 1e9

    print(f"[INFO] {n_ckpt} checkpoints x {steps_per_ckpt} steps x {n_env} envs = {total} transitions")
    print(f"[INFO] n_obs={n_obs} n_act={n_act} n_critic_obs={n_critic_obs}  buffer ~{est_gb:.1f} GB (CPU)")
    print(f"[INFO] actions: {'deterministic mean' if args_cli.deterministic else 'stochastic sample'}")

    # Env-major layout: each env's steps_per_ckpt steps stay contiguous, so the only discontinuities
    # are the segment boundaries flagged below. Writing step-major instead (all envs at t, then all
    # envs at t+1) would make EVERY adjacent pair a different trajectory and corrupt every n-step
    # sample. Flat index for (checkpoint c, env e, step t) = c*n_env*S + e*S + t.
    env_offsets = torch.arange(n_env) * steps_per_ckpt
    for ci, ckpt in enumerate(checkpoints):
        ckpt_base = ci * n_env * steps_per_ckpt
        print(f"[INFO] ({ci + 1}/{n_ckpt}) loading {ckpt}")
        runner.load(retrieve_file_path(ckpt))

        if args_cli.deterministic:
            policy = runner.get_inference_policy(device=env.unwrapped.device)
        else:
            # Mirror play.py --stochastic: same frozen normalizer, sample instead of mean.
            _actor = runner.actor.to(device)
            _actor.eval()
            _obs_norm = runner.obs_normalizer.to(device)
            _obs_norm.eval()

            def policy(obs, _actor=_actor, _obs_norm=_obs_norm):
                x = obs["actor_obs"]
                nx = _obs_norm(x, update=False) if runner.obs_normalization else x
                return _actor.explore(nx, deterministic=False)

        # inference_mode: the rollout below produces inference tensors, and IsaacLab's reset writes
        # in-place into asset buffers -- doing that outside inference mode raises on the 2nd
        # checkpoint ("Inplace update to inference tensor outside InferenceMode").
        with torch.inference_mode():
            if not args_cli.no_reset_between_checkpoints:
                env.reset()
            obs_td = env.get_observations().to(device)
            actor_obs = _flatten_obs_groups(obs_td, actor_obs_keys)
            critic_obs = _flatten_obs_groups(obs_td, critic_obs_keys)

        pbar = tqdm.tqdm(total=steps_per_ckpt, desc=f"ckpt {ci + 1}/{n_ckpt}")
        for t in range(steps_per_ckpt):
            with torch.inference_mode():
                actions = policy({"actor_obs": actor_obs})
                next_obs_td, rewards, dones, extras = env.step(actions.to(env.device))
                next_obs_td = next_obs_td.to(device)
                truncations = extras.get("time_outs", torch.zeros(n_env, dtype=torch.bool, device=device))
                next_actor_obs = _flatten_obs_groups(next_obs_td, actor_obs_keys)
                next_critic_obs = _flatten_obs_groups(next_obs_td, critic_obs_keys)

                idx = ckpt_base + env_offsets + t
                buf["observations"][0, idx] = actor_obs.float().cpu()
                buf["actions"][0, idx] = actions.float().cpu()
                buf["rewards"][0, idx] = rewards.float().cpu()
                buf["dones"][0, idx] = dones.long().cpu()
                buf["truncations"][0, idx] = truncations.to(device).long().cpu()
                buf["next_observations"][0, idx] = next_actor_obs.float().cpu()
                buf["critic_observations"][0, idx] = critic_obs.float().cpu()
                buf["next_critic_observations"][0, idx] = next_critic_obs.float().cpu()

                actor_obs = next_actor_obs
                critic_obs = next_critic_obs
            pbar.update(1)
        pbar.close()

        # Cut every env's segment so no n-step walk runs off its end into the next env's data
        # (or, at the block end, into the next checkpoint's).
        _stamp_segment_truncations(buf, ckpt_base + env_offsets + (steps_per_ckpt - 1))

    _validate_no_unflagged_seams(buf, steps_per_ckpt, args_cli.record_n_steps)

    payload = dict(buf)
    payload.update(
        {
            "ptr": total,
            "n_env": 1,
            "buffer_size": total,
            "n_steps": args_cli.record_n_steps,
            "global_step": 0,
            "checkpoints": list(checkpoints),
            "transitions_per_checkpoint": steps_per_ckpt * n_env,
            # Contiguous run length per env; boundaries are flagged truncated (see _stamp_segment_truncations).
            "segment_length": steps_per_ckpt,
            "task": args_cli.task,
            "stochastic": not args_cli.deterministic,
        }
    )
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    torch.save(payload, args_cli.output)
    print(f"[INFO] Saved {total} transitions from {n_ckpt} checkpoints to: {args_cli.output}")
    print(f"[INFO] Load with: agent.load_replay_buffer_path={args_cli.output} agent.buffer_size>={total} --num_envs 1")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
