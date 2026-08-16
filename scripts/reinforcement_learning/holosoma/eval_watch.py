# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch training run directories and evaluate every new checkpoint, on the side.

Why: the training-time ``charts/success_rate`` is ``successes / episodes`` over one
``logging_interval`` window. At ``--num_envs 1`` with 160-step episodes and the default
``logging_interval=100``, a window holds ~0.6 episodes, so every point is 0/1, 1/1, or 0/0 (nan).
That is not a noisy estimate, it is a binary sequence. Running the same policy over many parallel
envs instead gives a usable rate: 512 episodes at p=0.9 has a standard error of ~1.3%.

This runs as its own process (its own Isaac instance and JOB_TMPDIR, so it cannot collide with the
trainer), polls one or more run directories for new ``model_*.pt``, and evaluates each one at the
checkpoint cadence. Metrics land in ``eval_metrics.jsonl`` inside each run directory, and optionally
in a single wandb run namespaced per training run.

Success semantics are lifted from ``play.py --eval_episodes_per_env`` rather than reimplemented:
per-env episode tallies (surplus episodes discarded, so short successful episodes cannot inflate the
rate) and ``ever_success`` read off ``progress_context`` (an episode counts if it reached the success
configuration at any point, not only at the terminal step).

Example::

    python eval_watch.py --task <same task as training> --num_envs 256 --headless \\
        --episodes_per_env 2 \\
        --watch_dirs logs/rsl_rl/<exp>/<run_a> logs/rsl_rl/<exp>/<run_b> \\
        --wandb_project omnireset_fastsac
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate new checkpoints as they appear.")
parser.add_argument("--task", type=str, default=None, help="Task to evaluate on (use the training task).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs for evaluation (drives the sample size).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--watch_dirs",
    type=str,
    nargs="+",
    required=True,
    help="Run directories containing model_*.pt. Each is polled for new checkpoints.",
)
parser.add_argument(
    "--episodes_per_env",
    type=int,
    default=2,
    help="Episodes each env must complete per evaluation. Total episodes = num_envs * this.",
)
parser.add_argument("--poll_seconds", type=int, default=120, help="Seconds between directory scans.")
parser.add_argument(
    "--once",
    action="store_true",
    default=False,
    help="Evaluate everything currently present, then exit instead of polling forever.",
)
parser.add_argument(
    "--skip_existing",
    action="store_true",
    default=False,
    help="Ignore checkpoints already present at startup; only evaluate ones written from now on.",
)
parser.add_argument("--wandb_project", type=str, default=None, help="If set, log eval metrics to this wandb project.")
parser.add_argument("--wandb_name", type=str, default="eval-watch", help="Name for the eval job's wandb run.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# --watch_dirs is nargs="+", so it greedily eats any Hydra overrides that follow it on the command
# line. Left alone that silently drops them (the env would run with default objects) AND treats them
# as directories. Split them back out by shape: anything containing "=" is a Hydra override.
_dirs, _overrides = [], []
for _tok in args_cli.watch_dirs:
    (_overrides if "=" in _tok else _dirs).append(_tok)
if _overrides:
    print(f"[eval-watch] reclaimed {len(_overrides)} Hydra override(s) from --watch_dirs: {_overrides}")
args_cli.watch_dirs = _dirs
for _d in _dirs:
    if not os.path.isdir(_d):
        raise SystemExit(f"[eval-watch] --watch_dirs entry is not a directory: {_d}")

sys.argv = [sys.argv[0]] + hydra_args + _overrides

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import glob
import json
import os
import re
import time
import torch
import tqdm

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from vecenv_wrapper import HolosomaVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

CKPT_RE = re.compile(r"model_(\d+)\.pt$")


def _checkpoint_step(path: str) -> int:
    m = CKPT_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def _scan(watch_dirs: list[str]) -> list[tuple[str, str, int]]:
    """Return (run_dir, ckpt_path, step) for every model_*.pt found, oldest step first."""
    found: list[tuple[str, str, int]] = []
    for d in watch_dirs:
        for p in glob.glob(os.path.join(d, "model_*.pt")):
            step = _checkpoint_step(p)
            if step >= 0:
                found.append((d, p, step))
    return sorted(found, key=lambda t: (t[0], t[2]))


def _load_when_stable(path: str, device: str, settle_s: float = 3.0):
    """Load a checkpoint only once its size has stopped changing.

    The trainer writes these files while we poll, and a torch.load of a half-written file raises (or
    worse, is silently truncated). Returns None if it is still being written, so the caller retries
    on the next scan.
    """
    try:
        s1 = os.path.getsize(path)
        time.sleep(settle_s)
        if os.path.getsize(path) != s1:
            return None
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any read failure means "not ready yet"
        print(f"[eval-watch] not ready ({type(exc).__name__}): {path}")
        return None


@torch.inference_mode()
def _evaluate(env, policy, actor_obs_keys: list[str], episodes_per_env: int) -> dict:
    """Run until every env has completed ``episodes_per_env`` episodes; return summary metrics.

    Mirrors play.py --eval_episodes_per_env: per-env tallies with surplus discarded, and success
    defined as "reached the success configuration at any point in the episode".
    """
    device = env.unwrapped.device
    n_env = env.num_envs
    ever_success = torch.zeros(n_env, dtype=torch.bool, device=device)
    ep_count = torch.zeros(n_env, dtype=torch.long, device=device)
    succ_count = torch.zeros(n_env, dtype=torch.long, device=device)
    ep_steps = torch.zeros(n_env, dtype=torch.long, device=device)
    timeout_count = torch.zeros(n_env, dtype=torch.long, device=device)
    abnormal_count = torch.zeros(n_env, dtype=torch.long, device=device)
    other_count = torch.zeros(n_env, dtype=torch.long, device=device)
    durations: list[int] = []

    term_mgr = env.unwrapped.termination_manager
    term_names = list(term_mgr.active_terms)
    has_timeout = "time_out" in term_names
    has_abnormal = "abnormal_robot" in term_names
    zeros_b = torch.zeros(n_env, dtype=torch.bool, device=device)
    context_term = env.unwrapped.reward_manager.get_term_cfg("progress_context").func

    obs = env.get_observations()
    pbar = tqdm.tqdm(total=episodes_per_env * n_env, desc="eval episodes", unit="ep", leave=False)
    while bool((ep_count < episodes_per_env).any()):
        actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
        obs, _, dones, _ = env.step(policy({"actor_obs": actor_obs}))

        ever_success |= getattr(context_term, "orientation_aligned") & getattr(context_term, "position_aligned")
        ep_steps += 1
        done_mask = dones.bool()
        counting = done_mask & (ep_count < episodes_per_env)
        successes = ever_success & counting
        ep_count += counting.long()
        succ_count += successes.long()
        pbar.update(int(counting.sum().item()))

        failed = counting & ~successes
        timed_out = term_mgr.get_term("time_out").bool() if has_timeout else zeros_b
        abnormal = term_mgr.get_term("abnormal_robot").bool() if has_abnormal else zeros_b
        abnormal_f = failed & abnormal
        timeout_f = failed & timed_out & ~abnormal
        abnormal_count += abnormal_f.long()
        timeout_count += timeout_f.long()
        other_count += (failed & ~abnormal_f & ~timeout_f).long()
        if bool(counting.any()):
            durations.extend(ep_steps[counting].cpu().tolist())

        ever_success[done_mask] = False
        ep_steps[done_mask] = 0
    pbar.close()

    total = int(ep_count.sum().item())
    succ = int(succ_count.sum().item())
    rate = succ / total if total else 0.0
    # Binomial standard error: the whole point of this script is knowing how tight the estimate is.
    stderr = (rate * (1.0 - rate) / total) ** 0.5 if total else float("nan")
    return {
        "episodes": total,
        "successes": succ,
        "success_rate": rate,
        "success_rate_stderr": stderr,
        "timeout_rate": int(timeout_count.sum().item()) / total if total else 0.0,
        "abnormal_rate": int(abnormal_count.sum().item()) / total if total else 0.0,
        "other_rate": int(other_count.sum().item()) / total if total else 0.0,
        "mean_episode_length": (sum(durations) / len(durations)) if durations else 0.0,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")
    if not is_fastsac:
        raise ValueError("eval_watch.py supports FastSAC checkpoints only.")
    env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # use_cpu_rb: the agent allocates a training replay buffer in setup() that evaluation never
    # touches; keeping it off the GPU leaves room for PhysX.
    runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device, use_cpu_rb=True)
    runner.setup()
    actor_obs_keys = agent_cfg.actor_obs_keys

    # first_episode_termination kills a trickle of envs for the first max_episode_length steps, then
    # goes inert. In a watcher that means only the FIRST checkpoint of a session gets its episodes cut
    # short -- a bias that lands on one point of the curve and looks like a real dip. Refuse to hide it.
    if "first_episode_termination" in list(env.unwrapped.termination_manager.active_terms):
        print(
            "[eval-watch] WARNING: first_episode_termination is ACTIVE. It fires only during the "
            "first ~max_episode_length steps, so it biases the first checkpoint evaluated and not "
            "the rest. Pass env.terminations.first_episode_termination=null (as the training runs do)."
        )

    wb = None
    if args_cli.wandb_project:
        import wandb

        wb = wandb.init(project=args_cli.wandb_project, name=args_cli.wandb_name, config={
            "task": args_cli.task,
            "eval_num_envs": args_cli.num_envs,
            "episodes_per_env": args_cli.episodes_per_env,
            "watch_dirs": args_cli.watch_dirs,
        })

    done_ckpts: set[str] = set()
    if args_cli.skip_existing:
        done_ckpts = {p for _, p, _ in _scan(args_cli.watch_dirs)}
        print(f"[eval-watch] ignoring {len(done_ckpts)} pre-existing checkpoints")

    print(f"[eval-watch] watching {len(args_cli.watch_dirs)} dir(s), {args_cli.num_envs} envs x "
          f"{args_cli.episodes_per_env} episodes = {args_cli.num_envs * args_cli.episodes_per_env} episodes/eval")

    while True:
        pending = [t for t in _scan(args_cli.watch_dirs) if t[1] not in done_ckpts]
        for run_dir, ckpt, step in pending:
            payload = _load_when_stable(ckpt, agent_cfg.device)
            if payload is None:
                continue  # still being written; pick it up next scan
            run_name = os.path.basename(os.path.normpath(run_dir))
            runner.load(ckpt)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            # inference_mode: _evaluate leaves inference tensors behind, and IsaacLab's reset writes
            # in-place into asset buffers -- outside inference mode that raises on the 2nd checkpoint.
            with torch.inference_mode():
                env.reset()
            t0 = time.time()
            m = _evaluate(env, policy, actor_obs_keys, args_cli.episodes_per_env)
            m.update({"step": step, "run": run_name, "checkpoint": ckpt, "eval_seconds": round(time.time() - t0, 1)})

            print(f"[eval-watch] {run_name} step={step} success={m['success_rate']:.3f}"
                  f" +/-{m['success_rate_stderr']:.3f} over {m['episodes']} eps"
                  f" (timeout {m['timeout_rate']:.2f}, abnormal {m['abnormal_rate']:.2f})")
            with open(os.path.join(run_dir, "eval_metrics.jsonl"), "a") as f:
                f.write(json.dumps(m) + "\n")
            if wb is not None:
                wb.log({f"eval/{run_name}/{k}": v for k, v in m.items() if isinstance(v, (int, float))}, step=step)
            done_ckpts.add(ckpt)

        if args_cli.once:
            break
        time.sleep(args_cli.poll_seconds)

    if wb is not None:
        wb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
