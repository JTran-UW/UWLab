# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Confirm in the simulator that the restored exploration matches the pre-3.0 one.

Offline algebra says resampling the gSDE weight matrix every step reproduces the
heteroscedastic per-step noise the OmniReset task was tuned against. This checks
it on real rollouts: it builds the actual env and runner from the actual config,
then for every step compares three noise processes on the *same* observations and
the *same* penultimate features phi(s):

  ref-2.x   sigma * N(0, I), sigma = sqrt(phi^2 @ exp(log_std)^2)   [target]
  per-step  what the configured distribution actually sampled       [current]
  rollout   phi @ W with W held fixed, i.e. stock GsdeDistribution  [regression]

Equivalence means per-step matches ref-2.x on both marginal std and lag-1
autocorrelation, while rollout matches on std but not autocorrelation.

The pre-3.0 code cannot be run directly for comparison: the pinned rsl-rl 3.1.2
has no gSDE at all (actor_critic.py rejects any noise_std_type but scalar/log),
so `ref-2.x` is its formula reimplemented, driven by simulator-produced phi.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# `cli_args` lives beside the training scripts, not on the package path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402


def lag1(noise: torch.Tensor) -> torch.Tensor:
    """Mean per-action lag-1 autocorrelation of a (T, N, A) noise sequence."""
    a, b = noise[:-1].reshape(-1, noise.shape[-1]), noise[1:].reshape(-1, noise.shape[-1])
    return (((a - a.mean(0)) * (b - b.mean(0))).mean(0) / (a.std(0) * b.std(0) + 1e-8)).mean()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    # Same compatibility pass train.py/play.py apply, so the runner sees the
    # config exactly as a real training run would.
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)

    actor = getattr(runner.alg, "_raw_actor", None) or runner.alg.actor
    dist = actor.distribution
    print("\n" + "=" * 78)
    print("EXPLORATION EQUIVALENCE AUDIT (in simulator)")
    print("=" * 78)
    print(f"  task         : {args_cli.task}")
    print(f"  distribution : {type(dist).__module__}.{type(dist).__name__}")
    print(f"  checkpoint   : {args_cli.checkpoint or '<random init>'}")
    print(f"  envs={args_cli.num_envs}  steps={args_cli.steps}", flush=True)

    assert hasattr(dist, "sample_weights"), "not a gSDE-family distribution"

    obs, _ = env.get_observations() if isinstance(env.get_observations(), tuple) else (env.get_observations(), None)
    rec = {k: [] for k in ("ref", "per_step", "rollout", "sigma")}

    # W for the "rollout" (stock gSDE) arm, drawn once and held, mirroring
    # OnPolicyRunner's once-per-rollout sample_weights() call.
    dist.sample_weights()
    W = dist._exploration_matrix.clone()

    with torch.inference_mode():
        for _ in range(args_cli.steps):
            # Exactly how PPO.act() samples (ppo.py:142) -- anything else leaves
            # the distribution un-updated and silently measures nothing.
            actions = actor(obs, stochastic_output=True)
            phi = dist._cached_features
            mean, sigma = dist._distribution.mean, dist._distribution.stddev
            rec["per_step"].append((actions - mean).clone())
            rec["ref"].append((sigma * torch.randn_like(sigma)).clone())
            rec["rollout"].append((phi @ W.to(phi.device)).clone())
            rec["sigma"].append(sigma.clone())
            obs, _, _, _ = env.step(actions)

    s = {k: torch.stack(v).float() for k, v in rec.items()}
    print(f"\n  phi |.| mean = {phi.abs().mean():.4f}   sigma mean = {s['sigma'].mean():.4f}\n")
    print(f"  {'scheme':<26}{'noise std':>12}{'lag-1 autocorr':>18}{'|d noise|':>12}")
    print("  " + "-" * 66)
    for tag, key in [("ref-2.x  (target)", "ref"), ("per-step (current)", "per_step"), ("rollout  (stock gSDE)", "rollout")]:
        n = s[key]
        d = (n[1:] - n[:-1]).abs().mean()
        print(f"  {tag:<26}{n.std(dim=(0, 1)).mean():>12.4f}{lag1(n):>18.4f}{d:>12.4f}")

    # Equivalence is tested on the *normalized* noise z = noise/sigma, not the raw
    # noise. Both arms are conditionally N(0, sigma(s)^2) with the same sigma, but
    # sigma varies widely across states, so an empirical raw-std estimator is
    # dominated by the few highest-sigma steps and stays noisy even with many
    # samples. Dividing by sigma removes that weighting and leaves a clean test:
    # both z should be standard normal, per action.
    z_ref, z_cur = s["ref"] / s["sigma"], s["per_step"] / s["sigma"]
    print("\n  normalized noise z = noise/sigma, should be ~N(0,1) for an exact match:")
    for tag, z in [("ref-2.x ", z_ref), ("per-step", z_cur)]:
        f = z.flatten()
        print(f"     {tag}  mean={f.mean():+.4f}  std={f.std():.4f}  "
              f"kurtosis={((f - f.mean()) ** 4).mean() / f.var() ** 2:.3f}  "
              f"per-action std range=[{z.std(dim=(0, 1)).min():.4f}, {z.std(dim=(0, 1)).max():.4f}]")
    std_err = (z_cur.std(dim=(0, 1)) - z_ref.std(dim=(0, 1))).abs().max()
    ac = abs(lag1(z_cur))
    print(f"\n  max per-action |std(z_per-step) - std(z_ref-2.x)| = {std_err:.4f}   (Monte-Carlo floor ~0.005)")
    print(f"  |lag-1 autocorr| of per-step noise               = {ac:.4f}   (target ~0)")
    verdict = "EQUIVALENT" if std_err < 0.03 and ac < 0.05 else "NOT EQUIVALENT"
    print(f"  VERDICT: per-step vs ref-2.x -> {verdict}")
    print("=" * 78 + "\n", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
