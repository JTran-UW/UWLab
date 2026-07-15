"""Pure-strategy calibration for the BAMDP failure-rate injector.

Operational test from the proposal: before trusting mixed rollouts, run each
expert *in isolation* and confirm the realized failure rate matches its
target ``p_i``. The injection happens env-side (inside
``BAMDPLatentSampler._patched_process_action``) — this script just drives
the env with one expert from the ensemble and tallies forced-failure
triggers per episode.

Two modes:

**Single-strategy** (``--mode single``):
   Force ``theta = (p, p, ...)`` with the rescue forced to a *different*
   strategy (default: the only other strategy when K=2). Roll out the
   expert under test for ``num_episodes_per_p`` episodes; the
   discriminator should concentrate on the expert under test's class.
   Realized failure fraction should match ``p`` (within ~2 stderr).

**Joint** (``--mode joint``):
   Force ``theta = (p_0, p_1, ...)`` with NO rescue (sentinel
   ``force_rescue_idx = -1``). All strategies carry their target failure
   rate; no strategy is zeroed out. Run **each** expert in turn for
   ``num_episodes_per_p`` episodes and check the per-strategy realized
   rate matches its target — this is the sufficient test of "P =
   {p_1, …, p_K} is correctly enforced for every i simultaneously".

   In joint mode the env still enters a stall on trigger (open gripper +
   zero arm for ``stall_steps``) but rescue takeover is suppressed (no
   p was zeroed). After the stall the executed action returns to the
   learner's action. This lets us measure the realized rate from trigger
   counts without contaminating from rescue dynamics.
"""

from __future__ import annotations

# Pre-import numpy / numba BEFORE AppLauncher (Isaac Sim's Kit mutates sys.path).
import numpy  # noqa: F401
import numba  # noqa: F401

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pure-strategy calibration for the BAMDP injector.")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-BAMDPFailures-Train-v0",
)
parser.add_argument("--mode", choices=["single", "joint", "joint_asym"], default="single",
                    help=("`single`: one p_target, sampler zeros the rescue, expert under test drives. "
                          "`joint`: theta=(p,p) for both, no rescue, run each expert. "
                          "`joint_asym`: theta=(p_0, p_1) explicit, no rescue, run each expert — tests "
                          "per-strategy hazard attribution when the targets differ."))
parser.add_argument("--theta_pairs", type=str, default="",
                    help=("joint_asym only: comma-separated (p_0:p_1) pairs, e.g. `0.1:0.9,0.9:0.1`. "
                          "Each pair runs both experts."))
parser.add_argument("--expert_under_test", type=int, default=0,
                    help="single mode: index of the strategy whose calibration we're testing.")
parser.add_argument("--rescue_idx", type=int, default=None,
                    help="single mode only: force this strategy to be the rescue (default: other strategy when K=2).")
parser.add_argument("--p_grid", type=str, default="0.1,0.25,0.5,0.75,0.9")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--num_episodes_per_p", type=int, default=512)
parser.add_argument("--stall_steps", type=int, default=4)
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature on discriminator logits. T<1 sharpens, T>1 softens, T=1 is the trained softmax.")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="T → 0 limit: one-hot the highest-logit class. Overrides --temperature.")
parser.add_argument("--n_avg", type=float, default=50.0,
                    help="Mean episode length used by the hazard budget. Set close to the empirical mean ep length.")
parser.add_argument("--no_hazard_budget", action="store_true", default=False,
                    help="Ablation: fixed-schedule h_i = -ln(1-p_i) / n_avg (per step, constant). Skips the H_rem self-correction.")
parser.add_argument("--min_filter_len", type=int, default=10,
                    help="Episodes shorter than this don't have time to fire a trigger — reported separately as `long_realized`.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--insertive_object", type=str, default="peg")
parser.add_argument("--receptive_object", type=str, default="peghole")
parser.add_argument("--episode_length_s", type=float, default=16.0)
parser.add_argument("--output_json", type=str, default="logs/bamdp_calibration/calibration.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Force headless for calibration — no need for a viewer.
args_cli.headless = True
# Splice insertive/receptive into hydra_args so the variant lookup fires
# (assigning to env.scene.insertive_object directly would store a *string*
# where a RigidObjectCfg is expected, and Isaac Lab refuses).
hydra_args = list(hydra_args) + [
    f"env.scene.insertive_object={args_cli.insertive_object}",
    f"env.scene.receptive_object={args_cli.receptive_object}",
]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


def _parse_p_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.episode_length_s

    # Pin warmup_steps to 0 so the first step can fire — calibration cares
    # about the integrated rate, not whether the first step is suppressed.
    env_cfg.events.bamdp_latent_sampler.params["warmup_steps"] = 0
    env_cfg.events.bamdp_latent_sampler.params["stall_steps"] = args_cli.stall_steps
    env_cfg.events.bamdp_latent_sampler.params["discriminator_temperature"] = args_cli.temperature
    env_cfg.events.bamdp_latent_sampler.params["discriminator_argmax"] = bool(args_cli.argmax)
    env_cfg.events.bamdp_latent_sampler.params["n_avg"] = args_cli.n_avg
    env_cfg.events.bamdp_latent_sampler.params["use_hazard_budget"] = not args_cli.no_hazard_budget

    print(f"[calibration] building env {args_cli.task!r} with {args_cli.num_envs} envs...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    print("[calibration] env built.", flush=True)

    state = env.unwrapped.bamdp
    K = state.K
    p_grid = _parse_p_grid(args_cli.p_grid)
    device = state.device

    if args_cli.mode == "single":
        assert 0 <= args_cli.expert_under_test < K, f"expert_under_test must be in [0, {K})"
        rescue_idx = args_cli.rescue_idx
        if rescue_idx is None:
            if K != 2:
                raise ValueError(f"--rescue_idx required for K={K} (default only applies to K=2).")
            rescue_idx = 1 - args_cli.expert_under_test
        assert rescue_idx != args_cli.expert_under_test, "Rescue must differ from the expert under test."
        print(f"[calibration] mode=single  expert_under_test={args_cli.expert_under_test}  rescue_idx={rescue_idx}", flush=True)
    else:
        rescue_idx = -1  # sentinel: no rescue
        print(f"[calibration] mode={args_cli.mode}  (no rescue; all strategies hold their target p_i)", flush=True)

    # Build the list of (label, theta_tensor) sweep points.
    # - single / joint: one theta = (p, p, ...) per p in p_grid.
    # - joint_asym: one theta per pair, e.g. (0.1, 0.9) from "0.1:0.9".
    sweep_points: list[tuple[str, torch.Tensor]] = []
    if args_cli.mode == "joint_asym":
        if not args_cli.theta_pairs:
            raise ValueError("--mode joint_asym requires --theta_pairs.")
        for pair in args_cli.theta_pairs.split(","):
            parts = pair.strip().split(":")
            if len(parts) != K:
                raise ValueError(
                    f"Each --theta_pairs entry must have {K} values separated by ':', got {pair!r}"
                )
            theta = torch.tensor([float(x) for x in parts], dtype=torch.float32, device=device)
            label = "(" + ",".join(f"{float(x):.2f}" for x in parts) + ")"
            sweep_points.append((label, theta))
    else:
        for p in p_grid:
            theta = torch.full((K,), p, dtype=torch.float32, device=device)
            label = f"{p:.3f}"
            sweep_points.append((label, theta))

    results = []

    for sweep_label, sweep_theta in sweep_points:
        # Force theta and rescue. The sampler reads these on each reset.
        state.force_theta = sweep_theta.clone()
        if args_cli.mode == "single":
            state.force_rescue_idx = int(rescue_idx)
        else:
            state.force_rescue_idx = -1

        # Drive with the expert under test (single) or each expert in turn.
        experts_to_test = [args_cli.expert_under_test] if args_cli.mode == "single" else list(range(K))

        for e_under_test in experts_to_test:
            env.reset()

            triggered_this_ep = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
            total_done = 0
            total_failed = 0
            ep_lengths: list[int] = []
            ep_triggered_flags: list[bool] = []
            ep_term_causes: list[str] = []  # which termination term fired
            # Termination terms to track. "success" only exists on the train cfg;
            # the StudentEval variant nulls it. Probe what's active dynamically.
            term_mgr = env.unwrapped.termination_manager
            active_term_names = list(term_mgr.active_terms)

            max_total_episodes = args_cli.num_episodes_per_p

            # Each expert's per-strategy target is the corresponding entry in sweep_theta.
            p_target_for_expert = float(sweep_theta[e_under_test].item())
            tag = f"θ={sweep_label} e{e_under_test} (p_self={p_target_for_expert:.3f})"
            print(f"[calibration] {tag}: running until {max_total_episodes} eps complete...", flush=True)

            while total_done < max_total_episodes:
                pre_ep_len = env.unwrapped.episode_length_buf.clone()
                obs_buf = env.unwrapped.obs_buf
                proprio = obs_buf["proprio"].to(device)
                pc = obs_buf["pointcloud"].to(device)
                with torch.no_grad():
                    expert_action = state.experts[e_under_test](proprio, pc)

                action_dim = env.action_space.shape[-1]
                if expert_action.shape[-1] != action_dim:
                    raise RuntimeError(
                        f"Expert produces dim={expert_action.shape[-1]}, env needs {action_dim}"
                    )

                _, _, dones, _ = env.step(expert_action)
                triggered_this_ep |= state.last_triggered

                done = dones.bool()
                if done.any():
                    d_idx = done.nonzero(as_tuple=False).reshape(-1)
                    total_done += int(done.sum().item())
                    total_failed += int(triggered_this_ep[d_idx].sum().item())
                    # Snapshot which termination term fired this step (before
                    # the env auto-resets and clears the per-term buffers).
                    term_buffers = {n: term_mgr.get_term(n) for n in active_term_names}
                    for i in d_idx.tolist():
                        ep_lengths.append(int(pre_ep_len[i].item()) + 1)
                        ep_triggered_flags.append(bool(triggered_this_ep[i].item()))
                        causes = [n for n in active_term_names if bool(term_buffers[n][i].item())]
                        ep_term_causes.append(",".join(causes) if causes else "?")
                    triggered_this_ep[d_idx] = False

            realized = total_failed / max(total_done, 1)
            se = (realized * (1 - realized) / max(total_done, 1)) ** 0.5

            # Length-filtered: only episodes long enough to plausibly see triggers.
            # With n_avg=50, episodes shorter than min_filter_len steps spend
            # << target hazard so they don't probe the calibration.
            min_filter_len = int(args_cli.min_filter_len)
            long_idx = [i for i, L in enumerate(ep_lengths) if L >= min_filter_len]
            long_total = len(long_idx)
            long_failed = sum(1 for i in long_idx if ep_triggered_flags[i])
            long_realized = long_failed / max(long_total, 1)
            long_se = (long_realized * (1 - long_realized) / max(long_total, 1)) ** 0.5

            ep_len_mean = sum(ep_lengths) / max(len(ep_lengths), 1)

            # Termination cause breakdown.
            from collections import Counter
            cause_counts = Counter(ep_term_causes)
            cause_str = ", ".join(f"{k}={v}" for k, v in cause_counts.most_common())

            print(
                f"[calibration] {tag}  realized={realized:.3f} ± {se:.3f}  ({total_failed}/{total_done})  "
                f"| len>={min_filter_len}: {long_realized:.3f} ± {long_se:.3f}  ({long_failed}/{long_total})  "
                f"| mean_len={ep_len_mean:.1f}",
                flush=True,
            )
            print(f"[calibration] {tag}  term causes: {cause_str}", flush=True)
            results.append(
                dict(
                    mode=args_cli.mode,
                    expert_under_test=int(e_under_test),
                    rescue_idx=int(rescue_idx),
                    theta=sweep_theta.cpu().tolist(),
                    theta_label=sweep_label,
                    p_self=float(p_target_for_expert),
                    episodes=int(total_done),
                    failures=int(total_failed),
                    realized=float(realized),
                    std_err=float(se),
                    long_episodes=int(long_total),
                    long_failures=int(long_failed),
                    long_realized=float(long_realized),
                    long_std_err=float(long_se),
                    mean_episode_length=float(ep_len_mean),
                    min_filter_length=int(min_filter_len),
                    termination_causes=dict(cause_counts),
                )
            )

    out_path = Path(args_cli.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": args_cli.task,
        "mode": args_cli.mode,
        "K": int(K),
        "num_envs": int(args_cli.num_envs),
        "stall_steps": int(args_cli.stall_steps),
        "discriminator_temperature": float(args_cli.temperature),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[calibration] wrote {out_path}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()  # type: ignore[misc]
    finally:
        simulation_app.close()
