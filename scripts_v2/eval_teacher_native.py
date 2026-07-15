"""Quick eval: load a trained ScenePC teacher in its native env, roll out, track success rate.

Usage::

    python scripts_v2/eval_teacher_native.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-v0 \
        --num_envs 64 --num_steps 500 \
        --checkpoint checkpoints/pc_teacher/peg_pc_prevact.pt \
        --headless \
        env.scene.insertive_object=peg env.scene.receptive_object=peghole \
        env.curriculum.gravity_curriculum.params.reduction=monitor_mean \
        env.curriculum.gravity_curriculum.params.floor=0.1
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--stochastic", action="store_true", help="Use policy.act (stochastic) instead of act_inference (mean).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after sim app is up.
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from uwlab_tasks.utils.hydra import hydra_task_config

# Inject UWLab classes so eval(class_name) lookup works.
import rsl_rl.runners.on_policy_runner as _runner_module
from uwlab_rl.rsl_rl.actor_critic_encoder import ActorCriticWithEncoder
_runner_module.ActorCriticWithEncoder = ActorCriticWithEncoder


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Make agent_cfg compatible with installed rsl-rl version (strips unsupported fields like 'optimizer').
    sys.path.insert(0, "scripts/reinforcement_learning/rsl_rl")
    import cli_args  # type: ignore[import-not-found]
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[eval] loading checkpoint from {args_cli.checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    if args_cli.stochastic:
        # Stochastic: matches training-time PPO rollout behavior (gSDE samples).
        runner.eval_mode()
        runner.alg.policy.to(env.unwrapped.device)
        policy = runner.alg.policy.act
        print("[eval] using STOCHASTIC policy (act, gSDE sampling)")
    else:
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print("[eval] using DETERMINISTIC policy (act_inference, mean only)")

    # Use the termination_manager's per-term dones dict (more reliable than reading
    # progress_term.success post-step, which gets reset by reset_idx).
    term_mgr = env.unwrapped.termination_manager
    print(f"[eval] termination terms: {list(term_mgr.active_terms)}")

    # Curriculum diagnostic. CurriculumManager uses parallel _term_names/_term_cfgs lists.
    curr_mgr = env.unwrapped.curriculum_manager
    print(f"[eval] curriculum terms: {list(curr_mgr.active_terms)}")
    grav_func = None
    if "gravity_curriculum" in curr_mgr._term_names:
        idx = curr_mgr._term_names.index("gravity_curriculum")
        grav_func = curr_mgr._term_cfgs[idx].func
        print(
            f"[eval] gravity_curriculum attrs: "
            f"reduction={getattr(grav_func, 'reduction', 'N/A')}, "
            f"floor={getattr(grav_func, 'floor', 'N/A')}, "
            f"gravity_frac.shape={getattr(getattr(grav_func, 'gravity_frac', None), 'shape', 'N/A')}"
        )

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    num_episodes = 0
    num_successes = 0
    num_timeouts = 0
    num_abnormal = 0

    print(f"[eval] rolling out {args_cli.num_steps} env steps with {env.num_envs} envs...")
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            # Per-term done masks (from the most recent termination_manager.compute()).
            # These survive until reset_idx() runs at end of step. Read them right after step().
            done_idx = dones.view(-1).nonzero(as_tuple=False).view(-1)
            if done_idx.numel() > 0:
                num_episodes += int(done_idx.numel())
                # success term fired this step
                if "success" in term_mgr.active_terms:
                    succ_mask = term_mgr.get_term("success")
                    num_successes += int(succ_mask[done_idx].sum().item())
                if "early_success" in term_mgr.active_terms:
                    es_mask = term_mgr.get_term("early_success")
                    # early_success is also a success
                    succ2 = es_mask[done_idx]
                    if "success" in term_mgr.active_terms:
                        # avoid double-count: only add early_success that didn't also have success
                        succ_already = succ_mask[done_idx].bool()
                        succ2 = succ2 & ~succ_already
                    num_successes += int(succ2.sum().item())
                if "time_out" in term_mgr.active_terms:
                    to_mask = term_mgr.get_term("time_out")
                    num_timeouts += int(to_mask[done_idx].sum().item())
                if "abnormal_robot" in term_mgr.active_terms:
                    ab_mask = term_mgr.get_term("abnormal_robot")
                    num_abnormal += int(ab_mask[done_idx].sum().item())

            if (step + 1) % 50 == 0:
                rate = num_successes / max(num_episodes, 1)
                gf = "N/A"
                if grav_func is not None:
                    gf_t = getattr(grav_func, "gravity_frac", None)
                    if gf_t is not None:
                        try:
                            gf = f"{float(gf_t.mean() if hasattr(gf_t, 'mean') else gf_t):.3f}"
                        except Exception:
                            gf = str(gf_t)[:20]
                print(
                    f"[step {step + 1:>4d}] eps={num_episodes} succ={num_successes} "
                    f"timeout={num_timeouts} abnormal={num_abnormal} succ_rate={rate:.3f} grav_frac={gf}",
                    flush=True,
                )

    rate = num_successes / max(num_episodes, 1)
    print()
    print("=" * 60)
    print(f"FINAL: {num_successes}/{num_episodes} successes = {rate:.3f}")
    print(f"       timeouts={num_timeouts}, abnormal_robot={num_abnormal}")
    print("=" * 60, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
