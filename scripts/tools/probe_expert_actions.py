"""Roll out an RSL-RL PPO expert deterministically and report its action magnitudes and success rate."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
if not any("--enable_cameras" in a.option_strings for a in parser._actions):
    parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors.")
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    cmd = base.command_manager.get_term("task_command")
    horizon = int(base.max_episode_length)
    print(f"[probe] clip_actions={agent_cfg.clip_actions} horizon={horizon} action_dim={env.num_actions}", flush=True)
    obs = env.get_observations()
    acts = []
    once = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    done_once = torch.zeros_like(once)
    for _ in range(horizon):
        with torch.inference_mode():
            a = policy(obs)
            obs, _, dones, _ = env.step(a)
        acts.append(a[~done_once].float().cpu())
        once |= (cmd.orientation_aligned & cmd.position_aligned) & ~done_once
        done_once |= dones.bool()
    A = torch.cat(acts).abs()
    q = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0])
    print(f"[probe] {A.shape[0]} actions; |a| quantiles (rows = 50/90/99/99.9/100%) per dim:")
    print(torch.quantile(A[torch.randperm(A.shape[0])[:200000]], q, dim=0).numpy().round(3))
    for b in (1.0, 2.0, 3.0, 5.0):
        print(f"[probe] fraction of action components with |a| > {b}: {(A > b).float().mean().item():.5f}")
    print(f"[probe] success_once over the first episode: {once.float().mean().item():.4f} ({base.num_envs} envs)")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
