# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect an RLPD expert image buffer for flashsac_rgb.py by rolling out an RSL-RL state expert in the vision task.

Port of ManiSkill-opt-all reinforcement_learning/scripts/collect_rgb_expert.py: the expert acts on the privileged
``policy`` observation, the buffer stores the vision view (images + non-privileged state).
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect a FlashSAC-RGB expert buffer from an RSL-RL state expert.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL state expert checkpoint.")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_rounds", type=int, default=2, help="Episodes per env.")
parser.add_argument("--horizon", type=int, default=160)
parser.add_argument("--action_scale", type=str, default="10.0", help="Policy actions in [-1, 1] are scaled by this; expert actions are clipped to +-scale. "
                    "A comma-separated list gives one scale per action dim.")
parser.add_argument("--no_clip", action="store_true", help="Store expert actions unclipped (for an unsquashed --no_tanh actor).")
parser.add_argument("--state_group", type=str, default="vision_state", help="Observation group stored as the agent's low-dim state.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--expert_kind", type=str, choices=["rsl_rl", "legacy_mlp"], default="rsl_rl",
                    help="legacy_mlp: pre-5.x rsl_rl checkpoint (model_state_dict with actor.* + actor_obs_normalizer.*), mean action.")
parser.add_argument("--success_only", type=str, choices=["none", "strict", "lenient"], default="none",
                    help="Keep only episodes that succeeded (strict = task thresholds, lenient = 2.5 mm / 0.1 rad).")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # noqa: E402
from buffers import ImageReplayBuffer  # noqa: E402
from vision_env import IMAGE_KEYS, UWLabVisionVecEnv  # noqa: E402


def _legacy_mlp_policy(path: str, device):
    """Deterministic mean action of a pre-5.x rsl_rl actor: (obs - mean) / (std + 1e-2) -> Linear/ELU stack."""
    sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for n, i in enumerate(idx):
        lin = torch.nn.Linear(sd[f"actor.{i}.weight"].shape[1], sd[f"actor.{i}.weight"].shape[0]).to(device)
        lin.weight.data.copy_(sd[f"actor.{i}.weight"])
        lin.bias.data.copy_(sd[f"actor.{i}.bias"])
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(torch.nn.ELU())
    mlp = torch.nn.Sequential(*layers).eval()
    mean, std = sd["actor_obs_normalizer._mean"], sd["actor_obs_normalizer._std"]
    print(f"[collect] legacy actor {[l.weight.shape[1] for l in layers if isinstance(l, torch.nn.Linear)]} -> {layers[-1].weight.shape[0]}", flush=True)

    def policy(obs):
        return mlp((obs["policy"] - mean) / (std + 1e-2))

    return policy


@hydra_task_compose(args_cli.task, args_cli.agent, hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.expert_kind == "rsl_rl":
        runner = OnPolicyRunner(RslRlVecEnvWrapper(gym_env), agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(retrieve_file_path(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=gym_env.unwrapped.device)
    else:
        policy = _legacy_mlp_policy(retrieve_file_path(args_cli.checkpoint), gym_env.unwrapped.device)
    scale = [float(v) for v in args_cli.action_scale.split(",")] if "," in args_cli.action_scale else float(args_cli.action_scale)
    env = UWLabVisionVecEnv(gym_env, args_cli.horizon, scale, args_cli.state_group)
    device, n = env.device, env.num_envs
    size = n * args_cli.horizon * args_cli.num_rounds
    rgb_shape = env.single_observation_space["rgb"].shape
    state_dim = env.single_observation_space["state"].shape[-1]
    rb = ImageReplayBuffer(
        pixels_obs_space=rgb_shape, state_obs_dim=state_dim, act_dim=env.single_action_space.shape[0], size=size,
        horizon=args_cli.horizon, num_envs=n, n_frame_stack=1, n_steps=1, gamma=0.99, device=device,
    )
    obs, _ = env.reset()
    bound = torch.as_tensor(scale, dtype=torch.float32, device=device)
    raw_actions, successes, success_end, returns = [], [], [], []
    keep_rows = []
    with torch.inference_mode():
        for _ in range(args_cli.horizon * args_cli.num_rounds):
            raw = policy(TensorDict(env.obs_dict, batch_size=[n]))
            raw_actions.append(raw[::16].cpu())
            actions = raw / bound if args_cli.no_clip else (raw / bound).clamp(-1.0, 1.0)
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            real_next_obs = {k: v.clone() for k, v in next_obs.items()}
            if "final_info" in infos:
                for k in real_next_obs:
                    real_next_obs[k][truncs] = infos["final_observation"][k][truncs]
                episode = infos["final_info"]["episode"]
                successes.append(episode["success_once"].mean().item())
                success_end.append(episode["success_at_end"].mean().item())
                returns.append(episode["return"].mean().item())
                print(f"[collect] episode round {len(successes)}: " + ", ".join(f"{k} {v.float().mean().item():.3f}" for k, v in episode.items()), flush=True)
                key = {"strict": "success_once", "lenient": "lenient_success_once"}.get(args_cli.success_only)
                keep_rows.append(episode[key].bool() if key else torch.ones(n, dtype=torch.bool, device=device))
                tid = infos["final_info"]["task_id"]
                print("[collect]   by reset type (strict | lenient): " + ", ".join(
                    f"{t}: {episode['success_once'][tid == t].mean().item():.3f} | {episode['lenient_success_once'][tid == t].mean().item():.3f} (n={int((tid == t).sum())})"
                    for t in range(int(tid.max()) + 1) if (tid == t).any()), flush=True)
            rb.add(obs, real_next_obs, actions, rewards, terms)
            obs = next_obs
    assert rb.current_size == size and rb.step_num == 0 and rb.batch_start_episode == 0, "buffer must hold whole episodes"
    raw = torch.cat(raw_actions).abs()
    q = torch.quantile(raw.float(), torch.tensor([0.5, 0.9, 0.99]), dim=0)
    print(f"[collect] |expert action| per dim p50 {q[0].tolist()}\n          p90 {q[1].tolist()}\n          p99 {q[2].tolist()}")
    print(f"[collect] fraction of action entries beyond +-{scale}{' (kept, --no_clip)' if args_cli.no_clip else ' (clipped)'}: "
          f"{float((raw > bound.cpu()).float().mean()):.3f}")
    metadata = dict(
        task=args_cli.task, image_keys=list(env.image_keys), rgb_shape=tuple(rgb_shape), state_dim=state_dim,
        horizon=args_cli.horizon, action_scale=scale, expert_clip=not args_cli.no_clip, state_group=args_cli.state_group, n_frame_stack=1, checkpoint=args_cli.checkpoint,
        num_envs=n, num_rounds=args_cli.num_rounds, num_transitions=size, policy="deterministic_mean",
        success_once=float(np.mean(successes)), success_at_end=float(np.mean(success_end)), ret=float(np.mean(returns)),
        expert_kind=args_cli.expert_kind,
        reward_min=float(rb.rews_buf.min()), reward_max=float(rb.rews_buf.max()),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    d = rb.state_dict(metadata)
    if args_cli.success_only != "none":
        # buffer row = round * num_envs + env (ImageReplayBuffer.add fills one block of num_envs episodes per round)
        keep = torch.cat(keep_rows).cpu()
        rows = keep.nonzero().flatten()
        k, h = len(rows), args_cli.horizon
        for key in ("pixel_obs_buf", "state_obs_buf", "state_next_obs_buf", "acts_buf", "rews_buf", "done_buf"):
            d[key] = d[key][rows]
        d["idx_to_coord"] = torch.stack(torch.meshgrid(torch.arange(k), torch.arange(h), indexing="ij"), -1).reshape(-1, 2)
        d.update(max_size=k * h, num_envs=1, current_size=k * h, batch_start_episode=0, step_num=0)
        d["metadata"].update(success_only=args_cli.success_only, kept_episodes=k, kept_fraction=k / len(keep),
                             num_transitions=k * h)
        print(f"[collect] success filter ({args_cli.success_only}): kept {k}/{len(keep)} episodes ({k / len(keep):.3f})", flush=True)
    torch.save(d, args_cli.output)
    print(f"[collect] {metadata}")
    print(f"[collect] saved {args_cli.output}")
    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
