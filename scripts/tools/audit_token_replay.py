# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image audit of a DINO-token replay buffer: run the training rollout (expert envs, SGS, image aug) into the real buffer,
record the raw + augmented images behind every token observation, and match each stored slot to its image by exact tokens."""

import argparse
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--agent", default="flashsac_cfg_entry_point")
p.add_argument("--checkpoint", default="")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=330)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
args_cli, hydra_args = p.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os

import gymnasium as gym
import numpy as np
import torch
from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
from rsl_rl_flashsac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_rl.flashsac import make_flashsac_env
from uwlab_rl.flashsac.expert_envs import attach_expert_envs
from uwlab_tasks.utils.hydra import hydra_task_config


def sig(tokens):
    return tokens[:, ::7, ::8].half().flatten(1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    n = args_cli.num_envs
    env_cfg.scene.num_envs, env_cfg.seed = n, args_cli.seed
    agent_cfg.algorithm.replay_buffer_size = n * args_cli.steps
    agent_cfg.algorithm.buffer_min_length = n
    env = make_flashsac_env(gym.make(args_cli.task, cfg=env_cfg), FlashSACVecEnvWrapper, agent_cfg.clip_actions,
                            agent_cfg.action_bound, agent_cfg.action_bound_scale)
    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    alg, base = runner.alg, env.unwrapped
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
    attach_expert_envs(runner, env, args_cli.task, agent_cfg.device)
    reset = base.event_manager.get_term_cfg("reset_from_reset_states").func
    expert, monitor = reset.expert_mask.clone(), reset.sgs_monitor.clone()
    pick = lambda m, k: m.nonzero().flatten()[:k].tolist()
    policy_ids = pick(~expert & ~monitor, 6)
    keep = pick(expert, 4) + pick(monitor, 2) + policy_ids
    kt = torch.tensor(keep, device=base.device)

    om = base.observation_manager
    gi = om._group_obs_term_names["dino_tokens"].index("tokens")
    cfg = om._group_obs_term_cfgs["dino_tokens"][gi]
    inner, rec = cfg.func, {"raw": [], "aug": [], "sig": []}
    aug = inner.aug
    assert aug is not None, "the task's image augmentation is off"
    last = {}

    def spy(x):
        y = aug(x)
        last["raw"], last["aug"] = x[kt].to(torch.uint8).cpu(), y[kt].clamp(0, 255).round().to(torch.uint8).cpu()
        return y

    class Wrapped:
        def __call__(self, env_, **kw):
            out = inner(env_, **kw)
            rec["raw"].append(last["raw"]), rec["aug"].append(last["aug"]), rec["sig"].append(sig(out[kt]).cpu())
            return out

        def __getattr__(self, name):
            return getattr(inner, name)

    inner.aug = spy
    cfg.func = Wrapped()
    obs = env.get_observations().to(runner.device)
    alg.train_mode()
    states = []
    with torch.no_grad():
        for t in range(args_cli.steps):
            actions = alg.act(obs)
            obs, rew, dones, extras = env.step(actions)
            alg.process_env_step(obs, rew, dones, extras)
            if t % 80 == 0:
                print(f"[audit] step {t}, buffer {alg.replay_buffer._num_in_buffer}, token calls {len(rec['sig'])}", flush=True)
    rb = alg.replay_buffer
    n_steps = rb._num_in_buffer // n
    S = torch.stack(rec["sig"])  # [calls, keep, d]
    img = {k: torch.stack(rec[k]) for k in ("raw", "aug")}
    grp = next(g for g in rb._observations if "dino" in g)
    print(f"[audit] buffer {rb._num_in_buffer} transitions = {n_steps} steps x {n} envs, group {grp} {tuple(rb._observations[grp].shape)} {rb._observations[grp].dtype}, n_step {rb._n_step}", flush=True)
    out = {"keep": keep, "kind": ["expert" if expert[e] else "monitor" if monitor[e] else "policy" for e in keep], "n_steps": n_steps}
    raws, augs, rews, terms, truncs, exact = [], [], [], [], [], []
    for j, e in enumerate(keep):
        idx = torch.arange(n_steps) * n + e
        st = sig(rb._observations[grp][idx.to(rb._device)].float()).cpu()
        eq = torch.stack([(S[:, j] == st[k]).all(1) for k in range(n_steps)])  # [slots, calls]
        exact.append(eq.any(1).numpy())
        c = eq.float().argmax(1)
        raws.append(img["raw"][c, j]), augs.append(img["aug"][c, j])
        g = idx.to(rb._device)
        rews.append(rb._rewards[g].cpu()), terms.append(rb._terminateds[g].cpu()), truncs.append(rb._truncateds[g].cpu())
        print(f"[audit] env {e:3d} {out['kind'][j]:8s}: {int(eq.any(1).sum())}/{n_steps} stored observations match a recorded image exactly; reward sum {float(rews[-1].sum()):.2f}, terminated {int(terms[-1].sum())}, truncated {int(truncs[-1].sum())}", flush=True)
    np.savez_compressed(args_cli.out, raw=torch.stack(raws).numpy(), aug=torch.stack(augs).numpy(), reward=torch.stack(rews).float().numpy().squeeze(),
                        terminated=torch.stack(terms).float().numpy().squeeze(), truncated=torch.stack(truncs).float().numpy().squeeze(), exact=np.stack(exact), **{k: np.array(v) for k, v in out.items()})
    print(f"[audit] wrote {args_cli.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
