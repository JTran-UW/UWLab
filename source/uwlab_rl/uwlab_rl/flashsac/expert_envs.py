# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expert envs: a fixed subset of the training envs executes a frozen state expert's deterministic actions.

Their transitions go into the normal replay buffer (fresh expert data at the subset's share), replacing a fixed
expert buffer. The subset is the ``expert_mask`` of the ``reset_from_reset_states`` event (``expert_env_every``).
"""

from __future__ import annotations

import contextlib
import importlib.metadata as metadata
import os

import torch


def to_policy_space(env_actions: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Invert the FlashSAC wrapper's env-boundary map ``bias + scale * clamp(a, -1, 1)``."""
    return ((env_actions - bias) / scale).clamp(-1.0, 1.0)


@contextlib.contextmanager
def _single_process():
    # under torch.distributed.run rsl_rl's runner would start a second process group; the expert only does inference
    saved = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = "1"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = saved


def load_expert_policy(env, task: str, agent_entry: str, checkpoint: str, device: str):
    """The expert's deterministic inference policy (rsl_rl on-policy runner, as in collect_flashsac_expert_buffer.py)."""
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(task, agent_entry), metadata.version("rsl-rl-lib"))
    cfg.num_steps_per_env = 1  # inference only: the rollout storage holds every obs group (images included)
    with _single_process():
        runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=device)


def attach_expert_envs(runner, env, task: str, device: str) -> None:
    """Give ``runner.alg`` the expert for the reset event's expert envs (no-op without ``expert_policy_checkpoint``)."""
    alg = runner.alg
    checkpoint = getattr(alg, "expert_policy_checkpoint", "")
    if not checkpoint:
        return
    term = env.unwrapped.event_manager.get_term_cfg("reset_from_reset_states").func
    mask = getattr(term, "expert_mask", None)
    if mask is None or not bool(mask.any()):
        raise ValueError("expert_policy_checkpoint needs reset_from_reset_states.params.expert_env_every > 0")
    policy = load_expert_policy(env, task, alg.expert_policy_agent, checkpoint, device)
    alg.set_expert_envs(policy, mask.nonzero().flatten(), env.action_scale, env.action_bias)
    print(f"[expert_envs] {int(mask.sum())}/{mask.numel()} envs execute {checkpoint} ({alg.expert_obs_group} obs)",
          flush=True)
