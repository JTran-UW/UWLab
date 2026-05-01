# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tiny ``OnPolicyRunner`` subclass that hooks BCPPO's ``bind_env(env)`` after
algorithm construction.

Needed so BCPPO can read ``env.unwrapped.reward_manager.get_term_cfg(
'progress_context').func.success`` for per-pool success tracking on its eval
envs. rsl_rl's standard ``OnPolicyRunner`` doesn't pass env to the algorithm.
"""

from __future__ import annotations

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class BCPPORunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        if hasattr(self.alg, "bind_env"):
            self.alg.bind_env(env)
