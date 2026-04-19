# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DistillationRunner with a fixed per-env student/teacher pool split.

Assigns the first ``round(num_envs * student_fraction)`` envs to the student
pool and the rest to the teacher pool; the assignment is fixed for the entire
training run. The mask is:

* passed to :class:`DistillationDAgger` via ``student_mask`` so action routing
  honors the split, and
* stashed on ``env.unwrapped.pool_mask`` so the reset event
  (:class:`MultiResetManager`) can log ``Metrics/success_student_only`` and
  ``Metrics/success_teacher_only`` alongside the usual per-task success rates.

Per-pool success/length are also tracked in the runner's rollout loop as a
fallback: IsaacLab's ``ManagerBasedRLEnv._reset_idx`` wipes ``extras["log"]``
after the reset-mode event terms run, so ``MultiResetManager``'s writes are
only preserved by coincidence at certain rollout cadences. Tracking here is
cadence-independent and writes directly to the SummaryWriter each iter.
"""

from __future__ import annotations

from collections import deque

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.runners import DistillationRunner
from rsl_rl.utils import resolve_obs_groups


class DistillationRunnerSplit(DistillationRunner):
    """DistillationRunner that fixes a per-env student/teacher pool split."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Strip split-only keys before the parent sees the cfg.
        self.student_fraction = float(train_cfg.pop("student_fraction", 0.5))
        if not 0.0 <= self.student_fraction <= 1.0:
            raise ValueError(f"student_fraction must be in [0, 1]; got {self.student_fraction}")

        # DistillationRunner's __init__ builds the algorithm. We need the mask ready
        # before _construct_algorithm is called so DistillationDAgger sees it. The
        # mask depends on env.num_envs which is available right away.
        num_envs = env.num_envs
        num_student = round(num_envs * self.student_fraction)
        mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        mask[:num_student] = True
        self.student_mask = mask
        self.num_student = num_student
        self.num_teacher = num_envs - num_student

        # Expose mask to the env for the reset event's per-pool success logging.
        # Event-side ops are on env.device, runner-side ops on `device`; push a
        # per-device copy to avoid cross-device indexing failures.
        env.unwrapped.pool_mask = mask.to(env.unwrapped.device)

        # Inject student_mask into the algorithm cfg so _construct_algorithm
        # passes it to DistillationDAgger.__init__.
        train_cfg["algorithm"] = dict(train_cfg["algorithm"])
        train_cfg["algorithm"]["student_mask"] = mask

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # Rolling per-pool success buffers (cadence-independent). Populated in
        # the rollout loop below, logged to SummaryWriter each iteration.
        self._pool_buf_len = 1024
        self._student_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)
        self._teacher_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)

        print(
            f"[DistillationRunnerSplit] pool split: {num_student} student / "
            f"{self.num_teacher} teacher (fraction={self.student_fraction})"
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:  # type: ignore[override]
        """Rollout + update loop with per-pool success tracking injected."""
        import os
        import time
        from collections import deque

        import rsl_rl
        from rsl_rl.utils import store_code_state

        # Prepare logging (mirrors parent)
        self._prepare_logging_writer()
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer: deque[float] = deque(maxlen=100)
        lenbuffer: deque[float] = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Cache the underlying env's progress-context reward term for direct per-env success reads.
        # Accessed once; if unavailable, per-pool tracking is skipped for that run.
        reward_mgr = getattr(self.env.unwrapped, "reward_manager", None)
        progress_term = None
        if reward_mgr is not None:
            try:
                progress_term = reward_mgr.get_term_cfg("progress_context").func
            except Exception:
                progress_term = None

        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    # Per-pool success tracking (cadence-independent).
                    if progress_term is not None:
                        done_ids = dones.view(-1).nonzero(as_tuple=False).view(-1)
                        if done_ids.numel() > 0:
                            succ = progress_term.success[done_ids].float()
                            is_student_done = self.student_mask.to(dones.device)[done_ids]
                            student_succ = succ[is_student_done].detach().cpu().tolist()
                            teacher_succ = succ[~is_student_done].detach().cpu().tolist()
                            self._student_success_buf.extend(student_succ)
                            self._teacher_success_buf.extend(teacher_succ)

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Inject rolling per-pool success into ep_infos so it logs alongside
                # other Metrics/* keys (rsl_rl's log() uses the first ep_info's keys
                # as the schema, so inject into every entry for safety).
                pool_extras = {}
                if len(self._student_success_buf) > 0:
                    pool_extras["Metrics/success_student_only"] = sum(self._student_success_buf) / len(
                        self._student_success_buf
                    )
                if len(self._teacher_success_buf) > 0:
                    pool_extras["Metrics/success_teacher_only"] = sum(self._teacher_success_buf) / len(
                        self._teacher_success_buf
                    )
                if pool_extras:
                    if not ep_infos:
                        ep_infos.append(pool_extras)
                    else:
                        for ep in ep_infos:
                            ep.update(pool_extras)

                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
