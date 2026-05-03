# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRPO (Group Relative Policy Optimization) for finetuning a distilled policy.

Per DeepSeek's R1 recipe and Doorman's Phase-3 finetune of a DAgger student.
Key differences vs vanilla PPO:

* **No learned value function**. Advantages are computed from per-batch
  (or per-group) baseline of trajectory returns. Sidesteps the
  hard-to-learn V(depth) regression that BCPPO got stuck on.
* **KL penalty against a frozen reference policy** (the DAgger snapshot at
  init). Prevents the policy from drifting away from the distilled init when
  the reward signal is sparse.
* **Trajectory MC returns** (not GAE), with bootstrap-by-zero on incomplete
  trajectories — incomplete-episode rows get advantage ~0 (no gradient).

Implementation: subclass rsl_rl's PPO so we reuse its rollout / minibatch
generator / clipping logic. Override:

* ``__init__``: snapshot the policy as reference, store kl_coeff and DAgger
  checkpoint path (optional weight init).
* ``compute_returns``: replace GAE with truncated MC returns, per-batch
  advantage normalization. Skip critic value-target setup.
* ``update``: identical to PPO loop but adds ``kl_coeff * kl(student || ref)``
  per minibatch. Value-loss coefficient is forced to 0 (no critic update).

Init from DAgger: the DAgger student (StudentTeacherVision) keys are
``depth_encoder.*``, ``student.*``, ``student_obs_normalizer.*``, ``std`` /
``log_std``. ActorCriticDepth keys are ``depth_encoder.*``, ``actor.*``,
``actor_obs_normalizer.*``, ``std`` / ``log_std``. Loader renames + skips
the teacher / aux / std_head bits.
"""

from __future__ import annotations

import copy
from collections import deque

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


def load_dagger_into_policy(policy: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """Load DAgger StudentTeacherVision weights into an ActorCriticDepth policy.

    Renames ``student.*`` -> ``actor.*`` and ``student_obs_normalizer.*`` ->
    ``actor_obs_normalizer.*``. Drops teacher / aux_head / std_head keys.
    Critic + critic_obs_normalizer stay at random init (DAgger has no critic).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # rsl_rl checkpoints typically wrap state under 'model_state_dict' or 'model'.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    new_sd = {}
    skipped = []
    for k, v in sd.items():
        if k.startswith("teacher.") or k.startswith("aux_head.") or k.startswith("std_head."):
            skipped.append(k)
            continue
        if k.startswith("student."):
            new_sd[k.replace("student.", "actor.", 1)] = v
        elif k.startswith("student_obs_normalizer."):
            new_sd[k.replace("student_obs_normalizer.", "actor_obs_normalizer.", 1)] = v
        else:
            new_sd[k] = v
    # Some policies (e.g. ActorCriticDepth) override load_state_dict to return
    # a bool instead of the standard _IncompatibleKeys tuple. Bypass via
    # nn.Module.load_state_dict to always get the tuple back.
    result = nn.Module.load_state_dict(policy, new_sd, strict=False)
    missing = list(result.missing_keys) if hasattr(result, "missing_keys") else []
    unexpected = list(result.unexpected_keys) if hasattr(result, "unexpected_keys") else []
    print(
        f"[GRPO] Loaded DAgger checkpoint '{ckpt_path}': mapped {len(new_sd)} keys, "
        f"skipped {len(skipped)} (teacher/aux/std_head). missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"  missing keys (will use random init): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  unexpected keys (ignored): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")


class GRPO(PPO):
    def __init__(
        self,
        policy,
        kl_coeff: float = 0.04,
        kl_target: float | None = None,
        kl_coeff_min: float = 1e-6,
        kl_coeff_max: float = 10.0,
        init_from_dagger_path: str = "",
        clamp_advantage_quantile: float | None = None,
        group_size: int = 1,
        normalize_grouped_advantages: bool = False,
        boost_init_std: float = 0.0,
        lock_depth_encoder: bool = True,
        **kwargs,
    ) -> None:
        # Force value_loss_coef=0 so critic doesn't train (we don't use V).
        kwargs.setdefault("value_loss_coef", 0.0)
        super().__init__(policy, **kwargs)
        # Optionally init policy from DAgger checkpoint (renaming keys).
        if init_from_dagger_path:
            load_dagger_into_policy(self.policy, init_from_dagger_path)
            self.policy.to(self.device)
        # Optionally overwrite the policy std to a larger value to force
        # exploration. The DAgger checkpoint typically has a narrow std
        # (~0.10) which means very little exploration around the deterministic
        # action — on tasks where DAgger has 0% success (e.g. ZeroGAnywhere),
        # GRPO's per-group baseline is identically 0 and no gradient signal
        # exists. Boosting std restores exploration so some envs randomly
        # succeed and a baseline-relative advantage is computable.
        if boost_init_std > 0 and init_from_dagger_path:
            with torch.no_grad():
                if hasattr(self.policy, "std"):
                    self.policy.std.data.fill_(float(boost_init_std))
                    print(f"[GRPO] Overwrote policy.std → {boost_init_std} (exploration boost).")
                elif hasattr(self.policy, "log_std"):
                    self.policy.log_std.data.fill_(float(torch.log(torch.tensor(boost_init_std)).item()))
                    print(f"[GRPO] Overwrote policy.log_std → log({boost_init_std}).")
        # Freeze the obs normalizer so its running stats don't drift during finetune.
        # Without this, ``self.policy.actor_obs_normalizer`` keeps updating each env
        # step (via PPO.process_env_step) while the deepcopied reference's normalizer
        # stays frozen — causing the *normalized* obs to differ between current and
        # reference even with identical actor weights, which makes the KL term
        # explode (~3000 by iter 2 even at lr=1e-5). Freezing keeps both at the
        # well-trained DAgger normalizer state.
        for attr in ("actor_obs_normalizer", "critic_obs_normalizer", "student_obs_normalizer"):
            norm = getattr(self.policy, attr, None)
            if norm is not None and hasattr(norm, "count") and hasattr(norm, "until"):
                norm.until = int(norm.count)
                print(f"[GRPO] Froze policy.{attr} at count={int(norm.count)}.")
        # Lock depth_encoder to eval mode permanently so BatchNorm uses its
        # frozen running stats (not per-batch stats). Otherwise the same depth
        # image yields different encoder outputs in self.policy (train mode,
        # batch BN) vs reference_policy (eval mode, running BN) — another major
        # source of unwanted KL inflation.
        depth_encoder = getattr(self.policy, "depth_encoder", None)
        if depth_encoder is not None and lock_depth_encoder:
            # Lock encoder to eval mode so BN uses running stats not batch stats.
            # WARNING: if the DAgger ckpt was trained in BN-batch-stats mode (the
            # default for rsl_rl distillation runner), switching to running stats
            # at GRPO time changes the feature distribution → policy outputs
            # different actions → can crash the robot (we observed 83% abnormal
            # termination in 35001664 with this lock on). Default True for the
            # KL-explosion fix story; flip to False if the ckpt was trained
            # train-mode and you'd rather keep the encoder consistent.
            depth_encoder.eval()
            # Override .train() so the runner's `train_mode()` call doesn't flip
            # this back into BN-batch-stat mode.
            # NOTE: must NOT call self_mod.eval() here — eval() calls train(False)
            # which calls this override → infinite recursion. Set training flag
            # directly on every submodule instead.
            def _no_train(self_mod, mode=True):  # noqa: ARG001
                for m in self_mod.modules():
                    m.training = False
                return self_mod
            import types
            depth_encoder.train = types.MethodType(_no_train, depth_encoder)
            print("[GRPO] Locked policy.depth_encoder to eval mode (BN frozen, .train() overridden).")
        elif depth_encoder is not None:
            print("[GRPO] depth_encoder NOT locked (BN remains in train mode, batch-stats).")
        # Snapshot reference policy (frozen) AFTER potentially loading DAgger.
        # Disable gradient on reference params, but DO NOT call .eval() —
        # leaving reference in train mode keeps its BN layers in batch-stats
        # mode, matching self.policy (also in train mode). For the same input
        # obs_batch both encoders normalize using the same minibatch stats →
        # identical features → small KL between current and reference.
        # If we .eval() the reference, BN switches to running stats (frozen at
        # deepcopy time) while self.policy uses live batch stats → different
        # features → KL explodes (we observed ref_kl=7000+ in 35001682 with
        # this setup).
        self.reference_policy = copy.deepcopy(self.policy)
        for p in self.reference_policy.parameters():
            p.requires_grad = False
        # NOTE: not calling .eval() — see comment above.
        self.kl_coeff = float(kl_coeff)
        self.kl_target = kl_target  # if set, adapt kl_coeff toward this
        self.kl_coeff_min = float(kl_coeff_min)
        self.kl_coeff_max = float(kl_coeff_max)
        # Optional: clamp advantages to a quantile range to stabilize updates
        # when occasional huge returns dominate.
        self.clamp_advantage_quantile = clamp_advantage_quantile
        # Real grouped GRPO. group_size > 1 enables per-group baseline:
        # K envs in a group share a reset state (replicated by MultiResetManager).
        # Per-env first-episode return R_i is compared to the group mean R_g,
        # and advantage A_i = R_i - R_g is broadcast to all transitions of env i
        # within the first episode of the rollout. Subsequent episodes (where the
        # env has reset to an independent random state) get advantage 0 and
        # contribute no policy gradient. Setting group_size <= 1 falls back to
        # the original per-batch baseline path.
        self.group_size = int(group_size)
        self.normalize_grouped_advantages = bool(normalize_grouped_advantages)
        self._update_count = 0
        # Logging buffers
        self._buf_traj_return: deque[float] = deque(maxlen=2048)

    def compute_returns(self, obs) -> None:
        """Compute advantages.

        * group_size <= 1: fall back to truncated MC returns + per-batch normalization.
        * group_size > 1: real grouped GRPO. For each env, sum rewards within the
          *first* episode of the rollout (the only episode that started from the
          group-shared reset state). Compute baseline = mean first-episode return
          across the group. Advantage = R_i - baseline_g, broadcast to all
          first-episode transitions of env i; zero elsewhere.
        """
        T = self.storage.num_transitions_per_env
        rewards = self.storage.rewards  # [T, N, 1]
        dones = self.storage.dones.float()  # [T, N, 1]
        N = rewards.shape[1]

        # Always compute MC returns (used as `returns` field; logged for trajectory diagnostics).
        R = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            R = rewards[t] + self.gamma * (1.0 - dones[t]) * R
            self.storage.returns[t] = R
        with torch.no_grad():
            done_mask = (dones > 0).flatten()
            if done_mask.any():
                done_returns = self.storage.returns.flatten()[done_mask]
                self._buf_traj_return.extend(done_returns.detach().cpu().tolist())

        with torch.no_grad():
            if self.group_size > 1 and N % self.group_size == 0:
                K = self.group_size
                G = N // K
                # First-episode mask: 1 for steps before the env's first done; 0 from the
                # done step onward (so steps after the env reset to an out-of-group state
                # are excluded from both the return sum and the gradient).
                # prior_dones[t] = number of dones at steps [0, t-1].
                prior_dones = torch.cat([torch.zeros_like(dones[:1]), dones[:-1]], dim=0).cumsum(dim=0)
                first_ep_mask = (prior_dones < 1.0).float()  # [T, N, 1]
                # Per-env sum of rewards in first episode.
                R_per_env = (rewards * first_ep_mask).sum(dim=0).squeeze(-1)  # [N]
                # Group baseline: mean over each group.
                # Layout: env i is in group (i % G); leader is the env (i % G).
                # Reshape via group index: group_id[i] = i % G.
                group_id = torch.arange(N, device=R_per_env.device) % G  # [N], values in [0, G)
                # Sum returns per group, then divide by group size K.
                R_group_sum = torch.zeros(G, device=R_per_env.device).scatter_add_(
                    0, group_id, R_per_env
                )
                B_per_group = R_group_sum / K  # [G]
                B_per_env = B_per_group[group_id]  # [N]
                A_per_env = R_per_env - B_per_env  # [N]
                if self.normalize_grouped_advantages:
                    A_per_env = (A_per_env - A_per_env.mean()) / (A_per_env.std() + 1e-8)
                # Broadcast advantage to first-episode transitions only (zero after).
                A = (A_per_env.view(1, N, 1) * first_ep_mask)  # [T, N, 1]
                self.storage.advantages = A
                # Logging diagnostics.
                self._last_first_ep_frac = float(first_ep_mask.mean().item())
                self._last_group_baseline_mean = float(B_per_env.mean().item())
                self._last_advantage_abs_mean = float(A_per_env.abs().mean().item())
            else:
                # Fallback: original per-batch normalization on MC returns.
                A = self.storage.returns.clone()
                A_flat = A.flatten()
                A_mean = A_flat.mean()
                A_std = A_flat.std() + 1e-8
                A = (A - A_mean) / A_std
                if self.clamp_advantage_quantile is not None and 0 < self.clamp_advantage_quantile < 1:
                    lo = torch.quantile(A_flat, self.clamp_advantage_quantile)
                    hi = torch.quantile(A_flat, 1.0 - self.clamp_advantage_quantile)
                    A = A.clamp(min=lo, max=hi)
                self.storage.advantages = A

    def _ref_distribution(self, obs_batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu_ref, sigma_ref) for KL penalty. No grad."""
        with torch.no_grad():
            self.reference_policy.act(obs_batch)
            mu_ref = self.reference_policy.action_mean.detach()
            sigma_ref = self.reference_policy.action_std.detach()
        return mu_ref, sigma_ref

    @staticmethod
    def _kl_gaussian(mu1, sigma1, mu2, sigma2) -> torch.Tensor:
        """KL(N(mu1, sigma1) || N(mu2, sigma2)), summed over action dims, mean over batch."""
        var1 = sigma1.pow(2)
        var2 = sigma2.pow(2)
        kl = 0.5 * (
            (var1 + (mu1 - mu2).pow(2)) / (var2 + 1e-8)
            + torch.log(var2 + 1e-8)
            - torch.log(var1 + 1e-8)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def update(self) -> dict[str, float]:
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl_loss = 0.0
        mean_ref_kl = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,  # unused (no V update)
            advantages_batch,
            returns_batch,        # unused (no V update)
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            # Normalize per minibatch if requested (else GRPO uses per-batch from compute_returns).
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Forward student
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Surrogate (PPO clipped)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # KL against reference policy
            mu_ref, sigma_ref = self._ref_distribution(obs_batch)
            ref_kl = self._kl_gaussian(mu_batch, sigma_batch, mu_ref, sigma_ref)

            loss = surrogate_loss - self.entropy_coef * entropy_batch.mean() + self.kl_coeff * ref_kl

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl_loss += (self.kl_coeff * ref_kl).item()
            mean_ref_kl += ref_kl.item()

        n = self.num_learning_epochs * self.num_mini_batches
        mean_surrogate_loss /= n
        mean_entropy /= n
        mean_kl_loss /= n
        mean_ref_kl /= n

        # Adaptive KL coefficient (DeepSeek-style: target a fixed kl).
        # Capped to [kl_coeff_min, kl_coeff_max] to prevent runaway growth
        # when the policy keeps moving despite KL pull (gradient clipping
        # bounds per-step update, so KL never converges to target if reward
        # gradient also pushes the policy).
        if self.kl_target is not None:
            if mean_ref_kl > 1.5 * self.kl_target:
                self.kl_coeff = min(self.kl_coeff * 1.5, self.kl_coeff_max)
            elif mean_ref_kl < self.kl_target / 1.5:
                self.kl_coeff = max(self.kl_coeff / 1.5, self.kl_coeff_min)

        self._update_count += 1
        self.storage.clear()

        out = {
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_loss": mean_kl_loss,
            "ref_kl": mean_ref_kl,
            "kl_coeff": float(self.kl_coeff),
        }
        if len(self._buf_traj_return) > 0:
            out["mean_traj_return"] = sum(self._buf_traj_return) / len(self._buf_traj_return)
        if hasattr(self, "_last_first_ep_frac"):
            out["first_ep_frac"] = self._last_first_ep_frac
            out["group_baseline_mean"] = self._last_group_baseline_mean
            out["advantage_abs_mean"] = self._last_advantage_abs_mean
        return out
