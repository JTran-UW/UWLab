# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BAMDP-over-expert-failure-rates env-side machinery.

The BAMDP latent per episode is ``theta = (p_1, ..., p_K)`` — per-strategy
*failure* probability. The lowest p_i is rounded to 0 so one strategy is
always the "rescue" expert. The solver does not observe theta directly; it
only sees task obs + a cost signal (stall bit, rescue-active bit, failure
count) that lets it infer which strategy is the rescue and adapt.

**Everything runs env-side.** :class:`BAMDPLatentSampler` is a reset-mode
``EventTerm`` that:

  1. Samples theta + identifies the rescue strategy.
  2. Loads the K expert policies and the discriminator (once, at env init).
  3. Monkey-patches ``env.action_manager.process_action`` with a wrapper
     that:
       - reads w^(t) from the discriminator on the current obs,
       - mixes per-strategy hazards (path-independent),
       - samples a Bernoulli forced-failure,
       - swaps in a (zero-arm + open-gripper) stall action on trigger,
       - latches the rescue expert for every subsequent step,
       - stashes the rescue expert's intended action on
         ``env.bamdp.last_rescue_action`` so the demo recorder can use it
         as the supervision label.

This way ASTEROID's data-collection script, eval, and anything else that
calls ``env.step`` gets BAMDP semantics for free — the failure injection
is part of the environment, not the rollout caller.

State stored on ``env.bamdp`` (the BAMDPLatentSampler instance):

================  ==============================================================
``theta``                       (num_envs, K) per-env failure rates (rescue idx zeroed).
``rescue_idx``                  (num_envs,)   per-env rescue strategy index (argmin or forced).
``H_target``                    (num_envs, K) target cumulative hazard ``-ln(1-p_i)``.
``H_rem``                       (num_envs, K) remaining hazard budget; spent per step.
``stall_remaining``             (num_envs,)   steps left in a forced-failure stall.
``rescue_active``               (num_envs,)   bool — rescue has taken over.
``failure_count``               (num_envs,)   cumulative forced-failure triggers this episode.
``steps_since_failure``         (num_envs,)   steps since last trigger.
``last_rescue_action``          (num_envs, action_dim) the rescue's intended action this step.
``last_executed_action``        (num_envs, action_dim) what the env actually saw post-injection.
``last_w``                      (num_envs, K) discriminator weights this step.
``last_failure_prob``           (num_envs,)   per-step forced-failure probability this step.
``last_triggered``              (num_envs,)   bool — failure triggered this step.
================  ==============================================================
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


PC_DIM_DEFAULT = 1536  # 512 points * 3
ENCODER_OUT_DIM_DEFAULT = 32


# ---------------------------------------------------------------------------
# Expert + discriminator builders (rebuild from a flat state_dict).
# ---------------------------------------------------------------------------


def _build_mlp(state_dict: dict, prefix: str, activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    """Rebuild a (Linear → activation)+ stack from a flat state_dict.

    Same trick as ``scripts_v2/tools/convert_pc_expert_to_jit.py:build_mlp`` —
    walk ``<prefix>.<i>.weight`` keys in order, recover ``(in, out)`` from
    each tensor's shape, recreate the matching ``nn.Sequential`` with
    activations interleaved between Linears but NOT after the final Linear.
    """
    idx_re = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    indices = sorted(int(m.group(1)) for k in state_dict if (m := idx_re.match(k)))
    if not indices:
        raise KeyError(f"No Linear weights found under prefix {prefix!r} in state_dict.")
    layers: list[nn.Module] = []
    for i, idx in enumerate(indices):
        w = state_dict[f"{prefix}.{idx}.weight"]
        b = state_dict[f"{prefix}.{idx}.bias"]
        out_dim, in_dim = w.shape
        lin = nn.Linear(in_dim, out_dim)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(indices) - 1:
            layers.append(activation())
    return nn.Sequential(*layers)


class _ScenePCExpert(nn.Module):
    """One expert: pc_encoder + actor MLP, eval-only.

    Loads from an RSL-RL .pt checkpoint (state_dict has ``actor_encoders.pointcloud.*``,
    ``_group_normalizers.{proprio,pointcloud}._{mean,std}``, ``actor.*``). Mirrors
    ``ScenePCTeacherMean`` in scripts_v2/tools/convert_pc_expert_to_jit.py.
    """

    def __init__(self, checkpoint_path: str, proprio_dim: int, pc_dim: int, label: str = ""):
        super().__init__()
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        self.label = label or os.path.basename(checkpoint_path)
        self.proprio_dim = proprio_dim
        self.pc_dim = pc_dim

        self.register_buffer("proprio_mean", sd["_group_normalizers.proprio._mean"].clone())
        self.register_buffer("proprio_std", sd["_group_normalizers.proprio._std"].clone())
        self.register_buffer("pc_mean", sd["_group_normalizers.pointcloud._mean"].clone())
        self.register_buffer("pc_std", sd["_group_normalizers.pointcloud._std"].clone())

        self.pc_encoder = _build_mlp(sd, "actor_encoders.pointcloud")
        self.actor = _build_mlp(sd, "actor")

        self.eps = 1e-2

    def forward(self, proprio: torch.Tensor, pointcloud: torch.Tensor) -> torch.Tensor:
        proprio_n = (proprio - self.proprio_mean) / (self.proprio_std + self.eps)
        pc_n = (pointcloud - self.pc_mean) / (self.pc_std + self.eps)
        pc_enc = self.pc_encoder(pc_n)
        x = torch.cat([proprio_n, pc_enc], dim=-1)
        return self.actor(x)


class _ExpertDiscriminator(nn.Module):
    """Per-step P(expert = i | obs) classifier.

    Mirrors ``analysis/train_expert_classifier.py:ExpertClassifier``:
    z-score normalizer → pc MLP encoder → small MLP head over [proprio | pc_enc].
    Loaded from a checkpoint with keys ``state_dict``, ``proprio_dim``, ``pc_dim``.
    """

    def __init__(self, proprio_dim: int, pc_dim: int, num_classes: int):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.pc_dim = pc_dim
        self.num_classes = num_classes

        self.register_buffer("proprio_mean", torch.zeros(proprio_dim))
        self.register_buffer("proprio_std", torch.ones(proprio_dim))
        self.register_buffer("pc_mean", torch.zeros(pc_dim))
        self.register_buffer("pc_std", torch.ones(pc_dim))

        def _mlp(in_d: int, hidden: list[int], out_d: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            last = in_d
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ELU()]
                last = h
            layers += [nn.Linear(last, out_d)]
            return nn.Sequential(*layers)

        self.pc_encoder = _mlp(pc_dim, [256, 128], 32)
        self.head = _mlp(proprio_dim + 32, [128, 64], num_classes)

    @classmethod
    def load(cls, ckpt_path: str, device: torch.device) -> "_ExpertDiscriminator":
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = payload["state_dict"]
        proprio_dim = int(payload["proprio_dim"])
        pc_dim = int(payload["pc_dim"])
        head_bias_keys = [k for k in sd if k.startswith("head.") and k.endswith(".bias")]
        head_bias_keys.sort(key=lambda k: int(k.split(".")[1]))
        num_classes = int(sd[head_bias_keys[-1]].shape[0])
        model = cls(proprio_dim=proprio_dim, pc_dim=pc_dim, num_classes=num_classes).to(device)
        model.load_state_dict(sd)
        model.eval()
        model.requires_grad_(False)
        return model

    def forward(self, proprio: torch.Tensor, pointcloud: torch.Tensor) -> torch.Tensor:
        p = (proprio - self.proprio_mean) / self.proprio_std
        c = (pointcloud - self.pc_mean) / self.pc_std
        c = self.pc_encoder(c)
        return self.head(torch.cat([p, c], dim=-1))


# ---------------------------------------------------------------------------
# The main BAMDP term.
# ---------------------------------------------------------------------------


class BAMDPLatentSampler(ManagerTermBase):
    """Reset-mode event that owns the *entire* BAMDP layer.

    Beyond the reset hook (sample theta, set up budget) this class also:

      * Loads K experts + discriminator at sim-play time.
      * Monkey-patches ``env.action_manager.process_action`` with a
        wrapper that injects forced failures inside ``env.step``.
      * Exposes per-step state (stall remaining, rescue active, last
        rescue action, etc.) on ``env.bamdp`` for the recorder /
        observation-term consumers.

    cfg.params (forwarded by EventManager; declared in ``__call__``):

      ``num_strategies``          K. Must match discriminator num_classes.
      ``p_min``, ``p_max``        Sampling range for each p_i ~ Uniform.
      ``expert_specs``            List of dicts with ``checkpoint``, ``proprio_dim``,
                                  ``pc_dim``, optional ``label``. Order MUST match
                                  the discriminator's class index.
      ``discriminator_ckpt``      Path to ``best.pt`` from train_expert_classifier.py.
      ``proprio_obs_group``       Name of the env obs group with proprio.
      ``pointcloud_obs_group``    Name of the env obs group with the flat scene PC.
      ``stall_steps``             K_stall — length of the stall window.
      ``warmup_steps``            Steps from reset before failures can fire.
      ``discriminator_temperature`` Softmax temperature on disc logits.
      ``arm_action_dims``         Which action dims to zero during stall.
      ``gripper_action_dim``      Which dim is the gripper (``-1`` = last).
      ``open_gripper_value``      Action value that opens the gripper. ``BinaryJointAction``
                                  reads ``actions >= 0`` as open.
      ``force_theta`` (optional)  (K,) sequence — when set, every env's theta is this
                                  (used by calibration to pin a target).
      ``force_rescue_idx``        Optional int — fix the rescue strategy instead of argmin.
                                  ``-1`` is a sentinel meaning "no rescue" (used by
                                  joint-calibration, where we want both strategies to
                                  carry their target p_i and no strategy zeroed out).
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        params = dict(cfg.params or {})

        self.num_strategies: int = int(params.get("num_strategies", 2))
        self.p_min: float = float(params.get("p_min", 0.0))
        self.p_max: float = float(params.get("p_max", 1.0))
        self.n_avg: float = float(params.get("n_avg", 50.0))
        self.proprio_obs_group: str = str(params.get("proprio_obs_group", "proprio"))
        self.pointcloud_obs_group: str = str(params.get("pointcloud_obs_group", "pointcloud"))
        self.stall_steps: int = int(params.get("stall_steps", 4))
        self.warmup_steps: int = int(params.get("warmup_steps", 1))
        self.discriminator_temperature: float = float(params.get("discriminator_temperature", 1.0))
        # If True, take the argmax over discriminator logits — the T → 0 limit.
        # w_i ∈ {0, 1}, all hazard concentrates on the single highest-logit class.
        # No path-independence safety margin: a single misclassified step routes
        # the whole step's hazard to the wrong strategy.
        self.discriminator_argmax: bool = bool(params.get("discriminator_argmax", False))
        self.arm_action_dims = tuple(params.get("arm_action_dims", (0, 1, 2, 3, 4, 5)))
        self.gripper_action_dim: int = int(params.get("gripper_action_dim", -1))
        self.open_gripper_value: float = float(params.get("open_gripper_value", 1.0))
        self.fail_only_when_learner_in_control: bool = bool(
            params.get("fail_only_when_learner_in_control", True)
        )
        # If False, per-step hazard is fixed at -ln(1-p_i)/n_avg (the "without
        # budget" variant from the proposal) and we skip H_rem bookkeeping.
        # Biases realized rate slightly above target per Jensen on (1-f)^N
        # — kept as an ablation knob to verify the budget actually helps.
        self.use_hazard_budget: bool = bool(params.get("use_hazard_budget", True))

        # When True, the env's BAMDP layer is a no-op: no discriminator
        # inference, no hazard mixing, no stall, no rescue takeover. The
        # action the caller passes goes straight through to physics, and
        # ``last_rescue_action`` is set equal to the caller's action so the
        # recorder still produces a clean supervision stream (whatever
        # expert the caller chose to drive that env with).
        #
        # Used for iter-0 ASTEROID data collection: we want clean multi-
        # expert demos with no synthetic failures, training the BC student
        # to *represent* the multimodal expert distribution before any
        # in-context adaptation pressure kicks in.
        self.bamdp_disabled: bool = bool(params.get("bamdp_disabled", False))

        ft = params.get("force_theta")
        self.force_theta = None if ft is None else torch.as_tensor(ft, dtype=torch.float32, device=env.device)
        fri = params.get("force_rescue_idx")
        self.force_rescue_idx: int | None = None if fri is None else int(fri)

        expert_specs = params.get("expert_specs")
        discriminator_ckpt = params.get("discriminator_ckpt")
        if not expert_specs:
            raise ValueError("BAMDPLatentSampler requires `expert_specs` in cfg.params.")
        if not discriminator_ckpt:
            raise ValueError("BAMDPLatentSampler requires `discriminator_ckpt` in cfg.params.")
        if len(expert_specs) != self.num_strategies:
            raise ValueError(
                f"len(expert_specs)={len(expert_specs)} != num_strategies={self.num_strategies}"
            )

        # --- per-env state ---
        # NB: ``num_envs`` and ``device`` are read-only properties on
        # :class:`ManagerTermBase`; we can't shadow them. Use locals + the
        # parent property accessors at the few call sites that need them.
        n_envs = env.scene.num_envs
        device = env.device
        K = self.num_strategies
        self.K = K

        self.theta = torch.zeros(n_envs, K, device=device)
        self.rescue_idx = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.no_rescue_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self.H_target = torch.zeros(n_envs, K, device=device)
        self.H_rem = torch.zeros(n_envs, K, device=device)
        self.stall_remaining = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.rescue_active = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self.failure_count = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.steps_since_failure = torch.zeros(n_envs, dtype=torch.long, device=device)

        # Action-dim buffers — sized at first injection (we don't know action_dim
        # before action_manager finishes init). Allocated lazily.
        self.last_rescue_action: torch.Tensor | None = None
        self.last_executed_action: torch.Tensor | None = None
        self.last_w = torch.zeros(n_envs, K, device=device)
        self.last_failure_prob = torch.zeros(n_envs, device=device)
        self.last_triggered = torch.zeros(n_envs, dtype=torch.bool, device=device)

        # --- experts + discriminator ---
        self.experts = nn.ModuleList(
            [
                _ScenePCExpert(
                    checkpoint_path=spec["checkpoint"],
                    proprio_dim=int(spec.get("proprio_dim", 25)),
                    pc_dim=int(spec.get("pc_dim", PC_DIM_DEFAULT)),
                    label=str(spec.get("label", "")),
                )
                for spec in expert_specs
            ]
        ).to(device)
        self.experts.eval()
        self.experts.requires_grad_(False)

        self.discriminator = _ExpertDiscriminator.load(discriminator_ckpt, device)
        if self.discriminator.num_classes != K:
            raise ValueError(
                f"Discriminator num_classes={self.discriminator.num_classes} != K={K}."
                " Retrain the discriminator with matching class count."
            )

        # --- attach to env ---
        # The action_manager doesn't exist yet at __init__ time (the
        # timeline-PLAY callback that instantiates us fires BEFORE
        # ManagerBasedRLEnv.load_managers). We monkey-patch process_action
        # lazily on the first __call__ (which runs during the env's
        # initial reset, after load_managers + the action_manager are up).
        env.bamdp = self  # type: ignore[attr-defined]
        self._env = env
        self._original_process_action = None
        self._action_patch_installed = False

        # Compute max_episode_length from cfg (env.max_episode_length isn't
        # callable until load_managers has finished).
        sim_dt = float(env.cfg.sim.dt)
        decim = int(env.cfg.decimation)
        ep_len_s = float(env.cfg.episode_length_s)
        import math as _math

        self.max_episode_length: int = _math.ceil(ep_len_s / (sim_dt * decim))

    # ------------------------------------------------------------------
    # EventTerm reset hook.
    # ------------------------------------------------------------------

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int] | torch.Tensor,
        # The EventManager forwards cfg.params to __call__ for its static
        # signature check — declare every param key with a default so it's
        # absorbed. Values were already captured in __init__; we only
        # re-read those that may be live-mutated by calibration scripts.
        num_strategies: int = 2,
        p_min: float = 0.0,
        p_max: float = 1.0,
        n_avg: float = 90.0,
        expert_specs=None,
        discriminator_ckpt=None,
        proprio_obs_group: str = "proprio",
        pointcloud_obs_group: str = "pointcloud",
        stall_steps: int = 4,
        warmup_steps: int = 1,
        discriminator_temperature: float = 1.0,
        discriminator_argmax: bool = False,
        arm_action_dims=(0, 1, 2, 3, 4, 5),
        gripper_action_dim: int = -1,
        open_gripper_value: float = 1.0,
        fail_only_when_learner_in_control: bool = True,
        use_hazard_budget: bool = True,
        bamdp_disabled: bool = False,
        force_theta=None,
        force_rescue_idx=None,
    ) -> None:
        # First-reset hook: install the action_manager monkey-patch.
        # By now `load_managers` has finished and `env.action_manager` exists.
        if not self._action_patch_installed:
            self._original_process_action = self._env.action_manager.process_action
            self._env.action_manager.process_action = self._patched_process_action
            self._action_patch_installed = True

        # Coerce env_ids to a 1-D long tensor on the right device.
        if isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_t.numel() == 0:
            return
        n = int(env_ids_t.numel())
        K = self.K

        if self.force_theta is not None:
            theta = self.force_theta.unsqueeze(0).expand(n, K).clone()
        else:
            theta = torch.rand((n, K), device=self.device) * (self.p_max - self.p_min) + self.p_min

        # rescue index: argmin by default. force_rescue_idx in [0, K) pins it;
        # force_rescue_idx == -1 is the "no rescue" sentinel (no p gets zeroed),
        # used by joint-calibration to enforce both p_i targets simultaneously.
        if self.force_rescue_idx is not None and self.force_rescue_idx >= 0:
            rescue_idx = torch.full((n,), int(self.force_rescue_idx), dtype=torch.long, device=self.device)
            no_rescue = torch.zeros(n, dtype=torch.bool, device=self.device)
        elif self.force_rescue_idx == -1:
            rescue_idx = torch.zeros(n, dtype=torch.long, device=self.device)  # placeholder
            no_rescue = torch.ones(n, dtype=torch.bool, device=self.device)
        else:
            rescue_idx = theta.argmin(dim=-1)
            no_rescue = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Zero out the rescue strategy's p (rescue never fails).
        if not bool(no_rescue.all().item()):
            row_idx = torch.arange(n, device=self.device)
            theta[row_idx, rescue_idx] = torch.where(no_rescue, theta[row_idx, rescue_idx], torch.zeros_like(theta[row_idx, rescue_idx]))

        # Target cumulative hazard: H_i = -ln(1 - p_i).
        H_target = -torch.log1p(-theta.clamp(max=1.0 - 1e-6))

        self.theta[env_ids_t] = theta
        self.rescue_idx[env_ids_t] = rescue_idx
        self.no_rescue_mask[env_ids_t] = no_rescue
        self.H_target[env_ids_t] = H_target
        self.H_rem[env_ids_t] = H_target.clone()
        self.stall_remaining[env_ids_t] = 0
        self.rescue_active[env_ids_t] = False
        self.failure_count[env_ids_t] = 0
        self.steps_since_failure[env_ids_t] = 0
        if self.last_rescue_action is not None:
            self.last_rescue_action[env_ids_t] = 0.0
        if self.last_executed_action is not None:
            self.last_executed_action[env_ids_t] = 0.0

    # ------------------------------------------------------------------
    # The action-injection wrapper (env-side, transparent).
    # ------------------------------------------------------------------

    def _resolve_obs(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Pull proprio + pc tensors from the env's last computed obs_buf.

        Returns ``None`` if the obs hasn't been computed yet (very first
        call before reset() is done) — in which case we skip injection.
        """
        env = self._env
        obs_buf = getattr(env, "obs_buf", None)
        if obs_buf is None:
            return None
        proprio = obs_buf.get(self.proprio_obs_group)
        pc = obs_buf.get(self.pointcloud_obs_group)
        if proprio is None or pc is None:
            return None
        return proprio.to(self.device), pc.to(self.device)

    def _stall_action(self, template: torch.Tensor) -> torch.Tensor:
        out = template.new_zeros(template.shape)
        out[:, list(self.arm_action_dims)] = 0.0
        g = self.gripper_action_dim
        if g < 0:
            g = out.shape[-1] + g
        out[:, g] = self.open_gripper_value
        return out

    @torch.no_grad()
    def _patched_process_action(self, action: torch.Tensor) -> None:
        """Replacement for ``ActionManager.process_action``.

        Injects the BAMDP stall + rescue takeover before forwarding to the
        original process_action. All env.step callers (collect_demos,
        eval, play, …) go through this — no script-side opt-in needed.

        **The learner's pre-override action is restored into
        ``env.action_manager._action`` after dispatch.** This keeps
        downstream ``last_action`` / ``prev_actions`` observations
        consistent with what the *learner* chose, not what the env
        injected — required so the BAMDP failure-injection layer is
        truly part of "dynamics" and doesn't leak through past_action.
        Physics still acts on the overridden action (it's already been
        distributed to each action term's ``_raw_actions`` buffer).
        """
        # Lazy-allocate per-env action buffers now that we know the dim.
        if self.last_rescue_action is None:
            self.last_rescue_action = torch.zeros_like(action)
            self.last_executed_action = torch.zeros_like(action)

        action_learner = action.clone().to(self.device)

        # BAMDP layer disabled — pass action through verbatim. The caller's
        # action IS the supervision label (e.g., iter-0 multi-expert demos):
        # whatever expert was chosen to drive each env, that expert's action
        # gets recorded.
        if self.bamdp_disabled:
            self.last_rescue_action[:] = action_learner
            self.last_executed_action[:] = action_learner
            # Update obs_buf so the recorder picks up the current step's value.
            obs_buf = getattr(self._env, "obs_buf", None)
            if isinstance(obs_buf, dict):
                dc = obs_buf.get("data_collection")
                if isinstance(dc, dict):
                    dc["expert_action_mean"] = action_learner.clone()
                    dc["executed_action"] = action_learner.clone()
            self._original_process_action(action)
            return

        obs = self._resolve_obs()
        if obs is None:
            # Pre-reset call — just pass through unmodified. No rescue / no stall.
            self.last_executed_action[:] = action_learner
            self._original_process_action(action)
            # No swap needed: action == action_learner already.
            return

        proprio, pc = obs

        # 1. Discriminator weights.
        logits = self.discriminator(proprio, pc)
        if self.discriminator_argmax:
            # Hard one-hot — the T → 0 limit. Routes the whole step's hazard
            # onto the highest-logit class. Eliminates leakage but is brittle
            # to single misclassifications.
            class_idx = logits.argmax(dim=-1)
            w = torch.nn.functional.one_hot(class_idx, num_classes=self.K).to(logits.dtype)
        else:
            if self.discriminator_temperature != 1.0:
                logits = logits / self.discriminator_temperature
            w = torch.softmax(logits, dim=-1)  # (B, K)

        # 2. Per-strategy hazard.
        #    Two formulations (per the proposal):
        #
        #    - hazard budget (default): h_i = H_rem_i / n_rem_hat. Self-corrects
        #      for episode-length variation by spending the remaining target
        #      hazard evenly across the estimated remaining steps. We use
        #      ``n_rem_hat = (n_avg - episode_step).clamp(min=1)``.
        #
        #    - fixed schedule (use_hazard_budget=False): h_i = -ln(1-p_i)/n_avg,
        #      *constant* across the episode. Per the proposal, biases the
        #      realized rate slightly above target via Jensen's inequality on
        #      (1-f)^N. Mostly here as an ablation knob.
        episode_step = self._env.episode_length_buf.to(self.device).long()
        if self.use_hazard_budget:
            n_rem = (self.n_avg - episode_step.float()).clamp(min=1.0).unsqueeze(-1)
            h_per_strategy = self.H_rem / n_rem  # (B, K)
        else:
            # Fixed-schedule hazard from the target. Shape (B, K).
            n_avg = max(self.n_avg, 1.0)
            h_per_strategy = self.H_target / n_avg

        # 3. Mixed hazard + per-step failure probability.
        hazard = (w * h_per_strategy).sum(dim=-1)  # (B,)
        failure_prob = 1.0 - torch.exp(-hazard)

        # 4. Eligibility & sampling.
        eligible = (self.stall_remaining == 0) & (~self.rescue_active)
        eligible &= episode_step >= self.warmup_steps
        triggered = (torch.rand(self.num_envs, device=self.device) < failure_prob) & eligible

        if triggered.any():
            self.stall_remaining[triggered] = self.stall_steps
            self.failure_count[triggered] += 1
            self.steps_since_failure[triggered] = 0

        # 5. Build override.
        rescue_act = torch.zeros_like(action)
        # Compute rescue action for every env (cheap — small MLP). Mask afterward.
        all_actions = torch.stack([exp(proprio, pc) for exp in self.experts], dim=1)  # (B, K, A)
        row_idx = torch.arange(self.num_envs, device=self.device)
        rescue_act = all_actions[row_idx, self.rescue_idx]
        # When no_rescue_mask is set (calibration joint mode), there is no
        # rescue takeover — we don't define rescue actions, and rescue_active
        # is never latched. Zero them out for safety.
        if self.no_rescue_mask.any():
            rescue_act = torch.where(self.no_rescue_mask.unsqueeze(-1), torch.zeros_like(rescue_act), rescue_act)

        action_out = action.clone().to(self.device)

        in_stall = self.stall_remaining > 0
        if in_stall.any():
            stall_act = self._stall_action(action_out)
            action_out = torch.where(in_stall.unsqueeze(-1), stall_act, action_out)
            new_stall = (self.stall_remaining - 1).clamp(min=0)
            finished = in_stall & (new_stall == 0)
            self.stall_remaining = new_stall
            # Latch rescue_active only when no_rescue isn't in force AND finished stall.
            self.rescue_active |= finished & (~self.no_rescue_mask)

        # 6. Rescue takeover (post-stall, no_rescue envs never get here).
        active_rescue = self.rescue_active & ~in_stall
        if active_rescue.any():
            action_out = torch.where(active_rescue.unsqueeze(-1), rescue_act, action_out)

        # 7. Spend hazard budget on eligible envs only (in_stall / rescue / pre-warmup keep their budget).
        #    In fixed-schedule mode there's no budget to spend.
        if self.use_hazard_budget:
            spent = (w * h_per_strategy) * eligible.float().unsqueeze(-1)
            self.H_rem -= spent
            self.H_rem.clamp_(min=0.0)

        # 8. Update steps_since_failure (incremented except on trigger steps).
        self.steps_since_failure = torch.where(
            triggered, torch.zeros_like(self.steps_since_failure), self.steps_since_failure + 1
        )

        # 9. Cache for the recorder + observation-term consumers.
        self.last_w = w
        self.last_failure_prob = failure_prob
        self.last_triggered = triggered
        self.last_rescue_action[:] = rescue_act
        self.last_executed_action[:] = action_out

        # 9b. Patch the recorder's view of `data_collection` so this step's
        #     rescue action is the one logged. ObservationManager only
        #     refreshes at end-of-step; without this poke the recorder
        #     would log the *previous* step's rescue action against this
        #     step's obs (one-step lag misalignment).
        obs_buf = getattr(self._env, "obs_buf", None)
        if isinstance(obs_buf, dict):
            dc = obs_buf.get("data_collection")
            if isinstance(dc, dict):
                dc["expert_action_mean"] = rescue_act.clone()
                dc["executed_action"] = action_out.clone()

        # 10. Forward to the original process_action.
        self._original_process_action(action_out)

        # 11. Restore the LEARNER's action into the action manager's
        #     aggregate _action buffer so any obs term that reads
        #     ``env.action_manager.action`` (e.g. ``task_mdp.last_action``)
        #     sees the un-injected action. Physics already locked in
        #     ``action_out`` via each action term's ``_raw_actions``
        #     buffer above — this is purely an obs-side correction.
        #     The action_manager's ``_prev_action`` already shifted to
        #     whatever the previous step's restored ``_action`` was
        #     (also a learner action), so the past_action history stays
        #     consistent with the learner's own timeline.
        self._env.action_manager._action[:] = action_learner


# ---------------------------------------------------------------------------
# Observation functions (theta stays hidden; only cost signals are exposed).
# ---------------------------------------------------------------------------


def compute_rescue_action(
    bamdp: "BAMDPLatentSampler",
    proprio: torch.Tensor,
    pointcloud: torch.Tensor,
) -> torch.Tensor:
    """Per-env action from each env's rescue expert.

    Helper for scripts (e.g. demo collection) that need to drive the env
    with rescue actions BEFORE env.step runs the in-env discriminator+
    injector pass. After env.step ``bamdp.last_rescue_action`` holds the
    same value computed by the env's own pass, but for the action choice
    *this* step we have to compute it here.
    """
    with torch.no_grad():
        stacked = torch.stack([exp(proprio, pointcloud) for exp in bamdp.experts], dim=1)
    row_idx = torch.arange(stacked.shape[0], device=stacked.device)
    return stacked[row_idx, bamdp.rescue_idx]


def _get_state(env) -> BAMDPLatentSampler | None:
    return getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "bamdp", None)


def bamdp_stall_bit(env) -> torch.Tensor:
    """1.0 while a forced-failure stall is in progress for this env, else 0.0."""
    state = _get_state(env)
    if state is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return (state.stall_remaining > 0).float().unsqueeze(-1)


def bamdp_rescue_bit(env) -> torch.Tensor:
    """1.0 once the rescue expert has taken over for this env, else 0.0."""
    state = _get_state(env)
    if state is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return state.rescue_active.float().unsqueeze(-1)


def bamdp_failure_count(env) -> torch.Tensor:
    """Cumulative forced-failure triggers in this episode."""
    state = _get_state(env)
    if state is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return state.failure_count.float().unsqueeze(-1)


def bamdp_steps_since_failure(env) -> torch.Tensor:
    """Steps since the last forced-failure trigger in this episode."""
    state = _get_state(env)
    if state is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return state.steps_since_failure.float().unsqueeze(-1)


def bamdp_rescue_action(env) -> torch.Tensor:
    """Per-env rescue expert's intended action this step.

    Returned shape: ``(num_envs, action_dim)``. Used by the recorder as the
    supervision label (the BAMDP solver should learn to imitate the rescue
    strategy from step 0, irrespective of who actually executed each step).
    """
    state = _get_state(env)
    if state is None or state.last_rescue_action is None:
        # Shape unknown pre-init — return a placeholder vector matching the
        # env's action_dim if available, else 7d (6 arm + 1 gripper).
        action_dim = getattr(env, "action_space", None)
        a_dim = action_dim.shape[-1] if action_dim is not None else 7
        return torch.zeros(env.num_envs, a_dim, device=env.device)
    return state.last_rescue_action.clone()


def bamdp_executed_action(env) -> torch.Tensor:
    """Per-env action that was actually applied this step (post-injection).

    For sanity-check logging: this is what the env saw after the BAMDP
    layer rewrote the learner's action (stall + rescue takeover).
    """
    state = _get_state(env)
    if state is None or state.last_executed_action is None:
        action_dim = getattr(env, "action_space", None)
        a_dim = action_dim.shape[-1] if action_dim is not None else 7
        return torch.zeros(env.num_envs, a_dim, device=env.device)
    return state.last_executed_action.clone()


# Convenience: load expert_specs from a JSON file (used by env cfgs that
# don't want to inline the spec list).
def load_expert_specs_json(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)
