# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import time
import torch
from abc import ABC, abstractmethod
from typing import Any

try:
    from transformers import DynamicCache  # type: ignore
    _HAS_DYNAMIC_CACHE = True
except Exception:  # noqa: BLE001
    DynamicCache = None  # type: ignore
    _HAS_DYNAMIC_CACHE = False


class ObservationHistoryManager(ABC):
    """Abstract base class for managing observation history."""

    def __init__(self, num_envs: int, n_obs_steps: int, device: torch.device):
        self.num_envs = num_envs
        self.n_obs_steps = n_obs_steps
        self.device = device
        self.history = None
        self.needs_init = set()

    @abstractmethod
    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        """Initialize the history with the first observation."""
        pass

    @abstractmethod
    def update(self, processed_obs: dict[str, torch.Tensor]):
        """Update history with new observations."""
        pass

    @abstractmethod
    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        """Get observation batch for specific environments."""
        pass

    @abstractmethod
    def reset_envs(self, env_indices: list[int]):
        """Reset history for specific environments."""
        pass


class LowDimObservationHistory(ObservationHistoryManager):
    """Manages observation history for low-dimensional policies as a fixed rolling buffer."""

    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        obs_shape = processed_obs["obs"].shape
        history_shape = (self.num_envs, self.n_obs_steps, obs_shape[-1])
        self.history = torch.zeros(history_shape, device=self.device, dtype=processed_obs["obs"].dtype)

    def update(self, processed_obs: dict[str, torch.Tensor]):
        if self.history is None:
            self.initialize(processed_obs)
        if self.needs_init:
            for env_idx in list(self.needs_init):
                first_obs = processed_obs["obs"][env_idx : env_idx + 1]
                for step in range(self.n_obs_steps):
                    self.history[env_idx, step] = first_obs[0]
                self.needs_init.remove(env_idx)
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = processed_obs["obs"]

    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        if self.history is None:
            return {"obs": torch.zeros((len(env_indices), self.n_obs_steps, 0), device=self.device)}
        env_obs = self.history[env_indices]
        return {"obs": env_obs}

    def reset_envs(self, env_indices: list[int]):
        for i in env_indices:
            self.needs_init.add(i)


class ImageObservationHistory(ObservationHistoryManager):
    """Manages observation history for image-based policies as a fixed rolling buffer per key."""

    def __init__(self, num_envs: int, obs_keys: list[str], n_obs_steps: int, device: torch.device):
        super().__init__(num_envs, n_obs_steps, device)
        self.obs_keys = obs_keys

    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        self.history = {}
        for key in self.obs_keys:
            obs_shape = processed_obs[key].shape
            history_shape = (self.num_envs, self.n_obs_steps) + obs_shape[1:]
            self.history[key] = torch.zeros(history_shape, device=self.device, dtype=processed_obs[key].dtype)

    def update(self, processed_obs: dict[str, torch.Tensor]):
        if self.history is None:
            self.initialize(processed_obs)
        if self.needs_init and self.obs_keys is not None:
            for env_idx in list(self.needs_init):
                if env_idx < self.num_envs:
                    for key in self.obs_keys:
                        first_obs = processed_obs[key][env_idx : env_idx + 1]
                        for step in range(self.n_obs_steps):
                            self.history[key][env_idx, step] = first_obs[0]
                    self.needs_init.remove(env_idx)
        if self.obs_keys is not None:
            for key in self.obs_keys:
                self.history[key][:, :-1] = self.history[key][:, 1:].clone()
                self.history[key][:, -1] = processed_obs[key]

    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        if self.history is None or self.obs_keys is None:
            return {}
        obs_batch = {}
        for key in self.obs_keys:
            env_obs = self.history[key][env_indices]
            obs_batch[key] = env_obs
        return obs_batch

    def reset_envs(self, env_indices: list[int]):
        for i in env_indices:
            self.needs_init.add(i)


class ImageObservationSequence(ImageObservationHistory):
    """Stores full trajectory per-environment as lists and returns padded batches with attention masks.

    Used for transformer policies that perform in-context exploration: each env's trajectory grows
    from reset, and the policy receives a padded (B, max_len, ...) batch with an attention_mask of
    shape (B, max_len) where 1 = real observation, 0 = padding.
    """

    def initialize(self, processed_obs: dict[str, torch.Tensor]):
        self.history = {}
        for key in self.obs_keys:
            self.history[key] = [[] for _ in range(self.num_envs)]

    def update(self, processed_obs: dict[str, torch.Tensor], env_indices: list[int]):
        """Append the current observation to each listed env's trajectory, clearing any env flagged for init."""
        if self.history is None:
            self.initialize(processed_obs)

        if self.needs_init and self.obs_keys is not None:
            for env_idx in list(self.needs_init):
                if env_idx < self.num_envs:
                    for key in self.obs_keys:
                        self.history[key][env_idx].clear()
                    self.needs_init.remove(env_idx)

        if self.obs_keys is not None:
            for key in self.obs_keys:
                tensor = processed_obs[key]
                # `tensor` is pre-filtered to the exploration subset (shape: (len(env_indices), ...));
                # `idx` indexes into that subset while `env_idx` is the absolute env index into history.
                for idx, env_idx in enumerate(env_indices):
                    obs = tensor[idx]
                    self.history[key][env_idx].append(obs.detach().clone().to(self.device))

    def get_batch(self, env_indices: list[int]) -> dict[str, torch.Tensor]:
        if self.history is None or self.obs_keys is None:
            return {}

        lengths = [len(self.history[self.obs_keys[0]][env]) for env in env_indices]
        max_len = max(lengths) if lengths else 0

        obs_batch: dict[str, torch.Tensor] = {}

        lengths_tensor = torch.tensor(lengths, device=self.device)
        # (max_len,) < (B, 1) → (B, max_len) bool: True for positions < seq_len, i.e. real obs.
        attention_mask = (torch.arange(max_len, device=self.device).unsqueeze(0) < lengths_tensor.unsqueeze(1)).long()

        for key in self.obs_keys:
            obs_batch[key] = torch.nn.utils.rnn.pad_sequence(
                [torch.stack(self.history[key][env], dim=0) for env in env_indices],
                batch_first=True,
                padding_value=0.0,
            )

        obs_batch["attention_mask"] = attention_mask
        return obs_batch


class TransformerKVCacheManager:
    """Per-environment KV cache for a GPT-2-style transformer policy.

    This class is **token-structure agnostic**: it neither knows nor cares how many tokens
    the policy emits per environment step, what those tokens mean (obs, action, separator,
    …), or in what order they appear. It only tracks, per env:

    * a per-layer K/V store preallocated to shape
      ``(num_layers, num_envs, n_heads, max_seq_len, head_dim)`` for each of keys and values
    * an integer ``lengths[i]`` counting how many tokens have been appended for env ``i``

    The policy is responsible for building ``inputs_embeds``, ``attention_mask``, and
    ``position_ids``; it tells the manager via :meth:`append` how many tokens it added this
    step (``num_new_tokens``), and the manager writes the last ``num_new_tokens`` rows of the
    transformer's returned cache into each env's next free slots.

    This means a policy that e.g. interleaves ``[obs_t, act_t]`` pairs works out of the box:
    it just returns ``num_new_tokens=2`` from its step method and the manager appends both
    rows contiguously per env, correctly tracking per-env lengths even as envs reset at
    different wall-clock times.

    Resets are cheap: only the per-env length counter is zeroed; stale KV bytes are
    harmless because the policy-built ``attention_mask`` gates them out.
    """

    def __init__(
        self,
        num_envs: int,
        num_layers: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        storage_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Args:
            storage_dtype: On-device dtype used for the preallocated KV cache. Default
                ``bfloat16`` halves memory (~17GB → ~8.5GB at 1024 envs × 1024 max_seq_len
                × 4 layers × 4 heads × 64 head_dim) relative to fp32 with negligible
                precision loss for inference. Cast to ``compute_dtype`` at :meth:`gather`
                time before handing to the transformer.
            compute_dtype: Dtype the transformer expects for ``past_key_values`` on its
                forward. Should match the model's parameter dtype (fp32 for an unconverted
                GPT-2, fp16/bf16 if the model was cast for inference).

            Deferred TODO (fp8 cache): ``torch.float8_e4m3fn`` would further halve
            memory but needs per-tensor (or per-head/per-channel) scale factors to stay
            within its ~±448 range without saturating, verified fp8 indexing support on
            the deployed PyTorch, and fp8↔compute-dtype cast overhead on every
            gather/append. Only worth the engineering if bf16 cache becomes the
            bottleneck.
        """
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        # Env-first layout: putting ``num_envs`` in axis 0 makes env-indexed gathers /
        # scatters hit a contiguous per-env slab (all layers × heads × slots × dim),
        # letting us replace the old Python per-layer loop with a single tensor op.
        self.cache_k = torch.zeros(
            (num_envs, num_layers, n_heads, max_seq_len, head_dim),
            device=device, dtype=storage_dtype,
        )
        self.cache_v = torch.zeros(
            (num_envs, num_layers, n_heads, max_seq_len, head_dim),
            device=device, dtype=storage_dtype,
        )
        self.lengths = torch.zeros((num_envs,), device=device, dtype=torch.long)

    def _to_tensor(self, env_ids) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def reset(self, env_ids) -> None:
        if env_ids is None:
            self.lengths.zero_()
            return
        env_ids_t = self._to_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        self.lengths[env_ids_t] = 0

    def gather(self, env_ids):
        """Return (past_key_values, past_lengths, max_past) for the requested envs.

        ``past_key_values`` is a tuple of (K, V) per layer, each shaped
        ``(B, n_heads, max_past, head_dim)`` in ``compute_dtype``, or ``None`` when every
        env starts empty.
        """
        env_ids_t = self._to_tensor(env_ids)
        past_lengths = self.lengths[env_ids_t]
        max_past = int(past_lengths.max().item()) if past_lengths.numel() > 0 else 0
        if max_past == 0:
            return None, past_lengths, 0
        # One-shot env-indexed fetch: all layers / heads / slots at once.
        k = self.cache_k[env_ids_t, :, :, :max_past, :].to(self.compute_dtype)
        v = self.cache_v[env_ids_t, :, :, :max_past, :].to(self.compute_dtype)
        # Split by layer into HF's legacy per-layer (K, V) tuple list.
        past_kvs = tuple(
            (k[:, layer].contiguous(), v[:, layer].contiguous())
            for layer in range(self.num_layers)
        )
        # Deferred TODO (DynamicCache-native): HF converts this legacy tuple into a
        # ``DynamicCache`` internally on every forward. Storing/returning a
        # ``DynamicCache`` directly (mutated in-place via ``.update()``) would skip that
        # wrapping. ``from_legacy_cache`` is a thin reference-storing wrapper though, so
        # this is likely cosmetic — revisit if profiling shows it matters.
        if _HAS_DYNAMIC_CACHE:
            past_kvs = DynamicCache.from_legacy_cache(past_kvs)
        return past_kvs, past_lengths, max_past

    def append(
        self,
        env_ids,
        new_past_key_values,
        past_lengths: torch.Tensor,
        num_new_tokens: int = 1,
    ) -> None:
        """Write the last ``num_new_tokens`` rows of the model's returned cache per env.

        The policy is assumed to have appended ``num_new_tokens`` uniformly across the
        batch this step; the manager writes them into each env's slots
        ``[past_lengths[i], past_lengths[i] + num_new_tokens)``. Per-env ``lengths`` are
        bumped by ``num_new_tokens``. Out-of-bounds slots are clamped to
        ``max_seq_len - 1`` (last-slot overwrite) rather than raising, mirroring the
        behaviour on episode lengths that exceed the preallocated cache.

        Env-first cache layout lets the final scatter be a single advanced-indexed
        assignment per new-token offset, with the (layer, head, head_dim) axes swept
        vectorized in one kernel — no per-layer Python loop.
        """
        assert num_new_tokens >= 1, f"num_new_tokens must be >= 1, got {num_new_tokens}"
        env_ids_t = self._to_tensor(env_ids)
        # Stack the model's per-layer last-``num_new_tokens`` rows into a single
        # (B, num_layers, n_heads, num_new_tokens, head_dim) tensor. The stacking itself
        # iterates over layers in Python but only to collect references; the actual
        # memory traffic is one fused scatter below.
        k_stack = torch.stack(
            [new_past_key_values[layer][0][:, :, -num_new_tokens:, :]
             for layer in range(self.num_layers)],
            dim=1,
        ).to(self.storage_dtype)
        v_stack = torch.stack(
            [new_past_key_values[layer][1][:, :, -num_new_tokens:, :]
             for layer in range(self.num_layers)],
            dim=1,
        ).to(self.storage_dtype)
        # Advanced indexing with (env_ids, slots) tensor pair at non-contiguous axes
        # produces an LHS of shape (B, num_layers, n_heads, head_dim); RHS matches.
        # In the common case ``num_new_tokens == 1`` this is a single scatter.
        for j in range(num_new_tokens):
            slots = (past_lengths + j).clamp(max=self.max_seq_len - 1)
            self.cache_k[env_ids_t, :, :, slots, :] = k_stack[:, :, :, j, :]
            self.cache_v[env_ids_t, :, :, slots, :] = v_stack[:, :, :, j, :]
        self.lengths[env_ids_t] = (past_lengths + num_new_tokens).clamp(max=self.max_seq_len)


class _ProfileAccumulator:
    """Lightweight per-stage wall-clock accumulator for inference profiling.

    Tracks seconds spent in each labeled block (e.g. ``encode``, ``transformer``) plus per-call
    env counts, and emits a compact summary every ``print_every_calls`` invocations. Enabled by
    setting ``DIFFUSION_POLICY_PROFILE=1`` in the environment so runs not interested in timing
    pay nothing.
    """

    def __init__(self, name: str, enabled: bool, print_every_calls: int = 50, device: torch.device | None = None) -> None:
        self.name = name
        self.enabled = enabled
        self.print_every_calls = max(1, int(print_every_calls))
        self.device = device
        self.counts: dict[str, int] = {}
        self.total_s: dict[str, float] = {}
        self.calls = 0
        self.total_envs_stepped = 0
        self.since_last_envs = 0
        self.since_last_s: dict[str, float] = {}

    def step_start(self, num_envs: int) -> None:
        if not self.enabled:
            return
        self.calls += 1
        self.total_envs_stepped += num_envs
        self.since_last_envs += num_envs

    def _sync(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def time_block(self, label: str):
        prof = self

        class _Ctx:
            def __enter__(self_inner):
                if prof.enabled:
                    prof._sync()
                    self_inner._t0 = time.time()
                return self_inner

            def __exit__(self_inner, *a):
                if prof.enabled:
                    prof._sync()
                    dt = time.time() - self_inner._t0
                    prof.total_s[label] = prof.total_s.get(label, 0.0) + dt
                    prof.since_last_s[label] = prof.since_last_s.get(label, 0.0) + dt
                    prof.counts[label] = prof.counts.get(label, 0) + 1
                return False

        return _Ctx()

    def maybe_print(self) -> None:
        if not self.enabled:
            return
        if self.calls % self.print_every_calls != 0:
            return
        parts = [f"{label}={s * 1000 / max(1, self.print_every_calls):.2f}ms/call"
                 for label, s in self.since_last_s.items()]
        env_rate = self.since_last_envs / max(1e-9, sum(self.since_last_s.values())) if self.since_last_s else 0.0
        print(
            f"[profile:{self.name}] call={self.calls} envs_last_window={self.since_last_envs} "
            f"env_throughput={env_rate:.0f} envs/s " + " ".join(parts),
            flush=True,
        )
        self.since_last_envs = 0
        self.since_last_s = {}


class DiffusionPolicyWrapper:
    """Wraps diffusion policy to handle Isaac Lab environment observations and action execution."""

    def __init__(
        self,
        policy,
        device: torch.device,
        n_obs_steps: int = 2,
        num_envs: int = 1,
        mini_batch_size: int = 64,
        use_kv_cache: bool = True,
        kv_cache_max_seq_len: int | None = None,
        kv_cache_storage_dtype: torch.dtype = torch.bfloat16,
        profile_name: str = "exploration",
        sample_action: bool = False,
    ):
        """Initialize the policy wrapper.

        Args:
            policy: The diffusion policy to wrap.
            device: Device to run the policy on.
            n_obs_steps: Number of observation steps to maintain in history (unused by the
                transformer path, which keeps a full per-env trajectory).
            num_envs: Number of environments to handle.
            mini_batch_size: Batch size used to serialize inference calls across envs in
                :meth:`_get_action_chunks`. Bounds peak transformer activation memory while
                keeping per-step throughput high (default 64).
            use_kv_cache: If True *and* the policy is a GPT2-based transformer that exposes
                a ``kv_cached_step`` method, route inference through an incremental
                KV-cached path. The policy decides how many tokens to emit per step (via
                its ``_embed_new_step`` / ``num_new_tokens`` return), so this path works
                for any per-step token structure (1 obs token, interleaved obs+action,
                multi-modal fusion, …) without changes here. Brings per-step cost from
                O(T) to O(1) amortized.
            kv_cache_max_seq_len: Upper bound on the per-env cache length (pre-allocated). Defaults
                to the transformer's ``n_positions`` when available, else 1024. Episodes longer
                than this silently overwrite the last slot rather than crashing.
            kv_cache_storage_dtype: On-device dtype for the preallocated KV cache. Default
                ``bfloat16`` halves memory vs fp32 with negligible precision loss for
                inference; cast back to the model's compute dtype at gather time. Pass
                ``torch.float32`` to keep the storage precision (e.g. for numerical
                equivalence checks against a non-cached forward).
            profile_name: Prefix used for the optional profiling prints emitted when
                ``DIFFUSION_POLICY_PROFILE=1`` is set in the environment.
        """
        self.policy = policy
        self.device = device
        self.n_obs_steps = n_obs_steps
        self.num_envs = num_envs
        self.mini_batch_size = mini_batch_size

        # Persist the sampling-vs-greedy decision on the policy. Read by the
        # InterleavedTransformerImagePolicy's _decode_step / predict_action /
        # _ar_inference_loop. Heads that don't use this attribute are unaffected.
        if hasattr(policy, "sample_action"):
            policy.sample_action = bool(sample_action)
            print(
                f"[DiffusionPolicyWrapper] sample_action={bool(sample_action)} "
                f"({'stochastic' if sample_action else 'greedy'} inference).",
                flush=True,
            )

        self.is_image_policy = self._is_image_policy()
        self.is_transformer = self._is_transformer()
        if hasattr(policy.obs_encoder, "keys"):
            obs_keys = policy.obs_encoder.keys
        else:
            obs_keys = policy.obs_encoder.rgb_keys + policy.obs_encoder.low_dim_keys
        self._policy_obs_keys = list(obs_keys)
        if self.is_transformer:
            self.obs_history_manager = ImageObservationSequence(num_envs, obs_keys, n_obs_steps, device)
        elif self.is_image_policy:
            self.obs_history_manager = ImageObservationHistory(num_envs, obs_keys, n_obs_steps, device)
        else:
            self.obs_history_manager = LowDimObservationHistory(num_envs, n_obs_steps, device)

        # KV-cache setup (transformer-only). The policy is the sole authority on what a
        # "step" means in token space, so we only require it to expose ``kv_cached_step``.
        self.use_kv_cache = bool(use_kv_cache) and self.is_transformer and hasattr(
            policy, "kv_cached_step"
        )
        self.kv_cache: TransformerKVCacheManager | None = None
        if self.use_kv_cache:
            gpt2_cfg = getattr(self.policy, "transformer", None)
            if gpt2_cfg is None or not hasattr(gpt2_cfg, "config"):
                # The policy says it's a transformer but we can't introspect GPT2 dims; fall back.
                self.use_kv_cache = False
            else:
                cfg = gpt2_cfg.config
                num_layers = int(cfg.n_layer)
                n_heads = int(cfg.n_head)
                head_dim = int(cfg.n_embd) // n_heads
                if kv_cache_max_seq_len is None:
                    kv_cache_max_seq_len = int(getattr(cfg, "n_positions", 1024))
                # Use the model's actual parameter dtype as the compute dtype so the
                # cache contents match what the transformer expects on forward.
                try:
                    compute_dtype = next(self.policy.transformer.parameters()).dtype
                except StopIteration:
                    compute_dtype = torch.float32
                self.kv_cache = TransformerKVCacheManager(
                    num_envs=num_envs,
                    num_layers=num_layers,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    max_seq_len=int(kv_cache_max_seq_len),
                    device=device,
                    storage_dtype=kv_cache_storage_dtype,
                    compute_dtype=compute_dtype,
                )
                cache_bytes = (
                    2  # keys + values
                    * num_envs * num_layers * n_heads * int(kv_cache_max_seq_len) * head_dim
                    * torch.finfo(kv_cache_storage_dtype).bits // 8
                )
                print(
                    f"[DiffusionPolicyWrapper] KV cache enabled "
                    f"(layers={num_layers}, heads={n_heads}, head_dim={head_dim}, "
                    f"max_seq_len={kv_cache_max_seq_len}, num_envs={num_envs}, "
                    f"storage_dtype={kv_cache_storage_dtype}, compute_dtype={compute_dtype}, "
                    f"approx_mem={cache_bytes / 1e9:.2f}GB)",
                    flush=True,
                )

        self._kv_prev_action: torch.Tensor | None = None
        if self.use_kv_cache and self.kv_cache is not None and getattr(
            self.policy, "include_action_in_context", False
        ):
            da = int(getattr(self.policy, "action_dim", 0))
            self._kv_prev_action = torch.zeros(num_envs, da, device=device, dtype=torch.float32)
        else:
            if self.is_transformer:
                print(
                    "[DiffusionPolicyWrapper] KV cache disabled — re-encoding full trajectory each step.",
                    flush=True,
                )

        self.action_queue = [[] for _ in range(num_envs)]

        # Discrete-AR head reports ``outputs_raw_action``: action is already
        # decoded from bins, so we don't unnormalize. AR tokens stay on the
        # policy's transient sub-sequence and never enter the persistent cache.
        self._ar_discrete_head = self._detect_ar_discrete_head()
        if self._ar_discrete_head is not None:
            spec = self._ar_discrete_head.get_spec()
            print(
                f"[DiffusionPolicyWrapper] discrete AR action head detected "
                f"(num_bins={spec['num_bins']}, clip_val={spec['clip_val']}, "
                f"action_dim={spec['action_dim']}, gripper_dim={spec['gripper_dim']}). "
                f"Action tokens are NOT added to the main KV cache.",
                flush=True,
            )

        profile_enabled = os.environ.get("DIFFUSION_POLICY_PROFILE", "0") not in ("", "0", "false", "False")
        self._profile = _ProfileAccumulator(
            name=profile_name,
            enabled=profile_enabled,
            print_every_calls=int(os.environ.get("DIFFUSION_POLICY_PROFILE_EVERY", "50")),
            device=device,
        )

        self.policy.reset()

    # Discrete-AR head support. Duck-typed to avoid an import-time dependency
    # on diffusion_policy when the policy is continuous.
    def _detect_ar_discrete_head(self):
        head = getattr(self.policy, "output_head", None)
        if head is None or not getattr(head, "outputs_raw_action", False):
            return None
        if not hasattr(head, "get_spec") or not hasattr(head, "arm_bin_centers"):
            return None
        return head

    def save_discretize_spec(self, path: str) -> str | None:
        """Write the AR head's bin spec next to a checkpoint. No-op for continuous heads."""
        if self._ar_discrete_head is None:
            return None
        spec = self._ar_discrete_head.get_spec()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(spec, f, indent=2)
        return path

    def get_discretize_spec(self) -> dict | None:
        if self._ar_discrete_head is None:
            return None
        return self._ar_discrete_head.get_spec()

    def _is_transformer(self) -> bool:
        """Detect if the policy is a transformer-based model that requires attention_mask."""
        policy_class_name = self.policy.__class__.__name__.lower()
        transformer_indicators = ["transformer", "gpt", "bert", "dpt", "aawr"]
        return any(indicator in policy_class_name for indicator in transformer_indicators)

    def _is_image_policy(self) -> bool:
        """Detect if this is an image policy based on class name."""
        policy_class_name = self.policy.__class__.__name__.lower()
        image_policy_indicators = ["image", "hybrid", "video"]
        return any(indicator in policy_class_name for indicator in image_policy_indicators)

    def reset(self, reset_ids: torch.Tensor):
        """Reset the policy wrapper and clear observation history and action queue."""
        reset_indices = reset_ids.tolist() if hasattr(reset_ids, "tolist") else reset_ids
        for i in reset_indices:
            self.action_queue[i].clear()

        if isinstance(reset_indices, torch.Tensor):
            reset_indices = reset_indices.tolist()
        self.obs_history_manager.reset_envs(reset_indices)
        if self.kv_cache is not None:
            self.kv_cache.reset(reset_indices)
        if self._kv_prev_action is not None:
            rid = torch.tensor(reset_indices, device=self.device, dtype=torch.long)
            self._kv_prev_action[rid] = 0
        # IMPORTANT: In vectorized collection, envs reset asynchronously. A blanket
        # policy.reset() can wipe global model state (or RNG) for *all* envs when only
        # a subset resets, which can look like "OOD after reset" or sudden drops in
        # success. Prefer a per-env reset when supported; otherwise only reset the
        # full policy when every env resets together.
        try:
            self.policy.reset(reset_ids)
        except TypeError:
            try:
                all_envs = (len(reset_indices) == self.num_envs)
            except Exception:
                all_envs = False
            if all_envs:
                self.policy.reset()

    def predict_action(self, obs_dict: dict[str, Any], env_indices: list[int] | None = None) -> torch.Tensor:
        """Predict action given Isaac Lab environment observations.

        For transformer policies, ``obs_dict`` should be pre-filtered to only the envs in
        ``env_indices`` (so their per-env trajectories only grow on steps where this policy
        is actually executed). For other policies, ``obs_dict`` contains all envs and
        ``env_indices`` is ignored.

        Args:
            obs_dict: Raw observations from Isaac Lab environment.
            env_indices: Absolute env indices corresponding to rows of ``obs_dict`` (transformer path).

        Returns:
            Action tensor with shape (len(env_indices), action_dim) for the transformer path, or
            (num_envs, action_dim) otherwise.
        """
        # Fast path: KV-cached incremental inference for transformer policies.
        if self.is_transformer and self.use_kv_cache and self.kv_cache is not None:
            if env_indices is None:
                env_indices = list(range(self.num_envs))
            processed_obs = self._process_obs(obs_dict)
            return self._predict_action_kv_cached(processed_obs, env_indices)

        processed_obs = self._process_obs(obs_dict)

        if self.is_transformer:
            if env_indices is None:
                env_indices = list(range(self.num_envs))
            self.obs_history_manager.update(processed_obs, env_indices)
            need_new_actions = [i for i in range(self.num_envs) if len(self.action_queue[i]) == 0 and i in env_indices]
        else:
            self.obs_history_manager.update(processed_obs)
            need_new_actions = [i for i in range(self.num_envs) if len(self.action_queue[i]) == 0]

        self._profile.step_start(len(env_indices) if self.is_transformer else self.num_envs)
        if need_new_actions:
            with self._profile.time_block("legacy_full_seq"):
                new_actions = self._get_action_chunks(need_new_actions)
            for idx, env_idx in enumerate(need_new_actions):
                self.action_queue[env_idx].extend(new_actions[idx])
        self._profile.maybe_print()

        if self.is_transformer:
            actions = torch.zeros(
                len(env_indices), self.action_queue[env_indices[0]][0].shape[-1], device=self.device, dtype=torch.float32
            )
            for idx, env_idx in enumerate(env_indices):
                actions[idx] = self.action_queue[env_idx].pop(0)
            return actions

        actions = torch.zeros(self.num_envs, self.action_queue[0][0].shape[-1], device=self.device, dtype=torch.float32)
        for i in range(self.num_envs):
            actions[i] = self.action_queue[i].pop(0)

        return actions

    def _predict_action_kv_cached(
        self, processed_obs: dict[str, torch.Tensor], env_indices: list[int]
    ) -> torch.Tensor:
        """Mini-batched KV-cached inference.

        This wrapper is deliberately token-structure agnostic: it only gathers each env's
        padded past KVs, forwards them to ``policy.kv_cached_step``, then appends however
        many new tokens the policy says it added (via ``result["num_new_tokens"]``) into
        per-env storage. Attention mask and position-id construction lives inside the
        policy so that subclasses can change the token structure (e.g. interleaved
        obs+action tokens, separator tokens, multi-token fusion) without any edits here.

        When ``policy.include_action_in_context`` is True (concat or interleaved
        transformer), Isaac observations typically omit ``action``; this path injects
        ``chunk_inputs["action"]`` from ``_kv_prev_action`` (zeros after reset, then the
        last predicted action per env) so ``_embed_new_step`` receives the previous-step
        rollout action the policy was trained with.

        For interleaved (DT-style) policies, the per-step token count ``K >= 2``
        varies across envs: step 0 emits only ``[obs_0]`` (``num_new_tokens=1``)
        while step ``t >= 1`` emits ``[a_{t-1}, (r_{t-1}), obs_t]``
        (``num_new_tokens=K``). The KV manager's ``append`` takes a single scalar
        ``num_new_tokens`` per call, so this path buckets each mini-batch by
        is-first-step (``past_length == 0``) and runs up to two forwards per
        mini-batch to keep the scalar contract. Non-interleaved policies (K=1)
        skip bucketing.
        """
        assert self.kv_cache is not None
        self._profile.step_start(len(env_indices))
        B_total = len(env_indices)
        mini_batch_size = self.mini_batch_size
        out_actions: list[torch.Tensor] = []

        allowed_keys = set(self._policy_obs_keys)
        if getattr(self.policy, "include_action_in_context", False):
            allowed_keys.add("action")
        if getattr(self.policy, "include_reward_in_context", False):
            allowed_keys.add("reward")

        K = int(getattr(self.policy, "tokens_per_step", 1))
        needs_first_step_bucketing = K >= 2

        for start in range(0, B_total, mini_batch_size):
            end = min(start + mini_batch_size, B_total)
            chunk_ids = env_indices[start:end]
            chunk_len = len(chunk_ids)

            # Per-env slice, filtered to the keys the policy's encoder / normalizer know about.
            # The raw Isaac Lab obs dict contains extra fields (e.g. ``joint_vel``) that the
            # legacy path strips implicitly via ``obs_history_manager``; mirror that here so
            # ``_embed_new_step``'s normalizer lookup stays in-vocabulary.
            chunk_inputs: dict[str, torch.Tensor] = {
                key: value[start:end]
                for key, value in processed_obs.items()
                if key in allowed_keys
            }

            if self._kv_prev_action is not None:
                ids_t = torch.tensor(chunk_ids, device=self.device, dtype=torch.long)
                chunk_inputs["action"] = self._kv_prev_action[ids_t]

            if needs_first_step_bucketing:
                chunk_ids_t = torch.tensor(chunk_ids, device=self.device, dtype=torch.long)
                is_first = (self.kv_cache.lengths[chunk_ids_t] == 0)
                first_local = is_first.nonzero(as_tuple=False).flatten().tolist()
                rest_local = (~is_first).nonzero(as_tuple=False).flatten().tolist()
                bucket_locals = [lst for lst in (first_local, rest_local) if lst]
            else:
                bucket_locals = [list(range(chunk_len))]

            chunk_out: list[torch.Tensor | None] = [None] * chunk_len

            for local_idxs in bucket_locals:
                bucket_ids = [chunk_ids[i] for i in local_idxs]
                local_idxs_t = torch.tensor(local_idxs, device=self.device, dtype=torch.long)
                bucket_inputs = {k: v.index_select(0, local_idxs_t) for k, v in chunk_inputs.items()}

                with self._profile.time_block("kv_gather"):
                    past_kvs, past_lengths, max_past = self.kv_cache.gather(bucket_ids)

                with self._profile.time_block("kv_forward"), torch.no_grad():
                    result = self.policy.kv_cached_step(
                        bucket_inputs,
                        past_key_values=past_kvs,
                        past_lengths=past_lengths,
                        max_past=max_past,
                    )

                num_new_tokens = int(result.get("num_new_tokens", 1))
                with self._profile.time_block("kv_append"):
                    self.kv_cache.append(
                        bucket_ids,
                        result["past_key_values"],
                        past_lengths,
                        num_new_tokens=num_new_tokens,
                    )

                acts = result["action"].detach().to(torch.float32)
                if self._kv_prev_action is not None:
                    bucket_ids_t = torch.tensor(bucket_ids, device=self.device, dtype=torch.long)
                    self._kv_prev_action[bucket_ids_t] = acts

                for j, li in enumerate(local_idxs):
                    chunk_out[li] = acts[j]

            out_actions.append(torch.stack(chunk_out, dim=0))  # type: ignore[arg-type]

        self._profile.maybe_print()
        return torch.cat(out_actions, dim=0)

    def _process_obs(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert Isaac Lab observations to format expected by diffusion policy."""
        if isinstance(obs_dict, dict):
            obs = obs_dict.get("policy", obs_dict)
        else:
            obs = obs_dict

        if self.is_image_policy:
            return self._process_image_obs(obs)
        else:
            return self._process_lowdim_obs(obs)

    def _process_image_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process observations for image-based policies with batched operations."""
        processed_obs = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                tensor = value.to(self.device)
            else:
                tensor = torch.tensor(value, device=self.device)
            processed_obs[key] = tensor
        return processed_obs

    def _process_lowdim_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process observations for low-dimensional policies with batched operations."""
        obs_components = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, torch.Tensor):
                obs_components.append(value.to(self.device))
            else:
                obs_components.append(torch.tensor(value, device=self.device))

        if obs_components:
            obs_tensor = torch.cat(obs_components, dim=-1)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            processed_obs = {"obs": obs_tensor}
        else:
            processed_obs = {"obs": torch.zeros((self.num_envs, 0), device=self.device)}

        return processed_obs

    def _get_action_chunks(self, env_indices: list[int]) -> list[torch.Tensor]:
        """Get action chunks for specific environments, mini-batched to bound transformer memory."""
        mini_batch_size = self.mini_batch_size
        mini_batch_actions = []
        for start in range(0, len(env_indices), mini_batch_size):
            end = start + mini_batch_size
            mini_batch_envs = env_indices[start:end]
            obs_batch = self.obs_history_manager.get_batch(mini_batch_envs)
            with torch.no_grad():
                result = self.policy.predict_action(obs_batch)
            if isinstance(result, dict):
                action_chunk = result["action"]
            else:
                action_chunk = result
            mini_batch_actions.extend(action_chunk)
        action_chunk = torch.stack(mini_batch_actions, dim=0)

        action_chunks = []
        if action_chunk.ndim == 3:
            for i in range(action_chunk.shape[0]):
                env_action_chunk = action_chunk[i]
                action_chunks.append(env_action_chunk)
        else:
            # Single action per env: (B, Da) → unsqueeze to (B, 1, Da) as a per-env chunk of length 1.
            action_chunks = action_chunk.unsqueeze(1)

        return action_chunks
