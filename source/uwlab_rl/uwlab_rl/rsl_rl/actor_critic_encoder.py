from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticWithEncoder(ActorCritic):
    """ActorCritic with per-group MLP encoders for specified observation groups.

    Groups listed in ``encoder_groups`` are passed through separate MLP encoders
    (with per-group normalization) before concatenation.  Groups NOT listed pass
    through directly.  The main actor/critic MLPs operate on the concatenated
    encoded + pass-through features.
    """

    def __init__(self, obs: TensorDict, obs_groups: dict, num_actions: int,
                 encoder_groups: dict | None = None, **kwargs):
        self._encoder_groups = encoder_groups or {}
        self._do_normalize = kwargs.get("actor_obs_normalization", False)

        modified_obs = TensorDict({}, batch_size=obs.batch_size)
        for key in obs.keys():
            if key in self._encoder_groups:
                out_dim = self._encoder_groups[key]["output_dim"]
                modified_obs[key] = torch.zeros(obs.batch_size[0], out_dim)
            else:
                modified_obs[key] = obs[key]

        parent_kwargs = dict(kwargs)
        parent_kwargs["actor_obs_normalization"] = False
        parent_kwargs["critic_obs_normalization"] = False
        super().__init__(modified_obs, obs_groups, num_actions, **parent_kwargs)

        activation = kwargs.get("activation", "elu")
        self.actor_encoders = nn.ModuleDict()
        self.critic_encoders = nn.ModuleDict()
        self._group_normalizers = nn.ModuleDict()

        # NOTE: `set` iteration order depends on PYTHONHASHSEED which is randomized
        # per process. In distributed training this would build the encoder
        # ModuleDicts in different orders on different ranks, so
        # ``self.policy.parameters()`` would yield parameters in different orders
        # per rank and ``broadcast_parameters`` would deadlock on size-mismatched
        # collectives. Sort the groups to keep the construction order deterministic.
        all_groups = sorted(set(obs_groups.get("policy", [])) | set(obs_groups.get("critic", [])))
        for name in all_groups:
            raw_dim = obs[name].shape[-1]
            if self._do_normalize:
                self._group_normalizers[name] = EmpiricalNormalization(raw_dim)
            if name in self._encoder_groups:
                cfg = self._encoder_groups[name]
                self.actor_encoders[name] = MLP(raw_dim, cfg["output_dim"], cfg["hidden_dims"], activation)
                self.critic_encoders[name] = MLP(raw_dim, cfg["output_dim"], cfg["hidden_dims"], activation)

        self._print_summary(obs, obs_groups)

    def _print_summary(self, obs, obs_groups):
        print("ActorCriticWithEncoder:")
        for name in obs_groups.get("policy", []):
            raw = obs[name].shape[-1]
            if name in self._encoder_groups:
                out = self._encoder_groups[name]["output_dim"]
                print(f"  {name}: {raw}d -> encoder -> {out}d")
            else:
                print(f"  {name}: {raw}d (pass-through)")

    def _encode(self, obs: TensorDict, group_names: list[str], encoders: nn.ModuleDict) -> torch.Tensor:
        parts = []
        for name in group_names:
            x = obs[name]
            if name in self._group_normalizers:
                x = self._group_normalizers[name](x)
            if name in encoders:
                x = encoders[name](x)
            parts.append(x)
        return torch.cat(parts, dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode(obs, self.obs_groups["policy"], self.actor_encoders)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode(obs, self.obs_groups["critic"], self.critic_encoders)

    def update_normalization(self, obs: TensorDict) -> None:
        for name, normalizer in self._group_normalizers.items():
            if name in obs.keys():
                normalizer.update(obs[name])
