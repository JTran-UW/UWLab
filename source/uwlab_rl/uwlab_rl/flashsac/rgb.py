# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pixel FlashSAC on Holiday-Robot rsl_rl_flashsac, following ManiSkill ``flashsac_rgb.py``.

A shared CNN + state encoder feeds Holiday's FlashSAC actor / critic with a flat feature. The encoder is trained only by
the critic loss (online features with gradient for (s, s')); the actor and the next-action sample see detached online
features; the target critic sees an EMA target encoder. Images are stored in the replay buffer as 0..255 floats.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from isaaclab.utils import configclass
from rsl_rl.utils import resolve_obs_groups
from rsl_rl_flashsac.models import FlashSACActor, FlashSACCritic
from rsl_rl_flashsac.utils import resolve_callable

from .demo_mixing import FlashSACDemo, FlashSACDemoAlgorithmCfg
from .storage import PerGroupDtypeMemoryEfficientBuffer, PerGroupDtypeTorchUniformBuffer, parse_storage_dtypes

FEATURE_GROUP = "feature"


class UnitConv2d(nn.Conv2d):
    """Conv2d whose filters are renormalized to unit norm (per output channel) by ``normalize_parameters``."""

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        w = self.weight.view(self.out_channels, -1)
        w.div_(w.norm(dim=1, keepdim=True).clamp_min(1e-8))


class UnitLinear(nn.Linear):
    """Linear whose rows are renormalized to unit norm by ``normalize_parameters``."""

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.weight.div_(self.weight.norm(dim=1, keepdim=True).clamp_min(1e-8))


class PixelStateEncoder(nn.Module):
    """ManiSkill FlashSAC-RGB encoder: 4x (3x3 conv + ReLU + maxpool) -> 1x1 conv + ReLU -> unit linear to
    ``image_embed_dim``; state -> unit linear to ``state_embed_dim``; output = concat (no activation).
    ``num_cameras`` > 1 splits the stacked image channels into one conv stack per camera (flattened outputs concatenated);
    ``width`` multiplies every conv's channel count."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        state_dim: int,
        image_embed_dim: int,
        state_embed_dim: int,
        num_cameras: int = 1,
        width: float = 1.0,
    ):
        super().__init__()
        c, h, w = image_shape
        assert c % num_cameras == 0, f"{c} image channels do not split into {num_cameras} cameras"
        self.num_cameras = num_cameras

        def conv_stack(cin: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i, cout in enumerate(max(1, round(c * width)) for c in (16, 32, 64, 64)):
                pool = max(h // 32, 1) if i == 0 else 2
                layers += [UnitConv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.MaxPool2d(pool)]
                cin = cout
            return nn.Sequential(*layers, UnitConv2d(cin, max(1, round(64 * width)), 1), nn.ReLU(), nn.Flatten())

        if num_cameras == 1:
            self.conv = conv_stack(c)
        else:
            self.convs = nn.ModuleList(conv_stack(c // num_cameras) for _ in range(num_cameras))
        with torch.no_grad():
            flat = self._conv(torch.zeros(1, c, h, w)).shape[-1]
        self.image_fc = UnitLinear(flat, image_embed_dim, bias=False)
        self.state_fc = UnitLinear(state_dim, state_embed_dim, bias=False)
        self.out_dim = image_embed_dim + state_embed_dim
        self.normalize_parameters()

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

    def _conv(self, image: torch.Tensor) -> torch.Tensor:
        if self.num_cameras == 1:
            return self.conv(image)
        return torch.cat([conv(x) for conv, x in zip(self.convs, image.chunk(self.num_cameras, dim=1))], dim=-1)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.image_fc(self._conv(image / 255.0)), self.state_fc(state)], dim=-1)


class StateOnlyEncoder(nn.Module):
    """Debug bypass: feature = the raw state input (no CNN, no projection, no parameters)."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.out_dim = state_dim

    def normalize_parameters(self) -> None:
        pass

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return state


class FlashSACRGB(FlashSACDemo):
    """FlashSAC (+ optional expert mixing) on pixel + state observations through a critic-trained shared encoder."""

    def __init__(
        self,
        *args,
        encoder: PixelStateEncoder,
        image_group: str = "image",
        state_group: str = "state",
        encoder_ema: float = 0.05,
        image_embed_dim: int = 256,
        state_embed_dim: int = 64,
        encoder_type: str = "pixel",
        encoder_num_cameras: int = 1,
        encoder_width: float = 1.0,
        encoder_lr_scale: float = 1.0,
        aux_group: str = "",
        aux_head: str = "linear",
        aux_hidden_dim: int = 256,
        aux_weight: float = 1.0,
        aux_dim: int = 0,
        policy_ema: float = 0.0,
        **kwargs,
    ):
        del state_embed_dim, encoder_type, encoder_num_cameras, encoder_width  # consumed by construct_algorithm
        super().__init__(*args, **kwargs)
        self.encoder = encoder.to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.image_group, self.state_group = image_group, state_group
        self.encoder_ema = encoder_ema
        self.encoder_optimizer = self.encoder_scheduler = None
        if encoder_lr_scale == 0.0:
            self.encoder.requires_grad_(False)
        elif any(True for _ in self.encoder.parameters()):
            self.encoder_optimizer = torch.optim.Adam(
                self.encoder.parameters(), lr=encoder_lr_scale * self.critic_optimizer.param_groups[0]["initial_lr"]
            )
            # same LR schedule as the critic (stepped once per critic update)
            self.encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.encoder_optimizer, lr_lambda=self.critic_scheduler.lr_lambdas[0]
            )
        assert not self.use_amp, "FlashSACRGB keeps fp32 (use_amp=False)"
        # auxiliary state prediction: a head on the image embedding regresses the (running-normalized) aux_group obs
        self.aux_group, self.aux_weight, self.image_embed_dim = aux_group, aux_weight, image_embed_dim
        self.aux_head = self.aux_optimizer = None
        if aux_group:
            assert aux_head in ("linear", "mlp"), f"aux_head must be 'linear' or 'mlp', got {aux_head!r}"
            self.aux_head = (
                nn.Linear(image_embed_dim, aux_dim)
                if aux_head == "linear"
                else nn.Sequential(nn.Linear(image_embed_dim, aux_hidden_dim), nn.ReLU(), nn.Linear(aux_hidden_dim, aux_dim))
            ).to(self.device)
            self.aux_optimizer = torch.optim.Adam(self.aux_head.parameters(), lr=self.critic_optimizer.param_groups[0]["initial_lr"])
            self.aux_mean = torch.zeros(aux_dim, device=self.device)
            self.aux_m2 = torch.zeros(aux_dim, device=self.device)
            self.aux_count = torch.zeros((), device=self.device)
        # EMA copy of the policy (actor + encoder, incl. BatchNorm stats), updated after every gradient update
        self.policy_ema = policy_ema
        self.ema_actor = self.ema_encoder = None
        if policy_ema > 0.0:
            self.ema_actor = copy.deepcopy(self.actor).requires_grad_(False)
            self.ema_encoder = copy.deepcopy(self.encoder).requires_grad_(False)

    # -- features --------------------------------------------------------------------------------------------------
    def _encode(self, obs: TensorDict, encoder: nn.Module) -> torch.Tensor:
        return encoder(obs[self.image_group].float(), obs[self.state_group].float())

    def _feature_td(self, feature: torch.Tensor) -> TensorDict:
        return TensorDict({FEATURE_GROUP: feature}, batch_size=[feature.shape[0]])

    # -- rollout ---------------------------------------------------------------------------------------------------
    def act(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            feature = self._encode(obs, self.encoder)
        actions = super().act(self._feature_td(feature))
        self._last_obs = obs  # the replay buffer stores raw pixels + state
        return actions

    def act_random(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act_random(obs)
        self._last_obs = obs
        return actions

    def process_env_step(self, next_obs: TensorDict, rew: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        # substitute the pre-reset observation of timed-out envs here: the base class broadcasts the mask as
        # mask[:, None], which only fits flat observation groups
        if "time_outs" in extras and "time_outs_obs" in extras:
            mask = extras["time_outs"].reshape(-1).bool().to(next_obs.device)
            final = extras["time_outs_obs"]
            next_obs = TensorDict(
                {
                    k: torch.where(mask.view(-1, *[1] * (v.ndim - 1)), final[k].to(v.device), v)
                    for k, v in next_obs.items()
                },
                batch_size=next_obs.batch_size,
            )
            extras = {k: v for k, v in extras.items() if k != "time_outs_obs"}
        super().process_env_step(next_obs, rew, dones, extras)

    # -- learning --------------------------------------------------------------------------------------------------
    def update(self) -> dict:
        if not self.replay_buffer.can_sample():
            return {}
        sums = {"actor": 0.0, "critic": 0.0, "temperature": 0.0, "entropy": 0.0, "aux": 0.0}
        num_actor_updates = 0
        gamma_n = self.gamma**self.n_steps
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for _ in range(num_updates):
            batch = self._sample_batch()
            obs = batch["observation"].to(self.device)
            next_obs = batch["next_observation"].to(self.device)
            actions = batch["action"].to(self.device)
            rewards = batch["reward"].to(self.device)
            terminated = batch["terminated"].to(self.device)
            if self.reward_normalizer is not None:
                rewards = self.reward_normalizer.normalize_rewards(rewards)

            both = torch.cat([obs, next_obs], dim=0)
            online_all = self._encode(both, self.encoder)
            with torch.no_grad():
                target_all = self._encode(both, self.target_encoder)
            feat, next_feat = online_all.detach().chunk(2, dim=0)

            if self.update_step % self.actor_update_period == 0:
                actor_loss, entropy = self._update_actor(feat, next_feat, feat, actions)
                sums["actor"] += actor_loss.item()
                sums["temperature"] += self._update_temperature(entropy).item()
                sums["entropy"] += entropy.item()
                num_actor_updates += 1

            aux_target = obs[self.aux_group].float() if self.aux_group else None
            sums["critic"] += self._update_critic_rgb(
                online_all, target_all, next_feat, actions, rewards, terminated, gamma_n, aux_target
            ).item()
            with torch.no_grad():
                self._ema_update()
                if self.encoder_optimizer is not None:
                    torch._foreach_lerp_(
                        list(self.target_encoder.parameters()), list(self.encoder.parameters()), self.encoder_ema
                    )
                if self.ema_actor is not None:
                    self._policy_ema_update()
            self.update_step += 1
        losses = {
            "actor": sums["actor"] / max(num_actor_updates, 1),
            "critic": sums["critic"] / num_updates,
            "temperature": sums["temperature"] / max(num_actor_updates, 1),
            "entropy": sums["entropy"] / max(num_actor_updates, 1),
        }
        if self.aux_group:
            losses["aux_state"] = self._aux_loss_sum / num_updates
            self._aux_loss_sum = 0.0
        return losses

    @torch.no_grad()
    def _policy_ema_update(self) -> None:
        for ema, live in ((self.ema_actor, self.actor), (self.ema_encoder, self.encoder)):
            torch._foreach_lerp_(list(ema.parameters()), list(live.parameters()), self.policy_ema)
            for e, b in zip(ema.buffers(), live.buffers()):
                if e.is_floating_point():
                    e.lerp_(b, self.policy_ema)
                else:
                    e.copy_(b)

    @torch.no_grad()
    def _aux_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize aux targets by running (Welford, all batches seen) mean / std, updated with this batch."""
        n = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        total = self.aux_count + n
        delta = mean - self.aux_mean
        self.aux_mean += delta * n / total
        self.aux_m2 += var * n + delta.square() * self.aux_count * n / total
        self.aux_count = total
        return (x - self.aux_mean) / (self.aux_m2 / self.aux_count).sqrt().clamp_min(1e-4)

    def _update_critic_rgb(
        self,
        online_all: torch.Tensor,
        target_all: torch.Tensor,
        actor_next_obs: torch.Tensor,
        actions_batch: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        gamma_n: float,
        aux_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Holiday ``_update_critic`` with the target critic on target-encoder features and the online critic on online
        (gradient-carrying) features; the critic loss (plus ``aux_weight`` x the aux state-prediction loss) also steps
        the encoder."""
        with torch.no_grad():
            next_actions, next_log_probs = self._actor_sample(actor_next_obs, training=False)
            next_actions, next_log_probs = next_actions.clone(), next_log_probs.clone()
            next_actor_entropy = self.temperature() * next_log_probs
            act_all = torch.cat([actions_batch, next_actions], dim=0)
            qs_all, q_infos_all = self._critic_eval_target(target_all, act_all, training=True)
            next_qs = qs_all.chunk(2, dim=1)[1]
            next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
            next_q_log_probs = self._select_min_q(next_qs, next_q_log_probs)
            target_probs = self._compute_td_target(
                next_q_log_probs,
                rewards,
                terminated,
                next_actor_entropy,
                gamma_n,
                self.critic.num_bins,
                self.critic.min_v,
                self.critic.max_v,
            )
        _, pred_q_infos = self.critic.evaluate(online_all, act_all, training=True)
        pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]
        critic_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1).mean()
        loss = critic_loss
        if aux_target is not None:
            image_embed = online_all[: aux_target.shape[0], : self.image_embed_dim]
            aux_loss = F.mse_loss(self.aux_head(image_embed), self._aux_normalize(aux_target))
            loss = critic_loss + self.aux_weight * aux_loss
            self._aux_loss_sum = getattr(self, "_aux_loss_sum", 0.0) + aux_loss.item()
            self.aux_optimizer.zero_grad(set_to_none=True)

        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        if aux_target is not None:
            self.aux_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()
            self.encoder_scheduler.step()
        with torch.no_grad():
            self._critic_normalize()
            self.encoder.normalize_parameters()
        return critic_loss.detach()

    def train_mode(self) -> None:
        super().train_mode()
        self.encoder.train()
        if self.aux_head is not None:
            self.aux_head.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.encoder.eval()

    def save(self) -> dict:
        saved = super().save()
        saved["encoder_state_dict"] = self.encoder.state_dict()
        if self.aux_head is not None:
            saved["aux_head_state_dict"] = self.aux_head.state_dict()
            saved["aux_optimizer_state_dict"] = self.aux_optimizer.state_dict()
            saved["aux_normalizer"] = {"mean": self.aux_mean, "m2": self.aux_m2, "count": self.aux_count}
        saved["target_encoder_state_dict"] = self.target_encoder.state_dict()
        if self.ema_actor is not None:
            saved["ema_actor_state_dict"] = self.ema_actor.state_dict()
            saved["ema_encoder_state_dict"] = self.ema_encoder.state_dict()
        if self.encoder_optimizer is not None:
            saved["encoder_optimizer_state_dict"] = self.encoder_optimizer.state_dict()
            saved["encoder_scheduler_state_dict"] = self.encoder_scheduler.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        self.encoder.load_state_dict(loaded_dict["encoder_state_dict"], strict=strict)
        self.target_encoder.load_state_dict(loaded_dict["target_encoder_state_dict"], strict=strict)
        if self.encoder_optimizer is not None and (load_cfg is None or load_cfg.get("optimizer", True)):
            self.encoder_optimizer.load_state_dict(loaded_dict["encoder_optimizer_state_dict"])
            self.encoder_scheduler.load_state_dict(loaded_dict["encoder_scheduler_state_dict"])
        if self.aux_head is not None and "aux_head_state_dict" in loaded_dict:
            self.aux_head.load_state_dict(loaded_dict["aux_head_state_dict"], strict=strict)
            norm = loaded_dict["aux_normalizer"]
            self.aux_mean, self.aux_m2, self.aux_count = (norm[k].to(self.device) for k in ("mean", "m2", "count"))
            if load_cfg is None or load_cfg.get("optimizer", True):
                self.aux_optimizer.load_state_dict(loaded_dict["aux_optimizer_state_dict"])
        result = super().load(loaded_dict, load_cfg, strict)
        if self.ema_actor is not None:
            # resume the saved average, else start it from the loaded weights
            self.ema_actor.load_state_dict(loaded_dict.get("ema_actor_state_dict", self.actor.state_dict()))
            self.ema_encoder.load_state_dict(loaded_dict.get("ema_encoder_state_dict", self.encoder.state_dict()))
        return result

    def get_policy(self) -> nn.Module:
        return _PixelPolicy(self)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> FlashSACRGB:
        alg_cfg = cfg["algorithm"]
        alg_class = resolve_callable(alg_cfg.pop("class_name"))
        cfg["actor"].pop("class_name", None)
        cfg["critic"].pop("class_name", None)
        image_group, state_group = alg_cfg.get("image_group", "image"), alg_cfg.get("state_group", "state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        alg_cfg.setdefault("rnd_cfg", None)

        if alg_cfg.get("encoder_type", "pixel") == "state":
            encoder = StateOnlyEncoder(obs[state_group].shape[-1]).to(device)
        else:
            encoder = PixelStateEncoder(
                tuple(obs[image_group].shape[1:]),
                obs[state_group].shape[-1],
                alg_cfg.get("image_embed_dim", 256),
                alg_cfg.get("state_embed_dim", 64),
                num_cameras=alg_cfg.get("encoder_num_cameras", 1),
                width=alg_cfg.get("encoder_width", 1),
            ).to(device)
        feature_obs = TensorDict(
            {FEATURE_GROUP: torch.zeros(obs.batch_size[0], encoder.out_dim)}, batch_size=obs.batch_size
        )
        feature_groups = {"actor": [FEATURE_GROUP], "critic": [FEATURE_GROUP]}
        actor = FlashSACActor(feature_obs, feature_groups, "actor", env.num_actions, **cfg["actor"]).to(device)
        actor.set_action_scaling(env.action_bias.to(device), env.action_scale.to(device))
        critic = FlashSACCritic(
            feature_obs, feature_groups, "critic", 1, num_actions=env.num_actions, **cfg["critic"]
        ).to(device)

        buffer_class = (
            PerGroupDtypeMemoryEfficientBuffer
            if alg_cfg.get("buffer_optimize_memory_usage", True)
            else PerGroupDtypeTorchUniformBuffer
        )
        obs_dtype = alg_cfg.get("buffer_obs_dtype")
        replay_buffer = buffer_class(
            obs=obs,
            num_actions=env.num_actions,
            n_step=alg_cfg.get("n_steps", 1),
            gamma=alg_cfg.get("gamma", 0.99),
            max_length=alg_cfg.get("replay_buffer_size", 1_000_000),
            min_length=alg_cfg.get("buffer_min_length", 10_000),
            sample_batch_size=alg_cfg.get("mini_batch_size", 1024),
            device=alg_cfg.get("buffer_device") or device,
            obs_storage_dtype=parse_storage_dtypes(obs_dtype),
            store_groups=sorted({image_group, state_group} | ({alg_cfg["aux_group"]} if alg_cfg.get("aux_group") else set())),
        )
        if alg_cfg.get("aux_group"):
            alg_cfg["aux_dim"] = obs[alg_cfg["aux_group"]].shape[-1]
        if alg_cfg.get("learning_rate_decay_steps") is None:
            alg_cfg["learning_rate_decay_steps"] = (
                int(cfg["max_iterations"]) * alg_cfg.get("num_learning_epochs", 1) * alg_cfg.get("num_mini_batches", 1)
            )
        return alg_class(
            actor, critic, replay_buffer, encoder=encoder, device=device, **alg_cfg, multi_gpu_cfg=cfg.get("multi_gpu")
        )


class _PixelPolicy(nn.Module):
    """Deterministic policy on raw (image, state) observations, for evaluation."""

    def __init__(self, alg: FlashSACRGB):
        super().__init__()
        self.alg = alg

    @property
    def output_std(self) -> torch.Tensor:
        return self.alg.actor.output_std

    def forward(self, obs: TensorDict) -> torch.Tensor:
        feature = self.alg._encode(obs, self.alg.encoder)
        mean, _ = self.alg.actor.get_mean_and_std(feature, training=False)
        return torch.tanh(mean) * self.alg.actor.action_scale + self.alg.actor.action_bias


@configclass
class FlashSACRGBAlgorithmCfg(FlashSACDemoAlgorithmCfg):
    class_name: str = "uwlab_rl.flashsac:FlashSACRGB"
    image_group: str = "image"
    state_group: str = "state"
    image_embed_dim: int = 256
    state_embed_dim: int = 64
    encoder_ema: float = 0.05
    encoder_type: str = "pixel"
    """"pixel" (CNN + state projection) or "state" (debug bypass: the raw state input is the feature)."""
    encoder_num_cameras: int = 1
    """1 = one CNN over the channel-stacked cameras; N = a separate CNN per camera (channels split into N equal groups)."""
    encoder_width: float = 1.0
    """Channel multiplier for every conv layer (1 = 16/32/64/64 + 64; channels are rounded, so fractions work)."""
    encoder_lr_scale: float = 1.0
    """Encoder LR relative to the critic's; 0 freezes the encoder."""
    aux_group: str = ""
    """Obs group of privileged state targets for an auxiliary state-prediction loss on the image embedding ("" = off).
    The group is stored in the replay buffer (and must be in the expert buffer) but is never an actor / critic input."""
    aux_head: str = "linear"
    """Aux head: "linear", or "mlp" (two linear layers with ``aux_hidden_dim`` ReLU units)."""
    aux_hidden_dim: int = 256
    aux_weight: float = 1.0
    """Weight of the aux MSE (on running-normalized targets) added to the critic loss; it also trains the encoder."""
    policy_ema: float = 0.0
    """Rate of an EMA copy of the policy (actor + encoder, incl. BatchNorm stats) updated after every gradient update and
    saved as ``ema_actor_state_dict`` / ``ema_encoder_state_dict`` (0 = off). A frozen snapshot of the live policy can
    be much worse than the continually updated one; the average is the policy to evaluate / deploy."""
