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
import os

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


class MAPHead(nn.Module):
    """Multihead attention pooling over one camera's frozen patch tokens: ``num_queries`` learnable queries cross-attend
    to the projected tokens (+ learned position embeddings), then a residual MLP; output (B, num_queries * dim)."""

    def __init__(self, num_tokens: int, token_dim: int, dim: int, num_queries: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"map_dim {dim} does not split into {num_heads} heads"
        self.num_heads = num_heads
        self.norm_in = nn.LayerNorm(token_dim)
        self.proj = nn.Linear(token_dim, dim)
        self.pos = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm_in(tokens)) + self.pos
        k, v = self.kv(x).chunk(2, dim=-1)
        q = self.q(self.queries).expand(tokens.shape[0], -1, -1)
        heads = lambda t: t.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        y = self.out(F.scaled_dot_product_attention(heads(q), heads(k), heads(v)).transpose(1, 2).flatten(2))
        return self.norm_out(y + self.mlp(self.norm_mlp(y))).flatten(1)


class MAPStateEncoder(nn.Module):
    """Frozen-feature encoder: input (B, num_cameras * T, D) patch tokens (camera-major) -> one ``MAPHead`` per camera
    -> unit linear to ``image_embed_dim``; state -> unit linear to ``state_embed_dim``; output = concat."""

    def __init__(
        self,
        token_shape: tuple[int, int],
        state_dim: int,
        image_embed_dim: int,
        state_embed_dim: int,
        num_cameras: int = 1,
        num_queries: int = 8,
        dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        n, d = token_shape
        assert n % num_cameras == 0, f"{n} tokens do not split into {num_cameras} cameras"
        self.num_cameras = num_cameras
        self.heads = nn.ModuleList(MAPHead(n // num_cameras, d, dim, num_queries, num_heads) for _ in range(num_cameras))
        self.image_fc = UnitLinear(num_cameras * num_queries * dim, image_embed_dim, bias=False)
        self.state_fc = UnitLinear(state_dim, state_embed_dim, bias=False)
        self.out_dim = image_embed_dim + state_embed_dim
        self.normalize_parameters()

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

    def forward(self, tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([h(t) for h, t in zip(self.heads, tokens.chunk(self.num_cameras, dim=1))], dim=-1)
        return torch.cat([self.image_fc(pooled), self.state_fc(state)], dim=-1)


def load_timm_dino(model_dir: str):
    """A timm DINOv3 model from a folder named after its pretrained tag, holding ``model.safetensors``."""
    import sys

    timm_dir = os.environ.get("UWLAB_TIMM_DIR", "/workspace/uwlab/_timm")
    if os.path.isdir(timm_dir) and timm_dir not in sys.path:
        sys.path.insert(0, timm_dir)
    import timm

    arch = os.path.basename(model_dir.rstrip("/")).split(".")[0]
    return timm.create_model(arch, pretrained=True, num_classes=0,
                             pretrained_cfg_overlay=dict(file=os.path.join(model_dir, "model.safetensors")))


def dino_head_tokens(model: nn.Module, x: torch.Tensor, split_block: int) -> torch.Tensor:
    """Tokens (prefix + patches) after the first ``split_block`` blocks of a timm Eva / DINOv3 model."""
    x = model.patch_embed(x)
    x, rope = model._pos_embed(x)
    x = model.norm_pre(x)
    for blk in model.blocks[:split_block]:
        x = blk(x, rope=rope)
    return x


class DinoTailMAPEncoder(nn.Module):
    """Trainable tail of a DINOv3 ViT on cached head tokens: input (B, num_cameras * (P + g * g), D) from
    ``dino_head_tokens`` (camera-major); blocks ``split_block:`` + final norm (pretrained, ``backbone_parameters``)
    per camera, then one ``MAPHead`` per camera over the patch tokens -> unit linear to ``image_embed_dim``; state ->
    unit linear to ``state_embed_dim``; output = concat."""

    def __init__(
        self,
        token_shape: tuple[int, int],
        state_dim: int,
        image_embed_dim: int,
        state_embed_dim: int,
        model_dir: str,
        split_block: int = 10,
        num_cameras: int = 2,
        num_queries: int = 8,
        dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        model = load_timm_dino(model_dir).eval()
        n, d = token_shape
        self.num_cameras, self.prefix = num_cameras, model.num_prefix_tokens
        per_cam = n // num_cameras
        grid = int(round((per_cam - self.prefix) ** 0.5))
        assert n % num_cameras == 0 and grid * grid + self.prefix == per_cam, f"{n} tokens do not fit {num_cameras} cameras"
        with torch.no_grad():  # eval-mode rotary embedding (no training-time jitter) for the fixed patch grid
            rope = model.rope.get_embed(shape=(grid, grid)) if model.rope is not None else None
        self.register_buffer("rope", rope.detach().clone() if rope is not None else None)
        self.tail = nn.ModuleList(model.blocks[split_block:])
        self.norm = model.norm
        del model
        self.heads = nn.ModuleList(MAPHead(grid * grid, d, dim, num_queries, num_heads) for _ in range(num_cameras))
        self.image_fc = UnitLinear(num_cameras * num_queries * dim, image_embed_dim, bias=False)
        self.state_fc = UnitLinear(state_dim, state_embed_dim, bias=False)
        self.out_dim = image_embed_dim + state_embed_dim
        self.normalize_parameters()

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.tail.parameters()) + list(self.norm.parameters())

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

    def tail_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, cams * (P + T), D) head tokens -> (cams * B, T, D) final patch tokens (camera-major batch)."""
        x = torch.cat(tokens.chunk(self.num_cameras, dim=1), dim=0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tokens.is_cuda):
            for blk in self.tail:
                x = blk(x, rope=self.rope)
            x = self.norm(x)
        return x[:, self.prefix :].float()

    def forward(self, tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        patches = self.tail_tokens(tokens.float()).chunk(self.num_cameras, dim=0)
        pooled = torch.cat([h(t) for h, t in zip(self.heads, patches)], dim=-1)
        return torch.cat([self.image_fc(pooled), self.state_fc(state)], dim=-1)


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm with fixed (pretrained) running statistics; the affine scale / shift stay trainable."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.eps = bn.eps
        self.register_buffer("running_mean", bn.running_mean.detach().clone())
        self.register_buffer("running_var", bn.running_var.detach().clone())
        self.weight = nn.Parameter(bn.weight.detach().clone())
        self.bias = nn.Parameter(bn.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        shift = self.bias - self.running_mean * scale
        return x * scale.view(1, -1, 1, 1).to(x.dtype) + shift.view(1, -1, 1, 1).to(x.dtype)


def _freeze_batchnorm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child))
        else:
            _freeze_batchnorm(child)


class ResNetStateEncoder(nn.Module):
    """One torchvision ResNet-18 per camera (ImageNet weights from ``weights_path``, BatchNorm statistics frozen,
    bf16 autocast) on the 0..255 channel-stacked cameras -> unit linear to ``image_embed_dim``; state -> unit linear
    to ``state_embed_dim``; output = concat. ``backbone_parameters`` are the ResNet weights (own LR scale)."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        state_dim: int,
        image_embed_dim: int,
        state_embed_dim: int,
        num_cameras: int = 2,
        weights_path: str = "",
    ):
        super().__init__()
        from torchvision.models import resnet18

        assert image_shape[0] == 3 * num_cameras, f"{image_shape[0]} channels are not {num_cameras} RGB cameras"
        self.num_cameras = num_cameras
        trunks = []
        for _ in range(num_cameras):
            net = resnet18(weights=None)
            if weights_path:
                net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            net.fc = nn.Identity()
            _freeze_batchnorm(net)
            trunks.append(net)  # default layout: multi-GPU all-reduce flattens grads with .view(-1)
        self.trunks = nn.ModuleList(trunks)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.image_fc = UnitLinear(512 * num_cameras, image_embed_dim, bias=False)
        self.state_fc = UnitLinear(state_dim, state_embed_dim, bias=False)
        self.out_dim = image_embed_dim + state_embed_dim
        self.normalize_parameters()

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.trunks.parameters())

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "normalize_parameters"):
                m.normalize_parameters()

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        cams = (image / 255.0).chunk(self.num_cameras, dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=image.is_cuda):
            feats = [t((c - self.mean) / self.std) for t, c in zip(self.trunks, cams)]
        return torch.cat([self.image_fc(torch.cat(feats, dim=-1).float()), self.state_fc(state)], dim=-1)


class StateOnlyEncoder(nn.Module):
    """Debug bypass: feature = the raw state input (no CNN, no projection, no parameters)."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.out_dim = state_dim

    def normalize_parameters(self) -> None:
        pass

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return state


class OmniResetImageAug(nn.Module):
    """OmniReset sim2real distillation's training-time image randomization (diffusion_policy
    ``train_mlp_sim2real_image_with_aux_loss``): ColorJitter(0.1, 0.1, 0.1, 0.05) -> GaussianBlur(5, sigma 0.1-2.0) ->
    RandomGrayscale(0.05) -> Gaussian noise (std 0.01), per 3-channel camera, one random draw per call (as torchvision
    applies them to a batch tensor). Input / output in 0-255."""

    def __init__(self):
        super().__init__()
        import torchvision.transforms as T

        self.tf = T.Compose([T.ColorJitter(0.1, 0.1, 0.1, 0.05), T.GaussianBlur(5, (0.1, 2.0)), T.RandomGrayscale(0.05)])

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        cams = [self.tf(cam) for cam in (image.float() / 255.0).split(3, dim=1)]
        return torch.cat([cam + 0.01 * torch.randn_like(cam) for cam in cams], dim=1) * 255.0


class OmniResetImageAugV2(nn.Module):
    """``OmniResetImageAug`` with per-sample draws and a resolution-scaled blur (default 3x3, sigma 0.1-0.6 ~ the
    224 px k5 / sigma 0.1-2.0 blur at 64 px). Every image gets its own jitter / blur / grayscale draw (one random op
    order per call, as torchvision); with ``paired`` the batch is [obs; next_obs] and both halves share each draw.
    Vectorized: no per-image loop. Input / output in 0-255."""

    def __init__(self, blur_kernel: int = 3, blur_sigma: tuple[float, float] = (0.1, 0.6), jitter=(0.1, 0.1, 0.1, 0.05),
                 grayscale_p: float = 0.05, noise_std: float = 0.01):
        super().__init__()
        self.k, self.sigma, self.jitter = blur_kernel, blur_sigma, jitter
        self.grayscale_p, self.noise_std = grayscale_p, noise_std

    @staticmethod
    def _gray(x: torch.Tensor) -> torch.Tensor:
        return (0.2989 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

    def _cam(self, x: torch.Tensor, n: int, reps: int) -> torch.Tensor:
        from torchvision.transforms._functional_tensor import _hsv2rgb, _rgb2hsv

        dev = x.device

        def u(lo, hi):
            return (torch.rand(n, device=dev) * (hi - lo) + lo).repeat(reps).reshape(-1, 1, 1, 1)

        b, c, s, h = self.jitter
        for op in torch.randperm(4).tolist():
            if op == 0:
                x = (x * u(1 - b, 1 + b)).clamp(0, 1)
            elif op == 1:
                f = u(1 - c, 1 + c)
                x = (x * f + self._gray(x).mean(dim=(2, 3), keepdim=True) * (1 - f)).clamp(0, 1)
            elif op == 2:
                f = u(1 - s, 1 + s)
                x = (x * f + self._gray(x) * (1 - f)).clamp(0, 1)
            else:
                hsv = _rgb2hsv(x)
                hsv[:, :1] = (hsv[:, :1] + u(-h, h)) % 1.0
                x = _hsv2rgb(hsv)
        # per-image Gaussian blur: one grouped conv, reflect padding (as torchvision)
        r = self.k // 2
        t = torch.arange(-r, r + 1, device=dev, dtype=x.dtype)
        sig = u(*self.sigma).reshape(-1, 1)
        k1 = torch.exp(-0.5 * (t / sig) ** 2)
        k1 = k1 / k1.sum(dim=1, keepdim=True)
        k2 = (k1[:, :, None] * k1[:, None, :]).repeat_interleave(3, dim=0).unsqueeze(1)
        N, C, H, W = x.shape
        x = torch.nn.functional.pad(x, (r, r, r, r), mode="reflect").reshape(1, N * C, H + 2 * r, W + 2 * r)
        x = torch.nn.functional.conv2d(x, k2, groups=N * C).reshape(N, C, H, W)
        g = (torch.rand(n, device=dev) < self.grayscale_p).repeat(reps).reshape(-1, 1, 1, 1)
        return torch.where(g, self._gray(x).expand_as(x), x)

    @torch.no_grad()
    def forward(self, image: torch.Tensor, paired: bool = False) -> torch.Tensor:
        n = image.shape[0] // 2 if paired else image.shape[0]
        reps = 2 if paired else 1
        cams = [self._cam(cam, n, reps) for cam in (image.float() / 255.0).split(3, dim=1)]
        return torch.cat([cam + self.noise_std * torch.randn_like(cam) for cam in cams], dim=1) * 255.0


BC_GROUP = "expert_action"
"""Replay obs group holding the state expert's action for the stored observation (``expert_bc_weight`` > 0)."""


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
        image_aug: bool = False,
        image_aug_version: int = 1,
        image_aug_blur_sigma_max: float = 0.6,
        expert_bc_weight: float = 0.0,
        expert_bc_on_mean: bool = False,
        actor_eval_bn: bool = False,
        actor_q_warmup: int = 0,
        actor_q_ramp: int = 0,
        expert_bc_gate_zero: float = 0.0,
        expert_bc_gate_start: int = 0,
        critic_pick_every: int = 0,
        critic_pick_n: int = 16,
        actor_q_encoder: bool = False,
        expert_bc_loss: str = "mse",
        expert_bc_std_floor: tuple = (0.05, 0.03, 0.035, 0.06, 0.05, 0.035, 0.33),
        critic_entropy_backup: bool = True,
        expert_bc_weight_final: float = -1.0,
        expert_bc_anneal_start: int = 0,
        expert_bc_anneal_iterations: int = 0,
        expert_bc_gate_every: int = 0,
        expert_bc_gate_low: float = 0.70,
        expert_bc_gate_high: float = 0.76,
        expert_bc_gate_down: float = 0.85,
        expert_bc_gate_up: float = 1.5,
        expert_bc_weight_min: float = 0.01,
        expert_bc_gate_key: str = "Curriculum/reset_event_log/Metrics/monitor_env_end_of_episode_success_rate",
        expert_bc_encoder: bool = False,
        lr_schedule_per_update: bool = False,
        replay_audit_every: int = 0,
        replay_audit_envs: str = "0,1",
        replay_audit_steps: int = 160,
        replay_audit_batch: int = 64,
        map_queries: int = 8,
        map_dim: int = 128,
        map_heads: int = 4,
        encoder_weights: str = "",
        encoder_backbone_lr_scale: float = 0.1,
        dino_split_block: int = 10,
        **kwargs,
    ):
        del state_embed_dim, encoder_num_cameras, encoder_width, map_queries, map_dim, map_heads, encoder_weights  # used by construct_algorithm
        del dino_split_block
        self.token_input = encoder_type in ("map", "dino_tail")
        assert not (self.token_input and image_aug), "map encoder: augment images in the env, before the frozen encoder"
        super().__init__(*args, **kwargs)
        self.encoder = encoder.to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.image_group, self.state_group = image_group, state_group
        self.expert_bc_weight, self._bc_target, self._bc_loss_sum = float(expert_bc_weight), None, 0.0
        self.expert_bc_on_mean, self.critic_entropy_backup = bool(expert_bc_on_mean), bool(critic_entropy_backup)
        self.actor_eval_bn = bool(actor_eval_bn)
        self.actor_q_warmup, self.actor_q_ramp = int(actor_q_warmup), int(actor_q_ramp)
        self.bc_gate_zero = float(expert_bc_gate_zero)
        self.bc_gate_start = int(expert_bc_gate_start)
        self.critic_pick_every, self.critic_pick_n, self._pick_ids = int(critic_pick_every), int(critic_pick_n), None
        self._holdout_arms = self._store_ids = None
        assert expert_bc_loss in ("mse", "weighted_l2"), expert_bc_loss
        self.expert_bc_loss, self._bc_std_floor = expert_bc_loss, torch.tensor(list(expert_bc_std_floor), dtype=torch.float32)
        self._bc_mu_sum = self._bc_sigma_sum = 0.0
        self._pick_noise = None
        assert expert_bc_loss == "mse" or not (expert_bc_on_mean or actor_q_encoder), "weighted_l2 has its own mean/std terms"
        self.actor_q_encoder = bool(actor_q_encoder)
        assert not actor_q_encoder or (actor_eval_bn and expert_bc_on_mean and expert_bc_encoder), 'actor_q_encoder builds on the fixed BC path'
        # expert_bc_weight gates label storage; the loss uses _bc_weight_now (linear anneal, iterations of this process)
        self._bc_weight_now = self.expert_bc_weight
        self.expert_bc_weight_final = self.expert_bc_weight if expert_bc_weight_final < 0 else float(expert_bc_weight_final)
        self.expert_bc_anneal_start, self.expert_bc_anneal_iterations = int(expert_bc_anneal_start), int(expert_bc_anneal_iterations)
        # success-gated BC: every gate_every iterations, lower the weight while success > high, raise it when < low
        self.bc_gate_every, self.bc_gate_key = int(expert_bc_gate_every), expert_bc_gate_key
        self.bc_gate_band, self.bc_gate_steps = (float(expert_bc_gate_low), float(expert_bc_gate_high)), (float(expert_bc_gate_down), float(expert_bc_gate_up))
        self.bc_weight_min, self._bc_gate_value, self._bc_gate_last = float(expert_bc_weight_min), None, float("nan")
        assert not (self.bc_gate_every > 0 and self.expert_bc_anneal_iterations > 0), "use either the BC anneal or the BC gate"
        self.expert_bc_encoder, self._bc_feature, self._bc_encoder_grads = bool(expert_bc_encoder), None, None
        self.lr_schedule_per_update = bool(lr_schedule_per_update)
        self.encoder_ema = encoder_ema
        self.encoder_optimizer = self.encoder_scheduler = None
        if encoder_lr_scale == 0.0:
            self.encoder.requires_grad_(False)
        elif any(True for _ in self.encoder.parameters()):
            base_lr = encoder_lr_scale * self.critic_optimizer.param_groups[0]["initial_lr"]
            backbone = self.encoder.backbone_parameters() if hasattr(self.encoder, "backbone_parameters") else []
            ids = {id(p) for p in backbone}
            groups = [{"params": [p for p in self.encoder.parameters() if id(p) not in ids], "lr": base_lr}]
            if backbone:  # pretrained backbone at a lower LR than the new projection layers
                groups.append({"params": backbone, "lr": base_lr * encoder_backbone_lr_scale})
            self.encoder_optimizer = torch.optim.Adam(groups)
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
        self.image_aug = None
        self.image_aug_paired = image_aug_version == 2
        if image_aug:
            self.image_aug = (OmniResetImageAugV2(blur_sigma=(0.1, image_aug_blur_sigma_max)) if image_aug_version == 2
                              else OmniResetImageAug())
        self.replay_audit_every, self.replay_audit_steps, self.replay_audit_batch = replay_audit_every, replay_audit_steps, replay_audit_batch
        self.replay_audit_envs = [int(e) for e in str(replay_audit_envs).split(",") if e != ""]
        self.replay_audit_dir = ""  # set by train.py (<log_dir>/replay_audit)
        self._audit_calls = 0
        self._audit_num_envs = 0

    # -- replay audit ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _replay_audit(self, both: TensorDict, actions, rewards, terminated) -> None:
        """Save (1) the first ``replay_audit_batch`` update samples exactly as fed to the encoder (post-augmentation) and
        (2) the latest ``replay_audit_steps`` stored replay frames of ``replay_audit_envs``, in time order."""
        rb, k = self.replay_buffer, self.replay_audit_batch
        n = both.batch_size[0] // 2
        out = {"update_call": self._audit_calls, "update_step": self.update_step, "image_group": self.image_group,
               "batch_obs_image": both[self.image_group][:k].detach().cpu(),
               "batch_next_image": both[self.image_group][n:n + k].detach().cpu(),
               "batch_state": both[self.state_group][:k].detach().cpu(),
               "batch_action": actions[:k].detach().cpu(), "batch_reward": rewards[:k].detach().cpu(),
               "batch_terminated": terminated[:k].detach().cpu()}
        ne = self._audit_num_envs
        steps = min(self.replay_audit_steps, rb._num_in_buffer // max(ne, 1))
        if steps > 0 and ne > 0:
            base = (rb._current_idx - torch.arange(steps, 0, -1, device=rb._device) * ne) % rb._max_length
            series = {}
            for e in self.replay_audit_envs:
                idx = (base + e) % rb._max_length
                series[e] = {"image": rb._observations[self.image_group][idx].cpu(),
                             "reward": rb._rewards[idx].cpu(), "terminated": rb._terminateds[idx].cpu(),
                             "truncated": rb._truncateds[idx].cpu(),
                             "expert_env": bool(self.expert_env_ids is not None and (self.expert_env_ids == e).any())}
            out["series"] = series
        os.makedirs(self.replay_audit_dir, exist_ok=True)
        path = os.path.join(self.replay_audit_dir, f"audit_call{self._audit_calls:06d}.pt")
        torch.save(out, path)
        print(f"[replay_audit] wrote {path} (batch {k}, {steps} steps x envs {self.replay_audit_envs})", flush=True)

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
        if self.critic_pick_every > 0:
            actions = self._pick_by_critic(feature, actions)
        # the features carry no expert obs group, so the expert envs are applied here, on the raw obs
        actions = self._apply_expert_envs(obs, actions)
        self._last_obs = self._with_expert_label(obs)  # the replay buffer stores raw pixels + state
        return actions

    def set_holdout_arms(self, arms: torch.Tensor) -> None:
        """Eval-only envs (``arms`` >= 0: 0 plain sample, 1 best of N, 2 worst of N by the critic): never stored in replay."""
        assert self.critic_pick_every > 0, "holdout arms need critic_pick_every > 0"
        self._holdout_arms = arms.to(self.device)
        self._store_ids = (self._holdout_arms < 0).nonzero().flatten()

    @torch.no_grad()
    def _pick_by_critic(self, feature: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Every ``critic_pick_every``-th non-expert env (1 = all) acts with the best of ``critic_pick_n`` policy samples under
        the min of the two critics; holdout envs by their arm."""
        if self._pick_ids is None:
            free = torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device)
            if self.expert_env_ids is not None:
                free[self.expert_env_ids] = False
            arms = self._holdout_arms if self._holdout_arms is not None else torch.full_like(free, -1, dtype=torch.long)
            train = (free & (arms < 0)).nonzero().flatten()[self.critic_pick_every // 2 :: self.critic_pick_every]
            held = (arms >= 0).nonzero().flatten()
            self._pick_ids = torch.cat([train, held])
            self._pick_mode = torch.cat([torch.ones_like(train), arms[held]])  # 0 plain, 1 best, 2 worst
        f, n = feature[self._pick_ids], self.critic_pick_n
        mean, std = self.actor.get_mean_and_std(f, training=False)
        # candidate 0 rides the base exploration noise; the others are redrawn only when it is (same repeat length)
        if self._pick_noise is None or self._pick_noise.shape[:2] != (n, mean.shape[0]) or self._noise_repeat_count == 1:
            self._pick_noise = torch.randn(n, *mean.shape, device=mean.device)
        self._pick_noise[0] = self._cached_noise[self._pick_ids]
        cand = torch.tanh(mean + std * self._pick_noise)
        q = torch.minimum(*self.critic.evaluate(f.repeat(n, 1), cand.flatten(0, 1), training=False)[0]).view(n, -1)
        pick = torch.where(self._pick_mode == 1, q.argmax(0), torch.where(self._pick_mode == 2, q.argmin(0), torch.zeros_like(self._pick_mode)))
        actions = actions.clone()
        actions[self._pick_ids] = cand[pick, torch.arange(q.shape[1], device=q.device)].to(actions.dtype)
        self._last_actions = actions
        return actions

    def act_random(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act_random(obs)
        self._last_obs = self._with_expert_label(obs)
        return actions

    def _with_expert_label(self, obs: TensorDict) -> TensorDict:
        """``obs`` plus the BC target of ``expert_bc_weight`` for every env: the state expert's action (policy space), or
        for ``expert_bc_loss weighted_l2`` its Gaussian in the actor's pre-squash space, [mean_u | std_u]."""
        if self.expert_bc_weight <= 0.0:
            return obs
        assert self.expert_policy is not None, "expert_bc_weight needs expert_policy_checkpoint (expert envs)"
        from .expert_envs import to_policy_space

        with torch.no_grad():
            if self.expert_bc_loss == "weighted_l2":
                label = self._teacher_u(obs.select(self.expert_obs_group))
            else:
                label = to_policy_space(
                    self.expert_policy(obs.select(self.expert_obs_group)), self.expert_action_scale, self.expert_action_bias
                )
        out = obs.copy()
        out[BC_GROUP] = label.float()
        return out

    def _teacher_u(self, expert_obs: TensorDict) -> torch.Tensor:
        """The expert's (heteroscedastic Gaussian, env units) action distribution in the actor's pre-tanh space:
        mean_u = atanh(normalized mean), std_u = normalized std / (1 - mean^2) (first order), floored per dim."""
        pol = self.expert_policy
        pol.distribution.update(pol.mlp(pol.get_latent(expert_obs)))
        m = ((pol.output_mean - self.expert_action_bias) / self.expert_action_scale).clamp(-0.995, 0.995)
        std = pol.output_std / self.expert_action_scale / (1.0 - m * m)
        floor = self._bc_std_floor.to(std.device)
        if not getattr(self, "_teacher_logged", False):
            self._teacher_logged = True
            print(f"[bc] teacher std_u median per dim {[round(float(v), 4) for v in std.median(0).values]}, below floor "
                  f"{[round(float(v), 2) for v in (std < floor).float().mean(0)]}, above 1.0 {[round(float(v), 2) for v in (std > 1.0).float().mean(0)]}, |mean| at clamp per dim {[round(float(v), 2) for v in (m.abs() >= 0.995).float().mean(0)]}, all {float((m.abs() >= 0.995).float().mean()):.3f}", flush=True)
        # a saturated mean (|m| -> 1) blows the first-order std up; 1.0 pre-squash is already near-uniform
        return torch.cat([torch.atanh(m), torch.maximum(std, floor).clamp_max(1.0)], dim=-1)

    def _bc_gate_step(self) -> None:
        """Every ``bc_gate_every`` update calls: rank-mean of the latest gate metric, then scale the BC weight by
        ``down`` above the band or ``up`` below it, within [weight_min, expert_bc_weight]."""
        if self._audit_calls <= self.bc_gate_start or self._audit_calls % self.bc_gate_every != 0:
            return
        have = self._bc_gate_value is not None
        value = torch.tensor([self._bc_gate_value if have else 0.0, float(have)], device=self.device)
        if self.is_multi_gpu:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        if value[1] < (self.gpu_world_size if self.is_multi_gpu else 1):
            return  # wait until every rank has a reading
        s, (low, high), (down, up) = float(value[0] / value[1]), self.bc_gate_band, self.bc_gate_steps
        self._bc_gate_last = s
        if s > high:
            self._bc_weight_now = max(self._bc_weight_now * down, self.bc_weight_min)
            if self._bc_weight_now <= self.bc_gate_zero:
                self._bc_weight_now = 0.0
        elif s < low:
            self._bc_weight_now = min(max(self._bc_weight_now, self.bc_gate_zero) * up, self.expert_bc_weight)

    def process_env_step(self, next_obs: TensorDict, rew: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        if self.bc_gate_every > 0 and self.bc_gate_key in extras.get("log", {}):
            self._bc_gate_value = float(extras["log"][self.bc_gate_key])
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
        next_obs = self._with_expert_label(next_obs)
        if self._store_ids is not None:
            ids = self._store_ids
            next_obs, rew, dones = next_obs[ids], rew.reshape(-1)[ids], dones.reshape(-1)[ids]
            self._last_obs, self._last_actions = self._last_obs[ids], self._last_actions[ids]
            extras = {**extras, "time_outs": extras["time_outs"].reshape(-1)[ids]} if "time_outs" in extras else extras
        super().process_env_step(next_obs, rew, dones, extras)

    # -- learning --------------------------------------------------------------------------------------------------
    def update(self) -> dict:
        if not self.replay_buffer.can_sample():
            return {}
        check_every = int(os.environ.get("FLASHSAC_RGB_RANK_CHECK", "0")) if self.is_multi_gpu else 0
        self._rgb_check_calls = getattr(self, "_rgb_check_calls", 0) + 1
        if check_every and self._rgb_check_calls == 1:
            self._rgb_rank_check("before_first_update")
        sums = {"actor": 0.0, "critic": 0.0, "temperature": 0.0, "entropy": 0.0, "aux": 0.0}
        num_actor_updates = 0
        gamma_n = self.gamma**self.n_steps
        num_updates = self.num_learning_epochs * self.num_mini_batches
        self._audit_calls += 1
        if self.expert_bc_weight > 0.0 and self.bc_gate_every > 0:
            self._bc_gate_step()
        elif self.expert_bc_weight > 0.0:
            k = self._audit_calls - 1 - self.expert_bc_anneal_start
            frac = 0.0 if k < 0 else min(k / self.expert_bc_anneal_iterations, 1.0) if self.expert_bc_anneal_iterations > 0 else 1.0
            self._bc_weight_now = self.expert_bc_weight + frac * (self.expert_bc_weight_final - self.expert_bc_weight)
        audit_now = (self.replay_audit_every > 0 and self.replay_audit_dir and self.gpu_global_rank == 0
                     and self._audit_calls % self.replay_audit_every == 0 and not self.token_input)
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
            if self.image_aug is not None:
                aug_kw = {"paired": True} if self.image_aug_paired else {}
                both[self.image_group] = self.image_aug(both[self.image_group], **aug_kw)
            if audit_now:
                self._replay_audit(both, actions, rewards, terminated)
                audit_now = False
            online_all = self._encode(both, self.encoder)
            with torch.no_grad():
                target_all = self._encode(both, self.target_encoder)
            feat, next_feat = online_all.detach().chunk(2, dim=0)

            if self.update_step % self.actor_update_period == 0:
                self._bc_target = obs[BC_GROUP].float() if self._bc_weight_now > 0.0 else None
                # separate encoder pass for the BC term (compiled backward cannot retain the critic's encoder graph)
                self._bc_feature = both[: feat.shape[0]] if self.actor_q_encoder or (self._bc_target is not None and self.expert_bc_encoder) else None
                actor_loss, entropy = self._update_actor(feat, next_feat, feat, actions)
                sums["actor"] += actor_loss.item()
                sums["temperature"] += self._update_temperature(entropy).item()
                sums["entropy"] += entropy.item()
                num_actor_updates += 1
            elif self.lr_schedule_per_update:
                # actor/temperature schedules advance on every update, so they decay with the critic and encoder
                self.actor_scheduler.step()
                self.temperature_scheduler.step()

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
        losses["lr_actor"] = self.actor_optimizer.param_groups[0]["lr"]
        losses["lr_critic"] = self.critic_optimizer.param_groups[0]["lr"]
        losses["alpha"], losses["q_weight"] = self.alpha, self._q_weight()
        if self.expert_bc_weight > 0.0:
            losses["expert_bc_weight"] = self._bc_weight_now
            if self.bc_gate_every > 0:
                losses["expert_bc_gate_success"] = self._bc_gate_last
            losses["expert_bc"] = self._bc_loss_sum / max(num_actor_updates, 1)
            self._bc_loss_sum = 0.0
            if self.expert_bc_loss == "weighted_l2":
                losses["expert_bc_mu"] = self._bc_mu_sum / max(num_actor_updates, 1)
                losses["expert_bc_sigma"] = self._bc_sigma_sum / max(num_actor_updates, 1)
                self._bc_mu_sum = self._bc_sigma_sum = 0.0
        if self.is_multi_gpu:
            self._sync_running_stats()
        if check_every and self._rgb_check_calls % check_every == 0:
            self._rgb_rank_check(f"after_update_call_{self._rgb_check_calls}")
        if self.expert_buffer is not None:
            losses["expert_ratio"] = self.num_expert / self.replay_buffer._sample_batch_size
        return losses

    def _update_actor(self, actor_obs, actor_next_obs, critic_obs, actions_batch):
        """The base actor update plus ``expert_bc_weight`` x MSE between the sampled action and the state expert's
        action for the same observation (``self._bc_target``); unchanged without a target."""
        if self._bc_target is None and not self.actor_eval_bn:
            return super()._update_actor(actor_obs, actor_next_obs, critic_obs, actions_batch)
        with torch.autocast(device_type=self._device_type, dtype=torch.float16, enabled=self.use_amp):
            if self.actor_eval_bn:
                # every actor term uses the deployed (running-statistics) policy; the statistics track the update batch
                with torch.no_grad():
                    self.actor.get_mean_and_std(torch.cat([actor_obs, actor_next_obs], dim=0), training=True)
                grad_obs = self._encode(self._bc_feature, self.encoder) if self.actor_q_encoder else actor_obs
                actions, log_probs = self._actor_sample(grad_obs, training=False)
            else:
                actions_all, log_probs_all = self._actor_sample(torch.cat([actor_obs, actor_next_obs], dim=0), training=True)
                actions = torch.chunk(actions_all, 2, dim=0)[0]
                log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]
            self.critic.critic.requires_grad_(False)
            qs, _ = self._critic_eval(critic_obs, actions, training=False)
            q = torch.minimum(qs[0], qs[1])
            self.critic.critic.requires_grad_(True)
            actor_loss = (log_probs * self.temperature().detach() - self._q_weight() * q).mean()
            if self.actor_bc_alpha > 0:
                actor_loss = actor_loss + self.actor_bc_alpha * torch.abs(q).mean().detach() * ((actions - actions_batch) ** 2).mean()
            if self._bc_target is None:
                bc_loss = torch.zeros((), device=self.device)
            elif self.expert_bc_loss == "weighted_l2":
                # rsl_rl DistillationDAggerWeighted (DEXTRAH-style): teacher-std-weighted L2 norm on the mean + L2 norm on
                # the std, pre-squash, in the same BatchNorm mode as the SAC term
                n = actor_obs.shape[0]
                bc_feat = actor_obs if self._bc_feature is None else self._encode(self._bc_feature, self.encoder)
                if self.actor_eval_bn:
                    s_mu, s_sig = self.actor.get_mean_and_std(bc_feat, training=False)
                else:
                    norms = [m for m in self.actor.modules() if hasattr(m, "momentum") and hasattr(m, "running_mean")]
                    saved = [m.momentum for m in norms]
                    for m in norms:
                        m.momentum = 0.0
                    s_mu, s_sig = self.actor.get_mean_and_std(torch.cat([bc_feat, actor_next_obs], dim=0), training=True)
                    for m, v in zip(norms, saved):
                        m.momentum = v
                t_mu, t_sig = self._bc_target.chunk(2, dim=-1)
                mu_loss = torch.sqrt(((s_mu[:n].float() - t_mu) ** 2 / t_sig.clamp_min(1e-6) ** 2).sum(-1) + 1e-12).mean()
                sigma_loss = torch.norm(s_sig[:n].float() - t_sig, p=2, dim=-1).mean()
                bc_loss = mu_loss + sigma_loss
                self._bc_mu_sum += mu_loss.item()
                self._bc_sigma_sum += sigma_loss.item()
            elif self.expert_bc_on_mean:
                # tanh(mean) of the deployed policy: a sampled-action loss would shrink the std
                bc_feat = grad_obs if self.actor_q_encoder else actor_obs if self._bc_feature is None else self._encode(self._bc_feature, self.encoder)
                bc_loss = ((torch.tanh(self.actor.get_mean_and_std(bc_feat, training=False)[0]).float() - self._bc_target) ** 2).mean()
            elif self._bc_feature is not None:
                bc_actions, _ = self._actor_sample(self._encode(self._bc_feature, self.encoder), training=False)
                bc_loss = ((bc_actions.float() - self._bc_target) ** 2).mean()
            else:
                bc_loss = ((actions.float() - self._bc_target) ** 2).mean()
            actor_loss = actor_loss + self._bc_weight_now * bc_loss
            entropy = -log_probs.mean()
        self._bc_loss_sum += bc_loss.item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        encoder_params = list(self.encoder.parameters()) if self._bc_feature is not None else []
        for p in encoder_params:
            p.grad = None
        if self.use_amp:
            assert not encoder_params, "expert_bc_encoder does not support use_amp"
            self.grad_scaler.scale(actor_loss).backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor_parameters)
            self.grad_scaler.unscale_(self.actor_optimizer)
            self.grad_scaler.step(self.actor_optimizer)
            self.grad_scaler.update()
        else:
            actor_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor_parameters)
            self.actor_optimizer.step()
        if encoder_params:
            # BC encoder gradient joins the critic + aux gradient in this update's single encoder step
            self._bc_encoder_grads = [None if p.grad is None else p.grad.detach().clone() for p in encoder_params]
            for p in encoder_params:
                p.grad = None
        self._bc_feature = None
        self.actor_scheduler.step()
        with torch.no_grad():
            self._actor_normalize()
        return actor_loss.detach(), entropy.detach()

    def _q_weight(self) -> float:
        """Q term weight of the actor loss: 0 for ``actor_q_warmup`` update calls, then a ``actor_q_ramp`` linear ramp to 1."""
        k = self._audit_calls - 1 - self.actor_q_warmup
        if k < 0:
            return 0.0
        return min((k + 1) / self.actor_q_ramp, 1.0) if self.actor_q_ramp > 0 else 1.0

    def _critic_side_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters stepped from the critic loss (+ aux loss): critic, trainable encoder, aux head."""
        params = list(self.critic_parameters)
        if self.encoder_optimizer is not None:
            params += list(self.encoder.parameters())
        if self.aux_head is not None:
            params += list(self.aux_head.parameters())
        return params

    def _rgb_modules(self) -> list[nn.Module]:
        mods = [self.encoder, self.target_encoder]
        mods += [m for m in (self.aux_head, self.ema_actor, self.ema_encoder) if m is not None]
        return mods

    @torch.no_grad()
    def _broadcast_rgb_state(self) -> None:
        """Rank 0's encoder, target encoder, aux head (+ aux normalizer) and policy-EMA copies to every rank."""
        tensors = [t for m in self._rgb_modules() for t in list(m.parameters()) + list(m.buffers())]
        if self.aux_head is not None:
            tensors += [self.aux_mean, self.aux_m2, self.aux_count]
        if not tensors:
            return
        flat = torch.cat([t.reshape(-1).to(torch.float64) for t in tensors])
        torch.distributed.broadcast(flat, src=0)
        offset = 0
        for t in tensors:
            t.copy_(flat[offset : offset + t.numel()].view_as(t).to(t.dtype))
            offset += t.numel()

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        self._broadcast_rgb_state()

    @torch.no_grad()
    def _sync_running_stats(self) -> None:
        super()._sync_running_stats()
        if self.aux_head is not None:
            stats = torch.cat([self.aux_mean, self.aux_m2, self.aux_count.reshape(1)])
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            stats /= self.gpu_world_size
            n = self.aux_mean.numel()
            self.aux_mean.copy_(stats[:n])
            self.aux_m2.copy_(stats[n : 2 * n])
            self.aux_count.copy_(stats[2 * n])
        self._broadcast_rgb_state()

    @torch.no_grad()
    def _rgb_rank_check(self, tag: str) -> None:
        """Diagnostic (FLASHSAC_RGB_RANK_CHECK=K, multi-GPU): per module, max |w - w_rank0| and ||w - w_rank0|| / ||w_rank0||,
        then on rank 0's mini-batch the encoder-feature cosine, |Q - Q_rank0| and |a - a_rank0| of this rank's networks."""
        dist = torch.distributed
        mods = {"actor": self.actor, "temperature": self.temperature, "critic": self.critic.critic,
                "critic_target": self.critic.critic_target, "encoder": self.encoder, "target_encoder": self.target_encoder}
        if self.aux_head is not None:
            mods["aux_head"] = self.aux_head
        groups = {k: list(m.parameters()) for k, m in mods.items()}
        groups["bn_running_stats"] = self._running_stat_buffers()
        parts = []
        for name, tensors in groups.items():
            if not tensors:
                continue
            flat = torch.cat([t.reshape(-1).float() for t in tensors])
            ref = flat.clone()
            dist.broadcast(ref, src=0)
            parts.append(f"{name}={float((flat - ref).abs().max()):.3e}/{float((flat - ref).norm() / ref.norm().clamp_min(1e-12)):.3e}")
        batch = self._sample_batch()
        n = 256
        img = batch["observation"][self.image_group][:n].to(self.device).float().contiguous()
        st = batch["observation"][self.state_group][:n].to(self.device).float().contiguous()
        act = batch["action"][:n].to(self.device).float().contiguous()
        for t in (img, st, act):
            dist.broadcast(t, src=0)
        feat = self.encoder(img, st)
        q = self.critic.evaluate(feat, act, training=False)[0].mean(0)
        a = torch.tanh(self.actor.get_mean_and_std(self.actor.flatten_obs(self._feature_td(feat)), training=False)[0])
        ref_feat, ref_q, ref_a = feat.clone(), q.clone(), a.clone()
        for t in (ref_feat, ref_q, ref_a):
            dist.broadcast(t, src=0)
        cos = float(F.cosine_similarity(feat, ref_feat, dim=-1).mean())
        print(f"[rgb_rank_check] rank {self.gpu_global_rank} {tag} update_step {self.update_step} | max|dw|/rel_L2: "
              + " ".join(parts)
              + f" | same-batch: feat_cos {cos:.4f} |dQ| {float((q - ref_q).abs().mean()):.4e} (|Q| {float(ref_q.abs().mean()):.3f})"
              + f" |da| {float((a - ref_a).abs().mean()):.4e}", flush=True)

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
            next_actor_entropy = self.temperature() * next_log_probs * float(self.critic_entropy_backup)
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
        if self._bc_encoder_grads is not None:
            if os.environ.get("FLASHSAC_BC_DEBUG") and self.update_step % 200 == 0:
                norm = lambda gs: float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
                print(f"[bc_encoder] step {self.update_step} grad norm bc {norm(self._bc_encoder_grads):.4g} "
                      f"critic+aux {norm([p.grad for p in self.encoder.parameters()]):.4g}", flush=True)
            for p, g in zip(self.encoder.parameters(), self._bc_encoder_grads):
                if g is not None:
                    p.grad = g if p.grad is None else p.grad + g
            self._bc_encoder_grads = None
        if self.is_multi_gpu:
            self.reduce_parameters(self._critic_side_parameters())
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
        saved["expert_bc_weight_now"] = self._bc_weight_now
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
        if self.bc_gate_every > 0 and "expert_bc_weight_now" in loaded_dict:
            self._bc_weight_now = min(float(loaded_dict["expert_bc_weight_now"]), self.expert_bc_weight)
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
        elif alg_cfg.get("encoder_type") == "resnet18":
            encoder = ResNetStateEncoder(
                tuple(obs[image_group].shape[1:]),
                obs[state_group].shape[-1],
                alg_cfg.get("image_embed_dim", 256),
                alg_cfg.get("state_embed_dim", 64),
                num_cameras=alg_cfg.get("encoder_num_cameras", 2),
                weights_path=alg_cfg.get("encoder_weights", ""),
            ).to(device)
        elif alg_cfg.get("encoder_type") == "dino_tail":
            encoder = DinoTailMAPEncoder(
                tuple(obs[image_group].shape[1:]),
                obs[state_group].shape[-1],
                alg_cfg.get("image_embed_dim", 256),
                alg_cfg.get("state_embed_dim", 64),
                model_dir=alg_cfg["encoder_weights"],
                split_block=alg_cfg.get("dino_split_block", 10),
                num_cameras=alg_cfg.get("encoder_num_cameras", 2),
                num_queries=alg_cfg.get("map_queries", 8),
                dim=alg_cfg.get("map_dim", 128),
                num_heads=alg_cfg.get("map_heads", 4),
            ).to(device)
        elif alg_cfg.get("encoder_type") == "map":
            encoder = MAPStateEncoder(
                tuple(obs[image_group].shape[1:]),
                obs[state_group].shape[-1],
                alg_cfg.get("image_embed_dim", 256),
                alg_cfg.get("state_embed_dim", 64),
                num_cameras=alg_cfg.get("encoder_num_cameras", 1),
                num_queries=alg_cfg.get("map_queries", 8),
                dim=alg_cfg.get("map_dim", 128),
                num_heads=alg_cfg.get("map_heads", 4),
            ).to(device)
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
        store_groups = {image_group, state_group} | ({alg_cfg["aux_group"]} if alg_cfg.get("aux_group") else set())
        if alg_cfg.get("expert_bc_weight", 0.0) > 0.0:
            obs = obs.copy()
            dims = env.num_actions * (2 if alg_cfg.get("expert_bc_loss", "mse") == "weighted_l2" else 1)
            obs[BC_GROUP] = torch.zeros(obs.batch_size[0], dims, device=obs.device)
            store_groups.add(BC_GROUP)
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
            store_groups=sorted(store_groups),
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
    """"pixel" (CNN + state projection), "resnet18" (one ResNet-18 per camera, see ``encoder_weights``), "dino_tail"
    (last DINOv3 blocks + MAP on cached block tokens, see ``dino_split_block``), "map" (attention
    pooling over frozen patch tokens in ``image_group``, shape
    (num_cameras * T, D); ``encoder_num_cameras`` heads) or "state" (debug bypass: the raw state input is the feature)."""
    map_queries: int = 8
    """map encoder: learnable queries per camera."""
    map_dim: int = 128
    """map encoder: attention width (tokens are projected from D to this)."""
    map_heads: int = 4
    encoder_weights: str = ""
    """resnet18 encoder: torchvision state dict to start from (e.g. ImageNet resnet18-f37072fd.pth; "" = random init)."""
    encoder_backbone_lr_scale: float = 0.1
    """LR of a pretrained encoder backbone (resnet18 trunks, dino_tail blocks) relative to the encoder LR."""
    dino_split_block: int = 10
    """dino_tail encoder: blocks before this index run frozen in the env (cached tokens); the rest train here.
    ``encoder_weights`` is the DINOv3 model folder (as the env's ``dino_block_tokens`` term)."""
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
    expert_bc_weight: float = 0.0
    expert_bc_weight_final: float = -1.0
    """BC weight at the end of a linear anneal from ``expert_bc_weight`` (negative = constant). Labels are still stored."""
    expert_bc_anneal_start: int = 0
    """update() calls of this process (iterations once training starts) before the anneal starts; resume-safe."""
    expert_bc_anneal_iterations: int = 0
    expert_bc_gate_every: int = 0
    """Success-gated BC (0 = off): every N iterations, multiply the BC weight by ``expert_bc_gate_down`` while the
    rank-mean ``expert_bc_gate_key`` metric is above ``expert_bc_gate_high`` and by ``expert_bc_gate_up`` when it is below
    ``expert_bc_gate_low``, within [``expert_bc_weight_min``, ``expert_bc_weight``]. The weight is checkpointed."""
    expert_bc_gate_low: float = 0.70
    expert_bc_gate_high: float = 0.76
    expert_bc_gate_down: float = 0.85
    expert_bc_gate_up: float = 1.5
    expert_bc_weight_min: float = 0.01
    expert_bc_gate_key: str = "Curriculum/reset_event_log/Metrics/monitor_env_end_of_episode_success_rate"
    """Length of the linear anneal in iterations (0 = jump to the final weight at the start)."""
    """Weight of a behaviour-cloning MSE added to the actor loss: sampled action vs the state expert's action for the
    same observation, on every replayed transition (needs ``expert_policy_checkpoint``; 0 = off)."""
    expert_bc_on_mean: bool = False
    """BC on the deployed policy's tanh(mean) instead of a sampled action (which shrinks the std)."""
    actor_eval_bn: bool = False
    """SAC actor + entropy terms use running BatchNorm statistics (updated by a no-grad pass), like the deployed policy."""
    critic_pick_every: int = 0
    """Act with the critic's best of ``critic_pick_n`` policy samples in every k-th non-expert env (0 = off; 2, 4, 8, 16)."""
    critic_pick_n: int = 16
    actor_q_encoder: bool = False
    expert_bc_loss: str = "mse"
    """"mse" or "weighted_l2": teacher-std-weighted L2 norm on the mean + L2 norm on the std, pre-squash (DEXTRAH-style,
    as rsl_rl DistillationDAggerWeighted); needs an expert with a Gaussian output distribution."""
    expert_bc_std_floor: tuple = (0.05, 0.03, 0.035, 0.06, 0.05, 0.035, 0.33)
    """The whole actor loss (-Q + entropy + BC) trains the encoder through the actor's input (False: only BC does). The
    critic's own features stay detached in the actor loss."""
    expert_bc_gate_start: int = 0
    """Update calls (since start / resume) before the success gate acts (e.g. the critic warm-up)."""
    expert_bc_gate_zero: float = 0.0
    """Success gate: a weight lowered to this value or below becomes exactly 0 (and restarts from it on a drop)."""
    actor_q_warmup: int = 0
    """Update calls (since start / resume) with the Q term of the actor loss off: the critic learns first."""
    actor_q_ramp: int = 0
    """Update calls over which the Q term then ramps from 0 to 1."""
    critic_entropy_backup: bool = True
    """Entropy term in the critic target (False: plain returns, as RLPD)."""
    expert_bc_encoder: bool = False
    lr_schedule_per_update: bool = False
    """Advance the actor and temperature LR schedules on every update (not only on actor updates every
    ``actor_update_period``), so all networks follow the same schedule."""
    """BC term also trains the image encoder: a separate encoder + eval-mode actor pass on the update batch, whose
    encoder gradient is added to the critic + aux gradient of the same update (the SAC actor term stays detached)."""
    replay_audit_every: int = 0
    """Every N update() calls, rank 0 saves update-batch images (as fed to the encoder) + stored replay episodes (0 = off)."""
    replay_audit_envs: str = "0,1"
    replay_audit_steps: int = 160
    replay_audit_batch: int = 64
    image_aug: bool = False
    """Apply the OmniReset distillation image randomization (``OmniResetImageAug``) to the sampled update batches."""
    image_aug_version: int = 1
    """1 = OmniReset as-is (one draw per batch, k5 / sigma 0.1-2.0 blur); 2 = per-sample draws, k3 blur (``OmniResetImageAugV2``)."""
    image_aug_blur_sigma_max: float = 0.6
    """v2 only: blur sigma upper bound (lower bound 0.1)."""
    policy_ema: float = 0.0
    """Rate of an EMA copy of the policy (actor + encoder, incl. BatchNorm stats) updated after every gradient update and
    saved as ``ema_actor_state_dict`` / ``ema_encoder_state_dict`` (0 = off). A frozen snapshot of the live policy can
    be much worse than the continually updated one; the average is the policy to evaluate / deploy."""
