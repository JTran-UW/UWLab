"""Convert a ScenePC RL teacher checkpoint into a JIT-scripted teacher for DAgger.

Mirrors ``convert_state_expert_to_jit.py`` but for encoder-aware actors trained
with ``RslRlActorCriticWithEncoderCfg`` (proprio + scene_pc → shared MLP encoder
→ concat → actor MLP). The exported JIT exposes the same single-tensor
``forward(obs) → mean`` contract that ``StudentTeacherVision`` expects, but
internally splits the input into ``proprio`` (first ``proprio_dim`` features)
and ``pointcloud`` (remaining 1536 = 512 × 3 features), runs each through its
respective normalizer, encodes the PC, then runs the actor.

Usage::

    python scripts_v2/tools/convert_pc_expert_to_jit.py \
        --input  checkpoints/pc_teacher/peg_pc_prevact.pt \
        --output teachers/peg_pc_prevact_jit.pt \
        --proprio-dim 25
"""

from __future__ import annotations

import argparse
import re

import torch
import torch.nn as nn

PC_DIM = 512 * 3  # ScenePC always uses 512 points
ENCODER_OUT_DIM = 32


def build_mlp(sd: dict, prefix: str, activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    idx_re = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    indices = sorted(int(m.group(1)) for k in sd if (m := idx_re.match(k)))
    assert indices, f"no Linear weights found under prefix '{prefix}'"
    layers: list[nn.Module] = []
    for i, idx in enumerate(indices):
        w = sd[f"{prefix}.{idx}.weight"]
        b = sd[f"{prefix}.{idx}.bias"]
        out_dim, in_dim = w.shape
        lin = nn.Linear(in_dim, out_dim)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(indices) - 1:
            layers.append(activation())
    return nn.Sequential(*layers)


def split_actor(sd: dict, prefix: str = "actor") -> tuple[nn.Sequential, nn.Linear]:
    """Split the actor MLP into (features=all but last Linear, final=last Linear).

    Mirrors the `convert_state_expert_to_jit.py` MeanStdTeacher split. ``features``
    is the activations of the second-to-last Linear (64d for our 4-hidden actor);
    ``final`` is the last Linear (64 → 7) producing the mean.
    """
    full = build_mlp(sd, prefix)
    layers = list(full)
    assert isinstance(layers[-1], nn.Linear), f"last actor layer is not Linear: {type(layers[-1])}"
    final = layers[-1]
    # Drop the final Linear AND the activation that precedes the final Linear's "preceding" Linear...
    # actually build_mlp produces alternating Linear/activation, ending with Linear (no activation
    # after final). So layers[-1] is the final Linear; layers[:-1] is [Linear, act, Linear, act, ..., Linear, act].
    # Wait: that ends with activation. We want the features = output of second-to-last Linear after activation.
    # So features network = layers[:-1] which ends with the activation after the second-to-last Linear.
    features = nn.Sequential(*layers[:-1])
    assert isinstance(layers[-2], nn.ELU), f"expected activation before final Linear, got {type(layers[-2])}"
    return features, final


class ScenePCTeacherMean(nn.Module):
    """``forward(obs) -> mean``. For non-weighted DAgger / probe."""

    def __init__(self, sd: dict, proprio_dim: int):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.pc_dim = PC_DIM

        self.register_buffer("proprio_mean", sd["_group_normalizers.proprio._mean"].clone())
        self.register_buffer("proprio_std", sd["_group_normalizers.proprio._std"].clone())
        self.register_buffer("pc_mean", sd["_group_normalizers.pointcloud._mean"].clone())
        self.register_buffer("pc_std", sd["_group_normalizers.pointcloud._std"].clone())

        self.pc_encoder = build_mlp(sd, "actor_encoders.pointcloud")
        self.actor = build_mlp(sd, "actor")

        self.eps = 1e-2

        assert self.proprio_mean.shape == (1, proprio_dim)
        assert self.pc_mean.shape == (1, PC_DIM)
        actor_in = sd["actor.0.weight"].shape[1]
        assert actor_in == proprio_dim + ENCODER_OUT_DIM

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio = obs[..., : self.proprio_dim]
        pc = obs[..., self.proprio_dim :]
        proprio_n = (proprio - self.proprio_mean) / (self.proprio_std + self.eps)
        pc_n = (pc - self.pc_mean) / (self.pc_std + self.eps)
        pc_enc = self.pc_encoder(pc_n)
        x = torch.cat([proprio_n, pc_enc], dim=-1)
        return self.actor(x)


class ScenePCTeacherMeanStd(nn.Module):
    """``forward(obs) -> (mean, std)`` with gSDE state-dependent std.

    Splits actor into ``actor_features`` (all layers except final Linear) and
    ``actor_final`` (final Linear). Variance is ``features² @ exp(log_std)²``
    (UWLab gSDE exporter, ``uwlab_rl/rsl_rl/exporter.py:114-145``); std adds
    eps for numerical stability.
    """

    def __init__(self, sd: dict, proprio_dim: int):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.pc_dim = PC_DIM

        self.register_buffer("proprio_mean", sd["_group_normalizers.proprio._mean"].clone())
        self.register_buffer("proprio_std", sd["_group_normalizers.proprio._std"].clone())
        self.register_buffer("pc_mean", sd["_group_normalizers.pointcloud._mean"].clone())
        self.register_buffer("pc_std", sd["_group_normalizers.pointcloud._std"].clone())

        self.pc_encoder = build_mlp(sd, "actor_encoders.pointcloud")
        self.actor_features, self.actor_final = split_actor(sd, "actor")
        # gSDE log_std: shape (last_hidden_dim, num_actions) e.g. (64, 7).
        self.register_buffer("log_std", sd["log_std"].clone())

        self.eps = 1e-2
        self.std_eps = 1e-6  # added to variance before sqrt for numerical stability

        assert self.proprio_mean.shape == (1, proprio_dim)
        actor_in = sd["actor.0.weight"].shape[1]
        assert actor_in == proprio_dim + ENCODER_OUT_DIM
        feat_dim = self.actor_final.in_features
        n_actions = self.actor_final.out_features
        assert self.log_std.shape == (feat_dim, n_actions), (
            f"log_std shape {tuple(self.log_std.shape)} != ({feat_dim}, {n_actions})"
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = obs[..., : self.proprio_dim]
        pc = obs[..., self.proprio_dim :]
        proprio_n = (proprio - self.proprio_mean) / (self.proprio_std + self.eps)
        pc_n = (pc - self.pc_mean) / (self.pc_std + self.eps)
        pc_enc = self.pc_encoder(pc_n)
        x = torch.cat([proprio_n, pc_enc], dim=-1)
        feats = self.actor_features(x)  # (B, 64)
        mean = self.actor_final(feats)  # (B, 7)
        # gSDE: variance = features² @ exp(log_std)²; std = sqrt(variance + eps).
        var = (feats ** 2) @ (torch.exp(self.log_std) ** 2)
        std = torch.sqrt(var + self.std_eps)
        return mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--proprio-dim", type=int, required=True, help="25 for prev_act variant, 18 for noprevact")
    p.add_argument("--std", action="store_true", help="export ``forward(obs) -> (mean, std)`` with gSDE std")
    args = p.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if args.std:
        teacher = ScenePCTeacherMeanStd(sd, proprio_dim=args.proprio_dim)
    else:
        teacher = ScenePCTeacherMean(sd, proprio_dim=args.proprio_dim)
    teacher.eval()

    # Smoke test: forward pass with random input.
    test_obs = torch.zeros(2, args.proprio_dim + PC_DIM)
    out = teacher(test_obs)
    if args.std:
        assert isinstance(out, tuple) and len(out) == 2
        mean, std = out
        print(f"[smoke] mean shape {tuple(mean.shape)}, std shape {tuple(std.shape)}")
        assert mean.shape == (2, 7) and std.shape == (2, 7)
    else:
        print(f"[smoke] forward shape ok: input {tuple(test_obs.shape)} → output {tuple(out.shape)}")
        assert out.shape == (2, 7), f"expected (2, 7), got {tuple(out.shape)}"

    # JIT-script and save.
    scripted = torch.jit.script(teacher)
    # Verify scripted matches eager.
    out_eager = teacher(test_obs)
    out_scripted = scripted(test_obs)
    if args.std:
        diff = max((out_eager[0] - out_scripted[0]).abs().max().item(),
                   (out_eager[1] - out_scripted[1]).abs().max().item())
    else:
        diff = (out_eager - out_scripted).abs().max().item()
    print(f"[smoke] JIT vs eager max diff: {diff:.2e}")
    assert diff < 1e-5, f"JIT and eager diverge by {diff}"

    scripted.save(args.output)
    print(f"[ok] saved JIT teacher to {args.output}")
    print(f"     proprio_dim={args.proprio_dim}, pc_dim={PC_DIM}, total_input_dim={args.proprio_dim + PC_DIM}")
    print(f"     iter at conversion: {ckpt.get('iter', '?')}, mode={'mean+std' if args.std else 'mean-only'}")


if __name__ == "__main__":
    main()
