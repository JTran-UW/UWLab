"""
Standalone export: StudentTeacherVision checkpoint → self-contained JIT.

No Isaac Sim, no Replicator — only rsl_rl + tensordict + torchvision.

Loads student_teacher_vision.py and the export helpers directly from source
files (bypassing uwlab_rl/__init__.py which drags in omni.timeline).

Usage (from UWLab-patrick-private/, inside the isaac-sim container / patlab env):

    python export_vision_student.py \\
        --checkpoint logs/rsl_rl/.../model_195000.pt \\
        [--output_dir logs/rsl_rl/.../exported] \\
        [--filename rgb_policy.pt] \\
        [--teacher_jit teachers/seed20_sysid_jit.pt]

Reads agent.yaml from <ckpt_dir>/params/agent.yaml; infers shape params from
checkpoint weights; writes <output_dir>/<filename> + <stem>_meta.txt.
"""

import argparse
import copy
import importlib.util
import os
import pathlib
import sys
from typing import List

import torch
import torch.nn as nn
import yaml

# ── Load StudentTeacherVision directly from source (skip __init__.py) ────────
_REPO = pathlib.Path(__file__).parent
_STV_PATH = _REPO / "source/uwlab_rl/uwlab_rl/rsl_rl/student_teacher_vision.py"

def _load_from_file(module_name: str, file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_stv_mod = _load_from_file("_stv", _STV_PATH)
StudentTeacherVision = _stv_mod.StudentTeacherVision

# ── Inlined export helpers (from uwlab_rl/rsl_rl/exporter.py, pure torch) ────

class _VisionStudentJITExporter(nn.Module):
    """Self-contained JIT wrapper — deep-copies normalizer + encoder + student MLP."""

    def __init__(self, policy):
        super().__init__()
        self.normalizer    = copy.deepcopy(policy.student_obs_normalizer)
        self.depth_encoder = copy.deepcopy(policy.depth_encoder)
        self.student       = copy.deepcopy(policy.student)

    def forward(self, proprio: torch.Tensor, images: List[torch.Tensor]) -> torch.Tensor:
        proprio_norm = self.normalizer(proprio)
        feats: List[torch.Tensor] = [proprio_norm]
        for img in images:
            feats.append(self.depth_encoder(img))
        return self.student(torch.cat(feats, dim=-1))


def _first_conv2d(module: nn.Module) -> nn.Conv2d:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise ValueError("Expected a Conv2d in the depth encoder, found none")


def _depth_encoder_embed_dim(encoder: nn.Module) -> int:
    if hasattr(encoder, "proj") and isinstance(encoder.proj, nn.Linear):
        return int(encoder.proj.out_features)
    if hasattr(encoder, "head"):
        for m in encoder.head:
            if isinstance(m, nn.Linear):
                return int(m.out_features)
    raise ValueError("Could not determine embed_dim from encoder")


def _mlp_in_features(student: nn.Module) -> int:
    for m in student.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    raise ValueError("Could not find a Linear layer in the student MLP")


def export_vision_student_as_jit(
    policy,
    path: str,
    filename: str = "depth_policy.pt",
    vision_groups: List[str] | None = None,
) -> str:
    os.makedirs(path, exist_ok=True)
    groups       = list(vision_groups) if vision_groups is not None else list(policy.vision_groups)
    embed_dim    = _depth_encoder_embed_dim(policy.depth_encoder)
    student_in   = _mlp_in_features(policy.student)
    num_proprio  = int(student_in - len(groups) * embed_dim)
    num_actions  = int(policy.num_actions)
    first_conv   = _first_conv2d(policy.depth_encoder)
    image_channels = int(first_conv.in_channels)
    image_h, image_w = 224, 224

    wrapper  = _VisionStudentJITExporter(policy)
    wrapper.eval()
    out_path = os.path.join(path, filename)
    scripted = torch.jit.script(wrapper)
    scripted.save(out_path)

    meta_path = os.path.join(path, os.path.splitext(filename)[0] + "_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"num_proprio={num_proprio}\n")
        f.write(f"num_actions={num_actions}\n")
        f.write(f"image_h={image_h}\n")
        f.write(f"image_w={image_w}\n")
        f.write(f"image_channels={image_channels}\n")
        f.write("vision_groups=" + ",".join(groups) + "\n")

    print(f"[export_vision_student_as_jit] wrote {out_path} "
          f"(proprio={num_proprio}, actions={num_actions}, "
          f"image={image_channels}x{image_h}x{image_w}, groups={groups})")
    print(f"[export_vision_student_as_jit] wrote sidecar {meta_path}")
    return out_path


# ── Shape inference ───────────────────────────────────────────────────────────

def _infer_shapes(sd: dict, embed_dim: int, vision_groups: list, encoder_type: str):
    student_weight_keys = sorted(
        k for k in sd if k.startswith("student.") and k.endswith(".weight")
    )
    if not student_weight_keys:
        raise ValueError("No student.*.weight keys found in checkpoint.")

    student_in    = sd[student_weight_keys[0]].shape[1]
    num_actions   = sd[student_weight_keys[-1]].shape[0]
    vision_feat   = len(vision_groups) * embed_dim
    num_proprio   = student_in - vision_feat

    if encoder_type == "resnet18":
        in_channels = sd["depth_encoder.backbone.0.weight"].shape[1]
    else:
        in_channels = sd["depth_encoder.conv.0.weight"].shape[1]

    return num_proprio, num_actions, in_channels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export StudentTeacherVision checkpoint to JIT (no Isaac Sim).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model_*.pt")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <ckpt_dir>/exported/)")
    parser.add_argument("--filename", default=None,
                        help="JIT filename (default: rgb_policy.pt or depth_policy.pt)")
    parser.add_argument("--teacher_jit", default=None,
                        help="Override teacher JIT path (default: from agent.yaml)")
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    ckpt_dir = ckpt_path.parent

    yaml_path = ckpt_dir / "params" / "agent.yaml"
    if not yaml_path.exists():
        sys.exit(f"agent.yaml not found at {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    policy_cfg    = cfg["policy"]
    obs_groups    = cfg["obs_groups"]
    vision_groups = list(policy_cfg["vision_groups"])
    embed_dim     = int(policy_cfg.get("embed_dim", 128))
    encoder_type  = policy_cfg.get("encoder_type", "resnet18")
    teacher_jit   = args.teacher_jit or policy_cfg["teacher_jit_path"]

    print(f"Checkpoint   : {ckpt_path}")
    print(f"Teacher JIT  : {teacher_jit}")
    print(f"Vision groups: {vision_groups}  encoder: {encoder_type}  embed_dim: {embed_dim}")

    print("Loading checkpoint...")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd  = raw["model_state_dict"]
    print(f"  iter={raw.get('iter', '?')}")

    num_proprio, num_actions, in_channels = _infer_shapes(sd, embed_dim, vision_groups, encoder_type)
    print(f"Inferred: num_proprio={num_proprio}, num_actions={num_actions}, in_channels={in_channels}")

    # Dummy obs — StudentTeacherVision.__init__ only reads shapes from it.
    from tensordict import TensorDict
    policy_obs_keys = list(obs_groups.get("policy", ["proprio"]))
    dummy_obs = TensorDict(
        {
            **{g: torch.zeros(1, num_proprio) for g in policy_obs_keys},
            **{g: torch.zeros(1, in_channels, 224, 224) for g in vision_groups},
        },
        batch_size=[1],
    )

    print("Instantiating StudentTeacherVision...")
    print(f"encoder_imagenet_norm: {policy_cfg.get('encoder_imagenet_norm', False)}")
    policy = StudentTeacherVision(
        obs=dummy_obs,
        obs_groups=obs_groups,
        num_actions=num_actions,
        teacher_jit_path=teacher_jit,
        vision_groups=vision_groups,
        embed_dim=embed_dim,
        student_hidden_dims=policy_cfg.get("student_hidden_dims", [512, 256]),
        activation=policy_cfg.get("activation", "elu"),
        init_noise_std=float(policy_cfg.get("init_noise_std", 0.1)),
        noise_std_type=policy_cfg.get("noise_std_type", "scalar"),
        student_obs_normalization=bool(policy_cfg.get("student_obs_normalization", True)),
        encoder_type=encoder_type,
        encoder_imagenet_norm=bool(policy_cfg.get("encoder_imagenet_norm", False)),
        encoder_pretrained_path="",   # weights come from checkpoint; skip ImageNet load
        encoder_freeze_iters=0,
        aux_enabled=False,
        predict_std=bool(policy_cfg.get("predict_std", False)),
        teacher_returns_std=bool(policy_cfg.get("teacher_returns_std", False)),
    )

    print("Loading checkpoint weights (teacher keys skipped)...")
    policy.load_state_dict(sd, strict=False)
    policy.eval()

    out_dir  = args.output_dir or str(ckpt_dir / "exported")
    filename = args.filename or ("rgb_policy.pt" if in_channels == 3 else "depth_policy.pt")

    out_path = export_vision_student_as_jit(
        policy, path=out_dir, filename=filename, vision_groups=vision_groups)

    print(f"\nDone.")
    print(f"  JIT     → {out_path}")
    print(f"  sidecar → {pathlib.Path(out_path).stem}_meta.txt  (in same dir)")


if __name__ == "__main__":
    main()
