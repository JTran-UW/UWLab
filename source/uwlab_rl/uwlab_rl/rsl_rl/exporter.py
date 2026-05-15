# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
from typing import List

import torch
from torch import nn

from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter, _TorchPolicyExporter


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporterExtended(policy, normalizer)
    policy_exporter.export(path, filename)


def export_vision_student_as_jit(
    policy,
    path: str,
    filename: str = "depth_policy.pt",
    vision_groups: List[str] | None = None,
) -> str:
    """Export a ``StudentTeacherVision`` policy as a self-contained JIT module.

    Bundles the proprio normalizer + shared depth encoder + student MLP into one
    module so the real-robot eval (``robodiff_real`` env, no rsl_rl deps) can load
    it with just ``torch.jit.load`` and call ``forward(proprio, images)``.

    Forward signature (as scripted):
        forward(proprio: (B, num_proprio) float32,
                images: List[(B, 1, H, W) float32 in [0,1]]) -> (B, num_actions)

    Image list order MUST match the order in ``policy.vision_groups`` at training
    time (e.g. ``["side_depth", "wrist_depth"]``); the eval script reads
    ``vision_groups`` from the saved metadata file alongside the JIT.
    """
    os.makedirs(path, exist_ok=True)
    groups = list(vision_groups) if vision_groups is not None else list(policy.vision_groups)
    embed_dim = _depth_encoder_embed_dim(policy.depth_encoder)
    student_in = _mlp_in_features(policy.student)
    num_proprio = int(student_in - len(groups) * embed_dim)
    num_actions = int(policy.num_actions)
    first_conv = _first_conv2d(policy.depth_encoder)
    image_channels = int(first_conv.in_channels)
    # Encoder is fully convolutional → spatial dim isn't fixed by weights, but
    # sim always feeds 224x224 (DepthDAggerObservationsCfg.IMG_H/W). Hard-code.
    image_h, image_w = 224, 224

    wrapper = _VisionStudentJITExporter(policy)
    wrapper.eval()
    out_path = os.path.join(path, filename)
    scripted = torch.jit.script(wrapper)
    scripted.save(out_path)

    # Sidecar metadata: lets eval pick up obs layout without re-loading the runner.
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


def _first_conv2d(module: nn.Module) -> nn.Conv2d:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise ValueError("Expected a Conv2d in the depth encoder, found none")


def _depth_encoder_embed_dim(encoder: nn.Module) -> int:
    # DepthCNN.head = Sequential(Linear(flat, embed), ReLU); ResNet18Encoder.proj = Linear(512, embed).
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


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporterExtended(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _StateDependentPolicyMixin(nn.Module):
    """Mixin class to handle state-dependent policy logic."""

    def _setup_state_dependent_policy(self, policy):
        """Setup state-dependent policy components."""
        self.actor_features = self.actor[:-1]  # type: ignore
        self.actor_final = self.actor[-1]  # type: ignore

        self.register_buffer("log_std", policy.log_std.clone())
        self.epsilon = 1e-6

    def _setup_regular_policy(self, policy):
        """Setup regular policy components."""
        self.actor_features = self.actor[:-1]  # type: ignore
        self.actor_final = self.actor[-1]  # type: ignore

        if hasattr(policy, "std"):
            self.register_buffer("std", policy.std.clone())
        if hasattr(policy, "log_std"):
            self.register_buffer("log_std", policy.log_std.clone())
        if hasattr(policy, "noise_std_type"):
            self.noise_std_type = policy.noise_std_type
        else:
            self.noise_std_type = "scalar"

        # For GSDE, ensure epsilon is set
        if self.noise_std_type == "gsde":
            self.epsilon = 1e-6

    def _ensure_compatibility_attributes(self, policy):
        """Ensure all attributes exist for TorchScript compatibility."""
        if not hasattr(self, "std"):
            if hasattr(policy, "std"):
                self.register_buffer("std", policy.std.clone())
            else:
                # Create a default std tensor
                default_std = torch.ones(policy.num_actions if hasattr(policy, "num_actions") else 1)
                self.register_buffer("std", default_std)

        if not hasattr(self, "log_std"):
            if hasattr(policy, "log_std"):
                self.register_buffer("log_std", policy.log_std.clone())
            else:
                # Create a default log_std tensor
                default_log_std = torch.zeros(policy.num_actions if hasattr(policy, "num_actions") else 1)
                self.register_buffer("log_std", default_log_std)

        if not hasattr(self, "epsilon"):
            self.epsilon = 1e-6

        if not hasattr(self, "noise_std_type"):
            if hasattr(policy, "noise_std_type"):
                self.noise_std_type = policy.noise_std_type
            else:
                self.noise_std_type = "scalar"  # Default fallback

        # Ensure epsilon is set for GSDE
        if self.noise_std_type == "gsde" and not hasattr(self, "epsilon"):
            self.epsilon = 1e-6

    def _compute_distribution(self, observations):
        """Compute mean and std for distribution."""
        if self.is_state_dependent.item():  # type: ignore
            # Use the separated layers
            features = self.actor_features(observations)  # type: ignore
            mean = self.actor_final(features)  # type: ignore

            # Compute variance using exploration matrices and torch.mm
            variance = torch.mm(features**2, torch.exp(self.log_std) ** 2)  # type: ignore
            std = torch.sqrt(variance + self.epsilon)

            return mean, std
        else:
            # Regular ActorCritic logic
            mean = self.actor(observations)  # type: ignore

            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)  # type: ignore
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)  # type: ignore
            elif self.noise_std_type == "gsde":
                # GSDE: log_std is a matrix (hidden_dim, num_actions)
                # Compute features from actor[:-1] (all layers except last)
                features = self.actor_features(observations)  # type: ignore
                # Compute variance: variance = torch.mm(features**2, exp(log_std)**2)
                # features shape: (batch, hidden_dim), log_std shape: (hidden_dim, num_actions)
                variance = torch.mm(features**2, torch.exp(self.log_std) ** 2)  # type: ignore
                std = torch.sqrt(variance + self.epsilon)
            else:
                std = torch.ones_like(mean)

            return mean, std


class _TorchPolicyExporterExtended(_TorchPolicyExporter, _StateDependentPolicyMixin):
    def __init__(self, policy, normalizer=None):
        super().__init__(policy, normalizer)

        # Detect policy type
        is_state_dependent = hasattr(policy, "use_state_dependent_noise") and policy.use_state_dependent_noise
        self.register_buffer("is_state_dependent", torch.tensor(is_state_dependent, dtype=torch.bool))

        if is_state_dependent:
            self._setup_state_dependent_policy(policy)
        else:
            self._setup_regular_policy(policy)

        # Ensure all attributes exist for TorchScript compatibility
        self._ensure_compatibility_attributes(policy)

    @torch.jit.export
    def compute_distribution(self, x):
        observations = self.normalizer(x)
        return self._compute_distribution(observations)


class _VisionStudentJITExporter(nn.Module):
    """Self-contained JIT wrapper around ``StudentTeacherVision``.

    Inputs at inference time:
        proprio: (B, num_proprio) — concatenation of policy obs groups in the order
            given by ``policy.obs_groups['policy']``, already including history flatten.
        images:  list of (B, C, H, W) tensors in [0, 1], one per entry of
            ``policy.vision_groups`` and in the SAME order.

    Output: action mean (B, num_actions). No exploration noise — eval is deterministic.

    Holds deep-copied module references (normalizer, depth_encoder, student MLP)
    so the eval env doesn't need to import rsl_rl / uwlab_rl.
    """

    def __init__(self, policy):
        super().__init__()
        # Deep-copy submodules: source-policy mutation can't affect the export,
        # and the scripted graph owns its own parameters.
        self.normalizer = copy.deepcopy(policy.student_obs_normalizer)
        self.depth_encoder = copy.deepcopy(policy.depth_encoder)
        self.student = copy.deepcopy(policy.student)

    def forward(self, proprio: torch.Tensor, images: List[torch.Tensor]) -> torch.Tensor:
        # Mirror StudentTeacherVision._encode_student + .act_inference.
        proprio_norm = self.normalizer(proprio)
        feats: List[torch.Tensor] = [proprio_norm]
        for img in images:
            feats.append(self.depth_encoder(img))
        return self.student(torch.cat(feats, dim=-1))


class _OnnxPolicyExporterExtended(_OnnxPolicyExporter, _StateDependentPolicyMixin):
    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__(policy, normalizer, verbose)

        is_state_dependent = hasattr(policy, "use_state_dependent_noise") and policy.use_state_dependent_noise
        self.register_buffer("is_state_dependent", torch.tensor(is_state_dependent, dtype=torch.bool))

        if is_state_dependent:
            self._setup_state_dependent_policy(policy)
        else:
            self._setup_regular_policy(policy)

    @torch.jit.export
    def compute_distribution(self, x):
        observations = self.normalizer(x)
        return self._compute_distribution(observations)
