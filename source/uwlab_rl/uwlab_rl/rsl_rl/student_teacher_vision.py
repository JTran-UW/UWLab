# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision student + JIT teacher for rsl_rl ``DistillationRunner`` DAgger.

Differences vs vanilla ``rsl_rl.modules.StudentTeacher``:

* Student accepts a dict of proprio + per-camera depth images; uses a shared CNN
  encoder across cameras (feature-concat) then an MLP action head.
* Teacher is a loaded ``torch.jit.load`` module whose forward returns the action
  mean directly (the exporter bakes in the obs normalizer). No state-dict
  renaming; no MLP teacher reconstruction; gSDE std is dropped (we only use
  teacher mean for MSE-on-mean DAgger).
* ``__init__`` skips parent's 2D-shape assertion (images are ``B×C×H×W``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import MLP, EmpiricalNormalization


def _conv_out(hw: tuple[int, int], k: int, s: int) -> tuple[int, int]:
    return ((hw[0] - (k - 1) - 1) // s + 1, (hw[1] - (k - 1) - 1) // s + 1)


class DepthCNN(nn.Module):
    """4-layer conv encoder adapted from DEXTRAH's ``CustomCNN``.

    Input: ``(B, in_channels, H, W)`` depth image already normalized to [0, 1].
    Output: ``(B, embed_dim)`` feature vector.
    """

    def __init__(self, in_channels: int, height: int, width: int, embed_dim: int = 128):
        super().__init__()
        h, w = height, width

        h1, w1 = _conv_out((h, w), 6, 2)
        h2, w2 = _conv_out((h1, w1), 4, 2)
        h3, w3 = _conv_out((h2, w2), 4, 2)
        h4, w4 = _conv_out((h3, w3), 4, 2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.LayerNorm([16, h1, w1]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm([32, h2, w2]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm([64, h3, w3]),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm([128, h4, w4]),
        )
        self.flat_dim = 128 * h4 * w4
        self.head = nn.Sequential(
            nn.Linear(self.flat_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1))


class StudentTeacherVision(StudentTeacher):
    """Vision-enabled StudentTeacher: depth CNN student + JIT teacher."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_jit_path: str,
        vision_groups: list[str],
        embed_dim: int = 128,
        student_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        student_obs_normalization: bool = True,
        **kwargs,
    ) -> None:
        # Bypass StudentTeacher.__init__ (it asserts 1D obs and builds an MLP teacher).
        nn.Module.__init__(self)
        Normal.set_default_validate_args(False)

        self.obs_groups = obs_groups
        self.vision_groups = list(vision_groups)
        self.num_actions = num_actions

        # Proprio dim: concatenation of all groups listed under obs_groups["policy"].
        proprio = torch.cat([obs[g] for g in obs_groups["policy"]], dim=-1)
        assert proprio.ndim == 2, f"policy groups must be 1D; got shape {proprio.shape}"
        num_proprio = proprio.shape[-1]

        # Image shapes: all cameras must share the same (C, H, W).
        first_img = obs[self.vision_groups[0]]
        assert first_img.ndim == 4, (
            f"vision group {self.vision_groups[0]} must be 4D (B,C,H,W); got {first_img.shape}"
        )
        _, in_channels, H, W = first_img.shape
        for g in self.vision_groups[1:]:
            assert obs[g].shape == first_img.shape, (
                f"all vision groups must share shape; {g} has {obs[g].shape} vs {first_img.shape}"
            )

        # Shared depth encoder across cameras.
        self.depth_encoder = DepthCNN(in_channels, H, W, embed_dim=embed_dim)
        vision_feat_dim = len(self.vision_groups) * embed_dim

        # Action head: [proprio ; 3 × depth features] → num_actions.
        self.student = MLP(num_proprio + vision_feat_dim, num_actions, list(student_hidden_dims), activation)
        print(f"Student (proprio={num_proprio}, vision_feat={vision_feat_dim}): {self.student}")
        print(f"DepthCNN (C={in_channels}, H={H}, W={W}, embed={embed_dim})")

        # Normalize proprio (images already in [0,1] from process_image).
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_proprio)
        else:
            self.student_obs_normalizer = nn.Identity()

        # JIT teacher — normalizer baked in, forward returns action mean.
        self.teacher = torch.jit.load(teacher_jit_path)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_obs_normalization = False
        self.teacher_obs_normalizer = nn.Identity()
        self.loaded_teacher = True

        # Student exploration noise.
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"noise_std_type must be 'scalar' or 'log'; got {self.noise_std_type}")

        self.distribution = None

    def _encode_student(self, obs: TensorDict) -> torch.Tensor:
        proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        proprio = self.student_obs_normalizer(proprio)
        img_feats = [self.depth_encoder(obs[g]) for g in self.vision_groups]
        return torch.cat([proprio] + img_feats, dim=-1)

    def _encode_teacher(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["teacher"]], dim=-1)

    def _update_distribution(self, feat: torch.Tensor) -> None:
        mean = self.student(feat)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        feat = self._encode_student(obs)
        self._update_distribution(feat)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        feat = self._encode_student(obs)
        return self.student(feat)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self._encode_teacher(obs)
        with torch.no_grad():
            return self.teacher(teacher_obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_student(obs)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_teacher(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
            self.student_obs_normalizer.update(proprio)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.teacher.eval()
        return self

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Resume from a distillation checkpoint. Skips teacher keys (JIT already loaded)."""
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("teacher.")}
        nn.Module.load_state_dict(self, filtered, strict=False)
        self.loaded_teacher = True
        self.teacher.eval()
        return True
