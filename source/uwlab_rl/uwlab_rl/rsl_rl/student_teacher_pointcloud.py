# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Point-cloud student + JIT teacher for rsl_rl ``DistillationRunner`` DAgger.

Online-DAgger counterpart of the offline PointNet BC pipeline. The student is
literally a :class:`uwlab_rl.networks.point_net.PointNet` /
:class:`~uwlab_rl.networks.point_net.ResidualPointNet` — the SAME class the
Lightning BC trainer builds — so:

* iterating on the architecture (one source of truth in ``uwlab_rl.networks``)
  changes both offline BC and online DAgger at once, and
* a BC checkpoint's ``model.*`` state_dict loads straight into this student's
  ``student.*`` (see :meth:`load_bc_checkpoint`), enabling BC-init -> DAgger.

Differences vs vanilla ``rsl_rl.modules.StudentTeacher`` (mirrors the pattern in
``StudentTeacherVision`` / ``StudentTeacherMLP``):

* Student reads TWO kinds of policy obs groups: point-cloud group(s) named in
  ``pointcloud_groups`` (each a flat ``(B, N*point_dim)`` tensor, reshaped to
  ``(B, N, point_dim)``) and the remaining policy groups treated as flat proprio.
  The PointNet fuses them internally (set-encode + max-pool + proprio concat).
* Teacher is a frozen ``torch.jit.load`` module whose forward returns the action
  mean (normalizer baked in); ``__init__`` bypasses the parent's flat-MLP build.
* ``predict_std=True`` routes the PointNet's second output head (log-std) into
  ``act_inference_with_std`` for DEXTRAH-style weighted distillation; ``act``
  exploration noise still uses a separate scalar/log ``std`` param, exactly like
  ``StudentTeacherVision``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import EmpiricalNormalization

from uwlab_rl.networks.point_net import PointNet, ResidualPointNet

_ARCHITECTURES = {
    "point_net": PointNet,
    "residual_point_net": ResidualPointNet,
}


class StudentTeacherPointCloud(StudentTeacher):
    """PointNet student + JIT teacher (point-cloud -> point-cloud DAgger)."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_jit_path: str,
        pointcloud_groups: list[str],
        point_dim: int = 4,
        architecture: str = "residual_point_net",
        encoder_dims: tuple[int, ...] | list[int] = (256, 128),
        action_dims: tuple[int, ...] | list[int] = (512, 256),
        activation: str = "elu",  # accepted for signature parity; PointNet uses its own norms/acts
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        student_obs_normalization: bool = True,
        predict_std: bool = False,
        teacher_returns_std: bool = False,
        **kwargs,
    ) -> None:
        # Bypass StudentTeacher.__init__ (asserts 1D obs + builds an MLP teacher).
        nn.Module.__init__(self)
        Normal.set_default_validate_args(False)

        if architecture not in _ARCHITECTURES:
            raise ValueError(f"architecture must be one of {list(_ARCHITECTURES)}; got {architecture}")
        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.point_dim = int(point_dim)
        self.predict_std = bool(predict_std)

        # Split policy groups into point-cloud group(s) and proprio group(s).
        self.pointcloud_groups = list(pointcloud_groups)
        policy_groups = list(obs_groups["policy"])
        missing = [g for g in self.pointcloud_groups if g not in policy_groups]
        if missing:
            raise ValueError(f"pointcloud_groups {missing} not in obs_groups['policy']={policy_groups}")
        self.proprio_groups = [g for g in policy_groups if g not in self.pointcloud_groups]

        # Infer point count from the first PC group; validate divisibility + consistent point_dim.
        num_points = None
        for g in self.pointcloud_groups:
            flat = obs[g].shape[-1]
            if flat % self.point_dim != 0:
                raise ValueError(
                    f"pointcloud group '{g}' has flat dim {flat} not divisible by point_dim={self.point_dim}"
                )
            n = flat // self.point_dim
            num_points = n if num_points is None else num_points + n  # multiple PC groups concat along N
        self.num_points = num_points

        # Proprio dim = concat of the non-PC policy groups.
        if self.proprio_groups:
            proprio = torch.cat([obs[g] for g in self.proprio_groups], dim=-1)
            assert proprio.ndim == 2, f"proprio groups must be 1D; got shape {proprio.shape}"
            num_proprio = proprio.shape[-1]
        else:
            num_proprio = 0
        self.num_proprio = num_proprio

        # Student == the SAME PointNet class the BC trainer uses (single source of truth).
        self.student = _ARCHITECTURES[architecture](
            encoder_hidden_dims=list(encoder_dims),
            action_hidden_dims=list(action_dims),
            proprio_dim=num_proprio,
            action_dim=num_actions,
            predict_std=predict_std,
            point_dim=self.point_dim,
        )
        print(
            f"StudentTeacherPointCloud student [{architecture}] "
            f"(points={self.num_points}x{self.point_dim}, proprio={num_proprio}, act={num_actions}, "
            f"predict_std={predict_std}): {self.student}"
        )

        # Normalize proprio only; points stay in metric frame (PointNet set-encoder has its own norms).
        self.student_obs_normalization = student_obs_normalization and num_proprio > 0
        if self.student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_proprio)
        else:
            self.student_obs_normalizer = nn.Identity()

        # JIT teacher — normalizer baked in, forward returns action mean (or (mean, std)).
        self.teacher = torch.jit.load(teacher_jit_path)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_obs_normalization = False
        self.teacher_obs_normalizer = nn.Identity()
        self.loaded_teacher = True
        self.teacher_returns_std = bool(teacher_returns_std)

        # Student exploration noise (used by act(); independent of the predict_std head).
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"noise_std_type must be 'scalar' or 'log'; got {self.noise_std_type}")

        # Clamp range for the PointNet's predicted log-std (matches StudentTeacherVision).
        self.log_std_limits = (-5.0, 2.0)
        self.distribution = None

    # ------------------------------------------------------------------ obs plumbing

    def _split_student_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(points (B, N, point_dim), proprio (B, num_proprio))``."""
        pcs = []
        for g in self.pointcloud_groups:
            flat = obs[g]
            pcs.append(flat.view(flat.shape[0], -1, self.point_dim))
        points = torch.cat(pcs, dim=1) if len(pcs) > 1 else pcs[0]

        if self.proprio_groups:
            proprio = torch.cat([obs[g] for g in self.proprio_groups], dim=-1)
        else:
            proprio = points.new_zeros((points.shape[0], 0))
        proprio = self.student_obs_normalizer(proprio)
        return points, proprio

    def _student_forward(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(mean, log_std_or_None)`` from the student PointNet."""
        points, proprio = self._split_student_obs(obs)
        out = self.student(points, proprio)
        if self.predict_std:
            mean, log_std = out
            return mean, log_std
        return out, None

    def _encode_teacher(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["teacher"]], dim=-1)

    # ------------------------------------------------------------------ student API

    def _update_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        mean, _ = self._student_forward(obs)
        self._update_distribution(mean)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mean, _ = self._student_forward(obs)
        return mean

    def act_inference_with_std(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` from the student. Requires ``predict_std=True``."""
        if not self.predict_std:
            raise RuntimeError("act_inference_with_std called but predict_std=False")
        mean, log_std = self._student_forward(obs)
        std = torch.exp(log_std.clamp(*self.log_std_limits))
        return mean, std

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        # Flat concat for API parity (rsl_rl only uses this for shape/normalization bookkeeping).
        points, proprio = self._split_student_obs(obs)
        return torch.cat([points.reshape(points.shape[0], -1), proprio], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            proprio = torch.cat([obs[g] for g in self.proprio_groups], dim=-1)
            self.student_obs_normalizer.update(proprio)

    # ------------------------------------------------------------------ teacher API

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self._encode_teacher(obs)
        with torch.no_grad():
            out = self.teacher(teacher_obs)
        if isinstance(out, tuple):  # teacher_returns_std=True: (mean, std)
            return out[0]
        return out

    def evaluate_with_std(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` from the teacher JIT. Requires ``teacher_returns_std=True``."""
        if not self.teacher_returns_std:
            raise RuntimeError(
                "evaluate_with_std called but teacher_returns_std=False; "
                "re-export the teacher JIT with --std and set teacher_returns_std=True."
            )
        teacher_obs = self._encode_teacher(obs)
        with torch.no_grad():
            out = self.teacher(teacher_obs)
        assert isinstance(out, tuple) and len(out) == 2, f"teacher_returns_std=True but JIT returned {type(out)}"
        return out

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_teacher(obs)

    # ------------------------------------------------------------------ housekeeping

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

    def load_bc_checkpoint(self, path: str, strict: bool = False) -> None:
        """BC-init: load an offline PointNet-BC checkpoint's ``model.*`` weights into ``student``.

        The Lightning BC module stores its PointNet under ``self.model``; this maps
        ``model.<name>`` -> ``<name>`` and loads it into ``self.student`` (same class, so
        keys align exactly). ``strict=False`` tolerates a head-shape mismatch when the BC
        run used ``predict_std=False`` but this student uses ``predict_std=True`` (only the
        final action-head layer differs) — everything else transfers.

        NOTE (untested): does NOT transfer any input-normalization stats the BC pipeline
        may have baked in. For exact BC parity, match the student's proprio composition to
        the BC dataset and seed / disable ``student_obs_normalizer`` accordingly.
        """
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        student_sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
        if not student_sd:
            raise ValueError(f"no 'model.*' keys found in {path}; is this a PointNetBC checkpoint?")
        missing, unexpected = self.student.load_state_dict(student_sd, strict=strict)
        print(f"[StudentTeacherPointCloud] BC-init from {path}: missing={list(missing)} unexpected={list(unexpected)}")
