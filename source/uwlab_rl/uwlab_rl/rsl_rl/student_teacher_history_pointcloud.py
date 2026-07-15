# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""History-conditioned point-cloud student + JIT teacher for rsl_rl DAgger.

Sequence sibling of :class:`~uwlab_rl.rsl_rl.student_teacher_pointcloud.StudentTeacherPointCloud`:
the student is a :class:`uwlab_rl.networks.history_point_net.HistoryPointNet` — the SAME class the
offline sequence BC trainer (``train_point_net_seq.py``) builds — conditioned on a rolling window
of the last ``history_len`` (state, executed-action) pairs instead of a single frame.

Statefulness. The env obs carry only the CURRENT frame; the student keeps per-env rolling buffers
as its "hidden state":

* ``hist_tok``   (E, H, d_model) — per-frame state tokens (pooled PC feature ⊕ proprio), cached
  DETACHED so a policy step encodes ONE new frame and the Transformer runs over cached tokens
  (identical values to re-encoding the window at equal weights; gradient flows through the current
  frame's encoder only — the per-step DAgger cadence gives the encoder a dense 1-frame gradient
  anyway, exactly like the feed-forward student).
* ``hist_act``   (E, H, action_dim) — z-scored EXECUTED actions. The action actually applied to the
  env (student, teacher, or mixed — decided by the DAgger algorithm) is fed back via
  :meth:`record_executed_action`, which the algorithm calls right after choosing it.
* ``hist_valid`` (E, H) — real-vs-pad mask; a reset env restarts with only its current frame valid,
  matching the BC dataset's left-padded episode starts.

The rsl_rl recurrent hooks map onto the buffers: ``reset(dones)`` clears finished envs,
``get_hidden_states()``/``reset(hidden_states=...)`` snapshot/restore them around the update replay
(the algorithm replays the stored rollout in time order, so the buffers rebuild consistently), and
``detach_hidden_states`` is a no-op (buffers are stored detached by construction).

Normalization. Unlike the feed-forward student (online ``EmpiricalNormalization``, env-unit action
outputs), this student bakes the BC checkpoint's ``proprio_mean/std`` + ``action_mean/std`` in as
buffers: proprio is z-scored before token encoding, executed actions are z-scored before embedding,
and the network's (z-scored) output mean/std are DE-normalized back to env units. BC-init is
therefore value-exact, and the DAgger loss still compares env-unit means against the teacher.
Without a BC init the stats default to identity (mean 0 / std 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from uwlab_rl.networks.history_point_net import HistoryPointNet

from .student_teacher_pointcloud import StudentTeacherPointCloud


class StudentTeacherHistoryPointCloud(StudentTeacherPointCloud):
    """HistoryPointNet student + JIT teacher (rolling-window point-cloud DAgger)."""

    is_recurrent: bool = False  # statefulness handled via the reset/hidden-state hooks below

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_jit_path: str,
        pointcloud_groups: list[str],
        point_dim: int = 3,
        architecture: str = "history_point_net",
        encoder_dims: tuple[int, ...] | list[int] = (256, 512, 512),
        action_dims: tuple[int, ...] | list[int] = (512, 256),
        history_len: int = 16,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        transformer_dropout: float = 0.1,
        activation: str = "elu",  # signature parity; unused
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        student_obs_normalization: bool = False,  # signature parity; stats come from the BC ckpt
        predict_std: bool = False,
        teacher_returns_std: bool = False,
        bc_init_path: str = "",
        **kwargs,
    ) -> None:
        if architecture != "history_point_net":
            raise ValueError(f"StudentTeacherHistoryPointCloud only supports 'history_point_net'; got {architecture}")
        # Bypass ALL parent __init__s (they build feed-forward students / MLP teachers).
        nn.Module.__init__(self)
        Normal.set_default_validate_args(False)

        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.point_dim = int(point_dim)
        self.predict_std = bool(predict_std)
        self.history_len = int(history_len)

        # Split policy groups into point-cloud group(s) and proprio group(s) — parent logic verbatim.
        self.pointcloud_groups = list(pointcloud_groups)
        policy_groups = list(obs_groups["policy"])
        missing = [g for g in self.pointcloud_groups if g not in policy_groups]
        if missing:
            raise ValueError(f"pointcloud_groups {missing} not in obs_groups['policy']={policy_groups}")
        self.proprio_groups = [g for g in policy_groups if g not in self.pointcloud_groups]

        num_points = None
        for g in self.pointcloud_groups:
            flat = obs[g].shape[-1]
            if flat % self.point_dim != 0:
                raise ValueError(
                    f"pointcloud group '{g}' has flat dim {flat} not divisible by point_dim={self.point_dim}"
                )
            n = flat // self.point_dim
            num_points = n if num_points is None else num_points + n
        self.num_points = num_points

        if self.proprio_groups:
            proprio = torch.cat([obs[g] for g in self.proprio_groups], dim=-1)
            assert proprio.ndim == 2, f"proprio groups must be 1D; got shape {proprio.shape}"
            num_proprio = proprio.shape[-1]
        else:
            num_proprio = 0
        self.num_proprio = num_proprio

        # Student == the SAME HistoryPointNet class the sequence BC trainer uses.
        self.student = HistoryPointNet(
            encoder_hidden_dims=list(encoder_dims),
            action_hidden_dims=list(action_dims),
            proprio_dim=num_proprio,
            action_dim=num_actions,
            predict_std=predict_std,
            point_dim=self.point_dim,
            history_len=self.history_len,
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            transformer_dropout=float(transformer_dropout),
        )
        print(
            f"StudentTeacherHistoryPointCloud student (H={self.history_len}, d_model={d_model}, "
            f"layers={n_layers}, points={self.num_points}x{self.point_dim}, proprio={num_proprio}, "
            f"act={num_actions}, predict_std={predict_std})"
        )

        # Fixed z-scoring stats (overwritten by the BC ckpt in load_bc_checkpoint). The BC network
        # operates in z-scored units for proprio AND actions — identity fallback keeps a from-scratch
        # student functional.
        self.register_buffer("proprio_mean", torch.zeros(num_proprio))
        self.register_buffer("proprio_std", torch.ones(num_proprio))
        self.register_buffer("action_mean", torch.zeros(num_actions))
        self.register_buffer("action_std", torch.ones(num_actions))
        self.student_obs_normalization = False
        self.student_obs_normalizer = nn.Identity()

        # JIT teacher — identical to the feed-forward student.
        self.teacher = torch.jit.load(teacher_jit_path)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_obs_normalization = False
        self.teacher_obs_normalizer = nn.Identity()
        self.loaded_teacher = True
        self.teacher_returns_std = bool(teacher_returns_std)

        # Student exploration noise (act(); independent of the predict_std head).
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"noise_std_type must be 'scalar' or 'log'; got {self.noise_std_type}")

        self.log_std_limits = (-5.0, 2.0)
        self.distribution = None

        # Rolling history buffers, sized lazily on the first forward (num_envs from the obs batch).
        self._hist_tok: torch.Tensor | None = None
        self._hist_act: torch.Tensor | None = None
        self._hist_valid: torch.Tensor | None = None

        if bc_init_path:
            self.load_bc_checkpoint(bc_init_path, strict=False)

    # ------------------------------------------------------------------ history buffers

    def _ensure_buffers(self, batch: int, device: torch.device) -> None:
        if self._hist_tok is not None and self._hist_tok.shape[0] == batch:
            return
        H = self.history_len
        self._hist_tok = torch.zeros(batch, H, self.student.d_model, device=device)
        self._hist_act = torch.zeros(batch, H, self.num_actions, device=device)
        self._hist_valid = torch.zeros(batch, H, dtype=torch.bool, device=device)

    def _student_forward(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance the rolling window with the current frame and read the last-token action.

        Returns ``(mean, log_std_or_None)`` in ENV units (mean/std de-normalized by the BC action
        stats). Side effect: the history buffers roll forward one slot; the current frame's token is
        stored detached; its executed action is filled in later via :meth:`record_executed_action`.
        """
        points, proprio = self._split_student_obs(obs)  # proprio already z-scored via our override
        self._ensure_buffers(points.shape[0], points.device)

        tok = self.student.encode_state_frame(points, proprio)  # (B, d_model), grad flows here

        # Window for THIS forward: cached tokens shifted left + the fresh (grad-carrying) token.
        win_tok = torch.cat([self._hist_tok[:, 1:], tok.unsqueeze(1)], dim=1)
        win_act = torch.cat([self._hist_act[:, 1:], torch.zeros_like(self._hist_act[:, :1])], dim=1)
        win_valid = torch.cat(
            [self._hist_valid[:, 1:], torch.ones_like(self._hist_valid[:, :1])], dim=1
        )

        out = self.student.forward_from_tokens(win_tok, win_act, win_valid)
        if self.predict_std:
            mean_n, log_std_n = out[0][:, -1], out[1][:, -1]
        else:
            mean_n, log_std_n = out[:, -1], None

        # Commit the rolled buffers (detached token; action slot zeroed until recorded).
        self._hist_tok = win_tok.detach()
        self._hist_act = win_act
        self._hist_valid = win_valid

        mean = mean_n * self.action_std + self.action_mean
        return mean, log_std_n

    def _split_student_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Like the parent, but z-score proprio with the fixed BC stats."""
        pcs = []
        for g in self.pointcloud_groups:
            flat = obs[g]
            pcs.append(flat.view(flat.shape[0], -1, self.point_dim))
        points = torch.cat(pcs, dim=1) if len(pcs) > 1 else pcs[0]
        if self.proprio_groups:
            proprio = torch.cat([obs[g] for g in self.proprio_groups], dim=-1)
            proprio = (proprio - self.proprio_mean) / self.proprio_std
        else:
            proprio = points.new_zeros((points.shape[0], 0))
        return points, proprio

    def record_executed_action(self, action: torch.Tensor) -> None:
        """Store the EXECUTED env action (z-scored) into the current step's action slot.

        Called by the DAgger algorithm after it picks the applied action (student / teacher /
        mixed), both during rollout and during the time-ordered update replay."""
        if self._hist_act is None:
            return
        self._hist_act = self._hist_act.clone()
        self._hist_act[:, -1] = ((action - self.action_mean) / self.action_std).detach()

    # ------------------------------------------------------------------ student API

    def act_inference_with_std(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.predict_std:
            raise RuntimeError("act_inference_with_std called but predict_std=False")
        mean, log_std = self._student_forward(obs)
        std = torch.exp(log_std.clamp(*self.log_std_limits)) * self.action_std
        return mean, std

    def update_normalization(self, obs: TensorDict) -> None:
        pass  # fixed BC stats; nothing to update

    # ------------------------------------------------------------------ hidden-state hooks

    def reset(self, dones: torch.Tensor | None = None, hidden_states=(None, None)) -> None:
        # Restore a snapshot (update replay start). (None, None) is rsl_rl's "no snapshot" default —
        # at the very first update that means "rollout started from empty buffers": clear everything.
        if hidden_states is not None and not all(h is None for h in hidden_states):
            tok, act, valid = hidden_states
            self._hist_tok = tok.clone()
            self._hist_act = act.clone()
            self._hist_valid = valid.clone()
            return
        if dones is None:
            # Out-of-place: rollout-side buffers are inference tensors (the runner rolls out under
            # torch.inference_mode()), which reject in-place writes from the update side.
            if self._hist_tok is not None:
                self._hist_tok = torch.zeros_like(self._hist_tok)
                self._hist_act = torch.zeros_like(self._hist_act)
                self._hist_valid = torch.zeros_like(self._hist_valid)
            return
        if self._hist_tok is None:
            return
        # rsl_rl convention (matches Memory.reset): dones is a per-env 0/1 mask, not indices.
        mask = dones.view(-1).bool()
        if not bool(mask.any()):
            return
        keep = ~mask
        self._hist_tok = self._hist_tok * keep.view(-1, 1, 1)
        self._hist_act = self._hist_act * keep.view(-1, 1, 1)
        self._hist_valid = self._hist_valid & keep.view(-1, 1)

    def get_hidden_states(self):
        if self._hist_tok is None:
            return (None, None)
        return (self._hist_tok.clone(), self._hist_act.clone(), self._hist_valid.clone())

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass  # buffers are stored detached by construction

    # ------------------------------------------------------------------ BC init

    def load_bc_checkpoint(self, path: str, strict: bool = False) -> None:
        """Load a sequence-BC ckpt: network weights via the parent, plus the z-scoring stats."""
        super().load_bc_checkpoint(path, strict=strict)
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        stats = {k: sd[k] for k in ("proprio_mean", "proprio_std", "action_mean", "action_std") if k in sd}
        if len(stats) == 4:
            self.proprio_mean.copy_(stats["proprio_mean"].to(self.proprio_mean.device))
            self.proprio_std.copy_(stats["proprio_std"].to(self.proprio_std.device))
            self.action_mean.copy_(stats["action_mean"].to(self.action_mean.device))
            self.action_std.copy_(stats["action_std"].to(self.action_std.device))
            print(f"[StudentTeacherHistoryPointCloud] loaded BC z-scoring stats from {path}")
        else:
            print(
                f"[StudentTeacherHistoryPointCloud] WARNING: BC ckpt {path} has no z-scoring stats "
                f"(found {sorted(stats)}); keeping identity normalization"
            )
