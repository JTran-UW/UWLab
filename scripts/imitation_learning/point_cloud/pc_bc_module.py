# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PointNet behavior-cloning LightningModule (:class:`PointNetBC`) + policy builder.

Split out of ``train_point_net.py``. Wraps a raw point-cloud policy (``point_net.py`` /
``flatten_mlp.py`` / ``residual_point_net.py``) with proprio/action z-scoring, the optional
auxiliary pose probe, and train-time point-cloud dropout augmentation. ``_ARCHITECTURES`` and
``build_policy`` are exported so the trainer can validate the requested architecture and
``bc_utils.py`` mirrors the same registry at deploy time.
"""

from __future__ import annotations

import os
import sys

import torch

import lightning as L

# Policy modules live next to this file; make them importable regardless of caller cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flatten_mlp import FlattenMLP  # noqa: E402
from point_net import PointNet  # noqa: E402
from residual_point_net import ResidualPointNet  # noqa: E402

_ARCHITECTURES = {
    "point_net": PointNet,
    "flatten_mlp": FlattenMLP,
    "residual_point_net": ResidualPointNet,
}


def build_policy(
    architecture: str,
    encoder_dims: list[int],
    action_dims: list[int],
    proprio_dim: int,
    action_dim: int,
    predict_std: bool,
    point_dim: int,
    num_points: int,
):
    cls = _ARCHITECTURES[architecture]
    kwargs = dict(
        encoder_hidden_dims=encoder_dims,
        action_hidden_dims=action_dims,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        predict_std=predict_std,
        point_dim=point_dim,
    )
    if architecture == "flatten_mlp":
        kwargs["num_points"] = num_points
    return cls(**kwargs)


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(B,3) axis-angle -> (B,3,3) rotation matrices (Rodrigues)."""
    theta = aa.norm(dim=-1, keepdim=True)                       # (B,1)
    axis = aa / theta.clamp_min(1e-8)                           # (B,3) unit (arbitrary where theta~0)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(aa.shape[0], 3, 3)
    s = theta.sin().unsqueeze(-1)
    c = (1.0 - theta.cos()).unsqueeze(-1)
    return eye + s * K + c * (K @ K)


def _geodesic_deg(aa_pred: torch.Tensor, aa_gt: torch.Tensor) -> torch.Tensor:
    """Per-sample geodesic angle (degrees) between two axis-angle rotations."""
    R = _axis_angle_to_matrix(aa_pred).transpose(-1, -2) @ _axis_angle_to_matrix(aa_gt)
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((trace - 1) / 2).clamp(-1.0, 1.0)) * (180.0 / torch.pi)


class PointNetBC(L.LightningModule):
    """Behavior-cloning wrapper around a point-cloud policy with proprio/action z-scoring."""

    def __init__(
        self,
        architecture: str,
        proprio_dim: int,
        action_dim: int,
        encoder_dims: list[int],
        action_dims: list[int],
        predict_std: bool,
        lr: float,
        weight_decay: float,
        proprio_mean: torch.Tensor,
        proprio_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        point_dim: int,
        num_points: int,
        aux_dim: int = 0,
        aux_weight: float = 0.0,
        aux_mean: torch.Tensor | None = None,
        aux_std: torch.Tensor | None = None,
        action_weight: float = 1.0,
        aux_keys: list[str] | None = None,
        pc_parts: list[str] | None = None,
        pc_all_prim_names: list[str] | None = None,
        pc_pad_target: int | None = None,
        point_dropout: dict | None = None,
        prim_dropout: dict | None = None,
        append_prim_semantic: bool = False,
    ):
        super().__init__()
        # pc_parts / pc_all_prim_names / pc_pad_target travel in the checkpoint hparams so deploy
        # (bc_utils) can map the live per-prim cloud's prim ids, keep the SAME prims, and pad to
        # the same size.
        self.save_hyperparameters(
            ignore=["proprio_mean", "proprio_std", "action_mean", "action_std", "aux_mean", "aux_std"]
        )
        # Byte-offset slices into the concatenated aux vector, keyed by term name (each term is 6d:
        # pos[0:3] + axis_angle[3:6]). Used for per-asset val metrics (position mm, rotation deg).
        self.aux_slices = {}
        off = 0
        for k in (aux_keys or []):
            self.aux_slices[k] = (off, off + 6)
            off += 6
        self.model = build_policy(
            architecture=architecture,
            encoder_dims=encoder_dims,
            action_dims=action_dims,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            predict_std=predict_std,
            point_dim=point_dim,
            num_points=num_points,
        )
        # Normalization stats travel with the checkpoint (and back out at deploy time).
        self.register_buffer("proprio_mean", proprio_mean)
        self.register_buffer("proprio_std", proprio_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        # Auxiliary linear probe: pooled PC feature -> privileged object-pose targets. Enabled only
        # when the dataset has aux targets AND aux_weight>0. A single Linear (no nonlinearity) so the
        # representational burden falls on the shared encoder (gradients flow through `pooled`).
        self.use_aux = aux_dim > 0 and aux_weight > 0.0
        if self.use_aux:
            self.aux_probe = torch.nn.Linear(self.model.pooled_dim, aux_dim)
            self.register_buffer("aux_mean", aux_mean)
            self.register_buffer("aux_std", aux_std)

    # Segmentation-label value (4th channel) for each semantic category. Used to key the per-category
    # train-time point/prim dropout on the LEGACY seg-label path. Mirrors the collector's labels
    # (robot=0, insertive=-1, receptive=+1).
    _CAT_LABEL = {"robot": 0.0, "insertive": -1.0, "receptive": 1.0}
    # Object prim names; every other prim is treated as a robot part. Keep in sync with the collector.
    _OBJECT_PRIMS = ("insertive", "receptive")

    @classmethod
    def _prim_category(cls, name: str) -> str:
        return name if name in cls._OBJECT_PRIMS else "robot"

    def _augment_points(self, points: torch.Tensor, prim_id: torch.Tensor | None) -> torch.Tensor:
        """Train-time PC dropout augmentation (returns a possibly-modified copy).

        Two independent dropout mechanisms, both controlled by the per-category dicts
        ``prim_dropout`` / ``point_dropout`` (categories: ``robot``, ``insertive``, ``receptive``).
        "Dropout" means setting the whole point to 0s. Probabilities come from the training config.
        A no-op when neither dict is configured. Applied at TRAINING time only (val stays clean).

        PER-PRIM path (preferred; used when the batch carries ``prim_id``): each point knows its
        SOURCE PRIM, so ``prim_dropout[cat]`` drops EACH prim of that category INDEPENDENTLY
        (e.g. with ``robot: 0.3`` every robot link -- wrist_3, forearm, ... -- is dropped with
        prob 0.3 on its own, modelling a real-world cloud that misses some robot parts).
        ``point_dropout[cat]`` then zeroes individual points of that category. Order: prim -> point
        (point dropout also covers prim-dropped points, but the result is identical since both zero).

        LEGACY seg-label path (no prim_id, requires ``point_dim==4``): categories come from the 4th
        channel and ``prim_dropout[cat]`` drops the WHOLE category at once (not per prim).
        """
        pd = self.hparams.point_dropout or {}
        rd = self.hparams.prim_dropout or {}
        if not pd and not rd:
            return points
        if prim_id is not None and prim_id.numel() > 0:
            return self._augment_per_prim(points, prim_id, pd, rd)

        labels = points[..., 3]                                   # (B, N) per-point category label
        B, N = labels.shape
        drop = torch.zeros_like(labels, dtype=torch.bool)         # True -> zero this point
        for cat, lbl in self._CAT_LABEL.items():
            cat_mask = labels == lbl                              # (B, N) points in this category
            p_prim = float(rd.get(cat, 0.0))
            if p_prim > 0.0:                                      # drop the WHOLE category for a sample
                prim_drop = torch.rand(B, 1, device=points.device) < p_prim   # (B, 1) -> broadcast over N
                drop |= cat_mask & prim_drop
            p_pt = float(pd.get(cat, 0.0))
            if p_pt > 0.0:                                        # drop individual points
                pt_drop = torch.rand(B, N, device=points.device) < p_pt
                drop |= cat_mask & pt_drop
        points = points.clone()
        points[drop] = 0.0                                        # zero xyz + label together
        return points

    def _augment_per_prim(self, points: torch.Tensor, prim_id: torch.Tensor,
                          pd: dict, rd: dict) -> torch.Tensor:
        """Per-prim dropout: drop each individual prim (and/or its points) by its category rate.

        ``prim_id`` is (B, N) original prim ids (pad slots are -1). Prim-level Bernoulli draws are
        INDEPENDENT per (sample, prim), so each robot link drops on its own."""
        names = self.hparams.pc_all_prim_names or []
        P = len(names)
        device = points.device
        B, N = prim_id.shape
        pid = prim_id.to(device)
        valid = pid >= 0                                          # (B, N) real (non-pad) points
        pid_c = pid.clamp_min(0)                                  # safe gather index for pad slots
        cats = [self._prim_category(n) for n in names]
        drop = torch.zeros(B, N, dtype=torch.bool, device=device)

        prim_p = torch.tensor([float(rd.get(c, 0.0)) for c in cats], device=device)   # (P,)
        if bool((prim_p > 0).any()):                             # 1) per (sample, prim) independent drop
            prim_drop = torch.rand(B, P, device=device) < prim_p                       # (B, P)
            drop |= prim_drop.gather(1, pid_c)                                         # (B, N)

        pt_p = torch.tensor([float(pd.get(c, 0.0)) for c in cats], device=device)      # (P,)
        if bool((pt_p > 0).any()):                              # 2) per-point drop at the prim's rate
            drop |= torch.rand(B, N, device=device) < pt_p[pid_c]

        drop &= valid                                            # never "drop" pad slots (already 0)
        points = points.clone()
        points[drop] = 0.0
        return points

    def _action_loss(self, out, action):
        if self.model.predict_std:
            mean, log_std = out
            return torch.nn.functional.gaussian_nll_loss(mean, action, torch.exp(2 * log_std))
        return torch.nn.functional.mse_loss(out, action)

    def _step(self, batch, stage: str):
        points, proprio, action, aux = batch
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        action = (action - self.action_mean) / self.action_std
        # Single forward; also grab the pooled feature for the aux probe.
        out, pooled = self.model(points, proprio, return_pooled=True)
        action_loss = self._action_loss(out, action)

        bs = points.shape[0]
        total = self.hparams.action_weight * action_loss
        if self.use_aux:
            aux_pred_n = self.aux_probe(pooled)
            aux_t = (aux - self.aux_mean) / self.aux_std
            aux_loss = torch.nn.functional.mse_loss(aux_pred_n, aux_t)
            total = total + self.hparams.aux_weight * aux_loss
            self.log(f"{stage}/aux_loss", aux_loss, batch_size=bs, sync_dist=True)
            # Per-asset interpretable metrics (denormalized): position error (mm), rotation geodesic
            # (deg), and fraction-of-variance-unexplained (FVU, z-scored MSE -> 1=no info, 0=perfect).
            with torch.no_grad():
                aux_pred = aux_pred_n * self.aux_std + self.aux_mean
                for name, (a, b) in self.aux_slices.items():
                    if name == "insertive_in_receptive":
                        continue  # relational term; report the two absolute asset poses only
                    tag = "ins" if "insertive" in name else "rec"
                    pos_mm = (aux_pred[:, a:a + 3] - aux[:, a:a + 3]).norm(dim=-1).mean() * 1000.0
                    rot_deg = _geodesic_deg(aux_pred[:, a + 3:b], aux[:, a + 3:b]).mean()
                    fvu = (aux_pred_n[:, a:b] - aux_t[:, a:b]).pow(2).mean()
                    self.log(f"{stage}/{tag}_pos_mm", pos_mm, batch_size=bs, sync_dist=True)
                    self.log(f"{stage}/{tag}_rot_deg", rot_deg, batch_size=bs, sync_dist=True)
                    self.log(f"{stage}/{tag}_fvu", fvu, batch_size=bs, sync_dist=True)

        # An action-space (denormalized, per-dim RMS) error is the interpretable metric.
        with torch.no_grad():
            mean = out[0] if isinstance(out, tuple) else out
            act_mae = ((mean - action).abs() * self.action_std).mean()
        # val/loss = pure ACTION loss -> checkpoint selection + comparability stay on the BC objective,
        # independent of aux_weight. The optimizer still minimizes `total` (action + aux).
        self.log(f"{stage}/loss", action_loss, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log(f"{stage}/total_loss", total, batch_size=bs, sync_dist=True)
        self.log(f"{stage}/action_mae", act_mae, prog_bar=(stage == "val"), batch_size=bs, sync_dist=True)
        return total

    def training_step(self, batch, _):
        points, prim_id, proprio, action, aux = batch
        points = self._augment_points(points, prim_id)  # train-only dropout; val keeps the clean cloud
        return self._step((points, proprio, action, aux), "train")

    def validation_step(self, batch, _):
        points, prim_id, proprio, action, aux = batch
        return self._step((points, proprio, action, aux), "val")

    def on_validation_epoch_end(self):
        # Print the per-asset probe metrics so a pure state-extraction run is readable from stdout.
        if not self.use_aux:
            return
        m = self.trainer.callback_metrics
        def g(k):
            v = m.get(k)
            return float(v) if v is not None else float("nan")
        print(
            f"[probe] epoch {self.current_epoch:>3} | "
            f"INSERTIVE pos={g('val/ins_pos_mm'):6.1f}mm rot={g('val/ins_rot_deg'):5.1f}deg fvu={g('val/ins_fvu'):.3f} | "
            f"RECEPTIVE pos={g('val/rec_pos_mm'):6.1f}mm rot={g('val/rec_rot_deg'):5.1f}deg fvu={g('val/rec_fvu'):.3f}",
            flush=True,
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
