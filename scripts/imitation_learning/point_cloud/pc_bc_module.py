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
from diffusion_point_net import DiffusionActionPointNet  # noqa: E402
from flatten_mlp import FlattenMLP  # noqa: E402
from history_point_net import HistoryPointNet  # noqa: E402
from point_net import PointNet  # noqa: E402
from residual_point_net import ResidualPointNet  # noqa: E402

_ARCHITECTURES = {
    "point_net": PointNet,
    "flatten_mlp": FlattenMLP,
    "residual_point_net": ResidualPointNet,
    "diffusion_point_net": DiffusionActionPointNet,
    "history_point_net": HistoryPointNet,
}

# Architectures that consume a trajectory HISTORY (sequence windows) rather than a single timestep --
# they take the sequence dataset + step path (is_sequence) and have their own trainer
# (train_point_net_seq.py). Derived from the class flag so a new sequence policy registers itself.
_SEQUENCE_ARCHITECTURES = {name for name, cls in _ARCHITECTURES.items() if getattr(cls, "is_sequence", False)}


def build_policy(
    architecture: str,
    encoder_dims: list[int],
    action_dims: list[int],
    proprio_dim: int,
    action_dim: int,
    predict_std: bool,
    point_dim: int,
    num_points: int,
    num_train_timesteps: int = 100,
    num_sample_steps: int = 10,
    history_len: int = 8,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    transformer_dropout: float = 0.1,
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
    if architecture == "diffusion_point_net":
        kwargs["num_train_timesteps"] = num_train_timesteps
        kwargs["num_sample_steps"] = num_sample_steps
    if architecture == "history_point_net":
        kwargs["history_len"] = history_len
        kwargs["d_model"] = d_model
        kwargs["n_heads"] = n_heads
        kwargs["n_layers"] = n_layers
        kwargs["transformer_dropout"] = transformer_dropout
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
        action_var_weighting: bool = False,
        var_weight_sigma_floor: float = 0.05,
        has_expert_std: bool = False,
        aux_keys: list[str] | None = None,
        pc_parts: list[str] | None = None,
        pc_all_prim_names: list[str] | None = None,
        pc_pad_target: int | None = None,
        point_dropout: dict | None = None,
        prim_dropout: dict | None = None,
        append_prim_semantic: bool = False,
        num_train_timesteps: int = 100,
        num_sample_steps: int = 10,
        history_len: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        transformer_dropout: float = 0.1,
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
            num_train_timesteps=num_train_timesteps,
            num_sample_steps=num_sample_steps,
            history_len=history_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            transformer_dropout=transformer_dropout,
        )
        # DEXTRAH inverse-variance loss weighting needs the per-step expert std in the dataset.
        if action_var_weighting and not has_expert_std:
            raise ValueError(
                "action_var_weighting=True but the dataset has no per-step expert std "
                "('expert_action_std'). Re-collect with collect_pc_demos.py using an expert that "
                "reports std; existing mean-only datasets can't be relabeled offline."
            )
        # Action %-error metric buckets: the last `_late_steps` steps (precise insertion corrections)
        # vs the earlier free-space approach. eps floors the relative-error denominator and the clamp
        # caps the ratio, so a near-zero-action step can't blow the average up (numerical stability).
        self._late_steps = 10
        self._pct_eps = 1e-3
        self._pct_clamp = 10.0  # 1000%
        self.is_diffusion = getattr(self.model, "is_diffusion", False)
        # History-conditioned sequence policy (HistoryPointNet): its own dataset window path and
        # training/deploy step, mirroring the diffusion flag. Aux probe is disabled (below).
        self.is_sequence = getattr(self.model, "is_sequence", False)
        # Normalization stats travel with the checkpoint (and back out at deploy time).
        self.register_buffer("proprio_mean", proprio_mean)
        self.register_buffer("proprio_std", proprio_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        # Auxiliary linear probe: pooled PC feature -> privileged object-pose targets. Enabled only
        # when the dataset has aux targets AND aux_weight>0. A single Linear (no nonlinearity) so the
        # representational burden falls on the shared encoder (gradients flow through `pooled`).
        self.use_aux = aux_dim > 0 and aux_weight > 0.0 and not self.is_diffusion and not self.is_sequence
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

    def _action_loss(self, out, action, expert_std_raw):
        """BC action loss on z-scored actions.

        Default: plain MSE (mean head) or Gaussian-NLL (predict_std head).

        ``action_var_weighting`` (DEXTRAH inverse-variance trick): weight each action dim's squared
        error by ``(1/sigma_expert)^2``. The stored expert std is in RAW action units, so it is first
        converted to the z-scored space the loss lives in (divide by ``action_std``), floored (caps the
        max weight), inverted-squared, and NORMALIZED to mean 1 across the batch so the loss stays on
        the same scale as plain MSE -- only the emphasis across dims/samples changes (confident,
        low-variance expert steps -- e.g. the precise insertion corrections -- are up-weighted; diffuse
        free-space steps down-weighted). Weights are detached (they are data, not parameters). When the
        model also has a std head, the student std additionally regresses toward the expert std."""
        if self.hparams.action_var_weighting:
            mean = out[0] if isinstance(out, tuple) else out
            sigma_z = (expert_std_raw / self.action_std).clamp_min(self.hparams.var_weight_sigma_floor)
            w = sigma_z.reciprocal().pow(2)                       # (B, A) inverse variance
            w = (w / w.mean().clamp_min(1e-8)).detach()           # normalize scale ~ MSE; no grad
            loss = (w * (mean - action).pow(2)).mean()            # weighted MSE
            if isinstance(out, tuple):  # anchor the student std head to the expert std (z-space)
                loss = loss + torch.nn.functional.mse_loss(out[1], sigma_z.log())
            return loss
        if self.model.predict_std:
            mean, log_std = out
            return torch.nn.functional.gaussian_nll_loss(mean, action, torch.exp(2 * log_std))
        return torch.nn.functional.mse_loss(out, action)

    def _log_pct_error(self, mean_z, action_raw, steps_from_end, stage, bs):
        """Scale-invariant action %-error, bucketed by trajectory phase (test of whether the small
        late corrective actions are under-fit relative to the large free-space ones).

        Denormalizes the predicted action, takes the per-sample relative L2 error
        ``||pred - gt|| / max(||gt||, eps)`` (bounded by ``_pct_clamp``), and logs its mean over ALL
        steps, the LAST ``_late_steps`` steps (insertion corrections), and the steps BEFORE those
        (free-space approach). Cheap; runs under no_grad in both train and val."""
        mean_raw = mean_z * self.action_std + self.action_mean
        err = (mean_raw - action_raw).norm(dim=-1)
        denom = action_raw.norm(dim=-1).clamp_min(self._pct_eps)
        pct = (err / denom).clamp(max=self._pct_clamp) * 100.0            # percent, bounded
        late = steps_from_end < self._late_steps
        n_late = int(late.sum())
        self.log(f"{stage}/pct_err_all", pct.mean(), batch_size=bs, sync_dist=True)
        if n_late > 0:
            self.log(f"{stage}/pct_err_last{self._late_steps}", pct[late].mean(),
                     batch_size=n_late, sync_dist=True)
        if n_late < pct.shape[0]:
            self.log(f"{stage}/pct_err_early", pct[~late].mean(),
                     batch_size=pct.shape[0] - n_late, sync_dist=True)

    def _diffusion_step(self, points: torch.Tensor, proprio: torch.Tensor,
                        action: torch.Tensor, stage: str) -> torch.Tensor:
        """DDPM training step: minimize noise-prediction MSE. ``proprio``/``action`` are z-scored.
        Sampling is expensive, so the closed-loop action error is only computed on val."""
        action_loss = self.model.loss(points, proprio, action)
        total = self.hparams.action_weight * action_loss
        bs = points.shape[0]
        self.log(f"{stage}/loss", action_loss, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log(f"{stage}/total_loss", total, batch_size=bs, sync_dist=True)
        if stage == "val":
            with torch.no_grad():
                pred = self.model.sample(points, proprio)  # z-scored action via DDIM
                act_mae = ((pred - action).abs() * self.action_std).mean()
            self.log(f"{stage}/action_mae", act_mae, prog_bar=True, batch_size=bs, sync_dist=True)
        return total

    def _augment_seq(self, points: torch.Tensor, prim_id: torch.Tensor) -> torch.Tensor:
        """Apply the per-frame PC dropout augmentation across a history window.

        ``points`` is (B, H, N, point_dim) and ``prim_id`` (B, H, N) (or an empty (B, H, 0) for flat
        datasets). Flatten the time axis into the batch, reuse the single-frame ``_augment_points``,
        and fold time back. A no-op when no dropout is configured."""
        if not (self.hparams.point_dropout or self.hparams.prim_dropout):
            return points
        B, H, N, C = points.shape
        pts = points.reshape(B * H, N, C)
        pid = prim_id.reshape(B * H, prim_id.shape[-1]) if prim_id.numel() > 0 else None
        pts = self._augment_points(pts, pid)
        return pts.reshape(B, H, N, C)

    def _sequence_step(self, points, proprio, action, valid, stage: str):
        """History-conditioned Transformer step. Inputs carry a time axis: ``points`` (B,H,N,pd),
        ``proprio`` (B,H,proprio_dim), ``action`` (B,H,action_dim), ``valid`` (B,H). Proprio/action
        are z-scored (broadcast over the H axis), the model predicts an action at every state token,
        and the BC loss is masked to the valid (non-padded) positions."""
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        action_n = (action - self.action_mean) / self.action_std
        m = valid.float()
        denom = m.sum().clamp_min(1.0)
        out = self.model(points, proprio, action_n, valid)  # (B,H,ad) or (mean, log_std)
        if self.model.predict_std:
            mean, log_std = out
            per = torch.nn.functional.gaussian_nll_loss(
                mean, action_n, torch.exp(2 * log_std), reduction="none"
            ).mean(-1)
        else:
            mean = out
            per = torch.nn.functional.mse_loss(out, action_n, reduction="none").mean(-1)  # (B,H)
        action_loss = (per * m).sum() / denom
        total = self.hparams.action_weight * action_loss

        bs = points.shape[0]
        with torch.no_grad():
            ae = ((mean - action_n).abs() * self.action_std).mean(-1)  # (B,H) action-unit MAE
            act_mae = (ae * m).sum() / denom
        self.log(f"{stage}/loss", action_loss, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log(f"{stage}/total_loss", total, batch_size=bs, sync_dist=True)
        self.log(f"{stage}/action_mae", act_mae, prog_bar=(stage == "val"), batch_size=bs, sync_dist=True)
        return total

    def _step(self, batch, stage: str):
        points, proprio, action_raw, aux, steps_from_end, expert_std = batch
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        action = (action_raw - self.action_mean) / self.action_std
        if self.is_diffusion:
            return self._diffusion_step(points, proprio, action, stage)
        # Single forward; also grab the pooled feature for the aux probe.
        out, pooled = self.model(points, proprio, return_pooled=True)
        action_loss = self._action_loss(out, action, expert_std)

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
            # Scale-invariant %-error, split by trajectory phase (late insertion vs early free-space).
            self._log_pct_error(mean, action_raw, steps_from_end, stage, bs)
        # val/loss = pure ACTION loss -> checkpoint selection + comparability stay on the BC objective,
        # independent of aux_weight. The optimizer still minimizes `total` (action + aux).
        self.log(f"{stage}/loss", action_loss, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log(f"{stage}/total_loss", total, batch_size=bs, sync_dist=True)
        self.log(f"{stage}/action_mae", act_mae, prog_bar=(stage == "val"), batch_size=bs, sync_dist=True)
        return total

    def training_step(self, batch, _):
        if self.is_sequence:
            # History datasets yield (points, prim_id, proprio, action, valid) with a leading time axis.
            points, prim_id, proprio, action, valid = batch
            points = self._augment_seq(points, prim_id)  # train-only dropout; val keeps the clean cloud
            return self._sequence_step(points, proprio, action, valid, "train")
        points, prim_id, proprio, action, aux, steps_from_end, expert_std = batch
        points = self._augment_points(points, prim_id)  # train-only dropout; val keeps the clean cloud
        return self._step((points, proprio, action, aux, steps_from_end, expert_std), "train")

    def validation_step(self, batch, _):
        if self.is_sequence:
            points, prim_id, proprio, action, valid = batch
            return self._sequence_step(points, proprio, action, valid, "val")
        points, prim_id, proprio, action, aux, steps_from_end, expert_std = batch
        return self._step((points, proprio, action, aux, steps_from_end, expert_std), "val")

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
