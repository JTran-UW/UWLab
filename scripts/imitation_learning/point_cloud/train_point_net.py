# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train :class:`PointNet` (behavior cloning) on a collected point-cloud demo dataset.

Loads an HDF5 dataset produced by ``scripts/tools/sim2real/collect_pc_demos.py``
(robomimic-style: ``data/demo_*/obs/<term>`` + ``data/demo_*/actions``), flattens
every timestep into a flat (point-cloud, proprio, action) BC dataset, and trains the
PointNet with PyTorch Lightning. Logs to Weights & Biases.

The PC term is ``scene_pc`` (``(T, num_points, 3)``); every *other* obs term
(``joint_pos``, ``end_effector_pose``, ...) is concatenated -- in the order stored in
the dataset's ``obs_keys`` attr -- into the proprio vector. The whole dataset is loaded
into RAM (the machine has ~1 TB; a 100k-demo set is ~50 GB).

Usage::

    python scripts/imitation_learning/point_cloud/train_point_net.py \\
        --dataset demos/pc_demos.hdf5 --epochs 50 --batch_size 512 \\
        --wandb_project pc_bc --run_name pointnet_v0
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# PointNet lives next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from point_net import PointNet  # noqa: E402

# Obs terms that are point clouds (stored (T, num_points, 3)); everything else -> proprio,
# EXCEPT the weight terms below, which are recorded only for dataset re-weighting and are
# never fed to the model.
_PC_TERMS = {"scene_pc"}
# Per-step scalar(s) used purely as a sampling weight (e.g. EE contact-force magnitude), not
# a policy input. Excluded from proprio; surfaced as ``dataset.force`` for WeightedRandomSampler.
_WEIGHT_TERMS = {"wrist_force"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PCDemoDataset(Dataset):
    """All (point_cloud, proprio, action) timesteps from an HDF5 demo file, in RAM.

    ``proprio`` is the concatenation of every non-PC obs term, in the dataset's
    ``obs_keys`` order. If ``num_points`` is smaller than the stored cloud size, each
    ``__getitem__`` randomly subsamples that many points (cheap PC augmentation).
    """

    def __init__(self, path: str, num_points: int | None = None, success_only: bool = True):
        super().__init__()
        with h5py.File(path, "r") as f:
            data = f["data"]
            obs_keys = list(data.attrs["obs_keys"])
            pc_key = next(k for k in obs_keys if k in _PC_TERMS)
            self.proprio_keys = [k for k in obs_keys if k not in _PC_TERMS and k not in _WEIGHT_TERMS]
            # Optional per-step weight term (e.g. wrist_force); None on older datasets.
            self.weight_key = next((k for k in obs_keys if k in _WEIGHT_TERMS), None)

            demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
            if success_only:
                demos = [d for d in demos if bool(data[d].attrs.get("success", True))]

            total = sum(data[d]["actions"].shape[0] for d in demos)
            n_pts = data[demos[0]]["obs"][pc_key].shape[1]
            proprio_dim = sum(data[demos[0]]["obs"][k].shape[1] for k in self.proprio_keys)
            action_dim = data[demos[0]]["actions"].shape[1]

            # Preallocate once and fill (avoids the 2x peak of np.concatenate).
            self.points = np.empty((total, n_pts, 3), dtype=np.float32)
            self.proprio = np.empty((total, proprio_dim), dtype=np.float32)
            self.actions = np.empty((total, action_dim), dtype=np.float32)
            self.force = np.empty((total,), dtype=np.float32) if self.weight_key else None

            i = 0
            for d in demos:
                g = data[d]
                t = g["actions"].shape[0]
                self.points[i : i + t] = g["obs"][pc_key][:]
                self.proprio[i : i + t] = np.concatenate([g["obs"][k][:] for k in self.proprio_keys], axis=1)
                self.actions[i : i + t] = g["actions"][:]
                if self.weight_key is not None:
                    self.force[i : i + t] = np.asarray(g["obs"][self.weight_key][:]).reshape(t, -1)[:, 0]
                i += t

            self.expert_success_rate = float(data.attrs.get("expert_success_rate", float("nan")))

        self.points = torch.from_numpy(self.points)
        self.proprio = torch.from_numpy(self.proprio)
        self.actions = torch.from_numpy(self.actions)
        self.force = torch.from_numpy(self.force) if self.force is not None else None
        self.n_pts = n_pts
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.subsample = num_points if (num_points and num_points < n_pts) else None

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        pts = self.points[idx]
        if self.subsample is not None:
            sel = torch.randint(0, self.n_pts, (self.subsample,))
            pts = pts[sel]
        return pts, self.proprio[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class PointNetBC(L.LightningModule):
    """Behavior-cloning wrapper around :class:`PointNet` with proprio/action z-scoring."""

    def __init__(
        self,
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["proprio_mean", "proprio_std", "action_mean", "action_std"])
        self.model = PointNet(
            encoder_hidden_dims=encoder_dims,
            action_hidden_dims=action_dims,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            predict_std=predict_std,
        )
        # Normalization stats travel with the checkpoint (and back out at deploy time).
        self.register_buffer("proprio_mean", proprio_mean)
        self.register_buffer("proprio_std", proprio_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

    def _step(self, batch, stage: str):
        points, proprio, action = batch
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        action = (action - self.action_mean) / self.action_std
        loss = self.model.calculate_loss(points, proprio, action)
        # An action-space (denormalized, per-dim RMS) error is the interpretable metric.
        with torch.no_grad():
            pred = self.model(points, proprio)
            mean = pred[0] if isinstance(pred, tuple) else pred
            act_mae = ((mean - action).abs() * self.action_std).mean()
        bs = points.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log(f"{stage}/action_mae", act_mae, prog_bar=(stage == "val"), batch_size=bs, sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Config (dataclasses, populated from YAML + CLI overrides)
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Per-point set-encoder MLP (in=3 -> hidden dims); last dim is the pooled feature size.
    encoder_dims: list[int] = field(default_factory=lambda: [64, 128, 256])
    # Action-head MLP hidden dims (input is pooled_pc + proprio feats = 2 * encoder_dims[-1]).
    action_dims: list[int] = field(default_factory=lambda: [256, 128])
    # False: MSE head. True: Gaussian-NLL head that also predicts a per-dim std.
    predict_std: bool = False


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 512


@dataclass
class DataConfig:
    num_points: int | None = None  # None -> use the full stored cloud
    val_frac: float = 0.05
    num_workers: int = 8
    no_success_filter: bool = False  # False -> successful demos only
    seed: int = 0
    # --- force-weighted up-sampling (requires a `wrist_force` term in the dataset) ---
    # When on, training draws timesteps with prob proportional to a baseline-plus-force
    # weight: w_i = 1 + coef * (force_i / force_ref)^alpha, force_ref = the q-quantile of
    # train forces. The baseline 1 keeps every sample reachable; the second term up-weights
    # forceful (insertion/contact) timesteps. coef scales how much extra mass they get;
    # alpha sharpens it toward the very-most-forceful samples.
    force_weighting: bool = False
    force_weight_coef: float = 2.0
    force_weight_alpha: float = 1.0
    force_ref_quantile: float = 0.95


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | None) -> "TrainConfig":
        cfg = cls()
        if path and os.path.exists(path):
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            for section in ("model", "optim", "data"):
                for k, v in (raw.get(section) or {}).items():
                    cfg._set(k, v, where=f"{path}:{section}")
            print(f"[train] loaded config: {path}")
        return cfg

    def _set(self, key: str, value, where: str = "override"):
        """Set ``key`` on whichever section declares it (keys are unique across sections)."""
        for section in (self.model, self.optim, self.data):
            if hasattr(section, key):
                setattr(section, key, value)
                return
        raise KeyError(f"unknown config key '{key}' (from {where})")

    def apply_overrides(self, overrides: dict):
        for k, v in overrides.items():
            self._set(k, v, where="CLI")

    def as_flat_dict(self) -> dict:
        return {**asdict(self.model), **asdict(self.optim), **asdict(self.data)}


_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "base.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="Train PointNet (BC) on a PC demo dataset.")
    # --- runtime / IO (CLI-only) ---
    p.add_argument("--dataset", type=str, required=True, help="HDF5 demo file from collect_pc_demos.py.")
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help="YAML model/training config.")
    p.add_argument("--wandb_project", type=str, default="pc_bc")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="logs/pc_bc")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs (1 = single GPU).")
    # --- config overrides (default=SUPPRESS so only flags actually passed override the YAML) ---
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--encoder_dims", type=int, nargs="+")
    p.add_argument("--action_dims", type=int, nargs="+")
    p.add_argument("--predict_std", action="store_true", default=argparse.SUPPRESS,
                   help="Gaussian NLL head (mean+std) instead of MSE.")
    p.add_argument("--num_points", type=int, help="Subsample to this many points per cloud.")
    p.add_argument("--val_frac", type=float)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--no_success_filter", action="store_true", default=argparse.SUPPRESS,
                   help="Train on failed episodes too.")
    p.add_argument("--seed", type=int)
    ns = p.parse_args()

    cfg = TrainConfig.from_yaml(ns.config)
    override_keys = set(cfg.as_flat_dict())  # only these map into the dataclass
    overrides = {k: getattr(ns, k) for k in override_keys if getattr(ns, k, None) is not None}
    cfg.apply_overrides(overrides)
    runtime = SimpleNamespace(
        dataset=ns.dataset, wandb_project=ns.wandb_project, run_name=ns.run_name,
        out_dir=ns.out_dir, devices=ns.devices,
    )
    return runtime, cfg


def main():
    rt, cfg = parse_args()
    L.seed_everything(cfg.data.seed, workers=True)
    torch.set_float32_matmul_precision("high")  # H100 tensor cores

    full = PCDemoDataset(rt.dataset, num_points=cfg.data.num_points, success_only=not cfg.data.no_success_filter)
    print(
        f"[train] loaded {len(full)} timesteps | points={full.n_pts} proprio={full.proprio_dim} "
        f"action={full.action_dim} | expert_sr={full.expert_success_rate:.1f}%"
    )

    n_val = max(1, int(len(full) * cfg.data.val_frac))
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(cfg.data.seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=gen)

    # Normalize proprio/actions on the TRAIN split only.
    tr_idx = torch.as_tensor(train_set.indices)
    proprio_mean = full.proprio[tr_idx].mean(0)
    proprio_std = full.proprio[tr_idx].std(0).clamp_min(1e-6)
    action_mean = full.actions[tr_idx].mean(0)
    action_std = full.actions[tr_idx].std(0).clamp_min(1e-6)

    # Optional force-weighted up-sampling of the TRAIN split. The sampler indexes into
    # train_set (a Subset), so weights are aligned to train_set's order via tr_idx.
    train_sampler = None
    if cfg.data.force_weighting:
        if full.force is None:
            print("[train] WARNING: force_weighting=True but dataset has no 'wrist_force' term — "
                  "falling back to uniform shuffle.")
        else:
            f_tr = full.force[tr_idx].clamp_min(0.0)
            f_ref = torch.quantile(f_tr, cfg.data.force_ref_quantile).clamp_min(1e-6)
            w = 1.0 + cfg.data.force_weight_coef * (f_tr / f_ref).pow(cfg.data.force_weight_alpha)
            train_sampler = WeightedRandomSampler(w.double(), num_samples=len(f_tr), replacement=True)
            # Effective-sample-size fraction: 1.0 = uniform, lower = more concentrated.
            ess = (w.sum() ** 2 / (w * w).sum() / len(w)).item()
            print(
                f"[train] force-weighting ON | f_ref(q{cfg.data.force_ref_quantile:.2f})={f_ref:.2f}N "
                f"coef={cfg.data.force_weight_coef} alpha={cfg.data.force_weight_alpha} | "
                f"w mean={w.mean():.2f} max={w.max():.2f} | ESS={ess:.1%}"
            )

    common = dict(
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    # shuffle and sampler are mutually exclusive; shuffle only when no sampler is set.
    train_loader = DataLoader(
        train_set, sampler=train_sampler, shuffle=train_sampler is None, drop_last=True, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)

    model = PointNetBC(
        proprio_dim=full.proprio_dim,
        action_dim=full.action_dim,
        encoder_dims=cfg.model.encoder_dims,
        action_dims=cfg.model.action_dims,
        predict_std=cfg.model.predict_std,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    run_name = rt.run_name or os.path.splitext(os.path.basename(rt.dataset))[0]
    logger = WandbLogger(project=rt.wandb_project, name=run_name, save_dir=rt.out_dir)
    logger.log_hyperparams(
        {"dataset": rt.dataset, "n_timesteps": len(full), "expert_success_rate": full.expert_success_rate,
         **cfg.as_flat_dict()}
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(rt.out_dir, run_name),
        filename="pointnet-{epoch:03d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        max_epochs=cfg.optim.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=rt.devices,
        strategy="ddp" if rt.devices > 1 else "auto",
        logger=logger,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=25,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"[train] DONE. best val/loss={ckpt_cb.best_model_score:.4f} -> {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
