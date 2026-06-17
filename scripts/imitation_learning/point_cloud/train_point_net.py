# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a point-cloud BC policy (PointNet or FlattenMLP) on a collected demo dataset.

Loads an HDF5 dataset produced by ``scripts/tools/sim2real/collect_pc_demos.py``
(robomimic-style: ``data/demo_*/obs/<term>`` + ``data/demo_*/actions``), flattens
every timestep into a flat (point-cloud, proprio, action) BC dataset, and trains with
PyTorch Lightning. Logs to Weights & Biases.

The PC term is ``scene_pc`` (``(T, num_points, 3)``); every *other* obs term
(``joint_pos``, ``end_effector_pose``, ...) is concatenated -- in the order stored in
the dataset's ``obs_keys`` attr -- into the proprio vector. The whole dataset is loaded
into RAM (the machine has ~1 TB; a 100k-demo set is ~50 GB).

Usage::

    python scripts/imitation_learning/point_cloud/train_point_net.py \\
        --dataset demos/pc_demos.hdf5 --epochs 50 --batch_size 512 \\
        --wandb_project pc_bc --run_name pointnet_v0

    python scripts/imitation_learning/point_cloud/train_point_net.py \\
        --dataset demos/pc_demos.hdf5 --architecture flatten_mlp \\
        --wandb_project pc_bc --run_name flatten_mlp_v0
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
from torch.utils.data import DataLoader, Dataset, random_split

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Policy modules live next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flatten_mlp import FlattenMLP  # noqa: E402
from point_net import PointNet  # noqa: E402
from residual_point_net import ResidualPointNet  # noqa: E402

_ARCHITECTURES = {
    "point_net": PointNet,
    "flatten_mlp": FlattenMLP,
    "residual_point_net": ResidualPointNet,
}

# Obs terms that are point clouds (stored (T, num_points, point_dim)) -> the PointNet set input.
_PC_TERMS = {"scene_pc"}
# Proprio is an explicit ALLOWLIST (not "everything that isn't a PC"). This guarantees that
# recording extra obs terms in the collection env -- privileged poses, contact flags, etc.,
# kept "on the side" for future aux losses -- never silently changes the policy input. Any
# stored obs term not in _PC_TERMS or _PROPRIO_TERMS is aux metadata and is ignored by the
# model. {joint_pos, end_effector_pose} reproduces every existing dataset's 18d proprio.
_PROPRIO_TERMS = {"joint_pos", "end_effector_pose"}
# Privileged scene-state targets for the OPTIONAL auxiliary linear-probe loss (predicted from the
# pooled PC feature; NOT a policy input). Ordered list -> deterministic concat. Terms absent from a
# dataset are skipped; if none are present (or aux_weight=0) the aux loss is silently disabled.
_AUX_TERMS = ["insertive_asset_pose", "receptive_asset_pose", "insertive_in_receptive"]


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
            # Allowlist + preserve stored order (eval in bc_utils rebuilds the same order).
            self.proprio_keys = [k for k in obs_keys if k in _PROPRIO_TERMS]
            # Aux-probe targets present in this dataset, in the fixed _AUX_TERMS order.
            self.aux_keys = [k for k in _AUX_TERMS if k in obs_keys]

            demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
            if success_only:
                demos = [d for d in demos if bool(data[d].attrs.get("success", True))]

            total = sum(data[d]["actions"].shape[0] for d in demos)
            pc_shape = data[demos[0]]["obs"][pc_key].shape  # (T, n_pts, point_dim)
            n_pts = pc_shape[1]
            point_dim = pc_shape[2]  # 3 = xyz, 4 = xyz + segmentation label
            proprio_dim = sum(data[demos[0]]["obs"][k].shape[1] for k in self.proprio_keys)
            aux_dim = sum(data[demos[0]]["obs"][k].shape[1] for k in self.aux_keys)
            action_dim = data[demos[0]]["actions"].shape[1]

            # Preallocate once and fill (avoids the 2x peak of np.concatenate).
            self.points = np.empty((total, n_pts, point_dim), dtype=np.float32)
            self.proprio = np.empty((total, proprio_dim), dtype=np.float32)
            self.actions = np.empty((total, action_dim), dtype=np.float32)
            self.aux = np.empty((total, aux_dim), dtype=np.float32) if aux_dim else None

            i = 0
            for d in demos:
                g = data[d]
                t = g["actions"].shape[0]
                self.points[i : i + t] = g["obs"][pc_key][:]
                self.proprio[i : i + t] = np.concatenate([g["obs"][k][:] for k in self.proprio_keys], axis=1)
                self.actions[i : i + t] = g["actions"][:]
                if self.aux is not None:
                    self.aux[i : i + t] = np.concatenate([g["obs"][k][:] for k in self.aux_keys], axis=1)
                i += t

            self.expert_success_rate = float(data.attrs.get("expert_success_rate", float("nan")))

        self.points = torch.from_numpy(self.points)
        self.proprio = torch.from_numpy(self.proprio)
        self.actions = torch.from_numpy(self.actions)
        self.aux = torch.from_numpy(self.aux) if self.aux is not None else None
        self.n_pts = n_pts
        self.point_dim = point_dim
        self.proprio_dim = proprio_dim
        self.aux_dim = aux_dim
        self.action_dim = action_dim
        self.subsample = num_points if (num_points and num_points < n_pts) else None

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        pts = self.points[idx]
        if self.subsample is not None:
            sel = torch.randint(0, self.n_pts, (self.subsample,))
            pts = pts[sel]
        # aux is an empty (0,) tensor when the dataset has no aux-probe targets.
        aux = self.aux[idx] if self.aux is not None else self.points.new_zeros(0)
        return pts, self.proprio[idx], self.actions[idx], aux


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
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
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["proprio_mean", "proprio_std", "action_mean", "action_std", "aux_mean", "aux_std"]
        )
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

        total = action_loss
        bs = points.shape[0]
        if self.use_aux:
            aux_t = (aux - self.aux_mean) / self.aux_std
            aux_loss = torch.nn.functional.mse_loss(self.aux_probe(pooled), aux_t)
            total = action_loss + self.hparams.aux_weight * aux_loss
            self.log(f"{stage}/aux_loss", aux_loss, batch_size=bs, sync_dist=True)

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
    # point_net: per-point set encoder + max pool. flatten_mlp: single MLP on flattened PC.
    architecture: str = "point_net"
    # Encoder MLP hidden dims. PointNet: per-point in=point_dim. FlattenMLP: in=num_points*point_dim.
    encoder_dims: list[int] = field(default_factory=lambda: [64, 128, 256])
    # Action-head MLP hidden dims.
    action_dims: list[int] = field(default_factory=lambda: [256, 128])
    # False: MSE head. True: Gaussian-NLL head that also predicts a per-dim std.
    predict_std: bool = False


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 512
    # Weight on the auxiliary linear-probe loss (predict privileged object poses from the pooled
    # PC feature). 0 = off. The probe is linear, so gradients shape the ENCODER's representation.
    # val/loss (checkpoint-selection metric) stays the pure action loss regardless.
    aux_weight: float = 0.0


@dataclass
class DataConfig:
    num_points: int | None = None  # None -> use the full stored cloud
    val_frac: float = 0.05
    num_workers: int = 8
    no_success_filter: bool = False  # False -> successful demos only
    seed: int = 0


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
    p = argparse.ArgumentParser(description="Train a point-cloud BC policy on a PC demo dataset.")
    # --- runtime / IO (CLI-only) ---
    p.add_argument("--dataset", type=str, required=True, help="HDF5 demo file from collect_pc_demos.py.")
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help="YAML model/training config.")
    p.add_argument("--wandb_project", type=str, default="pc_bc")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="logs/pc_bc")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs (1 = single GPU).")
    p.add_argument("--ckpt_every_n_epochs", type=int, default=0,
                   help="Also save a checkpoint every N epochs (kept regardless of val/loss, in an "
                        "'epochs/' subdir) -- a ladder for analyzing how sim success tracks training "
                        "progress. 0 = off (only the val/loss top-k + last).")
    # --- config overrides (default=SUPPRESS so only flags actually passed override the YAML) ---
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--aux_weight", type=float, help="Weight on the aux object-pose probe loss (0=off).")
    p.add_argument("--architecture", type=str, choices=tuple(_ARCHITECTURES))
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
        out_dir=ns.out_dir, devices=ns.devices, ckpt_every_n_epochs=ns.ckpt_every_n_epochs,
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
    # Aux-probe targets: z-scored on the train split too (so the probe MSE is well-scaled).
    use_aux = full.aux_dim > 0 and cfg.optim.aux_weight > 0.0
    aux_mean = full.aux[tr_idx].mean(0) if use_aux else None
    aux_std = full.aux[tr_idx].std(0).clamp_min(1e-6) if use_aux else None

    common = dict(
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)

    if cfg.model.architecture not in _ARCHITECTURES:
        raise ValueError(f"unknown architecture '{cfg.model.architecture}', choose from {tuple(_ARCHITECTURES)}")
    num_points = cfg.data.num_points or full.n_pts

    model = PointNetBC(
        architecture=cfg.model.architecture,
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
        point_dim=full.point_dim,
        num_points=num_points,
        aux_dim=full.aux_dim,
        aux_weight=cfg.optim.aux_weight,
        aux_mean=aux_mean,
        aux_std=aux_std,
    )
    print(
        f"[train] architecture={cfg.model.architecture} num_points={num_points} "
        f"point_dim={full.point_dim} ({'xyz+seg' if full.point_dim == 4 else 'xyz'})"
    )
    if use_aux:
        print(f"[train] AUX probe ON: weight={cfg.optim.aux_weight} targets={full.aux_keys} (dim={full.aux_dim})")
    elif cfg.optim.aux_weight > 0:
        print(f"[train] WARNING: aux_weight={cfg.optim.aux_weight} but dataset has no aux targets {_AUX_TERMS} -- aux disabled")

    run_name = rt.run_name or os.path.splitext(os.path.basename(rt.dataset))[0]
    logger = WandbLogger(project=rt.wandb_project, name=run_name, save_dir=rt.out_dir)
    logger.log_hyperparams(
        {"dataset": rt.dataset, "n_timesteps": len(full), "expert_success_rate": full.expert_success_rate,
         **cfg.as_flat_dict()}
    )
    ckpt_prefix = cfg.model.architecture.replace("_", "-")
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(rt.out_dir, run_name),
        filename=f"{ckpt_prefix}-{{epoch:03d}}-{{val/loss:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="epoch")]
    # Optional epoch-ladder: keep every Nth-epoch checkpoint (independent of val/loss) so we
    # can measure how sim success evolves over training (e.g. does it peak before val/loss does?).
    if rt.ckpt_every_n_epochs > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(rt.out_dir, run_name, "epochs"),
            filename=f"{ckpt_prefix}-{{epoch:03d}}-{{val/loss:.4f}}",
            every_n_epochs=rt.ckpt_every_n_epochs,
            save_top_k=-1,  # keep them all
            auto_insert_metric_name=False,
        ))

    trainer = L.Trainer(
        max_epochs=cfg.optim.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=rt.devices,
        strategy="ddp" if rt.devices > 1 else "auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=25,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"[train] DONE. best val/loss={ckpt_cb.best_model_score:.4f} -> {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
