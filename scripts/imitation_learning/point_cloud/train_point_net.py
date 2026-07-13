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

This is the entry point for the FEED-FORWARD architectures (point_net / flatten_mlp /
residual_point_net / diffusion_point_net). The history-conditioned sequence policy has its own
entry point, ``train_point_net_seq.py``, which reuses the scaffolding here (``add_common_args`` /
``build_config`` / ``run_training``). The dataset lives in ``pc_dataset.py``
(:class:`PCDemoDataset`) and the Lightning model in ``pc_bc_module.py``
(:class:`PointNetBC`); ``bc_utils.py`` is the matching deploy/eval path.

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

import torch
import yaml
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Dataset + Lightning model live next to this script (importable regardless of caller cwd).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_bc_module import _ARCHITECTURES, _SEQUENCE_ARCHITECTURES, PointNetBC  # noqa: E402
from pc_dataset import _AUX_TERMS, PCDemoDataset  # noqa: E402


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
    # diffusion_point_net only: DDPM training steps + DDIM sampling steps (a few). Ignored otherwise.
    num_train_timesteps: int = 100
    num_sample_steps: int = 10


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 512
    # Aux object-pose probe loss weight (linear probe off the pooled feature; shapes the encoder).
    # 0 = off; val/loss stays the pure action loss regardless.
    aux_weight: float = 0.0
    # BC action loss weight. 1 = normal BC; 0 (with aux_weight>0) = pure state-extraction probe.
    action_weight: float = 1.0
    # DEXTRAH inverse-variance loss weighting: weight each action dim's supervision by
    # (1/expert_sigma)^2. Needs a dataset collected with per-step expert std (collect_pc_demos.py with
    # an expert that reports std). Weights are z-space-converted + normalized to mean 1 so the loss
    # scale matches plain MSE. False = plain MSE / Gaussian-NLL.
    action_var_weighting: bool = False
    # Floor on the (z-scored) expert std before inverting -> caps the max per-dim weight at (1/floor)^2.
    var_weight_sigma_floor: float = 0.05


@dataclass
class DataConfig:
    num_points: int | None = None  # None -> use the full stored cloud
    val_frac: float = 0.05
    num_workers: int = 8
    no_success_filter: bool = False  # False -> successful demos only
    seed: int = 0
    # First N of the 12 joint_pos dims to feed the policy. joint_pos = 6 arm + 6 gripper-mimic; the
    # gripper joints don't exist on the real robot, so use 6 for sim2real. None = all 12 (sim-only).
    joint_pos_dims: int | None = None
    # PER-PRIM datasets: prim names to keep (drops individual links / all robot links / an object).
    # None = every prim. Ignored for legacy flat datasets.
    pc_parts: list[str] | None = None
    # PER-PRIM datasets: append a per-point seg label derived from the prim (robot=0/insertive=-1/
    # receptive=+1) as a 4th channel -> model trains on [xyz, seg]. Deploy re-derives it. Flat: ignored.
    append_prim_semantic: bool = False


@dataclass
class AugConfig:
    # Train-only PC dropout (zeroes points; val stays clean). Per-category dicts {robot,insertive,
    # receptive}; needs a per-prim dataset or seg labels (point_dim==4). Order: prim -> point.
    # e.g. aug: {prim_dropout: {robot: 0.3}, point_dropout: {robot: 0.05}}
    #
    # prim_dropout: per-prim -> drop each prim of the category INDEPENDENTLY (robot:0.3 = each link
    # on its own); seg-label datasets -> drop the WHOLE category at once.
    prim_dropout: dict | None = None
    # point_dropout: prob each point of the category is zeroed independently (sparse/missing returns).
    point_dropout: dict | None = None


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)

    @classmethod
    def from_yaml(cls, path: str | None) -> "TrainConfig":
        cfg = cls()
        if path and os.path.exists(path):
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            for section in ("model", "optim", "data", "aug"):
                for k, v in (raw.get(section) or {}).items():
                    cfg._set(k, v, where=f"{path}:{section}")
            print(f"[train] loaded config: {path}")
        return cfg

    def _set(self, key: str, value, where: str = "override"):
        """Set ``key`` on whichever section declares it (keys are unique across sections)."""
        for section in (self.model, self.optim, self.data, self.aug):
            if hasattr(section, key):
                setattr(section, key, value)
                return
        raise KeyError(f"unknown config key '{key}' (from {where})")

    def apply_overrides(self, overrides: dict):
        for k, v in overrides.items():
            self._set(k, v, where="CLI")

    def as_flat_dict(self) -> dict:
        return {**asdict(self.model), **asdict(self.optim), **asdict(self.data), **asdict(self.aug)}


_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "base.yaml")
# Feed-forward architectures this entry point handles. The history-conditioned sequence policy has its
# own trainer (``train_point_net_seq.py``) since its config/CLI (window length, Transformer geometry)
# and its sequence dataset differ; both share the scaffolding below (add_common_args/run_training).
_FLAT_ARCHITECTURES = tuple(a for a in _ARCHITECTURES if a not in _SEQUENCE_ARCHITECTURES)


def add_common_args(p: argparse.ArgumentParser, default_config: str):
    """Register the runtime/IO args and the config-override flags shared by every PC-BC trainer.

    ``--architecture`` is intentionally NOT added here -- each entry point adds it with its own
    architecture set (feed-forward vs sequence). Override flags use ``default=None`` so only flags
    actually passed override the YAML (see :func:`build_config`)."""
    # --- runtime / IO (CLI-only) ---
    p.add_argument("--dataset", type=str, required=True, help="HDF5 demo file from collect_pc_demos.py.")
    p.add_argument("--config", type=str, default=default_config, help="YAML model/training config.")
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
    p.add_argument("--action_weight", type=float,
                   help="Weight on the BC action loss (1=normal). 0 + aux_weight>0 = pure pose probe.")
    p.add_argument("--action_var_weighting", action="store_true", default=argparse.SUPPRESS,
                   help="DEXTRAH inverse-variance loss weighting (needs a dataset with per-step expert std).")
    p.add_argument("--var_weight_sigma_floor", type=float,
                   help="Floor on the z-scored expert std before inverting (caps the max weight).")
    p.add_argument("--encoder_dims", type=int, nargs="+")
    p.add_argument("--action_dims", type=int, nargs="+")
    p.add_argument("--predict_std", action="store_true", default=argparse.SUPPRESS,
                   help="Gaussian NLL head (mean+std) instead of MSE.")
    p.add_argument("--num_points", type=int, help="Subsample to this many points per cloud.")
    p.add_argument("--joint_pos_dims", type=int,
                   help="Use only the first N of 12 joint_pos dims (6=arm only; last 6 gripper joints "
                        "do not exist on the real robot). Default: all 12.")
    p.add_argument("--pc_parts", type=str, nargs="+",
                   help="PER-PRIM datasets: prim names to use (e.g. wrist_3_link insertive receptive). "
                        "Default: all prims in the dataset.")
    p.add_argument("--append_prim_semantic", action="store_true", default=argparse.SUPPRESS,
                   help="PER-PRIM datasets: append a derived seg label (robot/insertive/receptive) as a "
                        "4th cloud channel (point_dim 3->4).")
    p.add_argument("--val_frac", type=float)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--no_success_filter", action="store_true", default=argparse.SUPPRESS,
                   help="Train on failed episodes too.")
    p.add_argument("--seed", type=int)
    return p


def build_config(ns, cfg_cls):
    """Build a (runtime, cfg) pair from parsed args: load the YAML, apply only the CLI flags that were
    actually passed (present as dataclass keys), and split off the runtime/IO namespace. ``cfg_cls`` is
    :class:`TrainConfig` for the flat trainer or its sequence subclass for ``train_point_net_seq.py``."""
    cfg = cfg_cls.from_yaml(ns.config)
    override_keys = set(cfg.as_flat_dict())  # only these map into the dataclass
    overrides = {k: getattr(ns, k) for k in override_keys if getattr(ns, k, None) is not None}
    cfg.apply_overrides(overrides)
    runtime = SimpleNamespace(
        dataset=ns.dataset, wandb_project=ns.wandb_project, run_name=ns.run_name,
        out_dir=ns.out_dir, devices=ns.devices, ckpt_every_n_epochs=ns.ckpt_every_n_epochs,
    )
    return runtime, cfg


def parse_args():
    p = argparse.ArgumentParser(description="Train a feed-forward point-cloud BC policy on a PC demo dataset.")
    add_common_args(p, _DEFAULT_CONFIG)
    p.add_argument("--architecture", type=str, choices=_FLAT_ARCHITECTURES)
    return build_config(p.parse_args(), TrainConfig)


def run_training(rt, cfg):
    """Shared training scaffolding for every PC-BC architecture: load the dataset, z-score on the train
    split, build the Lightning model, and fit. Sequence-only config fields (``history_len``, Transformer
    geometry) are read defensively with ``getattr`` so this runs unchanged for the flat trainer and for
    ``train_point_net_seq.py`` (whose cfg subclass carries those fields)."""
    L.seed_everything(cfg.data.seed, workers=True)
    torch.set_float32_matmul_precision("high")  # H100 tensor cores

    # The history-conditioned Transformer wants sequence windows; every other architecture wants the
    # flat single-timestep dataset. Keep arch and window length in lockstep so a mismatch (e.g. the
    # wrong trainer / a stray YAML key) fails loudly, not silently. history_len defaults to 1 (flat).
    history_len = getattr(cfg.data, "history_len", 1)
    is_sequence_arch = cfg.model.architecture in _SEQUENCE_ARCHITECTURES
    if is_sequence_arch and history_len <= 1:
        raise ValueError(f"architecture={cfg.model.architecture} needs data.history_len>1 (the window "
                         f"length); use train_point_net_seq.py and set --history_len (e.g. 8).")
    if not is_sequence_arch and history_len > 1:
        raise ValueError(f"history_len={history_len}>1 only applies to a sequence architecture "
                         f"({tuple(_SEQUENCE_ARCHITECTURES)}), not '{cfg.model.architecture}'.")

    full = PCDemoDataset(rt.dataset, num_points=cfg.data.num_points,
                         success_only=not cfg.data.no_success_filter, joint_pos_dims=cfg.data.joint_pos_dims,
                         pc_parts=cfg.data.pc_parts, append_prim_semantic=cfg.data.append_prim_semantic,
                         history_len=history_len)
    print(
        f"[train] loaded {len(full)} timesteps | points={full.n_pts} proprio={full.proprio_dim} "
        f"action={full.action_dim} | expert_sr={full.expert_success_rate:.1f}%"
        + (f" | history_len={history_len}" if history_len > 1 else "")
        + (f" | joint_pos_dims={cfg.data.joint_pos_dims} (arm-only, real-robot)" if cfg.data.joint_pos_dims else "")
    )
    if full.pc_parts is not None:
        print(f"[train] PER-PRIM cloud: parts={full.pc_parts} -> {full.n_pts} pts "
              f"(selected points zero-padded to pc_pad_target={full.pc_pad_target})"
              + (" | +derived seg channel ([xyz,seg], point_dim=4)" if full._append_seg else ""))

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

    # Point/prim dropout needs per-point category info: either a per-prim dataset (prim_id, preferred)
    # or the legacy segmentation-label channel (point_dim==4).
    if (cfg.aug.point_dropout or cfg.aug.prim_dropout) and not (full.per_prim or full.point_dim == 4):
        raise ValueError(
            f"point/prim dropout needs a per-prim dataset (scene_pc_prim_id) or segmentation labels "
            f"(point_dim==4); this dataset is flat with point_dim={full.point_dim}. Drop the aug config."
        )

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
        action_weight=cfg.optim.action_weight,
        action_var_weighting=cfg.optim.action_var_weighting,
        var_weight_sigma_floor=cfg.optim.var_weight_sigma_floor,
        has_expert_std=full.has_expert_std,
        aux_keys=full.aux_keys,
        pc_parts=full.pc_parts,
        pc_all_prim_names=full.pc_all_prim_names,
        pc_pad_target=full.pc_pad_target,
        point_dropout=cfg.aug.point_dropout,
        prim_dropout=cfg.aug.prim_dropout,
        append_prim_semantic=full._append_seg,
        num_train_timesteps=cfg.model.num_train_timesteps,
        num_sample_steps=cfg.model.num_sample_steps,
        # Sequence-only (history_point_net) knobs; PointNetBC ignores them for feed-forward archs.
        history_len=history_len,
        d_model=getattr(cfg.model, "d_model", 256),
        n_heads=getattr(cfg.model, "n_heads", 4),
        n_layers=getattr(cfg.model, "n_layers", 4),
        transformer_dropout=getattr(cfg.model, "transformer_dropout", 0.1),
    )
    print(
        f"[train] architecture={cfg.model.architecture} num_points={num_points} "
        f"point_dim={full.point_dim} ({'xyz+seg' if full.point_dim == 4 else 'xyz'})"
    )
    if cfg.optim.action_var_weighting:
        print(f"[train] DEXTRAH inverse-variance loss weighting ON "
              f"(expert std from dataset; sigma_floor={cfg.optim.var_weight_sigma_floor})")
    elif full.has_expert_std:
        print("[train] dataset has per-step expert std, but action_var_weighting is OFF (plain MSE). "
              "Pass --action_var_weighting to use it.")
    if cfg.aug.point_dropout or cfg.aug.prim_dropout:
        print(f"[train] PC DROPOUT (train-only): point={cfg.aug.point_dropout} prim={cfg.aug.prim_dropout}")
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


def main():
    rt, cfg = parse_args()
    run_training(rt, cfg)


if __name__ == "__main__":
    main()
