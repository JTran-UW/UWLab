# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a HISTORY-conditioned point-cloud BC policy (``history_point_net``) on a demo dataset.

Sibling of ``train_point_net.py`` for the sequence policy: where the flat trainer maps a single
(point-cloud, proprio) to an action, this one conditions on a short trajectory *history*. Each
timestep becomes a state token (pooled PointNet PC feature + proprio) interleaved with an action
token, and a causal Transformer reads the action off the LAST state token (see
:class:`history_point_net.HistoryPointNet`).

Kept separate so the flat trainer's config/CLI stay uncluttered: this file only adds the sequence
knobs -- the window length (``data.history_len``) and the Transformer geometry (``d_model`` /
``n_heads`` / ``n_layers`` / ``transformer_dropout``). Everything else -- the dataset load, train-split
z-scoring, checkpoint ladder, W&B logging, and the Lightning fit -- is the SHARED scaffolding imported
from ``train_point_net.py`` (``add_common_args`` / ``build_config`` / ``run_training``). The dataset
(``pc_dataset.py``) and Lightning model (``pc_bc_module.py``) route the sequence path off the
``history_len>1`` / ``is_sequence`` flags; ``bc_utils.py`` is the matching deploy/eval path (a rolling
per-env buffer -- no JIT export for this stateful policy).

Usage::

    python scripts/imitation_learning/point_cloud/train_point_net_seq.py \\
        --dataset demos/clean_scenepc_100k.hdf5 --history_len 8 \\
        --wandb_project pc_bc --run_name history_v0

    python scripts/imitation_learning/point_cloud/train_point_net_seq.py \\
        --dataset demos/clean_scenepc_100k.hdf5 \\
        --config scripts/imitation_learning/point_cloud/configs/pn_xl_residual_history_clean.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

# Shared config machinery + training scaffolding live in the flat trainer next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_bc_module import _SEQUENCE_ARCHITECTURES  # noqa: E402
from train_point_net import (  # noqa: E402
    DataConfig,
    ModelConfig,
    TrainConfig,
    add_common_args,
    build_config,
    run_training,
)

_DEFAULT_SEQ_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "base_seq.yaml")


# ---------------------------------------------------------------------------
# Sequence config: the flat sections + the history-only knobs (subclassed so the flat trainer's
# config stays free of them, while from_yaml / _set / as_flat_dict are inherited unchanged).
# ---------------------------------------------------------------------------
@dataclass
class SeqModelConfig(ModelConfig):
    architecture: str = "history_point_net"
    # Causal-Transformer geometry: token width / attention heads / layers / attention+FFN dropout.
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    transformer_dropout: float = 0.1


@dataclass
class SeqDataConfig(DataConfig):
    # Number of past timesteps per training window (>1 = sequence dataset; windows are clamped to the
    # episode start and left-padded). This is the defining knob of the sequence trainer.
    history_len: int = 8


@dataclass
class SeqTrainConfig(TrainConfig):
    model: SeqModelConfig = field(default_factory=SeqModelConfig)
    data: SeqDataConfig = field(default_factory=SeqDataConfig)


def parse_args():
    p = argparse.ArgumentParser(description="Train a history-conditioned point-cloud BC policy (Transformer).")
    add_common_args(p, _DEFAULT_SEQ_CONFIG)
    # Sequence architectures only (history_point_net today); default None -> keep the YAML/config value.
    p.add_argument("--architecture", type=str, choices=tuple(_SEQUENCE_ARCHITECTURES))
    p.add_argument("--history_len", type=int, help="Past timesteps per window (the sequence length).")
    p.add_argument("--d_model", type=int, help="Transformer token width.")
    p.add_argument("--n_heads", type=int, help="Attention heads.")
    p.add_argument("--n_layers", type=int, help="Transformer layers.")
    p.add_argument("--transformer_dropout", type=float, help="Attention/FFN dropout.")
    return build_config(p.parse_args(), SeqTrainConfig)


def main():
    rt, cfg = parse_args()
    run_training(rt, cfg)


if __name__ == "__main__":
    main()
