"""Persist and load per-iteration evaluation statistics for in-context runs.

Provides a light-weight JSON log schema used by ``run_incontext_exploration.py`` (and
its parallel sibling) to record what ``eval_distilled_policy.py`` produced for every
DAgger iteration, plus a helper used *inside* the eval subprocess to write a single
iteration's stats snapshot. ``plot_ablation_incontext.py`` consumes the resulting
log files to compare ablation runs without needing wandb.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable


@dataclass
class EvalIterationStats:
    """One row of eval metrics for a single DAgger iteration."""

    iteration: int
    episodes: int
    successful_episodes: int
    success_rate: float
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint: str | None = None
    task: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def flat(self) -> dict[str, Any]:
        """Flatten into a single dict suitable for DataFrame construction."""
        row: dict[str, Any] = {
            "iteration": self.iteration,
            "episodes": self.episodes,
            "successful_episodes": self.successful_episodes,
            "Metrics/success_rate": self.success_rate,
        }
        row.update(self.metrics)
        return row


class IncontextEvalLog:
    """Append-only JSON log of per-iteration eval stats for one DAgger run.

    The file stores run-level metadata (``exp_name``, arbitrary ``config`` dict) plus
    a sorted list of ``EvalIterationStats``. Each ``append`` replaces any prior entry
    for the same ``iteration`` so the orchestrator can safely re-run an iteration.
    """

    def __init__(
        self,
        log_path: str,
        exp_name: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.log_path = log_path
        self.exp_name = exp_name
        self.config: dict[str, Any] = dict(config) if config else {}
        self.iterations: list[EvalIterationStats] = []
        if os.path.exists(log_path):
            self._load()
        if config:
            # Merge caller-provided config over whatever was on disk.
            self.config.update(config)
        if exp_name:
            self.exp_name = exp_name

    def _load(self) -> None:
        with open(self.log_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.exp_name = payload.get("exp_name", self.exp_name)
        self.config = payload.get("config", {}) or {}
        self.iterations = [
            EvalIterationStats(**entry) for entry in payload.get("iterations", [])
        ]

    def append(self, stats: EvalIterationStats) -> None:
        """Upsert ``stats`` by iteration id, then persist to disk."""
        self.iterations = [it for it in self.iterations if it.iteration != stats.iteration]
        self.iterations.append(stats)
        self.iterations.sort(key=lambda it: it.iteration)
        self.save()

    def append_from_stats_file(self, stats_path: str, **overrides: Any) -> EvalIterationStats:
        """Load a single-iteration stats JSON (see :func:`write_eval_stats_file`) and append it."""
        stats = load_eval_stats_file(stats_path)
        for k, v in overrides.items():
            if v is not None and hasattr(stats, k):
                setattr(stats, k, v)
        self.append(stats)
        return stats

    def save(self) -> None:
        payload = {
            "exp_name": self.exp_name,
            "config": self.config,
            "iterations": [asdict(it) for it in self.iterations],
        }
        parent = os.path.dirname(os.path.abspath(self.log_path))
        os.makedirs(parent, exist_ok=True)
        tmp_path = self.log_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.log_path)

    @classmethod
    def load(cls, log_path: str) -> "IncontextEvalLog":
        """Load a log from disk; raises ``FileNotFoundError`` if missing."""
        if not os.path.exists(log_path):
            raise FileNotFoundError(log_path)
        log = cls(log_path)
        return log

    def iteration_keys(self) -> list[str]:
        """Return the union of metric keys across all stored iterations."""
        keys: set[str] = set()
        for it in self.iterations:
            keys.update(it.metrics.keys())
        return sorted(keys)


def _coerce_scalar(value: Any) -> float | None:
    """Best-effort coercion of arbitrary metric values to a Python float.

    Accepts torch tensors, numpy scalars / arrays, lists of numbers, and plain
    numbers. Returns ``None`` when the value cannot be reduced to a scalar float.
    """
    try:
        import torch  # local import to avoid a hard dependency at import-time

        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return float(value.mean())
        if isinstance(value, np.generic):
            return float(value.item())
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        try:
            floats = [float(v) for v in value]
            return float(sum(floats) / len(floats))
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_eval_stats_file(
    stats_path: str,
    episodes: int,
    successful_episodes: int,
    episode_metrics: dict[str, Iterable[Any]],
    iteration: int | None = None,
    checkpoint: str | None = None,
    task: str | None = None,
) -> None:
    """Write a single-evaluation stats snapshot as JSON.

    ``episode_metrics`` is the per-episode metric accumulator used by
    ``eval_distilled_policy.py`` (``dict[str, list[value]]``); entries are averaged
    over episodes. Non-numeric metrics are skipped silently.
    """
    averaged: dict[str, float] = {}
    for key, values in episode_metrics.items():
        floats: list[float] = []
        for v in values:
            scalar = _coerce_scalar(v)
            if scalar is not None:
                floats.append(scalar)
        if floats:
            averaged[key] = float(sum(floats) / len(floats))

    payload = {
        "iteration": int(iteration) if iteration is not None else 0,
        "episodes": int(episodes),
        "successful_episodes": int(successful_episodes),
        "success_rate": float(successful_episodes / episodes) if episodes > 0 else 0.0,
        "metrics": averaged,
        "checkpoint": checkpoint,
        "task": task,
        "timestamp": datetime.now().isoformat(),
    }
    parent = os.path.dirname(os.path.abspath(stats_path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = stats_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, stats_path)


def load_eval_stats_file(stats_path: str) -> EvalIterationStats:
    """Read a single-iteration stats JSON written by :func:`write_eval_stats_file`."""
    with open(stats_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return EvalIterationStats(
        iteration=int(payload.get("iteration", 0)),
        episodes=int(payload["episodes"]),
        successful_episodes=int(payload["successful_episodes"]),
        success_rate=float(payload.get("success_rate", 0.0)),
        metrics={k: float(v) for k, v in payload.get("metrics", {}).items()},
        checkpoint=payload.get("checkpoint"),
        task=payload.get("task"),
        timestamp=payload.get("timestamp", datetime.now().isoformat()),
    )


__all__ = [
    "EvalIterationStats",
    "IncontextEvalLog",
    "load_eval_stats_file",
    "write_eval_stats_file",
]
