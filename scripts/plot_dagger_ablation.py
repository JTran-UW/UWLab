"""Overlay loss + error curves for DAgger ablation runs from wandb.

Emits two plots:
  - scripts/dagger_ablation_nonweighted.png — all non-weighted ImageNet-pretrained variants
  - scripts/dagger_ablation_weighted.png    — weighted-L2 variants + non-weighted 1cam baseline
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import wandb

ENTITY = "patyin"

# All runs keyed by label. Each entry: (wandb project, wandb run name / SLURM job id).
ALL_RUNS = {
    "Depth 1cam + ImageNet":                 ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-v0",         "34718756"),
    "Depth 2cam + ImageNet (side+front)":    ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-2Cam-Split-v0",         "34722264"),
    "RGB   1cam + ImageNet":                 ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-Split-v0",                "34722265"),
    "Depth 1cam + ImageNet + aux (1x)":      ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Aux-v0",                "34722266"),
    "RGB   2cam + ImageNet (side+front)":    ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-2Cam-Pretrained-v0",      "34729126"),
    "Depth 1cam + ImageNet + aux (10x)":     ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Aux10x-v0",  "34729151"),
    "Depth 1cam + ImageNet + recurrent":     ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Recurrent-v0", "34738654"),
    "Depth 1cam + ImageNet + weighted":      ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Weighted-v0",  "34741084"),
    "RGB   2cam + ImageNet (side+wrist*)":   ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-Pretrained-v0",   "34744482"),
    "Depth 2cam + ImageNet (side+wrist*)":   ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-v0", "34744483"),
    "Depth 2cam + ImageNet + weighted (side+wrist*)": ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-v0", "34745402"),
    # Launched 2026-04-21 — weighted + one-extra-lever ablations
    "RGB   2cam + ImageNet + weighted (side+wrist*)":            ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-Pretrained-Weighted-v0",              "34753744"),
    "Depth 2cam + ImageNet + weighted + recurrent (side+wrist*)":("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-Recurrent-v0",  "34753761"),
    "Depth 2cam + ImageNet + weighted + split50 (side+wrist*)":  ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-Split50-v0",    "34753762"),
    "Depth 2cam + ImageNet + weighted + freeze5k (side+wrist*)": ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-Freeze5k-v0",   "34754067"),
}

NON_WEIGHTED_LABELS = [
    "Depth 1cam + ImageNet",
    "Depth 2cam + ImageNet (side+front)",
    "RGB   1cam + ImageNet",
    "Depth 1cam + ImageNet + aux (1x)",
    "RGB   2cam + ImageNet (side+front)",
    "Depth 1cam + ImageNet + aux (10x)",
    "Depth 1cam + ImageNet + recurrent",
    "RGB   2cam + ImageNet (side+wrist*)",
    "Depth 2cam + ImageNet (side+wrist*)",
]

WEIGHTED_LABELS = [
    "Depth 1cam + ImageNet (baseline)",  # aliased from "Depth 1cam + ImageNet"
    "Depth 1cam + ImageNet + weighted",
    "Depth 2cam + ImageNet + weighted (side+wrist*)",
    "RGB   2cam + ImageNet + weighted (side+wrist*)",
    "Depth 2cam + ImageNet + weighted + recurrent (side+wrist*)",
    "Depth 2cam + ImageNet + weighted + split50 (side+wrist*)",
    "Depth 2cam + ImageNet + weighted + freeze5k (side+wrist*)",
]
# Aliases for the weighted plot: remap labels so the baseline is marked as such.
WEIGHTED_ALIASES = {
    "Depth 1cam + ImageNet (baseline)": "Depth 1cam + ImageNet",
}

METRICS = [
    # (key, title, use_log, ylim, show_raw)
    ("Loss/behavior",                                       "Behavior loss (MSE on teacher mean)",    True,  (10.0, 100.0), False),
    ("Loss/aux",                                            "Aux pose loss (MSE on GT poses)",        True,  None,          True),
    ("Metrics/task_command/end_of_episode_pos_align_error", "EOE position align error (m)",           False, (0.0, 0.40),   False),
    ("Metrics/task_command/end_of_episode_rot_align_error", "EOE rotation align error (rad)",         False, (0.0, 3.20),   False),
    ("Metrics/success_student_eval",                        "Student success (eval pool)",            False, None,          True),
    ("Metrics/success_student_train",                       "Student success (train pool)",           False, None,          True),
]


def fetch_run_by_name(api, project: str, run_name: str):
    """Find a run in the project whose display name matches ``run_name``."""
    runs = list(api.runs(f"{ENTITY}/{project}", filters={"display_name": run_name}))
    if runs:
        return runs[0]
    for r in api.runs(f"{ENTITY}/{project}"):
        if r.name == run_name:
            return r
    return None


def plot_group(data: dict, group_labels: list[str], title: str, out_path: str):
    """Render `group_labels` as overlays into one multi-panel figure."""
    group_data = [(lbl, data[lbl]) for lbl in group_labels if lbl in data]
    if not group_data:
        print(f"no data for group {title}; skipping", file=sys.stderr)
        return

    ncols = 2
    nrows = (len(METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=False)
    axes = axes.flatten()

    # Wong palette (colorblind-safe, Nature Methods 2011); skip yellow #F0E442.
    CB_COLORS = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#000000",
    ]
    LINESTYLES = ["-", "-", "-", "--", ":", ":", "-."]
    MARKERS    = ["o", "s", "D", "^", "v", "P", "X"]

    SMOOTH = 200
    for ax_idx, (key, panel_title, use_log, ylim, show_raw) in enumerate(METRICS):
        ax = axes[ax_idx]
        any_plotted = False
        for i, (label, hist) in enumerate(group_data):
            if key not in hist.columns or hist[key].dropna().empty:
                continue
            y = hist[key].dropna()
            x = hist["_step"].loc[y.index]
            if len(y) >= SMOOTH:
                y_smooth = y.rolling(SMOOTH, min_periods=1).mean()
            else:
                y_smooth = y
            color = CB_COLORS[i % len(CB_COLORS)]
            ls = LINESTYLES[i % len(LINESTYLES)]
            marker = MARKERS[i % len(MARKERS)]
            if show_raw:
                ax.plot(x, y, color=color, alpha=0.12, linewidth=0.8, linestyle=ls)
            n = max(len(x) // 10, 1)
            ax.plot(
                x, y_smooth, label=label, color=color, linewidth=2.0,
                linestyle=ls, marker=marker, markevery=n, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6,
            )
            any_plotted = True
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("iteration")
        if use_log and any_plotted:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        if any_plotted:
            ax.legend(fontsize=7, loc="best", frameon=True)

    for j in range(len(METRICS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    api = wandb.Api()
    data = {}
    # Fetch each unique run once. For aliases, we duplicate the pandas frame under
    # the aliased label so downstream plotting treats them as separate entries.
    fetched = {}
    for label, (project, run_name) in ALL_RUNS.items():
        print(f"fetching {label} (run {run_name} in {project}) ...", file=sys.stderr)
        run = fetch_run_by_name(api, project, run_name)
        if run is None:
            print(f"  no run named {run_name}; skipping", file=sys.stderr)
            continue
        try:
            hist = run.history(samples=5000, pandas=True)
        except Exception as e:
            print(f"  history fetch failed: {e}", file=sys.stderr)
            continue
        data[label] = hist
        fetched[run_name] = hist
        print(f"  run {run.name} state={run.state} rows={len(hist)}", file=sys.stderr)

    # Populate aliases (e.g. baseline re-used across plots).
    for alias_label, canonical_label in WEIGHTED_ALIASES.items():
        if canonical_label in data:
            data[alias_label] = data[canonical_label]

    if not data:
        print("no data; exiting", file=sys.stderr)
        return

    plot_group(
        data,
        NON_WEIGHTED_LABELS,
        title="DAgger ablation — non-weighted variants (peg, Fast cadence, ResNet18 + ImageNet)",
        out_path="scripts/dagger_ablation_nonweighted.png",
    )
    plot_group(
        data,
        WEIGHTED_LABELS,
        title="DAgger ablation — weighted L2 variants vs 1cam baseline (peg, Fast cadence)",
        out_path="scripts/dagger_ablation_weighted.png",
    )


if __name__ == "__main__":
    main()
