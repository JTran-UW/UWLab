"""Overlay loss + error curves for the 6 DAgger ablation runs from wandb."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import wandb

ENTITY = "patyin"

RUNS = {
    # label: (wandb project name, wandb run name / SLURM job id)
    # Live ablations as of 2026-04-20 (scratch runs killed, replaced by pretrained variants)
    "Depth 1cam + ImageNet":                 ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-v0",         "34718756"),
    "Depth 2cam + ImageNet (side+front)":    ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-2Cam-Split-v0",         "34722264"),
    "RGB   1cam + ImageNet":                 ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-Split-v0",                "34722265"),
    "Depth 1cam + ImageNet + aux (1x)":      ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Aux-v0",                "34722266"),
    # Launched 2026-04-20 mid-day
    "RGB   2cam + ImageNet (side+front)":    ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-2Cam-Pretrained-v0",      "34729126"),
    "Depth 1cam + ImageNet + aux (10x)":     ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Aux10x-v0",  "34729151"),
    # Launched 2026-04-20 evening — DEXTRAH gap ablations
    "Depth 1cam + ImageNet + recurrent":     ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Recurrent-v0", "34738654"),
    "Depth 1cam + ImageNet + weighted":      ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Weighted-v0",  "34741084"),
    # Launched 2026-04-21 AM — wrist-cam fixed WristSide pretrained pair
    "RGB   2cam + ImageNet (side+wrist*)":   ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-Pretrained-v0",   "34744482"),
    "Depth 2cam + ImageNet (side+wrist*)":   ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-v0", "34744483"),
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
    # Fallback: try exact match against run.name
    for r in api.runs(f"{ENTITY}/{project}"):
        if r.name == run_name:
            return r
    return None


def main():
    api = wandb.Api()
    data = {}
    for label, (project, run_name) in RUNS.items():
        print(f"fetching {label} (run {run_name} in {project}) ...", file=sys.stderr)
        run = fetch_run_by_name(api, project, run_name)
        if run is None:
            print(f"  no run named {run_name}; skipping", file=sys.stderr)
            continue
        try:
            # Fetch full history — the keys= filter drops rows where ANY key is NaN,
            # which excludes runs that don't log Loss/aux.
            hist = run.history(samples=5000, pandas=True)
        except Exception as e:
            print(f"  history fetch failed: {e}", file=sys.stderr)
            continue
        data[label] = hist
        print(f"  run {run.name} state={run.state} rows={len(hist)}", file=sys.stderr)

    if not data:
        print("no data; exiting", file=sys.stderr)
        return

    ncols = 2
    nrows = (len(METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=False)
    axes = axes.flatten()

    # Wong's 8-color palette (colorblind-safe, Nature Methods 2011); skip yellow (#F0E442)
    # which is too light on white backgrounds for line plots.
    CB_COLORS = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#D55E00",  # vermillion
        "#000000",  # black
    ]
    # Second channel: linestyle + marker, so curves remain distinguishable in grayscale.
    LINESTYLES = ["-", "-", "-", "--", ":", ":", "-."]
    MARKERS    = ["o", "s", "D", "^", "v", "P", "X"]

    SMOOTH = 200  # rolling-mean window in iterations (smooths noisy per-iter metrics)
    for ax_idx, (key, title, use_log, ylim, show_raw) in enumerate(METRICS):
        ax = axes[ax_idx]
        any_plotted = False
        for i, (label, hist) in enumerate(data.items()):
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
            # Sparse markers (~10 across the curve) as a second perceptual channel.
            n = max(len(x) // 10, 1)
            ax.plot(
                x, y_smooth, label=label, color=color, linewidth=2.0,
                linestyle=ls, marker=marker, markevery=n, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6,
            )
            any_plotted = True
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration")
        if use_log and any_plotted:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        if any_plotted:
            ax.legend(fontsize=7, loc="best", frameon=True)

    # Hide any unused axes
    for j in range(len(METRICS), len(axes)):
        axes[j].axis("off")

    fig.suptitle("DAgger ablation — ResNet18 variants (peg, 512 envs, Fast cadence)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = "scripts/dagger_ablation.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
