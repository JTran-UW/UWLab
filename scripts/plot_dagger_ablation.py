"""Overlay loss + error curves for the 6 DAgger ablation runs from wandb."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import wandb

ENTITY = "patyin"

RUNS = {
    # label: (wandb project name, wandb run name / SLURM job id)
    # Scratch baselines (no pretrain, no aux)
    "Depth 1cam scratch":                 ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-ResNet18-v0",   "34715161"),
    "RGB   1cam scratch":                 ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-ResNet18-v0",     "34715162"),
    # Single-lever ablations
    "Depth 1cam + aux":                   ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Aux-v0",        "34718859"),
    "Depth 1cam + ImageNet":              ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-v0", "34718756"),
    # New (2026-04-20): combined levers
    "Depth 2cam + ImageNet":              ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-2Cam-Split-v0", "34722264"),
    "RGB   1cam + ImageNet":              ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-Split-v0",        "34722265"),
    "Depth 1cam + ImageNet + aux":        ("OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Aux-v0",        "34722266"),
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

    colors = plt.cm.tab10(range(len(data)))

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
            if show_raw:
                ax.plot(x, y, color=colors[i], alpha=0.15, linewidth=0.8)
            ax.plot(x, y_smooth, label=label, color=colors[i], linewidth=2.0)
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
