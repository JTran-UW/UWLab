"""Monitor_mean gravity reduction: 3 GPS ablations vs uniform baseline.

Pulls curves directly from W&B (no slurm logs needed).
Baseline (34590465 / o846wy36) is uniform sampling w/ mean-reduction gravity — solved ~iter 2100.
All 3 new runs share reduction=monitor_mean (commit 8c3e08b) + curriculum_target=0.5 + single-task peg.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import wandb

api = wandb.Api()

# (label, project, run_id, color, linestyle, linewidth)
PROJ = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-v0"
RUNS = [
    # current: GPS + monitor_mean (solid)
    ("Empirical GPS + monitor_mean (34692434)", PROJ, "zpfkjdkr", "#CC79A7", "-", 1.8),
    ("Classifier GPS + monitor_mean (34692435)", PROJ, "4kputdiu", "#0072B2", "-", 1.8),
    ("V_success GPS + monitor_mean (34692438)", PROJ, "pvjnai3t", "#D55E00", "-", 1.8),
    # prior: GPS + mean gravity (dashdot, same hue family)
    ("Empirical GPS + mean (34674405)", PROJ, "gsr6b2x6", "#CC79A7", "-.", 1.3),
    ("Classifier GPS + mean (34675161)", PROJ, "c8pkhvv5", "#0072B2", "-.", 1.3),
    # baseline: uniform routing + mean (dashed black)
    (
        "Uniform baseline (34590465)",
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-TerminateSuccess-HighSuccess-v0",
        "o846wy36",
        "#000000",
        "--",
        1.8,
    ),
]

KEYS = [
    "Metrics/task_0_success_rate",
    "Metrics/task_1_success_rate",
    "Curriculum/gravity_curriculum/gravity_frac",
]


def fetch(project: str, run_id: str):
    run = api.run(f"patyin/{project}/{run_id}")
    df = run.history(keys=KEYS, samples=5000, pandas=True)
    df = df.sort_values("_step")
    return df


parsed = {}
for label, proj, rid, *_ in RUNS:
    df = fetch(proj, rid)
    parsed[label] = df
    print(f"{label}: {len(df)} rows, last step={df['_step'].iloc[-1] if len(df) else 'NA'}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
for label, _, _, color, ls, lw in RUNS:
    df = parsed[label]
    if df is None or df.empty:
        continue
    axes[0].plot(df["_step"], df["Metrics/task_0_success_rate"], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
    axes[1].plot(df["_step"], df["Metrics/task_1_success_rate"], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
    axes[2].plot(df["_step"], df["Curriculum/gravity_curriculum/gravity_frac"], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

axes[0].set_title("rt0 (anywhere, hard) success rate")
axes[0].set_ylabel("success rate")
axes[0].set_ylim(-0.02, 1.0)
axes[1].set_title("rt1 (partial-assembly, easier) success rate")
axes[1].set_ylim(-0.02, 1.0)
axes[2].set_title("gravity_frac")
axes[2].set_ylim(-0.02, 1.02)
for ax in axes:
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("GPS + monitor_mean (solid) vs GPS + mean (dashdot) vs uniform+mean baseline (dashed) — peg ZeroG", fontsize=11)
fig.tight_layout()

out = Path("scripts/monitor_mean_vs_baseline.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
