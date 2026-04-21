"""State-only gravity-reduction ablations vs baseline (pulls from W&B).

Stories told:
- Uniform + mean reduction (baseline) solves peg ZeroG ~iter 2100 with ScenePC, ~iter 1660 state-only.
- Uniform + monitor_mean starves gravity_frac at 0 → rt0 stuck at ~0.01. Killed.
- Uniform + monitor_mean + floor=0.1 is the fix: floor lifts gravity immediately, rt0 learns,
  then monitor_mean gates subsequent ramp. Solves ~iter 1349 — competitive with the plain mean baseline.
- State+GPS + monitor_mean + short history windows: does GPS+short-window recover monitor_mean?
  Currently in-flight.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import wandb

api = wandb.Api()

PROJ_BASELINE = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-TerminateSuccess-HighSuccess-v0"
PROJ_STATE = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-v0"
PROJ_STATE_GPS = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-GPS-v0"

# (label, project, run_id, color, linestyle, linewidth)
RUNS = [
    (
        "Uniform + mean, ScenePC baseline (34590465)",
        PROJ_BASELINE,
        "o846wy36",
        "#000000",
        "--",
        1.8,
    ),
    (
        "State-only + uniform + mean (34707697)",
        PROJ_STATE,
        "42y90w99",
        "#009E73",
        "-",
        1.8,
    ),
    (
        "State-only + uniform + monitor_mean (34707702, killed)",
        PROJ_STATE,
        "ls3d0f42",
        "#D55E00",
        "-",
        1.8,
    ),
    (
        "State-only + uniform + monitor_mean + floor=0.1 (34722401)",
        PROJ_STATE,
        "a88ojkt0",
        "#0072B2",
        "-",
        2.0,
    ),
    (
        "State + GPS + monitor_mean + short hist (34722329)",
        PROJ_STATE_GPS,
        "ia6a9xr6",
        "#CC79A7",
        "-",
        1.8,
    ),
    (
        "State + GPS + monitor_mean + short hist + floor=0.1 (34728948)",
        PROJ_STATE_GPS,
        "a6jjl695",
        "#E69F00",
        "-",
        2.0,
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
    last_step = df["_step"].iloc[-1] if len(df) else "NA"
    last_rt0 = df["Metrics/task_0_success_rate"].iloc[-1] if len(df) else "NA"
    last_rt1 = df["Metrics/task_1_success_rate"].iloc[-1] if len(df) else "NA"
    last_g = df["Curriculum/gravity_curriculum/gravity_frac"].iloc[-1] if len(df) else "NA"
    print(f"{label}: {len(df)} rows, last iter={last_step} rt0={last_rt0} rt1={last_rt1} grav={last_g}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), sharex=True)
for label, _, _, color, ls, lw in RUNS:
    df = parsed[label]
    if df is None or df.empty:
        continue
    axes[0].plot(
        df["_step"], df["Metrics/task_0_success_rate"], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9
    )
    axes[1].plot(
        df["_step"], df["Metrics/task_1_success_rate"], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9
    )
    axes[2].plot(
        df["_step"],
        df["Curriculum/gravity_curriculum/gravity_frac"],
        label=label,
        color=color,
        linestyle=ls,
        linewidth=lw,
        alpha=0.9,
    )

axes[0].set_title("rt0 (anywhere, hard) success rate")
axes[0].set_ylabel("success rate")
axes[0].set_ylim(-0.02, 1.02)
axes[1].set_title("rt1 (partial-assembly, easier) success rate")
axes[1].set_ylim(-0.02, 1.02)
axes[2].set_title("gravity_frac")
axes[2].set_ylim(-0.02, 1.02)
for ax in axes:
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle(
    "Gravity reduction ablations (single-task peg ZeroG): floor=0.1 unblocks monitor_mean",
    fontsize=11,
)
fig.tight_layout()

out = Path("scripts/monitor_mean_vs_baseline.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
