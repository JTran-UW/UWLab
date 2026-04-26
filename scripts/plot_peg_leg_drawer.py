"""Winning config (uniform + monitor_mean + floor=0.1) across peg / leg / drawer.

Solid: scale=2 + ori=0.1 + multi-canonical PA dataset (commit 285d668, 2026-04-25
relaunch with HF cache fix). Dashed: yaw-fix runs (commit 86c1ee2) on the older
scale=8 dataset, kept for context.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import wandb

api = wandb.Api()

PROJ = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-v0"

# (label, run_id, color, linestyle)
RUNS = [
    # scale=2 + ori=0.1 + multi-canonical (commit 285d668)
    ("Peg scale=2 (34840064)",    "qxfktiu1", "#0072B2", "-"),
    ("Leg scale=2 (34840065)",    "c8lwfz9b", "#D55E00", "-"),
    ("Drawer scale=2 (34840066)", "52qcajzn", "#009E73", "-"),
    # Yaw-fix on older scale=8 dataset (commit 86c1ee2)
    ("Peg yaw-fix (34753110)",    "4icjogy3", "#0072B2", "--"),
    ("Leg yaw-fix (34754745)",    "rmp8ftoo", "#D55E00", "--"),
    ("Drawer yaw-fix (34752658)", "lxh3nxve", "#009E73", "--"),
]

KEYS = [
    "Metrics/task_0_success_rate",
    "Metrics/task_1_success_rate",
    "Curriculum/gravity_curriculum/gravity_frac",
    "Curriculum/gravity_curriculum/monitor_mean_rate",
]


def fetch(run_id: str):
    run = api.run(f"patyin/{PROJ}/{run_id}")
    available = [k for k in KEYS if k in run.summary]
    df = run.history(keys=available, samples=5000, pandas=True)
    for k in KEYS:
        if k not in df.columns:
            df[k] = float("nan")
    df = df.dropna(subset=["Metrics/task_0_success_rate"]).sort_values("_step")
    return df, run.state


parsed = {}
for label, rid, _c, _ls in RUNS:
    df, state = fetch(rid)
    parsed[label] = df
    last = df.iloc[-1]
    print(
        f"{label} [{state}]: iter={int(last['_step'])} "
        f"rt0={last['Metrics/task_0_success_rate']:.3f} "
        f"rt1={last['Metrics/task_1_success_rate']:.3f} "
        f"grav={last['Curriculum/gravity_curriculum/gravity_frac']:.3f} "
        f"mon={last['Curriculum/gravity_curriculum/monitor_mean_rate']:.3f}"
    )

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharex=True)
for label, rid, color, ls in RUNS:
    df = parsed[label]
    lw = 1.8 if ls == "-" else 1.2
    alpha = 1.0 if ls == "-" else 0.55
    axes[0].plot(df["_step"], df["Metrics/task_0_success_rate"], label=label, color=color, lw=lw, ls=ls, alpha=alpha)
    axes[1].plot(df["_step"], df["Metrics/task_1_success_rate"], label=label, color=color, lw=lw, ls=ls, alpha=alpha)
    axes[2].plot(df["_step"], df["Curriculum/gravity_curriculum/gravity_frac"], label=label, color=color, lw=lw, ls=ls, alpha=alpha)
    axes[3].plot(df["_step"], df["Curriculum/gravity_curriculum/monitor_mean_rate"], label=label, color=color, lw=lw, ls=ls, alpha=alpha)

axes[0].axhline(0.8, color="gray", ls=":", lw=0.8)
axes[3].axhline(0.8, color="gray", ls=":", lw=0.8, label="monitor gate=0.8")

axes[0].set_title("rt0 (anywhere, hard) success")
axes[1].set_title("rt1 (partial-assembly, easier) success")
axes[2].set_title("gravity_frac")
axes[3].set_title("monitor_mean_rate")
for ax in axes[:3]:
    ax.set_ylim(-0.02, 1.02)
axes[3].set_ylim(-0.02, 1.02)
for ax in axes:
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("success rate")
axes[0].legend(fontsize=8, loc="lower right")
axes[3].legend(fontsize=7, loc="lower right")

fig.suptitle(
    "Uniform + monitor_mean + floor=0.1 across OmniReset insertions "
    "— solid: scale=2 + multi-canonical (commit 285d668), dashed: yaw-fix scale=8",
    fontsize=12,
)
fig.tight_layout()

out = Path("scripts/peg_leg_drawer.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
