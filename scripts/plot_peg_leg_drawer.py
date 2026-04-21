"""Winning config (uniform + monitor_mean + floor=0.1) across peg / leg / drawer.

Peg solved by iter ~1349 (a88ojkt0). Drawer solved by iter ~700 (ottu8izh). Leg is stuck:
rt1 plateaus below 0.85 and rt0 never cracks, so monitor_mean never crosses 0.8 and
gravity stays pinned at the 0.1 floor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import wandb

api = wandb.Api()

PROJ = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-v0"

# (label, run_id, color)
RUNS = [
    ("Peg (34722401)",    "a88ojkt0", "#0072B2"),
    ("Leg (34741091)",    "dtud7prh", "#D55E00"),
    ("Drawer (34745030)", "ottu8izh", "#009E73"),
]

KEYS = [
    "Metrics/task_0_success_rate",
    "Metrics/task_1_success_rate",
    "Curriculum/gravity_curriculum/gravity_frac",
    "Curriculum/gravity_curriculum/monitor_mean_rate",
]


def fetch(run_id: str):
    run = api.run(f"patyin/{PROJ}/{run_id}")
    df = run.history(keys=KEYS, samples=5000, pandas=True)
    df = df.dropna(subset=["Metrics/task_0_success_rate"]).sort_values("_step")
    return df, run.state


parsed = {}
for label, rid, _ in RUNS:
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
for label, rid, color in RUNS:
    df = parsed[label]
    axes[0].plot(df["_step"], df["Metrics/task_0_success_rate"], label=label, color=color, lw=1.8)
    axes[1].plot(df["_step"], df["Metrics/task_1_success_rate"], label=label, color=color, lw=1.8)
    axes[2].plot(df["_step"], df["Curriculum/gravity_curriculum/gravity_frac"], label=label, color=color, lw=1.8)
    axes[3].plot(df["_step"], df["Curriculum/gravity_curriculum/monitor_mean_rate"], label=label, color=color, lw=1.8)

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
axes[0].legend(fontsize=9, loc="lower right")
axes[3].legend(fontsize=8, loc="lower right")

fig.suptitle(
    "Uniform + monitor_mean + floor=0.1 across OmniReset insertions (state-only, 80K envs)",
    fontsize=12,
)
fig.tight_layout()

out = Path("scripts/peg_leg_drawer.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
