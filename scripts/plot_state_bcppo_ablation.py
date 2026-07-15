"""State BCPPO ablation: 256 vs 16k envs x bc_coeff 0.001 vs 0.01.

All runs: peg only, gravity_floor=1.0 (no curriculum), bc_loss_type=mse.
"""
import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
PROJ = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-BCPPO-Sysid-v0"

JOBS = [
    ("256 envs, bc=0.001", "35004959", "tab:blue", "--"),
    ("256 envs, bc=0.01",  "35004999", "tab:orange", "--"),
    ("16k envs, bc=0.001", "35005011", "tab:blue", "-"),
    ("16k envs, bc=0.01",  "35005016", "tab:orange", "-"),
]

METRICS = [
    ("Metrics/success_student_train", "Success (train pool)"),
    ("Metrics/success_student_eval",  "Success (eval pool, no-grad)"),
]


def smooth(y, w=10):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def fetch(jid):
    runs = list(api.runs(f"patyin/{PROJ}", filters={"display_name": jid}))
    if not runs:
        return None
    return runs[0].history(samples=4000)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
SMOOTH = 8

for col_idx, (col, title) in enumerate(METRICS):
    ax = axes[col_idx]
    for label, jid, color, ls in JOBS:
        h = fetch(jid)
        if h is None or h.empty or col not in h.columns:
            print(f"skip {label} ({jid}): missing {col}")
            continue
        sub = h[["_step", col]].dropna()
        if sub.empty:
            continue
        x = sub["_step"].values
        y = sub[col].values
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.25, linestyle=ls)
        ys = smooth(y, SMOOTH)
        ax.plot(x[SMOOTH - 1:], ys, color=color, linewidth=2, linestyle=ls, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9)

plt.suptitle(
    "State BCPPO ablation - peg, gravity=1.0\n"
    "linestyle: -- 256 envs / - 16k envs   color: blue bc=0.001 / orange bc=0.01"
)
plt.tight_layout()
out = "scripts/state_bcppo_ablation.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved {out}")
