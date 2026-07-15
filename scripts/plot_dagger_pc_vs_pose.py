"""Plot DAgger success curves: PC-teacher (current) vs pose-teacher (4-canonical Lean, historical)."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# TSV columns: iter, st_train, st_eval, te_eval, te_train, mu_loss, behavior_loss
RUNS = {
    # Pose teacher (old, 4-canonical Lean, Anywhere-only resets; capped at ~55%)
    "34955211: 4-canon pose teacher, prev_act, Anywhere-only (old)":   ("/tmp/34955211.dagger.tsv", "#000000", "--"),
    # PC teacher (new, 8-canonical + seed-locked PC, Anywhere-only resets)
    "35176337: 8-canon PC teacher, prev_act, Anywhere-only":           ("/tmp/35176337.dagger.tsv", "#0072B2", "-"),
    "35176347: 8-canon PC teacher, NoPrevAct, Anywhere-only":          ("/tmp/35176347.dagger.tsv", "#D55E00", "-"),
}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            row = []
            for x in parts[:7]:
                if x == "":
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(x))
                    except ValueError:
                        row.append(np.nan)
            rows.append(row)
    return np.array(rows)


def smooth(y, k=50):
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    valid = np.isfinite(y)
    yp = np.where(valid, y, 0.0)
    return np.convolve(yp, kernel, mode="valid")


fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
ax_se, ax_te, ax_mu = axes

for label, (path, color, ls) in RUNS.items():
    data = load(path)
    if data.size == 0:
        continue
    it = data[:, 0]
    st_eval = data[:, 2]
    te_eval = data[:, 3]
    te_train = data[:, 4]
    mu = data[:, 5]

    # Subsample to make plot manageable.
    step = max(1, len(it) // 2000)
    sl = slice(None, None, step)

    ax_se.plot(it[sl], st_eval[sl], color=color, linestyle=ls, alpha=0.35, linewidth=0.8)
    if len(st_eval) > 50:
        ax_se.plot(it[49:][sl], smooth(st_eval, 50)[sl], color=color, linestyle=ls, label=label, linewidth=2)

    # Teacher rate: prefer teacher_eval (new runs), else teacher_train (old 50/50 runs).
    teacher = np.where(np.isfinite(te_eval), te_eval, te_train)
    ax_te.plot(it[sl], teacher[sl], color=color, linestyle=ls, alpha=0.35, linewidth=0.8)
    if len(teacher) > 50:
        ax_te.plot(it[49:][sl], smooth(teacher, 50)[sl], color=color, linestyle=ls, label=label, linewidth=2)

    ax_mu.plot(it[sl], mu[sl], color=color, linestyle=ls, alpha=0.35, linewidth=0.8)
    if len(mu) > 50:
        ax_mu.plot(it[49:][sl], smooth(mu, 50)[sl], color=color, linestyle=ls, label=label, linewidth=2)

ax_se.set_ylabel("success_student_eval\n(no-grad held-out)")
ax_se.set_ylim(-0.05, 1.05)
ax_se.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
ax_se.text(0.01, 0.51, "old DAgger ceiling ~50%", color="gray", fontsize=8, transform=ax_se.get_yaxis_transform())
ax_se.legend(loc="lower right", fontsize=9)
ax_se.grid(alpha=0.3)

ax_te.set_ylabel("teacher success rate\n(eval or train pool)")
ax_te.set_ylim(0, 1.05)
ax_te.grid(alpha=0.3)

ax_mu.set_ylabel("behavior_mu loss")
ax_mu.set_xlabel("DAgger iteration")
ax_mu.set_yscale("log")
ax_mu.grid(alpha=0.3, which="both")

fig.suptitle(
    "Lean DAgger: PC teacher (seed-locked, 8-canonical) vs pose teacher (4-canonical, historical)\n"
    "peg/peghole, single-task, Ur5e+Robotiq2f85, depth student",
    fontsize=12,
)
fig.tight_layout()
out = "/home/patrickhaoy/research/UWLab-private/scripts/dagger_pc_vs_pose.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")
