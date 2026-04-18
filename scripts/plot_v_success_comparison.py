"""Compare V_success vs classifier vs empirical vs uniform on rt0/rt1 success + gravity."""

import re
from pathlib import Path

import matplotlib.pyplot as plt

ITER_RE = re.compile(r"Learning iteration (\d+)/\d+")
TASK0_RE = re.compile(r"Metrics/task_0_success_rate:\s+([0-9.]+)")
TASK1_RE = re.compile(r"Metrics/task_1_success_rate:\s+([0-9.]+)")
GRAVITY_RE = re.compile(r"Curriculum/gravity_curriculum/gravity_frac:\s+([0-9.]+)")


def parse(path: Path) -> dict:
    iters, rt0, rt1, grav = [], [], [], []
    cur_iter = None
    cur_rt0 = cur_rt1 = cur_grav = None
    with path.open() as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                if cur_iter is not None and cur_rt0 is not None:
                    iters.append(cur_iter)
                    rt0.append(cur_rt0)
                    rt1.append(cur_rt1)
                    grav.append(cur_grav)
                cur_iter = int(m.group(1))
                cur_rt0 = cur_rt1 = cur_grav = None
                continue
            m = TASK0_RE.search(line)
            if m:
                cur_rt0 = float(m.group(1))
                continue
            m = TASK1_RE.search(line)
            if m:
                cur_rt1 = float(m.group(1))
                continue
            m = GRAVITY_RE.search(line)
            if m:
                cur_grav = float(m.group(1))
    if cur_iter is not None and cur_rt0 is not None:
        iters.append(cur_iter)
        rt0.append(cur_rt0)
        rt1.append(cur_rt1)
        grav.append(cur_grav)
    return {"iter": iters, "rt0": rt0, "rt1": rt1, "grav": grav}


# Okabe-Ito colorblind-safe palette + line-style variation for extra discriminability.
runs = {
    "V_success (34675699)": ("/tmp/patyin_logs/34675699.out", "#D55E00", "-", 1.8),       # vermillion
    "Classifier (34675161)": ("/tmp/patyin_logs/34675161.out", "#0072B2", "-", 1.8),      # blue
    "Empirical (34674405)": ("/tmp/patyin_logs/34674405.out", "#CC79A7", "-.", 1.8),      # reddish purple
    "Uniform (34675170)": ("/tmp/patyin_logs/34675170.out", "#000000", ":", 1.8),         # black dotted
    "Prior peg baseline (34590465)": ("/tmp/patyin_logs/34590465.out", "#F0E442", "--", 1.8),  # yellow, pre-3e980a8 solved run
}

parsed = {label: parse(Path(path)) for label, (path, _, _, _) in runs.items()}
for label, d in parsed.items():
    print(f"{label}: {len(d['iter'])} iters, last iter = {d['iter'][-1] if d['iter'] else 'NA'}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
for label, (_, color, ls, lw) in runs.items():
    d = parsed[label]
    axes[0].plot(d["iter"], d["rt0"], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
    axes[1].plot(d["iter"], d["rt1"], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
    axes[2].plot(d["iter"], d["grav"], label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)

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
fig.suptitle("V_success vs classifier vs empirical vs uniform (single-task peg ZeroG GPS)", fontsize=11)
fig.tight_layout()
out = Path("scripts/v_success_comparison.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
