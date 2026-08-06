"""Bar charts summarising the vision-observation speed/memory ablation ladder."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated categorical slots (light surface #fcfcfb): blue / orange / aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

CONFIGS = [
    "3 cam · hist 3\nrender 168×126 → 84²",
    "2 cam · hist 3\nrender 168×126 → 84²",
    "2 cam · no hist\nrender 168×126 → 84²",
    "2 cam · no hist\nrender 112×84 → 84²",
    "2 cam · no hist\nrender 112×84 → 32²",
]

# ms per iteration: env_step, sample_H2D (x5 updates), everything else
TIME = np.array([
    [196.212, 141.080, 73.202],
    [192.271, 128.335, 86.175],
    [169.129,  39.980, 72.113],
    [172.725,  35.035, 70.833],
    [140.941,   7.375, 65.965],
])
# host RSS GiB: Isaac Sim process footprint, replay buffer, other
# NB this is process RSS, NOT the sim state -- physics/render tensors are a SEPARATE
# ~14 GiB on the GPU. Both are deltas across the same env construction.
MEM = np.array([
    [12.70, 2.480, 0.770],
    [12.53, 1.650, 0.760],
    [12.52, 0.559, 0.771],
    [12.42, 0.553, 0.767],
    [12.37, 0.076, 0.754],
])
BUF = np.array([2.490, 1.670, 0.5655, 0.5530, 0.0944])   # GiB, exact from breakdown

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
})

fig, axes = plt.subplots(3, 1, figsize=(12.2, 14.0), sharey=True,
                         gridspec_kw={"height_ratios": [1, 1, 0.72], "hspace": 0.30})
y = np.arange(len(CONFIGS))[::-1]  # first config at top


def style(ax, xlabel, xmax):
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.65, len(CONFIGS) - 0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(CONFIGS, fontsize=10, color=INK, linespacing=1.35)
    ax.set_xlabel(xlabel, fontsize=10, color=INK2, labelpad=8)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")


def stacked(ax, data, labels, colors, unit, fmt="{:.0f}"):
    left = np.zeros(len(CONFIGS))
    for j, (lab, c) in enumerate(zip(labels, colors)):
        ax.barh(y, data[:, j], left=left, height=0.62, color=c, label=lab,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        # direct label inside any segment wide enough to hold it
        for i, (v, l0) in enumerate(zip(data[:, j], left)):
            if v / data.sum(axis=1).max() > 0.085:
                ax.text(l0 + v / 2, y[i], fmt.format(v), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold", zorder=4)
        left += data[:, j]
    totals = data.sum(axis=1)
    for i, t in enumerate(totals):
        ax.text(t + totals.max() * 0.015, y[i], fmt.format(t) + unit, va="center",
                fontsize=10.5, color=INK, fontweight="bold")
        if i > 0:
            ax.text(t + totals.max() * 0.135, y[i], f"{totals[0]/t:.2f}×", va="center",
                    fontsize=9.5, color="#006300", fontweight="bold")
    return totals


# ---- 1. iteration time -------------------------------------------------
ax = axes[0]
tot = stacked(ax, TIME, ["env_step (physics + render)", "buffer sample → GPU (×5 updates)", "everything else"],
              [BLUE, ORANGE, AQUA], " ms")
style(ax, "milliseconds per training iteration  ·  96 envs, UTD 5", 480)
ax.set_title("Iteration time — 1.92× faster end to end", fontsize=13.5, fontweight="bold",
             color=INK, loc="left", pad=30)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False,
          fontsize=9.5, labelcolor=INK2, ncol=3, handlelength=1.2, columnspacing=1.6)

# ---- 2. host memory ----------------------------------------------------
ax = axes[1]
stacked(ax, MEM, ["Isaac Sim process (Kit runtime, USD, caches)", "replay buffer", "other"],
        [BLUE, ORANGE, AQUA], " GiB", fmt="{:.2f}")
style(ax, "host RSS (GiB)", 19.2)
ax.set_title("Host memory — only 1.21×: the Isaac Sim process footprint dominates",
             fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=30)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False,
          fontsize=9.5, labelcolor=INK2, ncol=3, handlelength=1.2, columnspacing=1.6)

# ---- 3. replay buffer alone -------------------------------------------
ax = axes[2]
ax.barh(y, BUF, height=0.62, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
for i, v in enumerate(BUF):
    ax.text(v + 0.06, y[i], f"{v:.3f} GiB", va="center", fontsize=10.5, color=INK, fontweight="bold")
    if i > 0:
        ax.text(v + 0.60, y[i], f"{BUF[0]/v:.1f}×", va="center", fontsize=9.5,
                color="#006300", fontweight="bold")
style(ax, "replay buffer (GiB, CPU-resident)", 3.5)
ax.set_title("Replay buffer alone — 26× smaller (invisible above, but it drives the speedup)",
             fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=12)

fig.suptitle("Vision observation ablation — UR5e peg insertion, MR.Q asymmetric, RTX 4090",
             fontsize=15, fontweight="bold", color=INK, x=0.008, ha="left", y=0.988)
fig.text(0.008, 0.004,
         "Each row adds one change to the row above.  Observation: 63,504 → 2,048 elements (31×).  "
         "GPU memory moved only 15.36 → 14.79 GiB (3.7%).\n"
         "Host RSS above is the Isaac Sim PROCESS footprint (Kit runtime, USD stage, caches); "
         "the sim state itself is a separate ~14 GiB on the GPU.\n"
         "Row 4 buffer figure taken from the host-RSS line (its exact breakdown was truncated).  "
         "Timings from --profile_timing, which syncs around every block.",
         fontsize=8.6, color=MUTED, ha="left", va="bottom")

fig.subplots_adjust(left=0.185, right=0.985, top=0.905, bottom=0.085, hspace=0.95)
out = "/home/jtran/.claude/jobs/e31d500d/tmp/ablation_summary.png"
fig.savefig(out, dpi=170, facecolor=SURFACE)
print(f"saved {out}")
print(f"time  {tot[0]:.1f} -> {tot[-1]:.1f} ms  ({tot[0]/tot[-1]:.2f}x)")
print(f"host  {MEM.sum(axis=1)[0]:.2f} -> {MEM.sum(axis=1)[-1]:.2f} GiB  ({MEM.sum(axis=1)[0]/MEM.sum(axis=1)[-1]:.2f}x)")
print(f"buf   {BUF[0]:.2f} -> {BUF[-1]:.4f} GiB  ({BUF[0]/BUF[-1]:.1f}x)")
