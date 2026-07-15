"""Plot task_0 / task_1 / gravity_frac for the peg ScenePC teacher-retrain ablations."""

import matplotlib.pyplot as plt
import numpy as np

# Wong colorblind-safe palette (https://www.nature.com/articles/nmeth.1618).
# Skipping yellow which has poor contrast on white.
RUNS = {
    "34753110: state + 4-canonical (5×A40, 80K)":                       ("/tmp/34753110.tsv", "#000000"),
    "107905: ScenePC NO resample, prev_act (deploy 34%)":               ("/tmp/107905.tsv", "#009E73"),
    "107906: ScenePC NO resample, NoPrevAct (deploy 30%)":              ("/tmp/107906.tsv", "#CC79A7"),
    "109710 (killed): + FULL resample, prev_act":                       ("/tmp/109710.tsv", "#D55E00"),
    "109711 (killed): + FULL resample, NoPrevAct":                      ("/tmp/109711.tsv", "#E69F00"),
    "35149924 (live): SEED-LOCKED det. PC, prev_act":                   ("/tmp/35149924.tsv", "#0072B2"),
    "35141064 (live): SEED-LOCKED det. PC, NoPrevAct":                  ("/tmp/35141064.tsv", "#56B4E9"),
}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return np.array(rows)


def smooth(y, k=25):
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")


fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
ax_t0, ax_t1, ax_g = axes

for label, (path, color) in RUNS.items():
    data = load(path)
    it, t0, t1, g = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    K = 25
    it_s = it[K - 1:] if len(it) >= K else it

    ax_t0.plot(it, t0, color=color, alpha=0.25)
    ax_t0.plot(it_s, smooth(t0, K), color=color, label=label, linewidth=2)

    ax_t1.plot(it, t1, color=color, alpha=0.25)
    ax_t1.plot(it_s, smooth(t1, K), color=color, label=label, linewidth=2)

    ax_g.plot(it, g, color=color, label=label, linewidth=2)

ax_t0.set_ylabel("task_0 success\n(ZeroGAnywhere)")
ax_t0.set_ylim(-0.02, 1.05)
ax_t0.legend(loc="upper left", fontsize=9)
ax_t0.grid(alpha=0.3)

ax_t1.set_ylabel("task_1 success\n(PartialAssembly)")
ax_t1.set_ylim(0, 1)
ax_t1.grid(alpha=0.3)

ax_g.set_ylabel("gravity_frac")
ax_g.set_xlabel("Learning iteration")
ax_g.set_ylim(0, 1.05)
ax_g.axhline(0.1, color="gray", linestyle=":", linewidth=1, label="floor=0.1")
ax_g.legend(loc="upper right")
ax_g.grid(alpha=0.3)

fig.suptitle(
    "Peg teacher RL training: state+4-canonical baseline vs ScenePC+8-canonical retrains\n"
    "All ~80K envs, peg/peghole, uniform routing, monitor_mean curriculum, floor=0.1",
    fontsize=12,
)
fig.tight_layout()

out = "/home/patrickhaoy/research/UWLab-private/scripts/pc_teacher_retrain.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
