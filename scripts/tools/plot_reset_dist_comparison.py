"""Compare full-OmniReset vs reaching-only reset distributions over finetuning.

Two separate figures, deliberately: success rate (%) and episode duration (steps) have
unrelated scales, and putting them on one pair of y-axes would be a dual-axis chart.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1 and 2 of the validated palette (light surface #fcfcfb).
# Checked with validate_palette.js: CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1.
FULL = "#2a78d6"
REACH = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

SUCCESS_STEPS = [0, 1000, 2000, 3000, 4000]
SUCCESS_FULL = [81.1, 90.4, 90.4, 88.7, 81.0]
SUCCESS_REACH = [81.1, 80.5, 89.0, 66.4, 26.3]

# Duration has no step-0 measurement, so it starts at 1k.
DURATION_STEPS = [1000, 2000, 3000, 4000]
DURATION_FULL = [71.7, 53.5, 55.6, 67.1]
DURATION_REACH = [66.4, 54.1, 82.3, 130.3]

LABEL_FULL = "full omnireset reset distribution"
LABEL_REACH = "reset from reaching only"


def _style(ax, title, ylabel, xticks):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, pad=12, loc="left")
    ax.set_xlabel("steps", color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    # Ticks pinned to the sampled steps. Letting matplotlib choose puts them at 500-step
    # intervals, which the "k" formatter then renders as a duplicated 0k, 0k, 1k, 1k...
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(t / 1000)}k" for t in xticks])


def _series(ax, x, y, color, label):
    ax.plot(x, y, color=color, lw=2, marker="o", ms=8, label=label,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)


def make_plot(x_full, y_full, x_reach, y_reach, title, ylabel, out_path, annotate_last=True):
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    _series(ax, x_full, y_full, FULL, LABEL_FULL)
    _series(ax, x_reach, y_reach, REACH, LABEL_REACH)
    xticks = sorted(set(x_full) | set(x_reach))
    _style(ax, title, ylabel, xticks)
    span = max(xticks) - min(xticks)
    # Right margin sized to hold the endpoint labels inside the axes.
    ax.set_xlim(min(xticks) - 0.04 * span, max(xticks) + 0.22 * span)
    lo = min(y_full + y_reach)
    hi = max(y_full + y_reach)
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.10 * (hi - lo))
    if annotate_last:
        # Label only the endpoints -- that is where the two series diverge and where
        # the reader's question ("how far apart did they end up?") is answered.
        for x, y, c in ((x_full[-1], y_full[-1], FULL), (x_reach[-1], y_reach[-1], REACH)):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(10, -4),
                        color=c, fontsize=10, fontweight="bold")
    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"saved {out_path}")
    return fig


if __name__ == "__main__":
    make_plot(SUCCESS_STEPS, SUCCESS_FULL, SUCCESS_STEPS, SUCCESS_REACH,
              "Success rate by reset distribution", "success rate (%)",
              "reset_dist_success_rate.png")
    make_plot(DURATION_STEPS, DURATION_FULL, DURATION_STEPS, DURATION_REACH,
              "Average episode duration by reset distribution", "avg episode duration (steps)",
              "reset_dist_episode_duration.png")
