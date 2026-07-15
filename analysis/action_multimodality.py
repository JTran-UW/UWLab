#!/usr/bin/env python3
"""Action multimodality within each configuration.

For each config we pooled the first-episode states visited by ALL its seeds and relabeled every
state with EVERY seed's policy (see relabel_actions.py). So at each pooled state we have K action
vectors -- one per seed -- evaluated on the *same* state. The spread of those K actions measures how
much the seeds disagree on what to do, i.e. behavioral (action) multimodality, computed over a state
distribution that is unbiased toward any single seed.

This complements the state-visitation UMAP (umap_within_config.py): there the seeds visit nearly
identical insertive-object states (all silhouettes ~0/negative). Here we ask whether, at those shared
states, the policies nonetheless act differently.

Metrics per config (averaged over action dims unless noted):
  * self-error floor      -- a policy on its OWN states vs the stored on-policy action. The float16
                             obs-storage noise floor; the disagreement signal must dwarf it.
  * cross-policy std      -- std across the K seeds' actions at each state, mean over states.
  * disagreement fraction -- cross-policy (within-state) variance / total action variance over all
                             (state, policy). 0 = seeds agree everywhere (converged behavior);
                             ->1 = at any fixed state the seeds act as differently as the task's whole
                             action range. The clean scale-free "how multimodal" index.
  * pairwise matrix       -- mean per-state L2 action distance between each pair of seeds (reveals
                             whether seeds split into behavioral modes vs all-mutually-distinct).

Reads:  analysis/relabel_actions/<config>.npz
Writes: analysis/action_multimodality.png, analysis/action_multimodality_pairwise.png

Usage:
    python analysis/action_multimodality.py
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RELABEL_DIR = os.path.join(HERE, "relabel_actions")

# reuse the per-config base hues from the state-visitation UMAP for visual consistency
try:
    from umap_state_visitation import CONFIG_COLORS  # noqa: E402
except Exception:
    CONFIG_COLORS = {}

# Human-readable labels (letter code + description) and the A->G display order.
CONFIG_LABEL = {
    "sysid_floor01": ("A", "PC + gravity/sparse + sys id gains + 2 reset distributions"),
    "sysid_grav_4resets": ("B", "PC + gravity/sparse + sys id gains + 4 reset distributions"),
    "base_grav_2resets": ("C", "PC + gravity/sparse + 2 reset distributions"),
    "base_grav_4resets": ("D", "PC + gravity/sparse + 4 reset distributions"),
    "seq_resets__State-Sequential": ("E", "state + 4 reset distributions + fixed reset state ordering"),
    "seq_resets__State": ("F", "state + 4 reset distributions + fixed initial init"),
    "base_4resets": ("G", "state + 4 reset distribution"),
    "pc_4_resets": ("H", "PC + 4 reset distributions (normal gravity, no curriculum)"),
}


def analyze(npz):
    """Return a dict of per-config action-disagreement metrics from one relabel dump."""
    d = np.load(npz, allow_pickle=True)
    A = d["actions_by_policy"].astype(np.float64)  # [K, M, dim]
    ref = d["ref_action"].astype(np.float64)  # [M, dim]
    seeds = list(d["policy_seeds"])
    src = d["src_seed"]
    K, M, dim = A.shape

    # self-error floor: each policy vs the stored on-policy action on ITS OWN pooled states
    self_err = []
    for ki, s in enumerate(seeds):
        own = src == s
        if own.any():
            self_err.append(np.abs(A[ki, own] - ref[own]).mean())
    self_err = float(np.mean(self_err)) if self_err else float("nan")

    # cross-policy disagreement at each state
    var_within = A.var(0).mean(0)  # [dim] mean over states of cross-policy variance
    std_within = np.sqrt(A.var(0)).mean()  # scalar: mean cross-policy std
    # total variance of actions over (state, policy)
    var_total = A.reshape(K * M, dim).var(0)  # [dim]
    frac = float((var_within.sum()) / (var_total.sum() + 1e-12))  # disagreement fraction (dim-summed)

    # KxK pairwise mean per-state L2 action distance
    pair = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            pair[i, j] = np.linalg.norm(A[i] - A[j], axis=1).mean()

    return {
        "config": os.path.splitext(os.path.basename(npz))[0],
        "seeds": seeds,
        "K": K,
        "M": M,
        "dim": dim,
        "self_err": self_err,
        "std_within": float(std_within),
        "frac": frac,
        "var_within": var_within,
        "var_total": var_total,
        "pair": pair,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, "action_multimodality.png"))
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(RELABEL_DIR, "*.npz")))
    if not files:
        print(f"No relabel dumps in {RELABEL_DIR}/ -- run relabel_actions.py first.")
        return
    # order A->G by the letter code; unknown configs fall to the end
    def letter(cfg):
        return CONFIG_LABEL.get(cfg, ("Z", cfg))[0]

    res = [analyze(f) for f in files]
    res.sort(key=lambda r: letter(r["config"]))
    for r in res:
        r["letter"], r["desc"] = CONFIG_LABEL.get(r["config"], (r["config"], r["config"]))

    # ---- printed summary ----
    print(f"\n{'':>2} {'config':<30} {'K':>2} {'states':>7} {'self_err':>9} {'xpol_std':>9} {'disagree_frac':>13}")
    print("-" * 82)
    for r in res:
        print(f"{r['letter']:>2} {r['config']:<30} {r['K']:>2} {r['M']:>7} {r['self_err']:>9.4f} "
              f"{r['std_within']:>9.3f} {r['frac']:>13.3f}   {r['desc']}")
    print("-" * 82)
    print("disagree_frac = cross-policy(within-state) action variance / total action variance.")
    print("Higher = seeds act more differently at the SAME state (more behavioral multimodality).\n")

    # ---- Figure 1: headline disagreement bar chart ----
    from matplotlib.patches import Patch

    fig, ax1 = plt.subplots(figsize=(9, 7.5))
    letters = [r["letter"] for r in res]
    colors = [CONFIG_COLORS.get(r["config"], (0.4, 0.4, 0.4)) for r in res]
    x = np.arange(len(res))

    ax1.bar(x, [r["frac"] for r in res], color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(letters, fontsize=11)
    ax1.set_ylabel("cross-policy disagreement fraction")
    ax1.set_ylim(0, max(r["frac"] for r in res) * 1.18)
    ax1.set_title("Action multimodality within each configuration\n"
                  "(within-state action variance / total action variance)")
    ax1.grid(axis="y", alpha=0.3)
    for xi, r in zip(x, res):
        ax1.text(xi, r["frac"] + 0.005, f"{r['frac']:.2f}\n{r['K']} seeds", ha="center", va="bottom", fontsize=8)

    # shared legend mapping letter -> full description, in reserved space below the axes
    handles = [Patch(facecolor=CONFIG_COLORS.get(r["config"], (0.4, 0.4, 0.4)),
                     label=f"{r['letter']}.  {r['desc']}") for r in res]
    fig.subplots_adjust(bottom=0.32, top=0.90)
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.01), handlelength=1.2)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    # ---- Figure 2: per-config KxK pairwise action-distance heatmaps ----
    n = len(res)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig2, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    import textwrap

    for ax, r in zip(axes, res):
        im = ax.imshow(r["pair"], cmap="viridis")
        ax.set_xticks(range(r["K"])); ax.set_xticklabels([f"s{s}" for s in r["seeds"]], fontsize=8)
        ax.set_yticks(range(r["K"])); ax.set_yticklabels([f"s{s}" for s in r["seeds"]], fontsize=8)
        wrapped = "\n".join(textwrap.wrap(f"{r['letter']}. {r['desc']}", 38))
        ax.set_title(f"{wrapped}\n(mean pairwise action L2)", fontsize=9)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(len(res), len(axes)):
        axes[j].axis("off")
    fig2.suptitle("Per-config pairwise action distance between seeds (off-diagonal structure = behavioral modes)",
                  fontsize=13)
    fig2.tight_layout()
    out2 = os.path.splitext(args.out)[0] + "_pairwise.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
