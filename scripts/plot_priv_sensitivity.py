"""Plot per-action-dim privileged-sensitivity ratio vs PPO iteration.

Reads one or more PPO_RMA training logs (rsl_rl printed metrics, one block
per iteration). Each diagnostic block contains lines like
  ``Mean diag/priv_sens/dim{D}/ratio_delta_over_std loss: VALUE``
emitted every ``priv_sensitivity_log_every`` PPO iterations.

This script associates each diagnostic block with the most recently seen
``Learning iteration N/<MAX>`` marker, builds a per-dim trajectory, and
plots iter vs ratio (one line per action dim + the over-dims envelope).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
DIM_RE = re.compile(r"diag/priv_sens/dim(\d+)/ratio_delta_over_std loss:\s+([\-0-9.eE+]+)")
OVER_MAX_RE = re.compile(r"diag/priv_sens/over_dims_max_ratio loss:\s+([\-0-9.eE+]+)")
OVER_MIN_RE = re.compile(r"diag/priv_sens/over_dims_min_ratio loss:\s+([\-0-9.eE+]+)")


def parse_log(path: Path) -> dict[int, dict]:
    """Returns {iter -> {"dims": {d: ratio}, "over_max": v, "over_min": v}}."""
    out: dict[int, dict] = {}
    current_iter: int | None = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                current_iter = int(m.group(1))
                out.setdefault(current_iter, {"dims": {}, "over_max": None, "over_min": None})
                continue
            if current_iter is None:
                continue
            m = DIM_RE.search(line)
            if m:
                d = int(m.group(1))
                v = float(m.group(2))
                out[current_iter]["dims"][d] = v
                continue
            m = OVER_MAX_RE.search(line)
            if m:
                out[current_iter]["over_max"] = float(m.group(1))
                continue
            m = OVER_MIN_RE.search(line)
            if m:
                out[current_iter]["over_min"] = float(m.group(1))
    # Drop iterations without diagnostic blocks.
    return {k: v for k, v in out.items() if v["dims"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="one or more training log files")
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--title", default="Privileged-info sensitivity over training")
    args = parser.parse_args()

    # Merge multiple logs (later logs overwrite earlier on duplicate iters —
    # gives the v4-resume run priority over v3 at the overlap iter 500).
    merged: dict[int, dict] = {}
    for p in args.logs:
        merged.update(parse_log(Path(p)))

    if not merged:
        raise SystemExit("no diagnostic blocks found in any log")

    iters = sorted(merged.keys())
    num_dims = max(max(merged[i]["dims"].keys()) for i in iters) + 1

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.viridis
    for d in range(num_dims):
        xs = [i for i in iters if d in merged[i]["dims"]]
        ys = [merged[i]["dims"][d] for i in xs]
        color = cmap(d / max(1, num_dims - 1))
        # Action dims for RelCartesianOSCAction + gripper: 0-5 are arm
        # (3 trans + 3 rot via axis-angle), 6 is gripper.
        label = f"dim {d}" + (" (gripper)" if d == 6 else " (arm)")
        ax.plot(xs, ys, color=color, lw=1.5, alpha=0.9, label=label)

    # Over-dims envelope (thicker, on top).
    over_max_xs = [i for i in iters if merged[i].get("over_max") is not None]
    over_min_xs = [i for i in iters if merged[i].get("over_min") is not None]
    if over_max_xs:
        ax.plot(
            over_max_xs,
            [merged[i]["over_max"] for i in over_max_xs],
            color="black", lw=2.0, alpha=0.6, label="over-dims max", linestyle="--",
        )
    if over_min_xs:
        ax.plot(
            over_min_xs,
            [merged[i]["over_min"] for i in over_min_xs],
            color="gray", lw=2.0, alpha=0.6, label="over-dims min", linestyle="--",
        )

    ax.set_xlabel("PPO iteration")
    ax.set_ylabel(r"$|\Delta a|\;/\;\sigma_a$  (per-dim, batch)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.4)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_ylim(bottom=0.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}  ({len(iters)} iters, {num_dims} dims)")


if __name__ == "__main__":
    main()
