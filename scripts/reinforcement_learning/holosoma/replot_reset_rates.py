# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-plot per-reset OUTCOME-rate scatters (abnormal / timeout) from reset_success_rates.json files
and concat labeled panels. Pure post-processing; no simulator."""

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_critic_plots import concat_labeled  # noqa: E402


def rate_scatter(json_path: str, field: str, out_path: str, vmax: float = 0.5) -> tuple[float, int]:
    """3D scatter colored by per-reset rate of ``field`` ('abnormal' or 'failure'); returns
    (mean rate, #resets with rate>0)."""
    r = json.load(open(json_path))
    xyz = np.array([x["init_peg_xyz"] for x in r])
    rates = np.array([x[field] / x["n_env"] for x in r])
    hole = np.array(r[0]["peghole_xyz"])
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rates, cmap="RdYlGn_r", vmin=0.0, vmax=vmax, s=22, alpha=0.85)
    ax.scatter(hole[0], hole[1], hole[2], c="blue", s=200, marker="*", label="peghole (env 0)", zorder=10)
    name = "abnormal_robot" if field == "abnormal" else "timeout"
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1,
                 label=f"{name} rate over {r[0]['n_env']} trajectories (scale capped at {vmax:g})")
    ax.set_xlabel("x (m, env-local)")
    ax.set_ylabel("y (m, env-local)")
    ax.set_zlabel("z (m, env-local)")
    ax.set_title(f"Per-reset {name} rate  ({len(r)} resets, mean {rates.mean():.1%}, "
                 f"{int((rates > 0).sum())} resets > 0)")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(rates.mean()), int((rates > 0).sum())


def build(panels: list[tuple[str, str]], field: str, out_path: str, vmax: float = 0.5) -> None:
    imgs, labels = [], []
    for json_path, name in panels:
        img = os.path.join(os.path.dirname(json_path), f"reset_{field}_rate_panel.png")
        mean, npos = rate_scatter(json_path, field, img, vmax=vmax)
        imgs.append(img)
        labels.append(f"{name} — mean {mean:.1%}  ({npos} resets >0)")
    concat_labeled(imgs, labels, out_path)
    print("wrote", out_path)
    for lbl in labels:
        print("   ", lbl)
