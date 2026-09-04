# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot rendering for eval_critic.py, shared by the live run and the offline renderer.

``render_group`` draws the three per-reset figures from plain numpy arrays, so eval_critic.py can
either call it inline or dump the arrays to ``.npz`` (``--defer_plots``) and render later with
``render_eval_critic_plots.py``.
"""

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render_group(d: dict, output_path: str, peg_xy_path: str, q_traj_path: str | None,
                 peg_xlim, peg_ylim, ckpt_name: str) -> None:
    """Render Q-vs-MC (``output_path``), peg x-y trajectories (``peg_xy_path``) and, if
    ``q_traj_path`` is given, the leader env's Q vs return-to-go chart, for one reset group.

    ``d`` keys (numpy): q_dists/qt_dists [C, atoms], q_support/centers [atoms], bin_width, v_min,
    v_max, alpha, soft_mc/reward_mc/entropy_mc [n], mc_pmf [atoms], term_counts [3], term_type [n],
    peg_traj [T, n, 2], peghole_xy [2], q_lead/g_lead/act_lead [T], q_ppo_lead [T] or None,
    reset_idx, n_g, lead.
    """
    q_dists, qt_dists, q_support, centers = d["q_dists"], d["qt_dists"], d["q_support"], d["centers"]
    soft_mc, reward_mc, entropy_mc = d["soft_mc"], d["reward_mc"], d["entropy_mc"]
    term_counts = [int(c) for c in d["term_counts"]]
    n_g, reset_idx, lead = int(d["n_g"]), int(d["reset_idx"]), int(d["lead"])
    v_min, v_max, alpha, bin_width = float(d["v_min"]), float(d["v_max"]), float(d["alpha"]), float(d["bin_width"])
    peg_traj, peghole_xy = d["peg_traj"], d["peghole_xy"]

    # ---- Q vs total soft MC return (top), reward / entropy / terminations (bottom) ----
    num_critics = q_dists.shape[0]
    fig = plt.figure(figsize=(15, 8))
    ax_main = fig.add_subplot(2, 1, 1)
    ax_rew = fig.add_subplot(2, 3, 4)
    ax_ent = fig.add_subplot(2, 3, 5)
    ax_term = fig.add_subplot(2, 3, 6)
    ax_main.bar(centers, d["mc_pmf"], width=bin_width * 0.9, alpha=0.35,
                label="Soft MC return (reward + entropy)", color="tab:orange")
    online_colors = ["tab:blue", "tab:cyan", "tab:green", "tab:olive"]
    target_colors = ["tab:red", "tab:pink", "tab:purple", "tab:brown"]
    for c in range(num_critics):
        q_exp_c = float(np.sum(q_dists[c] * q_support))
        ax_main.plot(centers, q_dists[c], color=online_colors[c % len(online_colors)],
                     lw=1.8, label=f"Q online[{c}]  E={q_exp_c:.2f}")
        qt_exp_c = float(np.sum(qt_dists[c] * q_support))
        ax_main.plot(centers, qt_dists[c], color=target_colors[c % len(target_colors)],
                     lw=1.5, ls="--", label=f"Bellman target y[{c}] (s1,π(a1))  E={qt_exp_c:.2f}")
    q_exp_mean = float(np.mean([np.sum(q_dists[c] * q_support) for c in range(num_critics)]))
    ax_main.axvline(soft_mc.mean(), color="tab:orange", ls=":", lw=2, label=f"MC mean={soft_mc.mean():.2f}")
    ax_main.axvline(q_exp_mean, color="tab:blue", ls=":", lw=2, label=f"E[Q] mean={q_exp_mean:.2f}")
    ax_main.set_xlabel("Return")
    ax_main.set_ylabel("Probability")
    ax_main.set_title(
        f"Q (online + target, per critic) vs soft MC return — iter {reset_idx} | "
        f"soft MC mean={soft_mc.mean():.3f}  n={n_g}  (α={alpha:.4f})"
    )
    ax_main.legend(fontsize=8)
    ax_main.set_xlim(v_min, v_max)
    ax_rew.hist(reward_mc, bins=40, color="tab:green", alpha=0.8)
    ax_rew.set_title(f"Reward component  mean={reward_mc.mean():.3f}  std={reward_mc.std():.3f}")
    ax_rew.set_xlabel("Discounted reward return")
    ax_rew.set_ylabel("Count")
    ax_ent.hist(entropy_mc, bins=40, color="tab:purple", alpha=0.8)
    ax_ent.set_title(f"Entropy bonus  mean={entropy_mc.mean():.3f}  std={entropy_mc.std():.3f}")
    ax_ent.set_xlabel("Discounted −α·logπ return")
    ax_ent.set_ylabel("Count")
    term_labels = ["abnormal", "failure", "success"]
    ax_term.bar(term_labels, term_counts, color=["tab:red", "tab:gray", "tab:green"])
    for i, c in enumerate(term_counts):
        ax_term.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    ax_term.set_title(f"Terminations  (success {term_counts[2]}/{n_g} = {term_counts[2] / n_g:.1%})")
    ax_term.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

    # ---- Peg x-y trajectories: one line per env on a shared workspace canvas ----
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    max_lines = 256
    env_ids_plot = np.arange(n_g) if n_g <= max_lines else np.linspace(0, n_g - 1, max_lines).astype(int)
    term_np = d["term_type"]
    succ_ids = [e for e in env_ids_plot if term_np[e] == 2]
    fail_ids = [e for e in env_ids_plot if term_np[e] != 2]
    for ids, colour in ((fail_ids, "tab:red"), (succ_ids, "tab:green")):
        for e in ids:
            ax2.plot(peg_traj[:, e, 0], peg_traj[:, e, 1], color=colour, lw=0.5, alpha=0.35)
    ax2.plot([], [], color="tab:green", lw=1.5, label=f"success ({len(succ_ids)})")
    ax2.plot([], [], color="tab:red", lw=1.5, label=f"failure ({len(fail_ids)})")
    end_xy = []
    for e in env_ids_plot:
        valid = np.flatnonzero(~np.isnan(peg_traj[:, e, 0]))
        if valid.size:
            end_xy.append(peg_traj[valid[-1], e])
    if end_xy:
        end_xy = np.asarray(end_xy)
        ax2.scatter(end_xy[:, 0], end_xy[:, 1], color="gold", s=18, zorder=6,
                    edgecolors="black", linewidths=0.3, label="end")
    ax2.scatter(peghole_xy[0], peghole_xy[1], marker="*", color="black", s=250, zorder=8, label="peghole")
    ax2.scatter(peg_traj[0, 0, 0], peg_traj[0, 0, 1], color="tab:blue", s=60, zorder=9,
                edgecolors="white", linewidths=0.5, label="start (s0)")
    ax2.set_xlim(peg_xlim)
    ax2.set_ylim(peg_ylim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (env-local, m)")
    ax2.set_ylabel("y (env-local, m)")
    ax2.set_title(
        f"{ckpt_name}\n"
        f"success {term_counts[2]}/{n_g} = {term_counts[2] / n_g:.1%}"
        f"  —  iter {reset_idx}  ({len(env_ids_plot)}/{n_g} envs plotted, T={peg_traj.shape[0]})",
        fontsize=10,
    )
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(peg_xy_path, dpi=100)
    plt.close(fig2)

    if q_traj_path is None:
        return
    # ---- Leader env: critic Q vs soft return-to-go over the trajectory ----
    q0_plot, g0_plot = d["q_lead"].copy(), d["g_lead"].copy()
    inact0 = ~d["act_lead"].astype(bool)
    q0_plot[inact0] = np.nan
    g0_plot[inact0] = np.nan
    ts = np.arange(q0_plot.shape[0])
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    ax3.plot(ts, q0_plot, color="tab:blue", lw=2.0, label="Q(s_t, a_t) (critic)")
    ax3.plot(ts, g0_plot, color="tab:orange", lw=2.0, ls="--", label="soft return-to-go G_t (MC)")
    q_ppo_lead = d.get("q_ppo_lead")
    if q_ppo_lead is not None and np.ndim(q_ppo_lead) > 0:
        q_ppo0_plot = np.array(q_ppo_lead, dtype=float)
        q_ppo0_plot[inact0] = np.nan
        ax3.plot(ts, q_ppo0_plot, color="tab:green", lw=2.0, ls=":", label="Q(s_t, a*_ppo)")
    ax3.set_xlabel("timestep t")
    ax3.set_ylabel("value")
    ax3.set_ylim(0, v_max)
    ax3.set_title(f"env {lead}: critic Q vs MC return-to-go — iter {reset_idx}  (MC={soft_mc[0]:.2f})")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    fig3.savefig(q_traj_path, dpi=100)
    plt.close(fig3)


def save_reset_scatter(reset_records: list[dict], out_path: str, title_prefix: str = "") -> None:
    """3D scatter of each reset's initial peg xyz colored by its success rate (RdYlGn, 0..1)."""
    xyz = np.array([r["init_peg_xyz"] for r in reset_records])
    rates = np.array([r["success_rate"] for r in reset_records])
    hole = np.array(reset_records[0]["peghole_xyz"])
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rates, cmap="RdYlGn", vmin=0.0, vmax=1.0, s=22, alpha=0.85)
    ax.scatter(hole[0], hole[1], hole[2], c="blue", s=200, marker="*", label="peghole (env 0)", zorder=10)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=f"success rate over {reset_records[0]['n_env']} trajectories")
    ax.set_xlabel("x (m, env-local)")
    ax.set_ylabel("y (m, env-local)")
    ax.set_zlabel("z (m, env-local)")
    ax.set_title(
        f"{title_prefix}Per-reset success rate  ({len(reset_records)} resets, mean {rates.mean():.1%}, "
        f"min {rates.min():.0%}, max {rates.max():.0%})"
    )
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def concat_labeled(images: list[str], labels: list[str], out_path: str, header: int = 90, gap: int = 4) -> None:
    """Side-by-side concat of equal-size PNGs with a bold label above each and a gray divider."""
    from PIL import Image, ImageDraw, ImageFont

    imgs = [Image.open(p).convert("RGB") for p in images]
    w, h = imgs[0].size
    out = Image.new("RGB", (w * len(imgs) + gap * (len(imgs) - 1), h + header), "white")
    draw = ImageDraw.Draw(out)
    font_path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf")
    size = 34
    font = ImageFont.truetype(font_path, size)
    while size > 16 and max(font.getlength(t) for t in labels) > w - 24:
        size -= 2
        font = ImageFont.truetype(font_path, size)
    for i, (img, text) in enumerate(zip(imgs, labels)):
        x0 = i * (w + gap)
        out.paste(img, (x0, header))
        if i > 0:
            draw.rectangle([x0 - gap, 0, x0 - 1, h + header], fill=(128, 128, 128))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (w - tw) // 2, (header - th) // 2 - bbox[1]), text, fill=(20, 20, 20), font=font)
    out.save(out_path)


def render_rollout_npz(npz_path: str, peg_xlim, peg_ylim) -> int:
    """Render every group stored in one deferred-rollout ``.npz`` next to it; returns #groups."""
    z = np.load(npz_path, allow_pickle=True)
    plots_dir = os.path.dirname(npz_path)
    ckpt_name = str(z["ckpt_name"])
    n_groups = int(z["n_groups"])
    shared = {k: z[k] for k in ("q_support", "centers", "bin_width", "v_min", "v_max", "alpha")}
    for g in range(n_groups):
        d = dict(shared)
        for k in ("q_dists", "qt_dists", "soft_mc", "reward_mc", "entropy_mc", "mc_pmf", "term_counts",
                  "term_type", "peg_traj", "peghole_xy", "q_lead", "g_lead", "act_lead", "reset_idx", "n_g", "lead"):
            d[k] = z[f"g{g}_{k}"]
        d["q_ppo_lead"] = z[f"g{g}_q_ppo_lead"] if f"g{g}_q_ppo_lead" in z.files else None
        reset_idx = int(d["reset_idx"])
        render_group(
            d,
            os.path.join(plots_dir, f"iter{reset_idx:04d}_q_vs_mc.png"),
            os.path.join(plots_dir, f"iter{reset_idx:04d}_peg_xy.png"),
            os.path.join(plots_dir, f"iter{reset_idx:04d}_q_over_traj.png"),
            peg_xlim, peg_ylim, ckpt_name,
        )
    return n_groups
