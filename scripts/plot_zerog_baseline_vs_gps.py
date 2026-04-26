"""Plot ZeroG Baseline vs GPS: partial assembly & anywhere success rates side by side."""

import wandb
import matplotlib.pyplot as plt
import numpy as np

ENTITY = "patyin"

CONFIGS = {
    "Baseline": {
        "project": "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-Baseline-v0",
        "color": "#1f77b4",
    },
    "GPS": {
        "project": "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-v0",
        "color": "#ff7f0e",
    },
}

# task_0 = ZeroGAnywhere, task_1 = ZeroGPartialAssembly
METRICS = [
    ("Metrics/task_1_success_rate", "Partial Assembly Success Rate"),
    ("Metrics/task_0_success_rate", "Object Anywhere Success Rate"),
]

api = wandb.Api()


def get_combined_data(project, metric):
    """Pull metric from all runs in project, combine and deduplicate by step."""
    runs = api.runs(f"{ENTITY}/{project}", per_page=20)
    all_rows = []
    for run in runs:
        history = run.scan_history(keys=[metric, "_step"], page_size=1000)
        for r in history:
            if metric in r and r[metric] is not None:
                all_rows.append((r["_step"], r[metric]))
    if not all_rows:
        return None, None
    # Sort by step, deduplicate (keep last value per step)
    all_rows.sort(key=lambda x: x[0])
    step_dict = dict(all_rows)
    steps = np.array(sorted(step_dict.keys()))
    values = np.array([step_dict[s] for s in steps])
    return steps, values


def smooth(values, steps, window_frac=50):
    window = max(1, len(values) // window_frac)
    if window > 1:
        kernel = np.ones(window) / window
        values_smooth = np.convolve(values, kernel, mode="valid")
        steps_smooth = steps[window - 1 :]
    else:
        values_smooth = values
        steps_smooth = steps
    return steps_smooth, values_smooth


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (metric, title) in zip(axes, METRICS):
    for label, cfg in CONFIGS.items():
        steps, values = get_combined_data(cfg["project"], metric)
        if steps is None:
            print(f"No data for {label} / {metric}")
            continue
        print(f"{label} / {title}: {len(steps)} points, range [{steps[0]}, {steps[-1]}], final={values[-1]:.4f}")
        steps_s, values_s = smooth(values, steps)
        ax.plot(steps_s, values_s, label=label, linewidth=2, color=cfg["color"])
        # Light raw data
        ax.plot(steps, values, alpha=0.15, color=cfg["color"], linewidth=0.5)
    ax.set_xlabel("Updates", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

fig.suptitle("ZeroG State-Based: Baseline vs GPS", fontsize=14, y=1.02)
fig.tight_layout()

out_path = "scripts/zerog_baseline_vs_gps.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
