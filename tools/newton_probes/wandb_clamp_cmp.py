import wandb
api = wandb.Api()
runs = [r for r in api.runs("profjat-university-of-washington/isaaclab", order="-created_at")[:8] if "newton_control_clamp" in r.name]
keys = ["Metrics/task_command/end_of_episode_success_rate", "Metrics/task_command/task_3_success_rate",
        "Episode_Termination/abnormal_robot", "Policy/mean_noise_std", "Loss/entropy", "Loss/value_function",
        "Loss/surrogate", "Loss/learning_rate", "Perf/total_fps", "Train/mean_reward", "Train/mean_episode_length"]
for r in runs:
    print("====", r.id, r.name, r.state)
    rows = sorted((d for d in r.history(samples=5000, pandas=False) if "_step" in d), key=lambda d: d["_step"])
    have = [k for k in keys if any(k in d for d in rows)]
    print("cols:", [k.split("/")[-1] for k in have])
    for s in [5, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]:
        sub = [d for d in rows if d["_step"] <= s]
        if not sub: continue
        row = sub[-1]
        print(f"{int(row['_step']):4d}", " ".join(f"{k.split('/')[-1][:10]}={row[k]:.4g}" for k in have if row.get(k) is not None))
