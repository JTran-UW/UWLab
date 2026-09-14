import wandb
api = wandb.Api()
r = api.run("profjat-university-of-washington/isaaclab/b1092vqk")
keys = ["Metrics/task_command/end_of_episode_success_rate", "Episode_Termination/abnormal_robot", "Loss/entropy",
        "Loss/learning_rate", "Train/mean_reward", "Train/mean_episode_length"]
rows = sorted((d for d in r.history(samples=5000, pandas=False) if "_step" in d), key=lambda d: d["_step"])
print("==== PhysX control", r.id, r.name)
for s in [5, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400]:
    sub = [d for d in rows if d["_step"] <= s]
    if not sub: continue
    row = sub[-1]
    print(f"{int(row['_step']):4d}", " ".join(f"{k.split('/')[-1][:10]}={row[k]:.4g}" for k in keys if row.get(k) is not None))
