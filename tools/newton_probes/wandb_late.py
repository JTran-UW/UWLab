import wandb
api = wandb.Api()
ids = {"physx": "b1092vqk", "clamp0": "phcj5cq7", "clamp1": "8dha4e7a"}
keys = ["Metrics/task_command/end_of_episode_success_rate", "Metrics/task_command/task_0_success_rate", "Metrics/task_command/task_1_success_rate",
        "Episode_Termination/abnormal_robot", "Loss/entropy", "Loss/learning_rate", "Loss/value_function", "Policy/mean_noise_std", "Train/mean_episode_length"]
data = {}
for n, i in ids.items():
    r = api.run(f"profjat-university-of-washington/isaaclab/{i}")
    data[n] = sorted((d for d in r.history(samples=5000, pandas=False) if "_step" in d), key=lambda d: d["_step"])
for s in [400, 500, 600, 700, 750, 800]:
    print(f"--- it {s}")
    for n, rows in data.items():
        sub = [d for d in rows if d["_step"] <= s]
        if not sub: continue
        row = sub[-1]
        print(f"  {n:7s} {int(row['_step']):4d}", " ".join(f"{k.split('/')[-1][:9]}={row[k]:.3g}" for k in keys if row.get(k) is not None))
