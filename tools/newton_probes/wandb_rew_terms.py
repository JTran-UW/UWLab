import wandb
api = wandb.Api()
ids = {"physx": "b1092vqk", "clamp0": "phcj5cq7", "clamp1": "8dha4e7a"}
data = {}
for n, i in ids.items():
    r = api.run(f"profjat-university-of-washington/isaaclab/{i}")
    data[n] = sorted((d for d in r.history(samples=5000, pandas=False) if "_step" in d), key=lambda d: d["_step"])
allkeys = sorted({k for rows in data.values() for d in rows for k in d if k.startswith("Episode_Reward/") or k.startswith("Episode_Termination/")})
for s in [150, 300, 350]:
    print(f"--- it {s}")
    print(f"  {'term':45s} " + " ".join(f"{n:>9s}" for n in data))
    for k in allkeys:
        vals = []
        for n, rows in data.items():
            sub = [d for d in rows if d["_step"] <= s and d.get(k) is not None]
            vals.append(sub[-1][k] if sub else float("nan"))
        print(f"  {k[:45]:45s} " + " ".join(f"{v:9.4f}" for v in vals))
