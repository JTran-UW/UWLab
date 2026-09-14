import wandb, json, sys
api = wandb.Api()
ent, proj = "profjat-university-of-washington", "isaaclab"
runs = {rid: api.run(f"{ent}/{proj}/{rid}") for rid in sys.argv[1:]}
for rid, r in runs.items():
    print(f"== {rid}: name={r.name} state={r.state} created={r.created_at} runtime={r.summary.get('_runtime')}s steps={r.summary.get('_step')}")
    cfg = r.config
    for k in ("task", "num_envs", "seed", "max_iterations", "algorithm", "policy", "sim", "scene", "actions", "events"):
        if k in cfg: print(f"   cfg.{k}: {json.dumps(cfg[k], default=str)[:600]}")
    print("   top-level cfg keys:", list(cfg.keys())[:40])
# config diff
a, b = [runs[r].config for r in sys.argv[1:3]]
def flat(d, p=""):
    out = {}
    for k, v in (d or {}).items():
        kk = f"{p}.{k}" if p else k
        if isinstance(v, dict): out.update(flat(v, kk))
        else: out[kk] = v
    return out
fa, fb = flat(a), flat(b)
print("\n== CONFIG DIFF (a=%s, b=%s)" % tuple(sys.argv[1:3]))
for k in sorted(set(fa) | set(fb)):
    if fa.get(k) != fb.get(k) and not any(t in k for t in ("wandb", "log_dir", "run_name", "experiment", "time", "date")):
        print(f"   {k}: {str(fa.get(k))[:90]}  |  {str(fb.get(k))[:90]}")
