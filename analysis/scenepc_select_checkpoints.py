#!/usr/bin/env python3
"""Select converged checkpoints for the pc_4_resets (ScenePC-v0) seeds.

The ScenePC-v0 runs trained on slurm with requeues, so each seed has many wandb runs
(one per requeue) and each run keeps only its final ``model_<iter>.pt``. Later requeues
sometimes collapsed (final success ~0), so "latest run, latest checkpoint" — the rule
``wandb_crawl_configs.py`` uses — picks a bad policy. Instead, per seed:

  1. sort the seed's runs by creation time;
  2. for each run, read the ``Metrics/task_0_success_rate`` history and the success
     near each of its ``model_<iter>.pt`` checkpoints;
  3. pick the LATEST run with a checkpoint whose nearby success >= --threshold, and
     within it the highest-iter such checkpoint.

Writes ``analysis/scenepc_checkpoint_selection.json`` (seed -> run/step/ckpt/success),
downloads each chosen checkpoint to ``/tmp/eval_ckpts/<run_id>/`` (where
``eval_seed_diversity.prep_checkpoint`` expects it) plus a durable copy under
``analysis/scenepc_ckpts/``, and updates the project's entry in
``analysis/wandb_configs.json`` so the existing dispatchers pick up the chosen
(rather than latest) checkpoints.

Usage:
    /usr/bin/python3 analysis/scenepc_select_checkpoints.py --no_write   # inspect only
    /usr/bin/python3 analysis/scenepc_select_checkpoints.py             # select + download + merge
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict

# Repo root has a local `wandb/` run-output dir that shadows the installed package.
sys.path = [p for p in sys.path if p not in ("", ".", os.getcwd())]

from wandb import Api  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ENTITY = "learning-to-improve"
PROJECT = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-v0"
DESCRIPTOR = "pc_4_resets"
METRIC = "Metrics/task_0_success_rate"
NAME_RE = re.compile(r"^(\d+)_(.+)_s(\d+)$")
CKPT_RE = re.compile(r"^model_(\d+)\.pt$")

HOST_CKPT_DIR = "/tmp/eval_ckpts"                      # prep_checkpoint staging dir
DURABLE_DIR = os.path.join(HERE, "scenepc_ckpts")
SELECTION_JSON = os.path.join(HERE, "scenepc_checkpoint_selection.json")
CONFIGS_JSON = os.path.join(HERE, "wandb_configs.json")


def run_history(run):
    """[(step, success), ...] sampled history of METRIC, sorted by step."""
    rows = run.history(keys=[METRIC], samples=2000, pandas=False)
    pts = [(int(r["_step"]), float(r[METRIC])) for r in rows
           if r.get(METRIC) is not None and r.get("_step") is not None]
    return sorted(pts)


def success_near(pts, it, back=10, fwd=30):
    """MIN success in steps [it-back, it+fwd] (None if no points there).

    The min (not max) matters: several runs collapse off a cliff mid-run (success
    drops 0.98 -> 0.00 within ~2 iterations, e.g. after an in-place slurm requeue
    picked up mutated huggingface assets). A checkpoint saved just after the cliff
    is broken even though a +-window max still sees pre-cliff values. Requiring the
    window min >= threshold keeps only checkpoints with sustained success around
    (and shortly after) the save point.
    """
    vals = [v for s, v in pts if it - back <= s <= it + fwd]
    return min(vals) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--no_write", action="store_true", help="Print the per-run traces only.")
    args = p.parse_args()

    api = Api()
    by_seed = defaultdict(list)
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        m = NAME_RE.match(run.name)
        if m and m.group(2) == DESCRIPTOR:
            by_seed[int(m.group(3))].append(run)

    selection = {}
    for seed in sorted(by_seed):
        runs = sorted(by_seed[seed], key=lambda r: r.created_at or "")
        print(f"\n=== seed {seed}  ({len(runs)} runs) ===")
        candidates = []  # (created_at, run, ckpt_name, ckpt_iter, success)
        for run in runs:
            pts = run_history(run)
            ckpts = sorted(
                ((f.name, int(CKPT_RE.match(f.name).group(1))) for f in run.files()
                 if CKPT_RE.match(f.name)),
                key=lambda t: t[1],
            )
            span = f"steps {pts[0][0]}..{pts[-1][0]}" if pts else "no history"
            peak = max((v for _, v in pts), default=float("nan"))
            cstr = []
            for name, it in ckpts:
                s = success_near(pts, it)
                cstr.append(f"{name}@{'?' if s is None else f'{s:.3f}'}")
                if s is not None and s >= args.threshold:
                    candidates.append((run.created_at, run, name, it, s))
            print(f"  {run.id:<10} {run.created_at}  {span:<22} peak={peak:.3f}  ckpts: {', '.join(cstr) or '(none)'}")

        if not candidates:
            print(f"  [WARN] seed {seed}: no checkpoint with success >= {args.threshold} -- skipping")
            continue
        # latest qualifying run; within it the highest-iter qualifying checkpoint
        created, run, ckpt, it, succ = max(candidates, key=lambda c: (c[0], c[3]))
        n_dup = len(runs)
        print(f"  -> CHOSEN: run {run.id} ({created})  {ckpt}  success@ckpt={succ:.3f}")
        selection[seed] = {
            "seed": seed,
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": created,
            "n_dup_runs": n_dup,
            "latest_checkpoint": ckpt,
            "latest_checkpoint_iter": it,
            "success_at_checkpoint": succ,
            "step": it,
        }

    if args.no_write:
        return

    # download checkpoints (prep staging + durable copy)
    os.makedirs(DURABLE_DIR, exist_ok=True)
    for seed, e in selection.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{e['run_id']}")
        host_dir = os.path.join(HOST_CKPT_DIR, e["run_id"])
        os.makedirs(host_dir, exist_ok=True)
        host_path = os.path.join(host_dir, e["latest_checkpoint"])
        if not os.path.isfile(host_path):
            run.file(e["latest_checkpoint"]).download(root=host_dir, replace=True)
        durable = os.path.join(DURABLE_DIR, f"s{seed}__{e['run_id']}__{e['latest_checkpoint']}")
        shutil.copy2(host_path, durable)
        e["host_checkpoint"] = host_path
        e["durable_checkpoint"] = durable
        print(f"downloaded seed {seed}: {durable}")

    with open(SELECTION_JSON, "w") as f:
        json.dump({str(s): e for s, e in sorted(selection.items())}, f, indent=2)
    print(f"\nWrote {SELECTION_JSON}")

    # merge into wandb_configs.json so build_jobs() uses the chosen checkpoints
    cfgs = json.load(open(CONFIGS_JSON))
    key = f"{PROJECT}::{DESCRIPTOR}"
    cfgs[key] = {
        str(s): {k: e[k] for k in
                 ("seed", "run_id", "run_name", "state", "created_at", "n_dup_runs",
                  "latest_checkpoint", "latest_checkpoint_iter")}
        for s, e in sorted(selection.items())
    }
    with open(CONFIGS_JSON, "w") as f:
        json.dump(cfgs, f, indent=2)
    print(f"Updated {CONFIGS_JSON} [{key}]")


if __name__ == "__main__":
    main()
