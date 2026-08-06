"""Measure GPU/host memory as a function of num_envs, to find how many envs fit on a given GPU.

Reports DRIVER-level GPU usage (total - free), not torch's allocator counters: Isaac's physics and
render allocations are invisible to torch, and they dominate here. One process per (task, num_envs)
because the Isaac app can only be launched once per process.

Emits a single machine-readable RESULT line so a sweep can be parsed without scraping the log.
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--steps", type=int, default=20, help="steps to reach steady-state rendering")
AppLauncher.add_app_launcher_args(parser)
# docker/cluster/run_singularity.sh appends --distributed to every job unconditionally, and
# AppLauncher defines no such argument -- argparse would exit(2) before Isaac ever starts. This
# script is single-process, so the flag is simply dropped (same handling as cleanrl's
# vecenv_wrapper.py, where the identical injection killed jobs ~2 min in).
if "--distributed" in sys.argv:
    sys.argv = [a for a in sys.argv if a != "--distributed"]
    print("[launcher] ignoring --distributed (cluster-injected; this probe is single-process)")
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

GIB = 1024**3


def gpu_used_gib():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GIB, total / GIB


def host_rss_gib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / GIB
    return float("nan")


torch.manual_seed(0)
np.random.seed(0)

base_used, gpu_total = gpu_used_gib()

t0 = time.perf_counter()
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg)
env.reset(seed=0)
act = torch.zeros(env.unwrapped.action_space.shape, device="cuda:0")
for _ in range(args.steps):
    env.step(act)
torch.cuda.synchronize()
build_s = time.perf_counter() - t0

used, _ = gpu_used_gib()
host = host_rss_gib()

# steady-state step time, useful for judging whether more envs actually buy throughput
t0 = time.perf_counter()
for _ in range(20):
    env.step(act)
torch.cuda.synchronize()
step_ms = (time.perf_counter() - t0) / 20 * 1000

name = torch.cuda.get_device_name(0)
line = (
    f"RESULT task={args.task} num_envs={args.num_envs} gpu={name.replace(' ', '_')} "
    f"gpu_total_gib={gpu_total:.2f} gpu_used_gib={used:.3f} gpu_baseline_gib={base_used:.3f} "
    f"host_rss_gib={host:.2f} step_ms={step_ms:.2f} build_s={build_s:.1f}"
)
# flush=True is load-bearing: simulation_app.close() below tears the process down in a way that can
# drop a block-buffered stdout, which is exactly how job 38098368 exited 0 with an empty log.
print("\n" + line + "\n", flush=True)
sys.stdout.flush()

# Belt-and-braces: also write the result where it survives regardless of stdout capture. Only
# /workspace/uwlab/logs is bound to persistent GPFS -- the rest of the workspace is node-local
# scratch that run_singularity.sh deletes when the job ends.
try:
    import os

    os.makedirs("logs/env_probe", exist_ok=True)
    tag = os.environ.get("SLURM_JOB_ID", "local")
    with open(f"logs/env_probe/probe_{args.num_envs}env_{tag}.txt", "w") as f:
        f.write(line + "\n")
    print(f"[probe] result also written to logs/env_probe/probe_{args.num_envs}env_{tag}.txt", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[probe] could not write result file: {exc}", flush=True)

env.close()
simulation_app.close()
