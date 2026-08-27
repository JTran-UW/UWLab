# Hyak Cluster Reference

## Key Paths

| Location | Path |
|---|---|
| SLURM logs | `/mmfs1/gscratch/scrubbed/jtran/slurm_logs/` |
| UWLab code (per-job snapshot) | `/mmfs1/gscratch/scrubbed/jtran/uwlab_<timestamp>/` |
| Isaac Sim cache | `/mmfs1/gscratch/scrubbed/jtran/docker-isaac-sim/` |
| SIF container | `/mmfs1/gscratch/scrubbed/jtran/uw-lab-base.tar` |
| Holosoma source | `/mmfs1/gscratch/scrubbed/jtran/holosoma/` |
| Holosoma pip deps | `/mmfs1/gscratch/scrubbed/jtran/holosoma-deps/` |
| Expert replay buffer | `/mmfs1/gscratch/scrubbed/jtran/expert_rb/expert_transitions.pt` |
| SLURM log dir | `/mmfs1/gscratch/scrubbed/jtran/slurm_logs/` |

Inside the Singularity container, `/mmfs1` is bind-mounted at `/mmfs1`, so GPFS paths work directly.

---

## Submitting Jobs

### Basic pattern
```bash
NODES=1 GPUS_PER_NODE=4 ACCOUNT=weirdlab-ckpt bash docker/cluster/cluster_interface.sh job <train.py args>
```

**Do NOT put a `--` separator before the train.py args** (confirmed 2026-08-26, jobs
38916169/38916182/38918384): `job_args` is forwarded verbatim into the container, and a leading
`--` makes train.py's argparse treat everything after it as positionals — `--task` silently
becomes None and the job dies at hydra registration with
`AttributeError: 'NoneType' object has no attribute 'split'` right after Kit startup.

### Common job configs

**4-GPU ckpt (peg task, full)**
```bash
NODES=1 GPUS_PER_NODE=4 ACCOUNT=weirdlab-ckpt bash docker/cluster/cluster_interface.sh job -- \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-v0 \
    --num_envs 4096 --logger wandb --headless \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

**2-GPU gpu-l40s (easy task, with expert buffer)**
```bash
NODES=1 GPUS_PER_NODE=2 ACCOUNT=weirdlab PARTITION=gpu-l40s bash docker/cluster/cluster_interface.sh job -- \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-v0 \
    --num_envs 2048 --logger wandb --headless \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole \
    --expert_transitions /mmfs1/gscratch/scrubbed/jtran/expert_rb/expert_transitions.pt
```

### Overridable env vars

| Var | Default | Purpose |
|---|---|---|
| `NODES` | `1` | Number of nodes |
| `GPUS_PER_NODE` | `1` | GPUs per node |
| `ACCOUNT` | `weirdlab` | SLURM account |
| `PARTITION` | `ckpt` | SLURM partition |
| `CONSTRAINT` | `h200\|l40\|l40s\|a100` | GPU type filter |
| `TIME` | `24:00:00` | Wall time limit |
| `CPUS_PER_TASK` | `6` | CPUs per GPU |
| `MEM_PER_GPU` | `60G` | Memory per GPU |
| `CLUSTER_PYTHON_EXECUTABLE` | `scripts/reinforcement_learning/holosoma/train.py` | Script to run |

---

## GPU VRAM on Hyak

| GPU | VRAM | Notes |
|---|---|---|
| a40 / rtx6k | ~24 GB | Too small for 4096 envs — causes OOM |
| a100 | 40–80 GB | Safe |
| l40 / l40s | 48 GB | Safe, good throughput |
| h200 | 80 GB | Best, rarely available |

The peg task with 4096 envs uses ~22 GB per GPU — requires 40+ GB cards.

---

## Partitions

| Partition | Account | Notes |
|---|---|---|
| `ckpt` | `weirdlab-ckpt` | Preemptible, auto-requeues, free |
| `gpu-l40s` | `weirdlab` or `cse` | Dedicated, faster queue, costs allocation |
| `gpu-l40` | `weirdlab` or `cse` | L40 GPUs (48 GB), dedicated |
| `gpu-a100` | `cse` | A100 GPUs (40–80 GB), dedicated |
| `gpu-a40` | `weirdlab` | 24 GB GPUs — avoid for large jobs |

---

## Monitoring

```bash
# Check job status
ssh klone-login 'squeue -u profjat'

# Cancel a job
ssh klone-login 'scancel <job_id>'

# Check GPU availability
ssh klone-login 'sinfo -p ckpt --format="%N %G %T" | grep idle'
```

Use `/cluster-logs latest` slash command to read SLURM output/error logs.
Use `/cluster-status` slash command to check queue status.

---

## Requeue Behavior

Jobs on `ckpt` auto-requeue on preemption (SIGTERM) or time limit (SIGUSR1@30s).
On requeue: model checkpoint is restored, but WandB creates a **new run** (no run ID continuity).
The replay buffer is re-filled from scratch after each requeue.

---

## Expert Replay Buffer

Generated locally with:
```bash
python scripts/reinforcement_learning/holosoma/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-v0 \
    --num_envs 1024 \
    --checkpoint peg_state_rl_expert_seed42.pt \
    --record_transitions 1000 \
    --transitions_output expert_rb/expert_transitions.pt \
    --headless \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Upload once to GPFS:
```bash
ssh klone-login "mkdir -p /mmfs1/gscratch/scrubbed/jtran/expert_rb"
rsync -ahP expert_rb/expert_transitions.pt klone-login:/mmfs1/gscratch/scrubbed/jtran/expert_rb/
```

`expert_rb/` is excluded from code rsync via `.dockerignore`.
