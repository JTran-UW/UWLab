# Cluster Onboarding (Hyak / Tillicum)

How to go from a fresh checkout to running distributed training jobs on the UW
clusters. The same flow works for **Hyak (klone)** and **Tillicum** — just swap
`--cluster hyak` for `--cluster tillicum`. Examples below use Hyak.

## How it works (30-second mental model)

Training on the cluster runs inside an **Apptainer/Singularity container** built
from the UWLab Docker image (Isaac Sim + UWLab deps). You build that image
**once** (and rebuild only when dependencies change), convert it to a `.sif`, and
ship it to the cluster. From then on, each job:

1. **rsyncs your latest local code** to a timestamped dir on the cluster,
2. submits a SLURM job that runs the `.sif` on a GPU node,
3. **bind-mounts your synced code over the baked-in copy** — so day-to-day code
   edits do *not* require rebuilding the image.

You only rebuild + re-push the image when **dependencies** change (a new pip/apt
package, an Isaac Sim version bump). The container build is completely
independent of any local conda / `isaac_start.sh` workflow — it does not touch
your local env.

Pipeline: `cluster_interface.sh` (local) → SSH → `submit_job_slurm_<cluster>.sh`
(login node, emits `sbatch`) → `run_singularity.sh` (compute node, launches the
container under `torchrun`).

---

## One-time setup

### 1. SSH access to the cluster

The scripts SSH/scp/rsync to a host **alias** (`klone-login` for Hyak,
`tillicum` for Tillicum). Add it to `~/.ssh/config`:

```sshconfig
Host klone-login
    HostName klone.hyak.uw.edu
    User <your-uw-netid>
    # Reuse one authenticated connection so you only do Duo ONCE per job.
    # A single `job` opens ~5 connections; without this you'd get ~5 Duo prompts.
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Hyak enforces **Duo 2FA**. Set up an SSH key trusted by klone so submission is
non-interactive:

```bash
ssh-keygen -t ed25519            # if you don't already have a key
ssh-copy-id klone-login          # registers your public key on klone
ssh klone-login "echo ok"        # authenticate once; ControlPersist keeps it warm
```

Verify a passwordless (cached) connection works before going further.

### 2. Generate your user config (`.env.user`)

`.env.user` holds your NetID + lab account and is **git-ignored** (never commit
it). Generate it with the interactive wizard:

```bash
./docker/cluster/setup.sh
```

It prompts for:
- **UW NetID** (e.g. `yandabao`)
- **Hyak lab group / SLURM account** (e.g. `weirdlab`)
- (optional) SLURM defaults — partition, CPUs, mem, constraint, time

It derives all cluster paths automatically, e.g. on Hyak:
`/gscratch/<account>/<netid>/uwlab` (code+logs),
`.../docker-isaac-sim` (cache), `.../` (the `.sif`).

> Manual alternative: `cp docker/cluster/.env.user.template docker/cluster/.env.user`
> then fill in the two `<your-...>` placeholders. The interface refuses to run
> while placeholders remain.

### 3. WANDB API key (only if using `--logger wandb`)

Export it in your shell profile so it's forwarded to the cluster:

```bash
echo 'export WANDB_API_KEY="<your-key>"' >> ~/.bashrc
source ~/.bashrc
```

The submit script fails fast locally if `--logger wandb` is passed without this
set.

### 4. Validate

```bash
./docker/cluster/cluster_interface.sh --cluster hyak validate base
```

This checks your config resolves and that the `uw-lab-base.sif` image already
exists on the cluster. If it reports the image is missing, do the build + push
below (or copy a labmate's `.sif` — see shortcut at the end).

---

## Building the container image

> Needs **Docker** + access to NVIDIA NGC (the base image
> `nvcr.io/nvidia/isaac-sim:5.1.0` is pulled from a private registry).
> First time only: `docker login nvcr.io` (username `$oauthtoken`, password =
> your NGC API key from https://ngc.nvidia.com).

Build the Docker image (`uw-lab-base:latest`):

```bash
python docker/container.py start base
```

This builds the image per `docker/Dockerfile.base` (Isaac Sim base + apt deps +
UWLab code + `uwlab.sh --install` of all Python deps) and creates a detached
container. The build is large and can take a while the first time.

Useful related commands:
```bash
python docker/container.py enter base   # shell into the running container
python docker/container.py stop base    # stop + remove the container
```

> The image bakes in a copy of the repo, but jobs bind-mount your freshly-synced
> code over it — so you only rebuild when **dependencies** change, not for code
> edits.

---

## Pushing the image to the cluster

> Needs **Apptainer** installed locally:
> ```bash
> sudo apt update && sudo apt install -y software-properties-common
> sudo add-apt-repository -y ppa:apptainer/ppa
> sudo apt update && sudo apt install -y apptainer
> ```

```bash
./docker/cluster/cluster_interface.sh --cluster hyak push base
```

This converts the Docker image to a single `uw-lab-base.sif` (via
`apptainer build --fakeroot`), tars it, and scp's it to your `CLUSTER_SIF_PATH`
on the cluster (`/gscratch/<account>/<netid>/uw-lab-base.tar` on Hyak). Only
required on first setup and whenever the image changes.

---

## Launching jobs

```bash
./docker/cluster/cluster_interface.sh --cluster hyak job base \
    --gpus 4 --partition gpu-l40s --account weirdlab \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-PointCloud-128-SharedEncoder-v0 \
    --num_envs 8192 --logger wandb --headless --distributed
```

Tillicum equivalent:
```bash
./docker/cluster/cluster_interface.sh --cluster tillicum job base \
    --gpus 8 \
    --task <TASK> --num_envs 8192 --logger wandb --headless --distributed
```

**Cluster-level flags** (consumed by the launcher, not the training script):

| Flag | Meaning |
|------|---------|
| `--gpus N` | GPUs **per node** |
| `--nodes N` | number of nodes (default 1) |
| `--partition P` | SLURM partition (e.g. `gpu-l40s`, `gpu-a40`, `ckpt`) |
| `--account A` | SLURM account (e.g. `weirdlab`) |
| `--time HH:MM:SS` | wall-clock limit (default 24:00:00) |
| `--qos Q` | QOS (e.g. `ckpt-gpu` for scavenger GPU jobs) |

Everything **after** those (e.g. `--task`, `--num_envs`, `--logger`,
`--headless`, `--distributed`, and any `env.*` Hydra overrides) is forwarded
verbatim to `scripts/reinforcement_learning/rsl_rl/train.py`.

### Settings reference (CLI flags vs. environment variables)

Only the six flags above are parsed by `cluster_interface.sh`. The remaining
SLURM knobs (**CPUs, memory, GPU-type constraint, log dir**) have **no CLI flag**
— set them as environment variables in front of the command, e.g.:

```bash
MEM_PER_GPU=80G CPUS_PER_TASK=8 CONSTRAINT="l40s" \
  ./docker/cluster/cluster_interface.sh --cluster hyak job base --gpus 4 ...
```

**Precedence** (highest wins): exported env var → `HYAK_*` default in
`docker/cluster/.env.user` → hardcoded fallback in `submit_job_slurm_hyak.sh`.
So a one-off uses the env var; to change a default permanently, edit `.env.user`.

| Setting | Env var | `.env.user` default | CLI flag | Fallback | Notes |
|---|---|---|---|---|---|
| Nodes | `NODES` | — | `--nodes N` | 1 | |
| GPUs **per node** | `GPUS_PER_NODE` | — | `--gpus N` | 1 | total GPUs = nodes × this |
| Account | `ACCOUNT` | `HYAK_ACCOUNT` | `--account A` | `weirdlab` | |
| Partition | `PARTITION` | `HYAK_PARTITION` | `--partition P` | `gpu-a40` | comma-list allowed (`a,b,ckpt`) |
| QOS | `QOS` | — | `--qos Q` | *(empty)* | needed for ckpt GPU (`ckpt-gpu`) |
| Wall-clock | `TIME` | `HYAK_TIME` | `--time HH:MM:SS` | `24:00:00` | |
| **CPUs per GPU** | `CPUS_PER_TASK` | `HYAK_CPUS_PER_TASK` | *(none)* | `6` | `cpus-per-task = gpus × this` |
| **Mem per GPU** | `MEM_PER_GPU` | `HYAK_MEM_PER_GPU` | *(none)* | `60G` | total mem = gpus × this; **express in whole GB** (unit is stripped and re-appended as `G`) |
| GPU-type constraint | `CONSTRAINT` | `HYAK_CONSTRAINT` | *(none)* | `h200\|l40\|l40s\|a40` | SLURM feature expr: `\|`=OR, `&`=AND; empty = any GPU |
| SLURM log dir | `SLURM_LOGS_DIR` | `HYAK_SLURM_LOGS_DIR` | *(none)* | `<uwlab>/slurm_logs` | stdout/err `.out`/`.err` land here |

Setting **mem per GPU**, concretely:
```bash
# one-off (4 GPUs × 80G = 320G requested total):
MEM_PER_GPU=80G ./docker/cluster/cluster_interface.sh --cluster hyak job base --gpus 4 ...
# permanent default: edit HYAK_MEM_PER_GPU in docker/cluster/.env.user
```

Non-SLURM env vars: `WANDB_API_KEY` (and any `WANDB_*`) are forwarded into the
container when `--logger wandb` is used. Path/infra vars (`HYAK_LOGIN`,
`HYAK_DIR`, `HYAK_CACHE_DIR`, `HYAK_SIF_PATH`) live in `.env.user` and rarely
need changing. All env vars also apply to `sweep_seeds.sh` (it inherits your
environment), so `MEM_PER_GPU=80G ./sweep_seeds.sh --n 5 -- ...` covers the
whole sweep.

> **`--num_envs` is PER GPU.** Total envs = `num_envs × nodes × gpus`.
> For multi-task, cap at **8K envs/GPU** (16K overflows PhysX `collisionStackSize`).

Single-task variants are selected with Hydra overrides appended to the job args,
e.g. `env.scene.insertive_object=peg env.scene.receptive_object=peghole`.

### Hyak partition priority (weirdlab)

`gpu-l40s` > `gpu-l40` > (cse) `gpu-l40s` > `gpu-a40` > (cse) `gpu-a100` > `ckpt`.
`ckpt` is preemptible but the job auto-requeues (handled by the submit script).
cse partitions require `--time 24:00:00`.

---

## Monitoring & outputs

```bash
ssh klone-login "squeue -u <netid>"          # queue status (Hyak)
ssh tillicum    "squeue -u <netid>"           # Tillicum
```

- **SLURM stdout/err logs:** `/gscratch/<account>/<netid>/slurm_logs/` (Hyak).
- **Training logs / checkpoints:** under `uwlab/logs/` in the persistent
  `CLUSTER_UWLAB_DIR` (`/gscratch/<account>/<netid>/uwlab/logs`), synced back
  from the compute node.
- **W&B:** if `--logger wandb`, runs appear in your W&B project.

---

## Fast path: copy a labmate's image instead of building

Building the Isaac Sim image is heavy and needs Apptainer + NGC. Since the image
only needs rebuilding when **dependencies** change, a recent `.sif` from a
labmate is identical to one you'd build yourself. The SIF path is per-user, so
just copy it into yours (run on the login node):

```bash
ssh klone-login
ls /gscratch/weirdlab/*/uw-lab-base.tar                       # find an existing one
cp /gscratch/weirdlab/<labmate>/uw-lab-base.tar /gscratch/weirdlab/<netid>/
```

Then `validate` should pass and you can submit jobs without building/pushing.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `.env.user ... template placeholders` | Run `setup.sh` or fill in `<your-...>` values. |
| `Permission denied (...keyboard-interactive)` over SSH | Register your key with `ssh-copy-id klone-login`; check `~/.ssh/config` alias. |
| Repeated Duo prompts during one `job` | Add `ControlMaster/ControlPersist` to your SSH config (see setup step 1). |
| `image does not exist on the remote host` | Build + push, or copy a labmate's `.sif`. |
| `--logger wandb ... WANDB_API_KEY is not set` | `export WANDB_API_KEY` in your profile and re-source. |
| Job submitted but no GPU for a long time | Try a higher-priority partition, or `ckpt` (preemptible, auto-requeues). |

See `CLAUDE.md` for task-specific gotchas (`replicate_physics=False`, PhysX
material limits, shared-encoder requirement for point-cloud obs, etc.).
