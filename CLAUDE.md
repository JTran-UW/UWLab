# UWLab-private (Multi-Task Branch)

IsaacLab-based robotics framework for manipulation research. Built on NVIDIA Isaac Sim + IsaacLab.

**Branch**: `pat/multitask` — Multi-task OmniReset manipulation (UR5e + Robotiq 2F85)

## Project Structure

```
source/
  uwlab/              # Core library (assets, actuators, controllers, envs, managers, utils)
  uwlab_assets/       # Robot configs (franka, franka_xhand, ur5, xarm_leap, spot, etc.)
  uwlab_tasks/        # Task environments (manager-based)
    manipulation/
      omnireset/      # OmniReset manipulation tasks (THIS BRANCH'S FOCUS)
        config/ur5e_robotiq_2f85/
          __init__.py           # Gym registrations
          rl_state_cfg.py       # Single-task scene/obs/reward/event configs + PC ablation configs
          multitask_cfg.py      # Multi-task configs (scene, obs, rewards, events, curriculum)
          agents/rsl_rl_cfg.py  # PPO runner configs (standard + shared encoder)
        mdp/
          rewards.py        # ProgressContext, reward functions
          observations.py   # MeshPointCloud, TaskTypeOneHot, pose obs
          terminations.py   # object_off_table, success conditions
          commands.py        # TaskCommand with per-task thresholds
          events.py          # MultiResetManager (GPS curriculum, per-state sampling)
          success_monitor.py # SuccessMonitor + GPS sampling functions
          curriculums.py     # Curriculum terms
      dexhand/        # Franka+XHand dexterous grasping (pat/hand branch)
      factory_extension/  # Factory assembly tasks
      track_goal/     # Goal tracking tasks
    locomotion/       # Locomotion tasks
  uwlab_rl/           # RL wrappers (rsl_rl, skrl)
    uwlab_rl/rsl_rl/
      actor_critic_encoder.py  # ActorCriticWithEncoder (also in rsl_rl fork)
scripts/
  reinforcement_learning/rsl_rl/  # train.py, play.py, sweep.py
  simplify_mesh.py                # Blender-based mesh simplifier
```

## Environment

- **Conda env**: `patlab` (Python 3.11) + Isaac Sim setup (see below)
- **Python**: 3.11
- **IsaacSim**: 5.1.0 — installed at `/home/yandabao/isaacsim/`
- **Branch**: `pat/multitask`
- **RL library**: `patrickhaoy/rsl_rl` fork (has `ActorCriticWithEncoder` natively)
- **Reference codebase**: `/home/patrickhaoy/research/OctiLab` branch `feature/locomotion-clean`

### Local Launch Setup (yandabao machine)

`patlab` is a minimal Python 3.11 env — it does **not** have Isaac Sim packages installed directly. Isaac Sim packages (isaacsim, torch, omni.\*) are injected via `setup_conda_env.sh`. The correct launch sequence is:

```bash
conda activate patlab
source /home/yandabao/isaacsim/setup_conda_env.sh
cd /home/yandabao/UWLab-patrick-private
python scripts/reinforcement_learning/rsl_rl/train.py ...
```

For `nohup` / non-interactive background jobs, both steps must be explicit since shell init files are not sourced:

```bash
nohup bash -c "
  source /home/yandabao/miniforge3/etc/profile.d/conda.sh
  conda activate patlab
  source /home/yandabao/isaacsim/setup_conda_env.sh
  cd /home/yandabao/UWLab-patrick-private
  CUDA_VISIBLE_DEVICES=<id> python scripts/reinforcement_learning/rsl_rl/train.py ...
" > /tmp/run.log 2>&1 &
```

**Do not use `env_uwlab`** — it points at the UWLab-ICL IsaacLab install which has a broken `isaacsim` reference on this machine.

## Registered Multi-Task Environments

| Env ID | Description |
|--------|-------------|
| `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-v0` | Peg+Leg, original rewards, no curriculum |
| `...-MultiTask-Simplified-v0` | Peg+Leg, simplified rewards (success=50, fail=-25) |
| `...-MultiTask-Curriculum-v0` | Original rewards + coarse GPS curriculum |
| `...-MultiTask-Simplified-Curriculum-v0` | Simplified rewards + coarse GPS curriculum |
| `...-MultiTask-PerStateCurriculum-v0` | Original rewards + per-state GPS (80K partitions) |
| `...-MultiTask-Simplified-PerStateCurriculum-v0` | Simplified rewards + per-state GPS |
| `...-MultiTask-PointCloud-128-SharedEncoder-v0` | 128+128 PC, shared encoder, original rewards |

### Single-Task Ablation Environments

| Env ID | Description |
|--------|-------------|
| `...-State-Baseline-v0` | Pose obs only (43d), no PC |
| `...-State-PointCloud-v0` | 64-pt PC wrist frame (409d) |
| `...-State-PointCloud-BaseFrame-v0` | 64-pt PC base frame (409d) |
| `...-State-PointCloud-128-v0` | 128-pt PC flat MLP (786d) |
| `...-State-PointCloud-SharedEncoder-v0` | 64+64 PC shared encoder |
| `...-State-PointCloud-128-SharedEncoder-v0` | 128+128 PC shared encoder |

### Single-Task Baselines (via Hydra variants)

```bash
# Peg insertion
env.scene.insertive_object=peg env.scene.receptive_object=peghole
# Leg insertion
env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop
# Drawer
env.scene.insertive_object=fbdrawerbottom env.scene.receptive_object=fbdrawerbox
```

## Common Commands

### Training
```bash
# Single-task baseline (local)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Baseline-v0 \
    --num_envs 1 \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole

# DAgger / any env with cameras (local) — must pass --enable_cameras or Isaac Sim will crash
# on "A camera was spawned without the --enable_cameras flag". --video implicitly sets this,
# but plain training runs do not.
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-Pretrained-Weighted-PCTeacher-FullSysidDR-v0 \
    --num_envs 32 --logger wandb --headless --enable_cameras \
    agent.policy.teacher_jit_path=teachers/seed22_sysidenv.pt

# Multi-task (cluster, Hyak)
./docker/cluster/cluster_interface.sh --cluster hyak job base \
    --gpus 4 --partition gpu-l40s --account weirdlab \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-PointCloud-128-SharedEncoder-v0 \
    --num_envs 8192 --logger wandb --headless --distributed

# Multi-task (cluster, Tillicum)
./docker/cluster/cluster_interface.sh --cluster tillicum job base \
    --gpus 8 \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-PointCloud-128-SharedEncoder-v0 \
    --num_envs 8192 --logger wandb --headless --distributed
```

### Play/Eval
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Baseline-v0 \
    --num_envs 1 --checkpoint <path_to_pt> \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

## Code Style

- Line length: 120
- isort for imports (custom section ordering -- see pyproject.toml)
- pre-commit hooks configured (.pre-commit-config.yaml)
- pyright for type checking (basic mode)

## Cluster (Hyak / Tillicum)

Submit jobs via `cluster_interface.sh`. Command is `job`, not `submit`.

```bash
./docker/cluster/cluster_interface.sh --cluster <hyak|tillicum> job base \
    --gpus <N> [--time HH:MM:SS] [--partition P] [--account A] \
    --task <TASK> --num_envs <E> --logger wandb --headless [--distributed]
```

- **`--num_envs` is PER GPU**, not total. Total = num_envs x num_gpus.
- **Hyak GPU priority**: weirdlab L40S > weirdlab L40 > cse L40S > weirdlab A40 > cse A100 > ckpt
- **cse partitions**: always `--time 24:00:00`
- **Monitor**: `ssh klone-login "squeue -u patyin"`, `ssh tillicum "squeue -u patyin"`
- **Docker push** (only when deps change): `./docker/cluster/cluster_interface.sh --cluster hyak push base`

### Launching to the cluster from a git worktree

`cluster_interface.sh` rsyncs the **launch directory** (`-rLh`, honoring `.dockerignore`) to a fresh
timestamped dir and bind-mounts it at `/workspace/uwlab`, where the image editable-installs
`isaaclab_tasks` from `<launch dir>/_isaaclab/IsaacLab/...`. Several required dirs are **untracked**
(`_isaaclab`, `_isaac_sim`, `init_weights`, `teachers`, `models`, `demos`, `eval_ckpts`, `pc`), and
`git worktree add` does **not** copy untracked files — so a fresh worktree lacks them. Launching from
such a worktree makes **every rank crash ~30s in**, before Isaac boots, with
`ModuleNotFoundError: No module named 'isaaclab_tasks'`. The failure is silent-looking: SLURM logs it
`COMPLETED 0:0` (Elapsed ~00:00:35), the `.out` ends in `There was an error running python`, and
nothing ever reaches `wandb.init` — so no run appears.

**Before the first cluster launch from a new worktree**, symlink the vendored dirs the run needs.
rsync's `-L` dereferences symlinks into real dirs on the cluster (the symlink *target* path doesn't
exist there, so `-L` is what makes this work):

```bash
# _isaaclab is import-critical for EVERY task — always add it:
ln -sfn /mnt/storage/lti/UWLab-patrick-private/_isaaclab <worktree>/_isaaclab
# add these too only if the run references them (--init_weights, teacher jit, etc.):
# for d in init_weights teachers models; do ln -sfn /mnt/storage/lti/UWLab-patrick-private/$d <worktree>/$d; done
```

**Reproduce/verify the import without SSH** using the exact cluster image:

```bash
apptainer exec --bind /mnt/storage --bind <worktree>:/workspace/uwlab \
  docker/cluster/exports/uw-lab-base.sif \
  bash -c 'cd /workspace/uwlab && /isaac-sim/python.sh -c "import uwlab_tasks"'
# broken worktree -> ModuleNotFoundError: isaaclab_tasks
# fixed worktree  -> error moves on to "No module named 'omni.timeline'" (expected; no SimulationApp booted)
```

**wandb project = the task id** (not `experiment_name`, not `isaaclab`), so each task logs to its own
wandb project — new-task runs won't show up under an old task's project even when they do run.

### Tillicum vs Hyak launch flags

Tillicum (`--cluster tillicum`) differs from Hyak: use **`--qos normal`** (default; no `--partition`/`--account`
— those are Hyak-only), `MEM_PER_GPU` is **ignored** (the submit script fixes `--mem-per-cpu=25G` → 200G/GPU),
paths live under `/gpfs/scrubbed/<netid>/` (uses `CLUSTER_USER`=NetID, not the Hyak `yanda` dir quirk), and
SLURM logs land in `/gpfs/scrubbed/<netid>/slurm_logs`. SSH alias is `tillicum`. Before the first Tillicum
launch, `--cluster tillicum validate base` (checks config + image); if the image is missing there,
`--cluster tillicum push base` once (~12GB). H200 (141GB) vs Hyak L40 (48GB) is the reason to prefer Tillicum
for memory-heavy models (see the transformer-OOM gotcha below).

### Diagnosing why cluster jobs failed

`docker/cluster/diagnose_jobs.sh <name_regex> [--cluster hyak|tillicum] [--all] [--pull] [--lines N]` finds jobs
by run-name/task (matched against the `Parsed Script CLI Args` line in each `slurm_logs/*.out`, since SLURM
names every job `uwlab-dist-<datetime>`) and prints the ROOT-CAUSE reason — the Python traceback above
torchrun's `exitcode 1` summary, plus a `>>> exception:` line that surfaces the real error even when it sits
below the printed window (deep stacks). Catches the silent `COMPLETED 0:0` mode where python crashed but the
wrapper masked it. Match a single generation with `'run_id=<prefix>'`.

## Key Gotchas

- **`replicate_physics=False` required** for multi-task (`MultiAssetSpawnerCfg` with heterogeneous meshes)
- **Max 8K envs/GPU for multi-task** — 16K hits `collisionStackSize` int32 overflow on all GPUs
- **PhysX 64K materials limit** — stripped physics materials from object USDs; only robot materials remain (2/env)
- **`progress_context` weight must be non-zero** (0.1) — Isaac Lab skips calling reward terms with weight=0.0, breaking all downstream signals (success_reward, dense_success, per-task metrics, curriculum)
- **Shared encoder is required for PC obs** — flat MLP on raw PCs causes catastrophic collapse; shared encoder ([256,128]->32d) eliminates it
- **Wrist frame > base frame for PCs** — task is wrist-centric, wrist-frame PCs are nearly invariant to arm pose
- IsaacLab launcher is at `_isaaclab/IsaacLab/isaaclab.sh`, not repo root
- Code-only changes don't need Docker rebuild -- code is rsynced fresh each job. **EXCEPTION: `rsl_rl` is NOT rsynced code** — it's a pip'd git dependency baked into the image (`source/uwlab_rl/setup.py`: `rsl-rl-lib @ git+.../rsl_rl.git@<ref>`). Changing the fork requires repinning that ref **and rebuilding+pushing the image**, not a rsync.
- **`OnPolicyRunner.load() got an unexpected keyword argument 'strict'` on the cluster** — the image bakes `patrickhaoy/rsl_rl@main`, whose `load()` predates the `strict` kwarg; the strict-load fix lives only on `yandaboa/rsl_rl@pat/strict-load-ratio-diag`. Every **resume** run crashes at `train.py`'s `runner.load(resume_path, strict=False)` (SLURM masks it as `COMPLETED 0:0`). Locally fine because `patlab` imports the editable `/mnt/storage/lti/rsl_rl_patrick` (has the fix). Fixes: (1) `train.py` now calls `load()` signature-aware (falls back to plain `load()` for same-arch resumes → no rebuild needed); (2) `setup.py` repinned to the yandaboa fork branch — rebuild+push the image to bake in real strict-load (dim-mismatch padding). Diagnose with `diagnose_jobs.sh 'run_id=<prefix>'`.
- **Cluster launch from a git worktree needs `_isaaclab` symlinked in first** — `git worktree add` skips untracked dirs, so a fresh worktree lacks `_isaaclab`; without it every rank dies ~30s in with `ModuleNotFoundError: isaaclab_tasks` (SLURM `COMPLETED 0:0`, no wandb run). See "Launching to the cluster from a git worktree".
- **History-transformer PPO OOMs on 48GB L40 (not on 80GB H100)** — `BiasedState-History-Pretrain` at 16384 envs/GPU dies with `torch.OutOfMemoryError` in the transformer FFN during `alg.update()` (env sim eats ~20GB, leaving ~23GB for PyTorch). Runs on H200 (Tillicum, 141GB) unchanged, or on L40 with `agent.num_mini_batches=8`+ / lower `--num_envs`. `inference_chunk_size` does NOT help under grad (activations retained for backward). Not reproducible on the 80GB H100 dev box.

## Work Practices

- **Commit and push frequently** to `pat/multitask` with clean commit messages. Never leave work uncommitted.
- **Update memory regularly** (`~/.claude/projects/-home-patrickhaoy-research-UWLab-private/memory/`) so a new session can pick up where we left off. Include current experiment status, what's been tried, and next steps.
- **Update Notion** ([Multi-task / Long-horizon](https://www.notion.so/Multi-task-Long-horizon-31ba5cea7a8b806db44cd03fd9ea0ebf)) with experiment results, wandb links, and commit links. Do this every time a new experiment is launched or changes are made.

## Architecture Quick Reference

### Multi-Task Detection
All MDP terms auto-detect multi-task via `get_usd_paths_from_spawn_cfg()`. Single-task code paths preserved.

### Round-Robin Task Assignment
`task_type_ids = torch.arange(num_envs) % num_task_types` — even envs = peg, odd envs = leg.

### GPS Curriculum (Beta-shaped sampling)
Targets partitions near 33% success rate: `w(p) = p^(a-1) * (1-p)^(b-1)`, `a=1+kappa*target`, `b=1+kappa*(1-target)`. Per-state variant tracks 80K individual reset states.

### Shared Encoder Architecture
`ActorCriticWithEncoder`: per-group MLP encoders compress PC (768d -> 32d), main MLP sees proprio (21d) + encoded (32d) = 53d.

## Running Isaac Jobs (this machine — Yanda's setup)

**Preferred workflow for all Isaac Sim jobs (train / play / eval / collect) on this machine:**

Run inside the **`isaac-sim`** Docker container (image `isaac-sim:5.1.0`) using the **`patlab`**
conda env. NOT bare host conda, and NOT the `uw-lab-base` container. The container mounts
`/mnt/storage` and `/home/ubuntu` read-write at their real paths, so host files (including git
worktrees) are visible in-container at the same path.

**Interactive:**
```bash
bash /mnt/storage/lti/isaac-start.sh         # joins/starts the isaac-sim container, opens a shell
source /mnt/storage/lti/activate_patlab.sh   # conda activate patlab + Isaac setup_conda_env.sh
```

**Non-interactive / background (e.g. from an agent):**
```bash
docker exec -e CUDA_VISIBLE_DEVICES=<ids> isaac-sim bash -lc "
  source /mnt/storage/lti/activate_patlab.sh && \
  cd <run_dir> && python scripts/.../train.py ..."
```

**Rules:**
- **Run from a git worktree, not the main checkout** — parallel jobs and the user's working copy
  must never be disturbed. The worktree is visible in-container via the `/mnt/storage` mount, so
  no file copying is needed; `logs/` lands under `<run_dir>/logs/...` on the host (directly readable).
- **`patlab` editable-installs the `uwlab_*` packages from the MAIN checkout**, so worktree
  code under `source/` is NOT picked up unless you prepend it: after activating, run
  `export PYTHONPATH=<worktree>/source/uwlab_tasks:$PYTHONPATH` (add other `source/uwlab*`
  dirs if you edited them). `scripts/` runs by path from the worktree cwd, so it needs no
  PYTHONPATH. Without this, `import uwlab_tasks` resolves to the main checkout and worktree
  edits (e.g. new Hydra params) silently don't apply.
- **Pick idle GPUs** — check `nvidia-smi` first. GPUs 0,1 are reserved; use 2–7.
- **Distributed (multi-GPU):**
  `python -m torch.distributed.run --nnodes=1 --nproc_per_node=<N> --rdzv_backend=c10d
  --rdzv_endpoint=localhost:<port> scripts/.../train.py ... --distributed`.
  `--num_envs` is PER GPU. `CUDA_VISIBLE_DEVICES=6,7` → `cuda:0`=GPU6, `cuda:1`=GPU7.
