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

- **Conda env**: `env_uwlab`
- **Python**: 3.11
- **IsaacSim**: 5.1.0
- **Branch**: `pat/multitask`
- **RL library**: `patrickhaoy/rsl_rl` fork (has `ActorCriticWithEncoder` natively)
- **Reference codebase**: `/home/patrickhaoy/research/OctiLab` branch `feature/locomotion-clean`

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

## Key Gotchas

- **`replicate_physics=False` required** for multi-task (`MultiAssetSpawnerCfg` with heterogeneous meshes)
- **Max 8K envs/GPU for multi-task** — 16K hits `collisionStackSize` int32 overflow on all GPUs
- **PhysX 64K materials limit** — stripped physics materials from object USDs; only robot materials remain (2/env)
- **`progress_context` weight must be non-zero** (0.1) — Isaac Lab skips calling reward terms with weight=0.0, breaking all downstream signals (success_reward, dense_success, per-task metrics, curriculum)
- **Shared encoder is required for PC obs** — flat MLP on raw PCs causes catastrophic collapse; shared encoder ([256,128]->32d) eliminates it
- **Wrist frame > base frame for PCs** — task is wrist-centric, wrist-frame PCs are nearly invariant to arm pose
- IsaacLab launcher is at `_isaaclab/IsaacLab/isaaclab.sh`, not repo root
- Code-only changes don't need Docker rebuild -- code is rsynced fresh each job

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
