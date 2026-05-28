# RGB DAgger Ablations

## Completed Run (May 22–23, 2026)

Two runs were trained from `seed23_sysidenv.pt` (ScenePC teacher) on 32 envs with full sysid DR.

| Metric | GPU0 (no aux) | GPU1 (with aux) |
|---|---|---|
| Steps at stop | ~3.04M | ~5.48M |
| Final `success_student_eval` | 0.8672 | 0.8799 |
| `student_hidden_dims` | [512, 512, 512, 512] | [512, 512, 512, 512] |
| `student_fraction` | ~1.0 (tuple bug — see below) | ~1.0 (tuple bug) |
| `teacher_eval_fraction` | 0.5 | 0.5 |
| Aux loss | Off | On |

### Takeaways

- **Aux loss gave ~1.1pp at matched wall-clock time** but GPU0 closed to within 0.6pp by the time both were stopped. The aux advantage was real early on (~7–8pp at 2M steps) but diminished as no-aux training continued.
- **Sample efficiency**: GPU0 (no aux) reached ~87% at 3M steps; GPU1 (with aux) reached the same level at ~3.5M steps — so no-aux was ~15% more sample-efficient at the high-performance regime.
- **`student_fraction` bug**: `RGB_DAggerWristSidePretrainedWeightedRunnerCfg` had `student_fraction = 0.5,` (trailing comma → Python tuple `(0.5,)`). Effective behavior was undefined — likely fell back to the parent default `1.0` (pure student rollouts). Fixed to `student_fraction: float = 0.5` for upcoming runs.

---

## Planned Experiments

### Experiment 1 — Runner change, same env (GPU0)

**Hypothesis**: Explicit 50/50 teacher/student split in the training pool stabilizes early training and may improve final performance vs pure-student rollouts.

| Dimension | Value |
|---|---|
| Task | `RGB-DAgger-WristSide-Pretrained-Weighted-PCTeacher-FullSysidDR-v0` (existing) |
| Env cfg | `Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidTrainCfg` (unchanged) |
| Scene | DAgger wrist+side scene (RGB, 3 curtains) |
| Image resolution | 224×224 |
| Events / physics DR | ZeroG GPS sysid full DR (wide arm friction, mass, material, OSC gains) |
| Reset distribution | `ZeroGAnywhere` only |
| Teacher | `seed23_sysidenv.pt` (ScenePC JIT, returns std ✓) |
| Teacher obs | 25d proprio + 1536d ScenePC = 1561d |
| Runner | `RGB_DAggerWristSidePretrainedWeightedRunnerCfg` |
| **student_fraction** | **0.5** (fixed — was tuple bug, now proper 50/50 split) |
| student_hidden_dims | [512, 512, 512, 512] |
| aux_enabled | False |
| Loss | `DistillationDAggerWeighted` (inv-variance weighted L2 on μ + L2 on σ) |

**Key change vs completed run**: `student_fraction` is now correctly `0.5`, meaning half of the 32 training envs execute teacher actions each step. Previously all training envs ran the student (bug).

---

### Experiment 2 — Env change, same runner (GPU1)

**Hypothesis**: Using the data-collection scene (broader reset distribution, fixed sysid dynamics, state teacher) tests whether a simpler teacher + richer reset diversity can match or exceed the sysid-DR PC-teacher setup.

| Dimension | Value |
|---|---|
| Task | `RGB-DAgger-DataCollection-StateTeacher-v0` (**new**) |
| Env cfg | `Ur5eRobotiq2f85RGBDAggerDataCollectionStateCfg` (**new**) |
| Scene | DataCollection scene (RGB side+wrist only, 3 curtains, 240×320 cameras; front camera removed) |
| Image resolution | 224×224 (crop/resize via `process_image`) |
| Events / physics DR | FinetuneEval base (fixed sysid + OSC gains) + camera pose/focal DR + full visual DR |
| Reset distribution | 4-type: Anywhere / Grasped / AnywhereGrasped / PartiallyAssembled (25% each) |
| Teacher | `teachers/seed42_finetuned_std.pt` (state expert JIT, returns (mean, std) ✓) |
| Teacher obs | 215d state obs, history_length=5 (`DepthDAggerObservationsCfg.TeacherCfg`) |
| Runner | `RGB_DAggerWristSidePretrainedWeightedRunnerCfg` (unchanged) |
| student_fraction | 0.5 |
| student_hidden_dims | [512, 512, 512, 512] |
| aux_enabled | False |
| Loss | `DistillationDAggerWeighted` |

**Key changes vs completed run**: new env (DataCollection scene, state teacher, 4-reset-type sampling, no sysid physics DR).

**⚠️ Blocker**: runner has `teacher_returns_std=True`. Confirm `peg_state_rl_expert_finetuned_seed42.pt` was exported with `--std` before launching, otherwise `evaluate_with_std` crashes on the first update step.

---

## Env Comparison: Exp 1 vs Exp 2

| Dimension | Exp 1 | Exp 2 |
|---|---|---|
| Scene cameras | wrist + side (RGB) | wrist + side (RGB); front removed |
| Camera resolution | 480×640 (depth scene) → 224×224 crop | 240×320 → 224×224 crop |
| Physics DR | Full sysid DR (arm friction, mass, material, OSC gains) | Fixed sysid + OSC gains only |
| Reset distribution | ZeroGAnywhere only | 4-type (25% each) |
| Teacher type | ScenePC (1561d, JIT) | State (215d, history=5) |
| Teacher std | Returns std ✓ | Returns (mean, std) ✓ — exported via `convert_state_expert_to_jit.py --std` |

## Training Commands

**Exp 1 (GPU0):**
```bash
cd /home/yandabao/UWLab-patrick-private && CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n patlab \
  python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-Pretrained-Weighted-PCTeacher-FullSysidDR-v0 \
  --num_envs 32 --headless --enable_cameras \
  env.scene.insertive_object=peg env.scene.receptive_object=peghole \
  agent.policy.teacher_jit_path=teachers/seed23_sysidenv.pt \
  agent.policy.aux_enabled=False \
  agent.teacher_eval_fraction=0.5 \
  agent.save_interval=5000 \
  agent.logger=wandb \
  > /tmp/rgb_dagger_gpu0_exp1.log 2>&1
```

**Exp 2 (GPU1):**
```bash
cd /home/yandabao/UWLab-patrick-private && CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n patlab \
  python scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-DataCollection-StateTeacher-v0 \
  --num_envs 32 --headless --enable_cameras \
  env.scene.insertive_object=peg env.scene.receptive_object=peghole \
  agent.policy.teacher_jit_path=teachers/seed42_finetuned_std.pt \
  agent.policy.aux_enabled=False \
  agent.teacher_eval_fraction=0.5 \
  agent.save_interval=5000 \
  agent.logger=wandb \
  > /tmp/rgb_dagger_gpu1_exp2.log 2>&1
```
