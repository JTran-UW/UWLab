# Goal-Conditioned Task — Experiment Notes

Experiment-specific findings for the GC (GCMRM) peg task. Process lessons go in
`autoresearch-journal.md`; this file is for what we learned about THIS task.

## Session 2026-08-20 (overnight) — hunting signs of life

**Goal:** sustained increases in task success on the GC task; simplify until something moves.

### Starting position / config changes made first
- Success + dense reward made **insertive-object-only** (`include_ee=False`). Previously success
  required BOTH the object and the EE inside their thresholds -- a conjunction of two 6-DoF
  constraints, i.e. very sparse.
- **No intermediate rewards** in the default `GCRewardsCfg`.
- Added a **goal curriculum**: goals sampled near each env's own start pose, radii widening
  linearly to unbounded over `curriculum_steps` (default 100k env steps). Verified offline that
  the end of the schedule is an exact uniform draw.
- Added task `...-GC-Grasped-OffPolicy-v0`: resets restricted to `ObjectAnywhereEEGrasped`, so the
  object starts held and the task collapses to reaching.

### KEY PRIOR CONTEXT (important for interpreting results)
The non-GC peg task that DID learn from scratch
(`Full-State-Expert-N-Step-3-Rewards-Scaling-Sparse-No-Priv-Ratio005`, job 38484131) was seeded
with **expert transitions at `expert_ratio=0.05`**. GC has no expert buffer, so these runs are
pure from-scratch exploration on a sparser objective. Absence of expert seeding is a plausible
primary cause of any failure to launch, independent of curriculum/EE changes.

Known-good hyperparameters carried over from that run:
`--num_envs 4096 agent.handle_truncations=true agent.target_entropy_ratio=-1.0
agent.num_updates=2.0 agent.critic_learning_rate=3e-4 agent.actor_learning_rate=5e-5
agent.tau=0.125 agent.num_steps=3`

### PIVOT: PPO + multi-GPU (user direction, ~03:35)
FastSAC batch (38692649-52) cancelled before it ran. Rationale for PPO: there is no expert buffer
for the GC task, so off-policy seeding -- the thing that made the non-GC peg task work -- is
unavailable; PPO with many parallel envs is the natural regime for from-scratch exploration.
Moved to tillicum with 4 GPUs/job (user: "4 GPUs is acceptable"). These runs double as the first
real validation of `rank_isolate.py` beyond 2 ranks.

Cost note: tillicum bills GPU-hours. 4 arms x 4 GPUs ~= $16/hr.

### Batch 1 (2x2: reset distribution x curriculum), tillicum gpu-h200, PPO, 4 GPUs each
| Run | Task | Curriculum | Job |
|---|---|---|---|
| GC-PPO-4Path-Curriculum | GC-v0 | on | 250153 |
| GC-PPO-4Path-NoCurriculum | GC-v0 | off | 250154 |
| GC-PPO-Grasped-Curriculum | GC-Grasped-v0 | on | 250155 |
| GC-PPO-Grasped-NoCurriculum | GC-Grasped-v0 | off | 250156 |

PPO defaults (Base_PPORunnerCfg): num_steps_per_env 32, max_iterations 40000, lr 1e-4 adaptive,
entropy_coef 0.006, gamma 0.99, lam 0.95, desired_kl 0.01, 5 epochs / 4 minibatches,
actor+critic [512,256,128,64] elu, gSDE noise. 4096 envs per rank x 4 ranks = 16384 total.

Code propagation to tillicum VERIFIED (GC-Grasped-v0 registered, GCGraspedTrainEventCfg,
_sample_curriculum_goals, include_ee all present in the cluster copy).

Metrics to watch (from `wandb/run-*/files/wandb-summary.json`):
`charts/success_rate`, `metrics/task_N_success_rate` (per reset type),
`curriculum/goal_pos_radius`, `curriculum/goal_in_range_frac`.

### Results
(filled in as they arrive)

### CRITICAL FINDING (04:00) — success is geometrically unreachable at init
Local 4-iteration PPO run on `GC-Grasped-v0` (64 envs, workstation) reports:

```
Metrics/task_command/average_pos_align_error ~ 0.157 - 0.186 m   (threshold 0.03 m)
Metrics/task_command/average_rot_align_error ~ 1.13 - 1.57 rad   (threshold 0.2 rad)
Episode_Reward/success_reward = 0.0000
```

The object sits **5-6x further than the position threshold and ~7x the orientation threshold**
from its goal at initialisation. A sparse success term therefore cannot fire even once, so there
is nothing for PPO to bootstrap from -- this is a geometry problem, not (only) an algorithm or
reset-distribution problem. Insertive-only success and the curriculum do not by themselves fix it.

Consequence: difficulty levers matter more than reset distribution. Added and verified:
- `env.rewards.progress_context.params.require_orientation=false` (drop orientation conjunct)
- `env.rewards.progress_context.params.position_threshold=<m>` (sentinel -1.0 = USD metadata)
Both confirmed to bind via Hydra in the local run (no `Incorrect type` error; sentinel floats used
deliberately instead of None defaults).

Also: thresholds come from the receptive object's USD metadata, NOT config -- hence the sentinels.

### Batch 2 (04:00) — difficulty ladder, dataset_dir=./Datasets/OmniReset (fixed peghole)
`.dockerignore` excludes `Datasets/`, so the local dataset does NOT ride along with the code
rsync; it was pushed separately to both clusters. On tillicum the first push failed because rsync
creates only one directory level (`Datasets/OmniReset` is two) -- `mkdir -p` first.

| Arm | Cluster | Task | Difficulty | Curriculum | Job |
|---|---|---|---|---|---|
| A | tillicum 4GPU | grasped | standard | on | 250157 |
| B | tillicum 4GPU | grasped | pos-only, 0.15m | on | 250158 |
| C | tillicum 4GPU | 4-path | pos-only, 0.15m | on | 250159 |
| D | tillicum 4GPU | grasped | pos-only, 0.15m | off | 250160 |
| E | hyak 1GPU | grasped | standard | on | 38692716 |
| F | hyak 1GPU | grasped | pos-only, 0.15m | on | 38692717 |

Isolates: difficulty (A vs B), reset distribution (B vs C), curriculum (B vs D),
multi-GPU vs single (A vs E, B vs F).

NOTE: hyak arms are blocked on `AssocGrpJobsLimit` (user's own 2 gpu-l40s jobs hold the slots),
so tillicum is the primary path.

### Batch 3/4 (04:00) — pivot to 1-GPU for queue reasons + shaping hypothesis
Tillicum start estimates for four 4-GPU jobs were 04:35 / 08:10 / 10:36 / 13:01 -- only one would
run before the 9am deadline. Cause: 4 GPUs x 6 CPUs = 24 CPUs cannot backfill into "mixed" nodes
(most had only 8-16 idle CPUs). Cancelled the 4-GPU arms except one and resubmitted at 1 GPU;
those started within ~30 s. LESSON: on a fragmented partition, whole-node-ish asks starve.

**Second key hypothesis — the dense reward is nearly flat.**
`gc_dense_success_reward` uses `exp(-d/std)` with `std=1.0`. Over the working range d in [0, 0.2] m
that spans only ~0.15, so the position shaping supplies almost no gradient. At `std=0.1` the same
range spans ~0.87. Also, with `require_orientation=false` the dense term still rewarded ORIENTATION
that success no longer scores -- same incoherence as the EE issue. Added
`include_orientation` to `gc_dense_success_reward` (default True, so queued arms unaffected).

| Arm | Job | Task | pos thresh | curriculum | dense |
|---|---|---|---|---|---|
| B1 | 250163 | grasped | 0.15 | on | std 1.0 |
| G1 | 250173 | grasped | 0.30 | on | std 1.0 |
| D1 | 250166 | grasped | 0.15 | OFF | std 1.0 |
| C1 | 250167 | 4-path | 0.15 | on | std 1.0 |
| H1 | 250171 | grasped | 0.15 | on | std 0.1, pos-only shaping |
| I1 | 250172 | grasped | 0.15 | on | std 0.1 |
| B4 | 250158 | grasped | 0.15 | on | std 1.0, 4 GPUs |
| E  | 38692716 (hyak) | grasped | standard | on | std 1.0 |

G1 (orig 250165) died in 20 s to the CONCURRENT PIP RACE: several jobs starting together each run
`pip install --user -e holosoma` into the shared `holosoma-deps`, corrupting it
(`OSError ... warp/__init__.py`). Resubmitted as 250173 once the cache was warm.
LESSON: serialize job STARTS, not just submits, or give each job its own deps dir.

### FIRST SIGNAL (04:15, ~50 PPO iterations)
| Arm | job | pos thresh | curriculum | dense std | task_0_succ | succ_rew | avg_pos_err |
|---|---|---|---|---|---|---|---|
| G1 | 250173 | 0.30 | on | 1.0 | **0.742** | 0.792 | 0.136 |
| I1 | 250172 | 0.15 | on | 0.1 | 0.170 | 0.175 | 0.169 |
| D1 | 250166 | 0.15 | OFF | 1.0 | 0.160 | 0.197 | 0.212 |
| H1 | 250171 | 0.15 | on | 0.1 (pos-only shape) | 0.160 | 0.162 | 0.180 |

Loosening the success ball moved success from structurally-zero to measurable. Threshold is the
dominant lever, exactly as the local geometry check predicted.

**IMPORTANT metric caveat — two different success criteria are logged:**
- `metrics/task_N_success_rate` -> SuccessMonitor fed by `GCProgressContext.success`, i.e. OUR
  relaxed criterion (position-only, overridden threshold). This is the one to trust tonight.
- `Metrics/task_command/end_of_episode_success_rate` -> `TaskCommand`'s OWN success, computed in
  commands.py from the receptive object's USD metadata thresholds (0.03 m AND orientation). Our
  Hydra overrides live on the reward term, not the command, so this stays 0.0000 and is NOT
  evidence against the relaxed criterion working.

**Confound to control for:** with the curriculum active, goals are drawn near each env's start
state, so some envs begin ALREADY inside the success ball -- non-zero success at iteration ~0 is
partly sampling, not skill. D1 (curriculum off) also shows 0.16, so it is not purely that.
=> The evidence for "signs of life" must be a sustained INCREASE over iterations, not the level.
A CSV trend logger (`gc_trend.csv`, 10-min cadence) was started for exactly this.

### Failures this round (all one root cause: shared holosoma-deps pip race)
- G1 (250165) died 20 s: `OSError ... warp/__init__.py`. Resubmitted -> 250173, fine.
- C1 (250167) crashed importing wandb from the corrupted deps dir.
- B1 (250163) hung 15 min in `pip install` with 0 iterations; killed.
Resubmitted B1/C1 spaced 4 min apart. Long-term fix: per-job deps dir, or pre-warm before a batch.

## SIGNS OF LIFE CONFIRMED (04:43) — sustained increases on every arm

| Arm | job | dense std | dense orient | pos thresh | curriculum | succ first->last | delta |
|---|---|---|---|---|---|---|---|
| H1 | 250171 | **0.1** | **excluded** | 0.15 | on | 0.166 -> 0.616 | **+0.450** |
| D1 | 250166 | 1.0 | incl | 0.15 | OFF | 0.161 -> 0.414 | +0.253 |
| I1 | 250172 | **0.1** | incl | 0.15 | on | 0.176 -> 0.410 | +0.234 |
| G1 | 250173 | 1.0 | incl | 0.30 | on | 0.732 -> 0.903 | +0.170 |
| C1r | 250176 | 1.0 | incl | 0.15 (4-path) | on | 0.244 -> 0.313 | +0.069 |
| B1r | 250174 | 1.0 | incl | 0.15 | on | 0.152 -> 0.181 | +0.028 |

### The controlled comparison (B1r vs I1 vs H1)
These three differ ONLY in the dense reward term. Everything else -- task, reset distribution,
threshold, curriculum, PPO hyperparameters -- is identical:
- std 1.0, orientation in dense  -> +0.028
- std 0.1, orientation in dense  -> +0.234  (~8x)
- std 0.1, orientation dropped   -> +0.450  (~16x)

**Conclusion: the dense reward's length-scale was the binding constraint, not the reset
distribution and not the curriculum.** `exp(-d/std)` with std=1.0 is nearly flat over the 0-0.2 m
working range (spans ~0.15); at std=0.1 it spans ~0.87. Dropping the orientation term when success
is position-only removes shaping that points where success does not score.

Curriculum is NOT the driver: D1 (curriculum off) rose +0.253, more than curriculum-on arms at the
same shaping (B1r +0.028, I1 +0.234 -- I1 differs by std, not curriculum).

### Caveat
`average_pos_align_error` rose slightly on several arms while success rose. Not necessarily
contradictory: success is sampled by the SuccessMonitor at episode end, pos_err is an episode
average, and with the curriculum active the goal difficulty itself widens over time. Do not read
pos_err as a clean progress metric while the curriculum is moving.

### UPDATE 04:50 (n=4) — two corrections and a new result

**Correction 1: the "8x/16x" shaping effect was overstated at n=3.**
B1r (std 1.0) was +0.028 at n=3 but +0.276 at n=4 -- it learns, just slower. Matched-time deltas:
H1 (std 0.1, pos-only shaping) +0.701 > I1 (std 0.1) +0.575 > B1r (std 1.0) +0.276.
The ORDERING is stable and reproduced at every poll, but the honest effect size is ~2.5x, not 16x.
(Classic partial-data trap: do not quote a ratio from the first two points of a rising curve.)

**Correction 2: the reset distribution DOES matter.**
C1r (4-path, std 1.0) is FLAT: 0.244 -> 0.252 (+0.008) while every grasped-only arm rises steeply.
So the earlier claim "not the reset distribution" was wrong -- it was true for the shaping
comparison (all grasped) but does not generalise. Restricting to ObjectAnywhereEEGrasped (the
user's suggested simplification) is doing real work: pre-grasped is learnable, approach-and-grasp
is not yet.

Current standings (n=4):
| Arm | succ | delta |
|---|---|---|
| G1 0.30 m | 0.951 | +0.218 |
| H1 recipe 0.15 m | 0.867 | +0.701 |
| I1 std0.1 0.15 m | 0.752 | +0.575 |
| D1 no-curriculum | 0.696 | +0.535 |
| B1r std1.0 | 0.428 | +0.276 |
| C1r 4-path | 0.252 | +0.008 (FLAT) |

### Batch 5 in flight (climbing back toward the real task from the H1 recipe)
| Arm | job | change from recipe |
|---|---|---|
| J1 | 250189 | threshold 0.15 -> 0.05 |
| K1 | 250200 | grasped -> 4-PATH (does the recipe rescue C1r's failure?) |
| L1 | 250204 | orientation restored to SUCCESS |
| M1 | 250205 | dense std 0.1 -> 0.05 |

K1 is now the most informative arm in flight.

### Batch 5 first readings (04:58, n=1 each -- directional only)
| Arm | change from H1 recipe | first succ |
|---|---|---|
| G1 (ref) | thresh 0.30 | 0.995 (saturated) |
| H1 (ref) | -- | 0.936 |
| I1 (ref) | dense keeps orientation | 0.902 |
| K1 | grasped -> 4-PATH | 0.332 |
| M1 | dense std 0.1 -> 0.05 | 0.210 |
| J1 | thresh 0.15 -> 0.05 | 0.010 |
| L1 | orientation back in SUCCESS | 0.000 |

L1 = 0.000 and J1 = 0.010 reproduce the original wall on purpose: measured errors are ~1.1-1.3 rad
(vs 0.2 threshold) and ~0.16-0.21 m (vs 0.05), so those criteria are simply unsatisfiable at init.
This is direct confirmation that the ORIGINAL task definition (0.03 m AND orientation) is
geometrically unreachable, not merely hard.

K1 (0.332) starting above C1r (0.260, flat) hints the sharp-shaping recipe helps the 4-path case
too, but n=1 -- needs the trend before claiming it.

Bookkeeping: renaming arms mid-run split each trend series in two (same job id, two names).
Restored original names and dropped the 4 duplicate rows. LESSON: never rename a run key mid-run.

## MAJOR CORRECTION 05:13 (n=5-6) — the shaping hypothesis does NOT survive

All grasped-only arms CONVERGE to ~0.91-0.93 regardless of dense-reward shaping or curriculum:

| Arm | dense std | dense orient | curriculum | succ @05:13 |
|---|---|---|---|---|
| B1r | 1.0 | incl | on | 0.925 |
| H1 | 0.1 | excl | on | 0.925 |
| I1 | 0.1 | incl | on | 0.911 |
| D1 | 1.0 | incl | OFF | 0.908 |
| B1dup | 1.0 | incl | on | 0.920 |
| G1 (0.30 m) | 1.0 | incl | on | 0.974 |
| M1 | 0.05 | excl | on | 0.167 and FALLING |

So `std` changed early learning SPEED, not the asymptote -- and std=0.05 is actively worse.
**Retract the earlier "sharper shaping is the key lever" claim.** It was an artifact of comparing
rising curves at a fixed early time. Curriculum likewise has no visible effect (D1 off = 0.908).

### What ACTUALLY separates success from failure
1. **Reset distribution** — grasped-only 0.91-0.97 vs 4-path 0.27-0.36 (C1r +0.026, K1 +0.024,
   both crawling). Dominant factor. The user's suggested simplification is what unlocked it.
2. **Position threshold** — 0.15 m -> 0.92; 0.10 m -> (N1, launched to bracket); 0.05 m -> 0.004 flat.
3. **Orientation in success** — L1 stays at exactly 0.000. Fatal on its own.

### METRIC TRAP (important, nearly misread)
`Metrics/task_command/average_pos_align_error` is `TaskCommand`'s distance to the PEG-INSERTION
assembled pose, NOT to the sampled GC goal. It rises while GC success rises because they measure
different targets. Only `metrics/task_N_success_rate` (SuccessMonitor fed by GCProgressContext)
tracks the goal-conditioned objective. Do not plot task_command pos_err as GC progress.

### 05:28 (n=7-8 on the mature arms) — settled picture, plus a caution on early verdicts

Grasped-only + position-only + 0.15 m now at **0.92-0.95** (I1 0.950, H1 0.945, B1dup 0.934,
D1 0.933, B1r 0.924); G1 at 0.30 m is 0.985. All from ~0.16 at start. This is the headline result.

Still stuck: 4-path (C1r 0.264, K1 0.325) and orientation-in-success (L1 exactly 0.000, 4 polls).

**Caution: early verdicts flip.** M1 (std 0.05) read "falling" at n=2 and is now +0.328 (0.538).
J1 (0.05 m) read flat at n=2 and is now rising (0.064). Only n>=6 verdicts have held. Do not
report a direction from fewer than ~6 polls -- this is the third time tonight that a small-n
reading would have produced a wrong claim.

Launched O1 (4-path, threshold 0.30, recipe) to disambiguate:
- if O1 rises -> 4-path is learnable and the blocker is target tightness on a harder start state
- if O1 stays flat -> something about the 4-path distribution itself blocks learning

### MULTI-GPU (05:30) — rank_isolate works; NCCL is the remaining blocker
B4 (250158, 4 GPUs) reported `COMPLETED` exit 0 after 2:56 — the deceptive-success pattern again.
Actual sequence:
- **All 4 ranks initialised rank_isolate correctly** (rank 0 shared caches; ranks 1-3 private
  HOME/TMPDIR under /tmp/rank_isolate_250158/rank_N). All created their Isaac envs.
  => `rank_isolate.py` DOES fix the Kit lock-file contention it was written for. First confirmation
  it works beyond 1 rank.
- Then died at the FIRST collective:
  `rsl_rl/algorithms/ppo.py:423 broadcast_parameters -> torch.distributed.broadcast` (NCCL).

This is a DIFFERENT failure from the lock-file hang. Added to `submit_job_slurm_tillicum.sh`
(passed through `singularity --containall` via APPTAINERENV_*):
`NCCL_DEBUG=INFO`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_ASYNC_ERROR_HANDLING=1`.
Retry launched: P4 = 250234 (4 GPUs, 2048 envs/rank).

NOTE the whole night's science ran on 1-GPU arms, so multi-GPU is not blocking the result.

### 05:43 — two more early verdicts overturned; one real blocker isolated

| Arm | reading | was |
|---|---|---|
| J1 (thresh 0.05) | 0.010 -> **0.378** RISING | "flat/failed" at n=2 |
| M1 (dense std 0.05) | 0.210 -> **0.772** RISING | "falling" at n=2 |
| O1 (4-PATH @ 0.30) | **0.684** first reading | -- |
| grasped arms | 0.94 - 0.99 | -- |
| L1 (orientation in success) | **0.000** (5 polls) | unchanged |

**Revised conclusions:**
1. Position-only success is learnable across the whole threshold range tried (0.30, 0.15, 0.10?,
   0.05). Tighter = slower, not impossible. J1 at 0.05 m reaching 0.378 kills the earlier
   "0.05 m is unreachable" reading.
2. **4-path is NOT broken.** O1 (4-path, 0.30 m) hit 0.684 immediately vs 4-path@0.15 stuck ~0.30.
   The 4-path difficulty is target tightness given harder start states, not the distribution
   per se. => Earlier claim "reset distribution is the dominant factor" is itself too strong;
   it interacts with threshold.
3. **Orientation is the one robust blocker** -- but partly SELF-INFLICTED, see below.

### Self-inflicted confound in L1 (important)
L1 = recipe (dense `include_orientation=false`) + `require_orientation=true` in SUCCESS.
So success demanded orientation while the dense term provided NO orientation gradient at all.
Additionally `gc_dense_success_reward` uses ONE `std` for both position and angle: at std=0.1 an
orientation error of ~1.2 rad gives exp(-12) ~ 0, so even with include_orientation=true the angle
term would be dead. **Design flaw: position (metres) and angle (radians) need separate length
scales.**

Launched L2: dense std=0.5 WITH orientation, success orientation_threshold=0.5 rad, pos 0.15.
If L2 moves, orientation is learnable given shaping at the right scale, and the fix is to give
`gc_dense_success_reward` separate position/orientation stds.

### 05:58 — position-only is learnable at EVERY threshold tried
| threshold | best success | arm |
|---|---|---|
| 0.30 m | 0.977 | G1 |
| 0.15 m | 0.93 - 0.96 | H1/I1/B1r/D1/M1 |
| 0.10 m | 0.498 rising | N1 |
| **0.05 m** | **0.809** | J1 (was 0.010 at n=2!) |

Retract "0.05 m is unreachable" -- J1 went 0.010 -> 0.809. Tighter thresholds are SLOWER to take
off, not impossible. Launched Q1 at the REAL 0.03 m threshold to find the actual floor.

4-path plateaus rather than climbing: O1 (0.30 m) flat at 0.681, C1r (0.15) 0.305, K1 (0.15) 0.275.
So 4-path is qualitatively different from grasped-only, which reaches 0.93-0.98.

### MULTI-GPU: closing as an open issue (do not sink more time)
P4 (250234) with NCCL_P2P_DISABLE / NCCL_IB_DISABLE / NCCL_DEBUG=INFO failed IDENTICALLY to B4:
`rsl_rl/algorithms/ppo.py:423 broadcast_parameters -> torch.distributed.broadcast`, 2:25, and
SLURM again reported COMPLETED exit 0. No `NCCL INFO` lines appeared in either stream, which
suggests the APPTAINERENV_* vars did not actually reach the container -- verify that first next
time (e.g. `printenv | grep NCCL` inside the container) before trying more NCCL flags.

Confirmed positive: rank_isolate.py works at 4 ranks (all ranks built Isaac envs).
Blocker is the collective, not the cache isolation.
All of tonight's science ran on 1-GPU arms, so this did not gate the result.

### 06:43 — near-final state

**Threshold ladder (grasped-only, position-only success):**
| pos threshold | success | n |
|---|---|---|
| 0.30 m | 0.978 | 15 |
| 0.15 m | 0.93 - 0.96 | 11-14 |
| 0.10 m | 0.900 | 9 |
| 0.05 m | 0.894 | 11 |
| 0.03 m (REAL) | 0.003 flat | 4 (pre-takeoff, inconclusive) |

**Orientation — the L1 vs L2 contrast is the cleanest result of the night:**
- L1: success requires orientation, dense does NOT shape it -> **exactly 0.000 for 11 polls**
- L2: success requires orientation (0.5 rad), dense shapes it at std=0.5 -> **0.064 -> 0.158 rising**
Orientation IS learnable, but only when the dense term shapes it at a scale appropriate to
radians, and it climbs far more slowly than position (0.16 vs 0.95 in comparable wall-clock).
=> Concrete code fix for later: `gc_dense_success_reward` shares ONE `std` between metres and
radians. It needs separate `pos_std` / `rot_std`.

**4-path remains the unsolved case:** C1r 0.226, K1 0.281, O1 (0.30 m) 0.663 -- all flat/falling
after an initial rise. Grasped-only reaches 0.93-0.98 under identical settings.

## BREAKTHROUGH 06:58 — both "blockers" fall

| Arm | what it tests | reading |
|---|---|---|
| **Q1 (250238)** | **REAL 0.03 m threshold** | **0.003 -> 0.752** (n=6) |
| **L2 (250236)** | orientation, SHAPED, 0.5 rad | **0.064 -> 0.623** (n=8) |
| M1 | 0.15 m, std 0.05 | 0.966 |
| H1 | 0.15 m recipe | 0.961 |
| G1 | 0.30 m | 0.982 |
| N1 | 0.10 m | 0.924 |
| J1 | 0.05 m | 0.874 |
| O1 | 4-path @ 0.30 | 0.746 RISING |
| C1r / K1 | 4-path @ 0.15 | 0.305 / 0.284 |

Q1 sat at 0.003 for FOUR polls before taking off to 0.752. Fourth instance tonight of a small-n
reading being wrong. The takeoff delay grows as the threshold tightens; it is not a failure mode.

**Revised headline: the position criterion at its REAL tolerance (0.03 m) is learnable
(0.752), and orientation is learnable when shaped (0.623). The only unsolved case is the
4-path reset distribution at tight thresholds.**

### Code: split shaping length-scales (implemented 07:00)
`gc_dense_success_reward` now takes `rot_std` (default -1.0 = "use std"), so position (metres) and
orientation (radians) get independent scales. Previously one `std` served both: sharpening position
to 0.1 made the angle term exp(-1.2/0.1) ~ 0. Backwards compatible -- existing arms unaffected.

Launched R1: the FULL REAL TASK on grasped-only resets --
pos 0.03 m AND orientation 0.2 rad, dense std=0.05 (position) + rot_std=0.5 (orientation).

### Cleanup finding 08:45 — the tillicum submit script REQUEUES on time-limit
Four arms that hit their `TIME` limit reappeared as PENDING at 0:00, i.e. requeued to restart
training FROM SCRATCH. Cause: `submit_job_slurm_tillicum.sh` installs the `#SBATCH --signal=B:USR1@30`
trap and its handler calls `scontrol requeue $SLURM_JOB_ID`, even though I deliberately left
`REQUEUE_FLAG` empty (no `#SBATCH --requeue`). The flag governs preemption requeue; the TRAP is
independent and still fires.

Consequence: any completed run silently restarts and burns GPU-hours at zero success. On a BILLED
cluster that is a real cost. Fix: gate the trap on the same `REQUEUE` variable, e.g.
`if [ -n "${REQUEUE_FLAG}" ]; then trap ... ; fi`, or drop `--signal` when requeue is off.

All jobs cancelled at 08:45 (with `scontrol update Requeue=0` first so they stayed cancelled).

## 2026-08-20 (evening) — KEYPOINT OBJECTIVE WORKS; curriculum closes its loop

### The keypoint objective (user's idea) is what unblocked orientation
Axis keypoints along the peg's local z (spin-invariant by construction, matching the base task's
`euler_xy_distance` + "# yaw could be different"). Success = ALL keypoints within `keypoint_threshold`
metres; dense = `exp(-mean_kp_dist / std)`. One quantity in METRES, so one std and one threshold --
this removes the whole class of metres-vs-radians bugs that cost hours earlier.

**Root cause it fixed:** `GCProgressContext` used full `quat_error_magnitude` (yaw included) while the
base peg task deliberately uses euler-XY only. The GC task had silently inherited a STRICTER
orientation criterion than the task it derives from.

### Head-to-head, 4x16k (65,536 envs), identical except the objective
| iter | keypoint succ | pose succ |
|---|---|---|
| ~40 | 0.0035 | 0.0013 |
| ~80 | 0.0945 | 0.0023 |
| 120 | 0.4294 | -- |
| 159 | 0.5961 | -- |
| 197 | **0.8000** | -- |
| final pose (132 iters) | -- | **0.0003** |

~41x at matched iterations; pose flat across TWO independent launches (92 and 132 iters).

### The adaptive curriculum promoted for the first time
At succ 0.80 >= promote_at 0.70, radius went 0.0500 -> 0.0650 (= x1.3 expand factor). Expect a
sawtooth from here: wider radius -> harder -> success dips -> recovers -> promotes again.

### Correction recorded: I recommended killing this run at iteration 41
At iter 41 the keypoint run read succ 0.0035 / meanrew -0.22 and I proposed killing both jobs and
rebalancing the reward (weight 0.1->1.0, std 0.05->0.15). By iter 197 it was at 0.80 with meanrew
+10.64. The reward arithmetic was correct but was NOT the binding constraint. FIFTH instance
tonight of a pre-6-poll reading pointing the wrong way. The rule stands: do not act on a trend
before ~6 polls, no matter how mechanistic the explanation looks.

### Multi-GPU (fixed today)
Isaac's CUDA init poisons any NCCL communicator created afterwards -> first collective dies with
`enqueue.cc:76 Cuda failure 'invalid argument'`. Fix in `rsl_rl/train.py`: under `--distributed`,
set device + `init_process_group` + one throwaway `all_reduce` BEFORE `AppLauncher`. Scaling is
near-linear: 1 GPU x 16k = 23.2k steps/s, 4 GPU x 16k = 91.6k steps/s (3.95x).
Single-GPU env ceiling: 16,384 fits comfortably; 32,768 inconclusive (startup exceeded time limit,
no OOM observed).

## 2026-08-21 — k-NN rank curriculum: works, but the schedule was ~20x too fast

### The index (job 250789, 4-path, resume model_800)
Replacing reject-sampling with a per-dataset k-NN index over MEAN KEYPOINT distance immediately
unblocked paths that had never learned:
  t1 ObjectRestingEEGrasped went 0.38 (declining, old design) -> 0.88 within ~90 iterations.
This CORRECTS the earlier inference that resting's difficulty was "physics, not goal geometry".
Its neighbours were always 14x denser than the anywhere path's (1.3 mm vs 18.4 mm); the old
reject-sampler simply could never reach them (~5 cm floor). Density measurement was right, the
conclusion drawn from it was wrong.

### But the schedule outran learning
`curriculum_cooldown` counts `common_step_counter`, and one PPO iteration = `num_steps_per_env`
(32) steps. cooldown=400 therefore allowed a promotion every ~12.5 ITERATIONS, and rank 8 -> 256 at
x1.3 is only ~13 promotions => the entire curriculum could be traversed in ~165 iterations. For
scale, the grasped-only run needed ~150 iterations to go 0.003 -> 0.80 at FIXED difficulty.

Observed failure (per-path, iterations are offset by the resume at 800):
| iter | rank t0/t1/t2/t3 | succ t0/t1/t2/t3 |
|---|---|---|
| 940 | 8/138/180/180 | 0.61/0.75/0.90/0.94 |
| 950 | 8/180/235/235 | 0.61/0.66/0.90/0.94 |
| 970 | 8/180/256/256 | 0.60/0.53/0.80/**0.43** |
| 1250 | 8/180/256/256 | 0.57/0.57/0.89/0.46 (still stranded 280 iters later) |

### Two absorbing states the controller could not escape
- **Over-promoted paths strand in the dead band.** t3 fell 0.94 -> 0.43, but demote_at was 0.30, so
  0.43 never triggered a back-off.
- **Under-performing paths strand at the floor.** t0 sat at 0.59 at rank 8: below promote_at (0.7),
  already at rank_start, so nothing could move.

### Retune (now the defaults in all three GC event cfgs)
| param | was | now | reason |
|---|---|---|---|
| curriculum_cooldown | 400 | 6400 | ~200 iterations per promotion, matching observed adaptation |
| curriculum_demote_at | 0.30 | 0.55 | a path falling 0.94 -> 0.45 must back off |
| curriculum_expand_factor | 1.30 | 1.15 | ~26 steps across the rank range instead of 13 |

Better long-term: RELATIVE demotion (back off when a path drops materially below its own recent
best) rather than an absolute floor, which structurally cannot see a 0.94 -> 0.60 slide.

### Also found: curriculum state does NOT survive a SLURM requeue
`rsl_rl/train.py` auto-resumes the POLICY on requeue (SLURM_RESTART_COUNT), but curriculum radii /
rank limits live in per-process instance attributes, so they silently reset to the start value.
Job 250490 came back with radius 0.098 instead of 1.0 while the policy continued from model_1200 --
i.e. the task got much easier and the success numbers were no longer comparable.
Mitigation used: disable requeue (`scontrol update JobId=<id> Requeue=0`) on curriculum runs.
Proper fix: persist `_rank_limit` to the run dir and reload at init (~10 lines, not yet done).

NOTE also corrected: requeue does NOT restart training from scratch for rsl_rl train.py -- it
auto-resumes from the latest checkpoint. It only loops from zero when the job dies before writing
any checkpoint (as the 32k env probe did).
