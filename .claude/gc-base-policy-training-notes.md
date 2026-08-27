# Training the Base Goal-Conditioned Policy — Process Notes

*(2026-08-21 → 08-24. UR5e + Robotiq 2F85, peg on Peg__PegHole, state obs, PPO/rsl_rl,
4×H200 × 16,384 envs on tillicum. Final checkpoint: `checkpoints/gc_3path_saturated_model_8900.pt`.)*

## Outcome

A single policy that moves the peg from any state in three reset regimes
(**ObjectAnywhereEEGrasped**, **ObjectRestingEEGrasped**, **ObjectPartiallyAssembledEEGrasped**)
to **any goal pose drawn uniformly from the corresponding reset dataset**, scored on keypoints
(4 axis keypoints, max dist < 2 cm), at **0.96–0.97 terminal success per path**. Verified against
a local Isaac eval to within 1 s.e. of the wandb numbers.

## 1. Objective design — the decisions that made it learnable

- **Success = insertive object only** (`include_ee=False`). Requiring the EE to also hit a goal
  makes success a conjunction of two 6-DoF constraints; dropped early, can be re-added later.
- **Keypoint criterion instead of pose.** `max_k ||kp_cur - kp_goal|| < 0.02` over 4 points on the
  peg's axis. One quantity in metres couples position+tilt (no metres-vs-radians threshold pair),
  and spin about the peg axis is free by construction (solid of revolution) — the orientation
  conjunct had been a hard wall under `quat_error_magnitude`.
- **Rewards** (GCRewardsCfg): sparse dwell `success_reward` (w=1.0, +1/step in the ball) dominates
  ~89% of return; shaping `dense_success_reward = exp(-mean_kp_dist/0.05)` (w=0.1) pulls peg→goal;
  `ee_asset_distance_tanh` (w=0.1, from the base omnireset task) pulls hand→peg; small action/vel
  penalties. Gradient chain: hand→peg→goal→stay.
- **Identity goals** (`identity_goal_prob=0.1`): 10% of episodes the goal IS the start state.
  A standing example of success from step 0. Note: NOT free terminal successes — the monitor
  scores at episode end, and with identity_prob=1.0 the anywhere path still only scored 0.03
  (drift). They do not inflate the metric the curricula gate on.
- **No success termination in training** (policy must hold the goal all episode). Play/eval task
  adds `consecutive_success_state` (10 steps ≈ 1 s) for watchability.

## 2. Curriculum — one axis, run as a setpoint controller

**Rank (k-NN) curriculum only.** Goals are drawn uniformly from each start state's `rank_limit`
nearest dataset neighbours, ranked by **mean keypoint distance** (the same quantity success is
scored on). Tolerance stays fixed at 0.02. We briefly ran a tolerance curriculum in parallel —
redundant: success is `dist < tol`, so rank scales the numerator and tolerance the denominator;
two knobs on one difficulty axis fight each other.

Why an index instead of reject-sampling: only ~1.5% of random state pairs lie within 5 cm, but
every state has a neighbour at ~1.8 cm (anywhere) / ~1.3 mm (resting). Close goals exist; uniform
sampling can't find them. **No new data was ever collected — indexing the existing 10k states was
sufficient.**

**Controller settings (validated):**

| knob | value | why |
|---|---|---|
| `promote_at` | 0.85 | setpoint, not timer: climbs until success falls to 0.85, then holds |
| `demote_at` | 0.55 | live floor (the collapsed run's 0.3 left a dead band where a path stranded) |
| `expand_factor` | 1.15 (+1) | small steps; the collapse used 1.5 and blew 8→256 in 7 steps |
| `curriculum_cooldown` | 1600 steps (50 iters) | measured post-promotion recovery is 0–12 iters → ~4× margin |
| `knn_k` | **4096** | see the cliff below |
| per-path gating | per dataset | a strong path must never buy promotions a weak path pays for |

**The knn_k cliff (root cause of the original collapse).** `rank_limit >= knn_k` falls back to a
uniform draw over the WHOLE dataset. With knn_k=256 on ~10k states, rank 255→uniform jumps mean
goal distance 0.065→0.211 m (anywhere; 6.2× for resting) — a bigger difficulty step than the
entire preceding curriculum. knn_k=4096 puts rank-4096 (~0.19 m) adjacent to uniform (~0.21 m).
The old run's "collapse at high rank" was this cliff, not pacing.

**Cooldown premise, corrected.** The long cooldowns (6400) were justified by "the 100-episode
success window needs time to refresh" — but at 16k envs ~100 envs reset *per step*, so the ring
refreshes every ~2 steps. The real constraint is policy adaptation time, which we measured at
0–12 iterations across 12 promotions. Measure the lag; don't assume it.

## 3. How the runs actually went (chained 8 h walls, resume each time)

1. **253182** — 2-path, knn_k=4096 fix, rank 16→(207). Both paths climb, zero demotions.
2. **253764** — resumed via new `curriculum_rank_starts="r0:r1"`; anywhere reached **4096
   (uniform) at 0.92–0.97**; resting stalled at rank ~99–132 for ~1400 iters (≈0.75–0.84).
3. **254790** — resting **broke out**: 132→543→…→4096 in ~12 h while *gaining* success. The
   plateau was competence accumulating, not a wall. Both paths at uniform, ~0.95.
4. **256608** — added the third path (partially-assembled) at 50% sampling weight, old paths kept
   at 4096 as anti-forgetting replay. task_2 started at **0.73 from pure transfer**, climbed
   64→1095 without ever dipping below 0.91.
5. **258848** — task_2 1095→**4096**; final consolidation to 0.969/0.968/0.975.

Pattern worth remembering: **paths sprint, park at their frontier for a long time, then break
out and sprint again.** The setpoint controller makes parking safe (trains at the frontier
instead of collapsing or idling), so plateaus cost nothing but time.

## 4. Verification practices that caught real bugs

- **Test the untested safety mechanism.** `demote` had never fired in any run; exercised the real
  method in isolation (promote/hold/demote/floor/recover/clamp/cooldown) before relying on it.
  This surfaced that `rank_start` doubled as the demote floor → added `curriculum_rank_floor`.
- **Reproduce the wandb metric independently.** Local Isaac eval, same distribution (frozen ranks,
  same reset mix, identity goals, terminal scoring, path labels captured at episode START — the
  in-step reset re-samples task_id): matched wandb within 1 s.e. on both paths.
- **Look at what the policy actually does.** Resting-only motion probe: 96% of episodes move the
  peg >5 cm, but mean lift is 4 mm — it *slides*, it does not lift. Killed the "resting is hard
  because of lifting" hypothesis. Also: ~0.2–0.3 m excursions for ~2 cm goals (path-length
  penalty is available headroom).
- **Metrics literacy:** `metrics/task_N_success_rate` is TERMINAL (episode-end) success over a
  100-episode ring per path. `Metrics/task_command/end_of_episode_success_rate` (~0.17) is the
  OLD fixed-goal criterion — peg assembled in the hole — logged by the inherited TaskCommand;
  it is the free "true insertion rate" tracker for later insertion work, not a GC metric.

## 5. Infrastructure traps (all bit us)

- **Requeue trap ignored `Requeue: disabled`**: the tillicum submit script installed a
  USR1/TERM `scontrol requeue` handler unconditionally → job 252047 silently re-ran 3×, each
  restart loading the ORIGINAL `--resume_path` (explicit path beats auto-resume) and resetting
  the curriculum. Fixed: trap now gated on `REQUEUE`.
- **Curriculum state does not survive relaunch** → `curriculum_rank_starts` per-task resume
  ("4096:4096:1095", **colon** separator: commas make Hydra parse a list and quoting dies over
  the cluster ssh hop; a single value parses as int and fails the str type-check — use scalar
  `curriculum_rank_start` for one path).
- Known wart: first curriculum adjustment fires immediately on an empty success buffer (rate≈0)
  → one spurious ÷1.15 demote at startup, self-recovers in 1–2 windows. Fix if it ever matters:
  init `_rank_cooldown` to `+cooldown` instead of `-inf`.
- Flaky local CUDA illegal-memory-access at Isaac startup when other jobs hold the GPU — retry
  once before debugging.
- torch.distributed rendezvous can fail transiently at submit (C10d store); resubmit gets a new
  port derived from the new job id.

## 6. Where this leaves us / next rungs

- **Tolerance tightening** (0.02→0.01) is now the meaningful single axis — machinery exists
  (`keypoint_threshold_start/min`, `threshold_*`), verified, currently disabled.
- **Assembled-pose goals**: condition on the hole pose to convert GC competence into commanded
  insertion; watch `end_of_episode_success_rate` rise from ~0.17.
- **Re-add EE criterion** (`include_ee=True`) for the stricter original objective.
- **Efficiency**: path-length/workspace penalty to kill the 10× excursions.

Costs: 5 × 8 h × 4×H200 ≈ $145 total on the weirdlab account for the whole campaign.
