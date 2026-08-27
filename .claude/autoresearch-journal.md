# Autoresearch Journal

Shared, append-only notes on running autonomous RL research sessions in this repo.
**Any agent may read and modify this file.** Add entries at the top of the log; keep the
checklist at the bottom current by promoting lessons that have bitten more than once.

---

## Log

### 2026-08-20 — four lessons from a long autonomous session

**1. "The job is progressing" is not "the user can see it".**
Asked why runs looked dead, I re-read SLURM logs, saw iterations advancing, and said they were fine.
They were fine -- and invisible: wandb had printed `Fatal error while uploading data ... will not be
synced`, so the charts were frozen while stdout kept flowing. ~35 min of 4-GPU time burned on runs
nobody could watch.
→ **Fix:** when someone reports a monitoring surface is dead, check THAT surface, not a proxy for it.
Watchers should assert on the display path (wandb project URL + error count), not only on job state.

**2. Do not act on a trend before ~6 polls, however mechanistic the story.**
FIVE times in one session an early reading pointed the wrong way: a "16x" shaping effect that became
~2.5x; "std=0.05 is worse" (it caught up); "0.05 m is unreachable" (reached 0.90); "the reset
distribution doesn't matter" (it was the dominant factor); and worst, a recommendation to KILL a run
at iteration 41 on succ=0.0035/meanrew=-0.22 with a correct-looking reward-imbalance calculation
attached -- by iteration 197 it was at 0.80 and climbing to 0.92.
→ **Fix:** the arithmetic being right does not make it the binding constraint. Wait for the trend.

**3. Verify a setting ARRIVED before interpreting its absence of effect.**
Three separate attempts "enabled" NCCL_DEBUG and got no NCCL output; I twice concluded things about
the container from that silence. The var was being dropped one hop earlier -- `cluster_interface.sh`
forwards only a fixed whitelist over ssh. Same shape as the `lr=0.0` and `expert_ratio` traps.
→ **Fix:** for any knob that crosses a boundary (ssh -> sbatch -> srun -> container), confirm arrival
(`printenv | grep X` inside, or an echo of the parsed value) before reasoning about its effect.

**4. A cheap decisive test beats a confident diagnosis.**
A NCCL failure looked like a driver/build mismatch (`NCCL 2.29.7+cuda13.2` on driver 13.0.3); I
estimated a half-day container rebuild. A 2-minute bare 2-GPU `all_reduce` in the same container
PASSED, killing that theory instantly, and the real cause -- Isaac's CUDA init poisoning any NCCL
communicator created afterwards -- was found by one more A/B. Total: five short jobs.
→ **Fix:** when a hypothesis implies expensive work, find the 2-minute experiment that falsifies it
first. Prefer elimination over explanation.


### 2026-08-18 — A boolean-OR metric hid a single-camera fault for ~6 hours

**Task:** get RGB DAgger training on Hyak (rendering correct, loss down, success up).

**What happened:** every cluster run reported `Episode_Termination/corrupted_camera = 1.0`
and `Mean episode length = 1.0` — every episode dying at step 1. I spent roughly six hours
attributing this to antialiasing: DLAA needs NGX, the container logs
`Failed to create NGX context`, so the story was coherent. I tried `FXAA` (broke rendering
*locally* too), `Off`, `null`, `rendering_mode` presets, and disabling the DL denoiser.
`null` genuinely fixed a real bug — but the cluster stayed at exactly 1.0.

The actual cause: `enable_ambient_occlusion=True` makes **one camera** (`wrist_camera`)
render all-zero inside the container. `side_camera` was rendering perfectly the entire time.

**Root mistake:** `corrupted_camera_detected` ORs `std < threshold` across all cameras, so
ONE blank sensor is indistinguishable from "the whole renderer is broken". I reasoned about
the aggregate signal for hours without ever measuring a pixel. A 10-line debug print
(`px_min/px_max/std` per camera) collapsed the search space instantly:
```
side_camera  std=53.7  px_max=236.0   <- fine
wrist_camera std= 0.0  px_max=  0.0   <- all zero
```
→ **Fix:** when a boolean/aggregate health flag fires, instrument the *components* before
theorising about the subsystem. Ask "which input is bad?" before "which feature is broken?".

**Second-order mistake:** my first version of that instrumentation used one shared counter
incremented once per camera, so the `% 200` throttle always aliased onto the same (healthy)
camera — hiding the answer for another round. Throttle per-key, not globally.

**Third:** I twice quoted a first metric reading as a result (`teacher_eval 0.911`,
`student_eval 0.333`) that halved once the rolling deques filled. These success metrics are
rolling windows; a reading before a few hundred iterations is meaningless. I also called the
trend "flat/declining" from windows taken before the recovery leg existed.
→ **Fix:** report windowed means with the window stated, never a point reading, and never
before ~500 iterations.

**What worked:** front-loading parallel arms spanning a hypothesis space instead of testing
serially (3 AA arms, then 3 render-path arms, then a single-feature ablation) — the ablation
that isolated ambient occlusion cost one extra round because both arms ran concurrently.
Verifying the PR backport had actually rsynced to the cluster *before* interpreting its
result (it had; the fix simply targets a different subsystem). Monitor-based heartbeats:
`sleep`-based background wake-ups got killed mid-session and would have broken the chain
silently — a persistent Monitor loop survived.


### 2026-08-17 — loguru logs to stderr; SLURM splits it into a separate file

Spent several exchanges concluding a working code fix "had not propagated to the cluster" because its
confirmation line was absent from the job log. It was there the whole time — in the `.err` file.
`submit_job_slurm_hyak.sh` sets `--output=%x-%j.out` AND `--error=%x-%j.err` (lines 65-66), and
holosoma logs via **loguru, which writes to stderr**. So every `logger.info(...)` from the agent
(`Loaded replay buffer`, `Optimizer hyperparameters re-applied`, `no_learning=True`,
`reset_optimizers=True`) lands in `.err`, while `print()` from the scripts lands in `.out`.

Cost: doubted a correct fix, nearly relaunched valid runs, and earlier in the same session used the
same missing-line evidence to suspect a stale cluster copy.

→ **Fix:** when checking whether a run picked up a code change, grep **both** streams:
`grep -h <pattern> /path/slurm_logs/*-<jobid>.{out,err}`. Absence from `.out` alone means nothing.
The wandb run directory's `files/output.log` also contains both streams merged, which is a good
single place to look.

### 2026-08-16 — Overnight session failed to run its experiment loop

**What was asked:** run autonomously overnight — observe baselines, form hypotheses, launch
arms to Hyak `ckpt`, read metrics, repeat — and report at 9am with citations and a chart.

**What happened:** one baseline launched, then nothing for seven hours. Zero sweep arms.

#### Mistakes, most costly first

**1. Scheduled a single wake-up.** This session type only resumes when a background task
completes. The 9am alarm was the *only* pending task, so after launching the baseline the turn
ended and nothing brought the agent back until morning.
→ **Fix:** for multi-hour autonomy, chain short wake-ups (30-60 min). Each one reads state,
decides, launches, and *schedules the next one before ending the turn*. The deadline alarm is a
deadline, not a schedule. Losing a wake-up chain silently costs the entire session.

**2. Launched one job when the queue could hold many.** Even with no wake-ups, a batch of 4-6
arms would have produced a comparison by morning. Serial-only is the worst case for an agent
that may not wake up.
→ **Fix:** front-load a batch of arms spanning the hypothesis space, *then* iterate. Cheap
insurance against your own scheduling failing.

**3. Depended on a code fix without verifying it propagated.** An LR fix was made and the
baseline launched in the same turn; its confirmation log line never appeared in the job log,
noticed only next morning. A config-only flag is the dangerous shape: Hydra accepts
`agent.foo=true` happily even if the agent code that reads `foo` is stale on the cluster.
→ **Fix:** after any behavioural change, grep the *cluster copy* of the source for the change
AND grep the running job's log for its effect, in the same turn you rely on it.

**4. Ran submits in the foreground.** A 10-minute tool timeout killed a submit mid-rsync;
`squeue` was empty and it looked like a cluster failure.
→ **Fix:** always launch cluster submits detached (`run_in_background`). Never conclude "the
submit failed" without checking `squeue`/`sacct` first — the local shell dying is not the job
failing, and blind resubmission risks duplicates.

#### Analysis mistakes from the same period

**5. Analysed a partially-written file.** Copied a `tfevents` mid-run (49 KB of an eventual
734 KB) and reported a "6.2% decline, 3.2× the noise floor". The complete file showed **+2.2%**.
Every number in that analysis was wrong.
→ **Fix:** before analysing an artifact from a live run, check size/mtime, and re-check after a
delay. State "run still in progress, partial data" whenever it is.

**6. Resolved runs by name glob.** Two jobs shared a `--run_name`, so `ls -dt <pattern> | head -1`
silently picked the wrong directory and the analysis was attributed to the wrong job.
→ **Fix:** resolve run → directory via the **wandb run id** (`logs/**/wandb/run-*-<id>`) or the
SLURM job id. Never by run name alone; names are not unique.

**7. Trusted `sacct` elapsed time.** It reported 6:27 for a job whose `tfevents` had been written
across 7.5 hours. Preemptible partitions with `--requeue` make the accounting misleading.
→ **Fix:** cross-check job duration against artifact mtimes before reasoning about it.

**8. Proposed mechanisms before measuring them.** Twice in a row: "the observation normalizer is
drifting" (measured: 0.1% of a std over 1000 steps — negligible, count was 5.9e9) and "the upload
is slow because multi-GB blobs leak past `.dockerignore`" (measured: 63 MB, 2002 files, all
exclusions correct). Both were plausible, confident, and wrong.
→ **Fix:** a mechanism claim needs a number attached *before* it is offered. Prefer "here is the
measurement" over "here is the likely cause". Cheap offline probes usually exist — checkpoint
diffs, dry runs, isolated fixtures.

**9. Interpreted a "frozen" experiment for hours before checking the invariant.** Runs configured
with `lr=0.0` were assumed frozen; they were training the whole time (`Optimizer.load_state_dict`
restores `param_groups`, including `lr`). 16/18 actor and 28/29 critic tensors had changed.
→ **Fix:** when an experiment's validity rests on "X is disabled", verify X directly — bit-compare
the relevant state before/after — *before* interpreting any curve. This is the single highest-value
habit; it invalidated ~2 days of conclusions.

---

## Conventions

**Run names** describe the configuration, not a serial index. No `B0`/`B1`/`exp3` prefixes —
they carry no information and go stale the moment an arm is re-run. Lead with the task variant,
then the swept variable, then fixed context:

```
Gap-ER025-Prefill110k-NE1-Eval1024
NoGap-ER050-Prefill110k-NE1-Eval1024
Gap-OrigExpertRB2048-ER005-NoPrefill-Resume110k-NE1-Eval1024
```

Names are NOT unique across submissions — always resolve a run to its directory by wandb id or
SLURM job id (see log entry 2026-08-16 #6).

---

## Checklist

**Before ending any turn in an autonomous session**
- [ ] A wake-up is scheduled that will resume the loop (not just the final deadline).
      Use a **persistent Monitor loop**, not a `sleep`-based background task — those get
      killed and break the chain silently (2026-08-18).
- [ ] Any aggregate/boolean health metric that is firing has been decomposed to the
      component level (which camera? which rank? which env?) BEFORE theorising about
      which subsystem is broken (2026-08-18)
- [ ] No result quoted from a single reading of a rolling-window metric; report windowed
      means with the window stated, and not before ~500 iterations (2026-08-18)
- [ ] A gap between two arms is only a result if it is STABLE across windows and
      survives a seed replicate. A slow crossing looks exactly like a stable gap when
      you stop measuring halfway through it — this produced a claimed +0.055 effect
      that had fully reversed 2000 iterations later (2026-08-18)
- [ ] Every claim made this turn that depends on a code change has been verified on the cluster
      (grep BOTH `.out` and `.err` — loguru goes to stderr)
- [ ] Work is queued that produces value even if no further wake-up ever fires

**Before launching an experiment**
- [ ] Launch detached; confirm the job id from `squeue`, not from the local shell exiting
- [ ] The variable under test is the *only* thing that differs from the control
- [ ] The invariant the experiment assumes ("frozen", "disabled", "5% mixture") is verified, not configured
- [ ] Serialize submits — concurrent ones race on the shared `uwlab_latest` rsync and the shared `pip install`

**Before reporting a number**
- [ ] The source artifact is complete (size/mtime stable), or the analysis says it is partial
- [ ] The run was resolved by wandb/job id, not by name glob
- [ ] The metric measures what the claim needs (batch-mean Q ≠ critic quality; terminal-state
      success ≠ ever-succeeded; per-window ratios quantise hard at small denominators)
- [ ] A negative control exists that *should* fail, and does

**Cluster specifics (Hyak)**
- [ ] `scancel` alone resurrects requeue-enabled jobs — use `scontrol update JobId=<id> Requeue=0` first
- [ ] `COMPLETED` + exit 0 can mean the container's python died; check the log, not the state
- [ ] Login-node latency varies by ~100× (0.9 s to 90 s); slow submits are usually congestion, not config
