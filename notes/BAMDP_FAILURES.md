# BAMDP over expert failure rates

Central log for the BAMDP-over-expert-failure-rates research direction.
**One file per direction.** Append new sections and status entries here; do
not spawn per-experiment markdown files. Read this before launching or
interpreting any BAMDPFailures run.

The doc has four parts:

1. **Setup** — where the BAMDP machinery lives and what each piece does.
2. **Operating** — launch templates, knobs, gotchas.
3. **Active runs** — what's training right now.
4. **Experiment history** — newest-first, with verdicts.

---

## 1. Setup

The BAMDP latent is a per-strategy *failure* probability vector `θ = (p_1, …, p_K)`
sampled at each reset. The lowest entry is rounded to 0 → that strategy is
the **rescue expert** and never fails. During a learner rollout, a trained
discriminator outputs per-step weights `w^(t)` over which strategy the
trajectory "looks like"; the per-step hazards `h_i = -ln(1-p_i)/n_i` are
mixed in **hazard space** (not probability space), giving
`f^(t) = 1 - exp(-Σ_i w_i h_i)`. When a forced failure triggers, the env
enters a `stall_steps`-long zero-arm + open-gripper stall, after which the
rescue expert takes over for the rest of the episode. Per the proposal, the
adaptation signal lives entirely in the cost gap between "rescue early" and
"rescue after stall".

### Pieces

| Layer | Path | What |
|---|---|---|
| Env latent | `source/uwlab_tasks/.../omnireset/mdp/bamdp_failures.py:BAMDPLatentSampler` | Reset-mode event term: samples θ, identifies rescue idx, initialises hazard budget. Attaches itself to `env.bamdp` so the wrapper can find it. Supports `force_theta` / `force_rescue_idx` overrides for calibration runs. |
| Cost obs | `…/mdp/bamdp_failures.py:bamdp_{stall,rescue,failure_count,steps_since_failure}_bit` | Per-step bits exposed in the `bamdp_meta` obs group. θ itself is intentionally **hidden** — the solver must infer rescue from these signals. |
| Multi-expert ensemble | `source/uwlab_rl/uwlab_rl/wrappers/bamdp_failures.py:MultiExpertEnsemble` | Loads K PPO actor-critic checkpoints (ScenePC + MLP encoder, same architecture as the seeds 20/21 experts). Dispatches per-env deterministic actions by `rescue_idx`. |
| Discriminator | `…/wrappers/bamdp_failures.py:ExpertDiscriminator` | Mirrors `analysis/train_expert_classifier.py:ExpertClassifier` — proprio + pointcloud → 2 logits. Loaded from `analysis/expert_classifier_runs/seeds20_21_v2_unbiased/best.pt`. |
| Injector | `…/wrappers/bamdp_failures.py:BAMDPFailureInjector` | State machine. Each step: discriminator weights → mixed hazard → Bernoulli forced-failure → stall countdown → rescue takeover. Exposes `injector.rescue_action(obs)` for the supervision label. |
| Env-side wrapper | `…/wrappers/bamdp_failures.py:BAMDPFailureEnvWrapper` | Thin duck-typed wrapper that routes `step()` through the injector. Optional — production loops typically call the injector directly so they can record both the learner action and the rescue supervision label. |
| Env cfg | `source/uwlab_tasks/.../omnireset/config/ur5e_robotiq_2f85/bamdp_failures_cfg.py` | `Ur5eRobotiq2f85BAMDPFailuresTrainCfg` (data-collection) + `…StudentEvalCfg` (eval). Both inherit `ZeroGScenePCSysidSim2RealTrainCfg`. |
| Gym IDs | `…/omnireset/config/ur5e_robotiq_2f85/__init__.py` | `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-BAMDPFailures-Train-v0` and `…-StudentEval-v0`. Both routed to `ScenePCPPORunnerCfg`. |
| Calibration | `analysis/bamdp_calibration.py` | Pure-strategy calibration sweep — required by the proposal before trusting mixed rollouts. |

### Why hazard-space mixing

Probabilities don't compose linearly across steps; survival does. Mixing
hazards then re-exponentiating yields the closed-form path-independent
guarantee
`P(survive) = exp(-Σ_i T_i h_i) = ∏_i (1 - p_i)^(T_i / n_i)`
where `T_i = Σ_t w_i^(t)` is the time-integrated discriminator mass each
strategy attracts. Only `T_i` matters, not the ordering of `w^(t)` — so
a calibrated discriminator + correct hazard math gives exact `p_i` for a
trajectory that the discriminator confidently classifies as strategy `i`.

The injector uses the **hazard budget** variant (proposal section
"Hazard budget"): tracks a per-env per-strategy `H_rem = H_target` and
spends it as `h_i^(t) = H_rem / max(1, n_rem_hat)`. This self-corrects
for episode-length variation that would otherwise bias the realized rate
(Jensen on the convex map `(1-f)^N`).

### Where the data lives (post-collection)

Demos are recorded as zarr / hdf5 (per the existing distillation pipeline;
the BAMDP cfg inherits the parent's recorder set so this is unchanged).
The supervision target field comes from `injector.rescue_action(obs)` —
the rescue expert's intended action at every step, regardless of what
actually executed. The student is trained to imitate the rescue strategy
from step 0; the forced failures are test stimuli on the env, not part
of the demo.

---

## 2. Operating

### Asset paths (seeds 20/21 experts + their discriminator)

```text
Expert A (label "seed_21"):
  /mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-13_04-11-11/model_4000.pt

Expert B (label "seed_20"):
  /mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-13_04-19-11/model_3100.pt

Discriminator:
  /mnt/storage/lti/UWLab-patrick-private/analysis/expert_classifier_runs/seeds20_21_v2_unbiased/best.pt
```

**Class index convention.** The discriminator was trained with
`label = 0` for expert A (seed 21) and `label = 1` for expert B (seed 20)
(see `analysis/train_expert_classifier.py` line ~53). The injector's
`expert_specs` order MUST match this convention — class 0 → A, class 1 → B.

### Knobs

| Override | What | Default |
|---|---|---|
| `BAMDPLatentSampler.params.num_strategies` | K | 2 |
| `BAMDPLatentSampler.params.p_min`, `p_max` | range of each `p_i` ~ Uniform | 0.0, 1.0 |
| `BAMDPLatentSampler.params.n_avg` | seed mean-episode-length | 90 |
| `BAMDPLatentSampler.params.force_theta` | fix θ for every env (calibration only) | None |
| `BAMDPLatentSampler.params.force_rescue_idx` | pin rescue strategy | None (=argmin) |
| `BAMDPFailureInjectorCfg.stall_steps` | K_stall — stall length post-trigger | 4 |
| `BAMDPFailureInjectorCfg.warmup_steps` | suppress failures for first N steps after reset | 1 |
| `BAMDPFailureInjectorCfg.discriminator_temperature` | softmax temperature on classifier logits | 1.0 |
| `BAMDPFailureInjectorCfg.open_gripper_value` | gripper action during stall (≥ 0 ⇒ open) | 1.0 |

### Specs file for `--expert_specs` (JSON)

```json
[
  {"label": "seed_21",
   "checkpoint": "/mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-13_04-11-11/model_4000.pt",
   "proprio_dim": 25, "pc_dim": 1536},
  {"label": "seed_20",
   "checkpoint": "/mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-13_04-19-11/model_3100.pt",
   "proprio_dim": 25, "pc_dim": 1536}
]
```

### Calibration sweep (required before any mixed rollout)

```bash
bash /mnt/storage/lti/isaac-start.sh
source /mnt/storage/lti/activate_patlab.sh

python analysis/bamdp_calibration.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-BAMDPFailures-Train-v0 \
  --expert_under_test 0 \
  --expert_specs configs/bamdp_expert_specs.json \
  --discriminator_ckpt analysis/expert_classifier_runs/seeds20_21_v2_unbiased/best.pt \
  --num_envs 256 --num_episodes_per_p 512 \
  --p_grid 0.1,0.25,0.5,0.75,0.9 \
  --output_json logs/bamdp_calibration/seed21.json \
  --headless

# Repeat with --expert_under_test 1 for seed_20.
```

What to look for in the output JSON: `realized` should be within ~2σ
(`std_err`) of `p_target` for every row. Sample noise at N=512 is roughly
±2 pp at p=0.5; tighten N if the gap looks small.

If the gap is persistent and large:
- Check the discriminator: maybe `temperature > 1` is washing class
  probabilities too much. Drop to 0.5 or 0.25 to sharpen.
- Check `n_avg`: if real episode length differs strongly from the
  initial budget seed, you'll see the budget under/over-spend on the
  *first* few steps — the rolling `H_rem / n_rem_hat` recipe catches up
  but a bias remains.
- Check the recorded action: maybe the expert's action is occasionally
  saturating the OSC limits, which would make its rollout look unlike
  itself to the discriminator.

### Data collection / training

The injector is designed to be invoked between the runner's chosen action
and `env.step`. For patlab's online DAgger
(`uwlab_rl.rsl_rl.DistillationDAgger`) the integration looks like:

```python
# After constructing the env + the runner:
injector_cfg = BAMDPFailureInjectorCfg(
    expert_specs=specs,
    discriminator_ckpt="analysis/expert_classifier_runs/seeds20_21_v2_unbiased/best.pt",
)
injector = BAMDPFailureInjector(env.unwrapped, injector_cfg)

# In the rollout loop (or via a custom `DistillationDAgger.act` override):
out = injector.step(obs, learner_action=runner_policy.act(obs))
# Drive env with the (possibly overridden) action:
obs, rew, done, info = env.step(out["action"])
# Use the rescue expert as the supervision label, regardless of who acted:
runner_storage.transition.privileged_actions = out["rescue_action"]
```

For ASTEROID-style offline demo collection (zarr → diffusion-policy
training): the recorder records every executed step, but the
`expert_action` field that the trainer reads should be set from
`out["rescue_action"]` rather than the inverse-process-actions copy of
the executed action. See the corresponding hook in
`scripts_v2/tools/collect_demos_asteroid.py` (UWLab-ICL) — the patlab
port should follow the same pattern.

### Gotchas (don't re-hit these)

1. **Discriminator class order MUST match `expert_specs` order.** The
   trained classifier labels seed_21 as class 0 and seed_20 as class 1
   (see `train_expert_classifier.py:54`). If you pass the experts in the
   opposite order, the per-strategy hazard mixing is inverted — every
   "fail strategy 0" target gets enforced as "fail strategy 1" and the
   calibration sweep will look like a perfect mirror of the targets.
   Always cross-reference the spec file label against the discriminator's
   training convention.

2. **The rescue expert never has failures because `p_rescue = 0`.** A
   trivial gotcha but worth stating: if you set every `p_i = 1.0`, the
   sampler still rounds the argmin down to 0 so one strategy is always
   the rescue. To stress-test "everything fails", you'd need to disable
   the rescue mechanic entirely — there's no flag for that yet, and the
   proposal explicitly says the rescue is part of the spec.

3. **Hazard budget self-corrects, but only if you don't truncate
   trajectories externally.** The injector decrements `H_rem` by
   `w_i * h_i` only on **eligible** envs (not in stall, not in rescue,
   past warmup). If you truncate episodes mid-rollout (e.g., for video
   recording) the unspent budget is lost — and the realized rate
   under-shoots. For calibration sweeps and full-length rollouts the
   self-correction works; for video debug you'll see a downward bias.

4. **Forced failures only fire while the learner is in control.** The
   `fail_only_when_learner_in_control` flag (default True) suppresses
   triggers after the rescue takes over. This matches the proposal's
   "single-shot rescue" semantics: once rescue is latched, the env runs
   the rescue strategy to the end of the episode regardless of further
   discriminator weights. If you want repeated stall/rescue cycles within
   an episode, flip the flag (and update the proposal accordingly).

5. **`bamdp_meta` is *not* in the policy obs group by default.** The
   parent ScenePC actor uses `obs_groups = {"policy": ["proprio", "pointcloud"]}`.
   The `bamdp_meta` group exists on the env but isn't routed to the
   policy unless you explicitly add it to the runner's obs_groups. For
   the BAMDP solver, add `"bamdp_meta"` to `obs_groups["policy"]` in
   your runner cfg, otherwise the cost signal is invisible to the
   network and there's no adaptation pressure.

6. **Re-importing `bamdp_failures.py` inside the container.** The mdp
   `__init__.py` exports `BAMDPLatentSampler` + the four obs functions
   so the parent module-level `from task_mdp import …` patterns work.
   If you add a new obs term, remember to re-export in
   `omnireset/mdp/__init__.py`.

---

## 3. Active runs

(empty — no full training in flight. Smoke-runs are documented in §4.)

### Iter-0 semantics (different from iter > 0)

**Iter-0 demos are pure multi-modal expert demos with NO BAMDP perturbation.**
The orchestrator (`run_incontext_bamdp.py`) passes `--bamdp_disabled --multi_expert`
to the collection script when `iteration == 0`, which:

  1. Turns the env-side BAMDP layer into a no-op (`bamdp_disabled` cfg
     param). No discriminator inference, no hazard mixing, no stall, no
     rescue takeover — the action the caller passes goes straight to physics.
     `last_rescue_action` is set to the caller's own action so the recorder
     still produces a clean supervision stream.

  2. Drives each env with a *random* expert (`multi_expert` script flag).
     The script maintains a per-env `assigned_expert` tensor sampled
     uniformly from `[0, K)`, resamples it on every reset, and dispatches
     the appropriate expert at every step. Demos end up as a roughly-balanced
     mix across the K experts — k-means(k=2) on per-episode mean actions
     splits cleanly into the two expert styles in our seeds 20/21 setup.

Why: iter-0 is asking the BC student to *represent the multimodal expert
distribution*, not to imitate any single "rescue". Forcing the student
through a one-hot supervision label at iter-0 would mode-collapse it before
in-context adaptation has a chance to break the tie via cost feedback.

**Iter > 0 demos use the full BAMDP layer.** The exploration policy from
the previous iteration's checkpoint drives `max_exploration_horizon` of
each episode; the env's BAMDPLatentSampler samples a fresh θ per reset,
the discriminator + hazard math fires forced failures, and the rescue
expert takes over after a stall to finish the episode. Recorded action
label is the rescue expert's counterfactual (read from
`env.bamdp.last_rescue_action`).

### End-to-end smoke (one-iteration) launch template

```bash
bash /mnt/storage/lti/isaac-start.sh
source /mnt/storage/lti/activate_patlab.sh

python run_incontext_bamdp.py \
    --num_envs 32 --num_demos 20 --train_steps 100 \
    --num_eval_envs 16 --num_eval_episodes 16 \
    --episode_length_s 12.0 \
    --config_name in_context_adaptation_interleave_smoke.yaml
```

The orchestrator auto-applies the iter-0 semantics above. Total wall
time ≈ 3 min for a complete iter-0 round-trip (60s collect, 60s train,
60s eval).

### Knob cheatsheet (collect_demos_asteroid.py)

| Flag | Effect |
|---|---|
| `--bamdp_disabled` | BAMDP layer is a no-op. Pair with `--multi_expert` for iter-0. |
| `--multi_expert` | Random expert per env per episode; recorded label = that expert's action. |
| `--exploration_checkpoint <ckpt>` | Diffusion-policy student that drives the warmup portion. Iter > 0 only. |
| `--max_exploration_horizon <frac>` | Fraction of episode driven by the exploration policy (default 0). |
| `--n_avg <N>` | Mean episode length used by the hazard budget (default 25 — matches seeds 20/21). |
| `--temperature <T>` | Discriminator softmax temperature; T<1 sharpens. Default 0.3. |
| `--argmax` | Hard one-hot discriminator (T → 0 limit). |

---

## 4. Experiment history (newest first)

### 2026-05-21 — iter-0 multi-modal demos + bamdp_disabled mode

Added `bamdp_disabled` cfg knob (env layer becomes a no-op) and
`--multi_expert` script flag (random per-env per-episode expert
assignment, resampled on reset). Orchestrator auto-applies both for
iter-0. End-to-end smoke still passes; iter-0 zarr now contains a 50/50
mix of seed_20 and seed_21 demos (verified via k-means(k=2) on
per-episode mean actions — split is 15/15 with visibly separated
centers).

### 2026-05-20 — end-to-end smoke green

One-iteration smoke (rescue-only collection + 100 train steps + 16 eval
episodes) runs cleanly in ~3 min. Outputs:
- 20 demos collected with ~83% admit rate (rescue-only iter-0).
- Diffusion student checkpoint at step 51.
- EOE success rate = 0% on 16 eval episodes (expected — under-trained).

Pipeline verified. Remaining work for a real run: dial up `train_steps`
(target ~50k for the discrete-AR head per UWLab-ICL), iterate N>0 with the
exploration policy, and add the multi-GPU pipelining.

Calibration also done across single + joint + joint_asym modes; full
write-up + sharpening sweep landed earlier today.

### 2026-05-20 — infrastructure landed

- `bamdp_failures.py` (env-side, MDP terms) committed.
- `wrappers/bamdp_failures.py` (action-side, injector + ensemble +
  discriminator loader) committed.
- `bamdp_failures_cfg.py` + two gym registrations committed.
- `analysis/bamdp_calibration.py` (pure-strategy calibration sweep)
  committed.
- BAMDP_FAILURES.md (this file) committed.

Not yet run end-to-end. Pending validations:
- Calibration sweep against seeds 20/21 to confirm the discriminator +
  hazard math reproduce the target `p_i`. The seeds 20/21 discriminator
  is 91.5% per-state / 100% per-episode (per
  `UWLab-ICL/EXPERT_DISCRIMINATOR.md`), so we expect the calibration
  curve to be close to the diagonal but biased slightly upward (the
  ~8.5% of states classified into the wrong class still leak hazard
  to the other strategy).
- Online DAgger integration with `DistillationDAgger` — patlab's runner
  expects `transition.privileged_actions = teacher_action`. The
  `injector.rescue_action(obs)` value should flow into that field.

---

## Open design questions / followups

- **Discriminator temperature.** The proposal lists "discriminator weights
  used raw vs. temperature-sharpened" as an open design decision. We
  default to raw (`temperature=1.0`). Once we have calibration numbers
  we'll know if a sharper temperature is needed.
- **Remaining-length estimate `n_rem_hat`.** Currently uses
  `max_episode_length - episode_length_buf`. If episodes consistently
  truncate-on-success well before max, the budget over-spends early
  (because `n_rem_hat` is too large) and under-spends near the end.
  We may want a learned / cfg'd estimate; see the calibration script
  for the empirical check.
- **K > 2 path.** The discriminator is currently 2-class. Adding a third
  strategy requires retraining `analysis/train_expert_classifier.py`
  with the third expert's data. The injector raises if K and
  `discriminator.num_classes` disagree, so this can't silently break.

---

## Files of interest (for grep / future Claude)

- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/bamdp_failures.py`
- `source/uwlab_rl/uwlab_rl/wrappers/bamdp_failures.py`
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/bamdp_failures_cfg.py`
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/__init__.py` (gym registrations)
- `analysis/bamdp_calibration.py`
- Discriminator + experts referenced in `/mnt/storage/lti/UWLab-ICL/EXPERT_DISCRIMINATOR.md`
- Proposal (the spec this code realises) — verbatim in the conversation
  that produced this commit.
