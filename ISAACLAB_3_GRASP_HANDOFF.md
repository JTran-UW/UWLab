# OmniReset on IsaacLab 3.0 — grasping investigation handoff

**Status: unresolved.** The learning plateau is not fixed. One bug is found and
validated (reset-state quaternion convention); a second (gripper joint direction)
is measured but my attempted fixes do **not** produce correct behaviour, and the
story around it is internally inconsistent. Read the *Uncertainty* section before
acting on anything here.

Companion doc: `ISAACLAB_3_MIGRATION.md` (the port itself). This doc covers only
the post-migration "policy won't learn" investigation.

---

## 1. The symptom

Training on 3.0 plateaus far below 2.x, at matched iterations and matched env
counts. Reference runs (wandb, project `profjat-university-of-washington/isaaclab`):

| run | what | at it 700 | at it 1500–3000 |
|---|---|---|---|
| `mh4o92uy` | **2.x, 64k envs** (the target) | eoe 0.928, t0 0.83, t1 0.96 | eoe 0.973 |
| `nws8p39u` | 3.0, 57k envs | eoe 0.449, **t0 0.000**, t1 0.13 | **flat** eoe ~0.45 |
| `agkxyse8` | 3.0 canonical-parity, 57k envs | — | 2,173 it: eoe 0.432, **t0 0.0000** |

`eoe` = `Metrics/task_command/end_of_episode_success_rate`.
Task indices (verified, `events.py` ~line 1070, ordered by `reset_types`):

- **t0** `ObjectAnywhereEEAnywhere` — must acquire a grasp → **pinned at exactly 0.000**
- **t1** `ObjectRestingEEGrasped` — grasped, object on the table → **capped ~0.14**
- **t2** `ObjectAnywhereEEGrasped` — grasped, airborne → trains to ~0.87
- **t3** `ObjectPartiallyAssembledEEGrasped` — seated in hole → trains to ~0.78

**The two tasks that need a real grasp are the two that fail.** 2.x and 3.0 are
statistically identical through iteration ~150 and diverge from ~250 onward.

---

## 2. Bug #1 — reset-state quaternion convention (FOUND, FIX VALIDATED)

`reset_from_reset_states` wrote `root_pose` from the dataset verbatim. The stored
quaternions are IsaacLab 2.x `(w,x,y,z)`; 3.0 consumes `(x,y,z,w)`. A 2.x identity
`(1,0,0,0)` read as xyzw is a **180° rotation about X**.

Isolation (spawn vs reset), `scripts_v2/tools/diagnostics/` + ad-hoc probe:

```
BEFORE reset (spawn):   robot base +Z = (0, 0, +1)   upright
AFTER  reset (dataset): robot base +Z = (0, 0, -1)   upside down
  ur5_metal_support  +Z = (0, 0, -1)        inverted
  table              +Z = (0, +1, 0)        on its side
  receptive_object   +Z = (0, -0.18, -0.98) inverted
```

**Fix:** `_reset_state_quat_to_xyzw()` in
`source/uwlab_tasks/.../omnireset/mdp/events.py`, applied to both the articulation
and rigid-object branches of the reset. After it, robot / plate / table / peghole
all read `+Z = (0,0,+1)`.

**Effect on object retention** (`diagnostics/peg_retention_height.py`, 128 envs,
gripper commanded closed, 40 steps, measuring peg height only — no frame
assumptions):

| reset type | before | after quat fix |
|---|---|---|
| ObjectRestingEEGrasped | 100% fell 0.85 m | **0.0%** |
| ObjectPartiallyAssembledEEGrasped | 82% fell | **0.0%** |
| ObjectAnywhereEEGrasped | 99.2% fell 0.94 m | 60.2% |

Before the fix objects fell *through* the work surface to the ground plane at
−0.868 (landing at −0.838 = ground + half the 0.06 m peg). After, they rest on
the surface at z≈0.

**Confidence: high.** Directly observed (the user spotted the inverted scene in
the GUI), isolated to the reset event, and the fix flips every asset from
inverted to upright.

**Caveat:** t1/t3 now "hold" partly because the object is *supported by the table
or seated in the hole*, not necessarily because it is gripped. Do not read 0.0%
as proof of a working grasp.

---

## 3. Bug #2 — gripper joint direction (MEASURED, FIXES NOT WORKING)

### What is measured and reproducible

`diagnostics/gripper_direction.py` sweeps `finger_joint` with **no object
present** and reports the separation of the two `inner_finger` bodies:

| `finger_joint` | pad separation |
|---|---|
| 0.0000 | 0.0070 m |
| 0.2000 | 0.0243 m |
| 0.4000 | 0.0452 m |
| 0.6000 | 0.0676 m |
| 0.7854 | 0.0890 m |
| 0.8727 (limit) | 0.0902 m |

Monotonic. The Robotiq 2F-85 opens to ~85 mm, so **0.785 looks like OPEN and 0.0
like CLOSED** on this USD — i.e. reversed from the real hardware, where joint
angle 0 is open.

Consistent with that reading:

- `ROBOTIQ_GRIPPER_BINARY_ACTIONS` declares `open=0.0, close=0.785398`.
- Robot metadata declares `finger_open_joint_angle: 0.0`.
- Recorded `Peg` grasps (`Datasets/OmniReset/Grasps/Peg/grasps.pt`, 3,868 entries)
  have `finger_joint` mean **0.4812**, min 0.3726, max 0.7374 → ~42–84 mm of
  aperture on a **30 mm** peg. None of them would be gripping.
- Recorded EE-grasped reset states saturate at 0.7806 ≈ the close command.

### What was changed (and did NOT work)

1. `uwlab_assets/robots/ur5e_robotiq_gripper/actions.py` — swapped
   `open_command_expr` / `close_command_expr`.
2. `omnireset/mdp/events.py` `_open_gripper` — mirrored metadata's
   `finger_open_joint_angle` across the joint range (`hi - (v - lo)`).
3. Same site — write the whole 6-joint linkage together, signs
   `finger +, right_outer_knuckle +, left_inner_knuckle +, right_inner_knuckle −,
   left_inner_finger_knuckle −, right_inner_finger_knuckle −`, because writing
   `finger_joint` alone violates the USD mimic constraints (observed as fingers
   jittering and clipping through the hand).

**After all three, the user reports the gripper still opens when it should
close.** Grasp sampling on `fbleg` still drops every object.

### Why the story is incomplete — read this

- **It contradicts the change itself.** After the swap, "close" commands
  `finger_joint = 0.0`. If the hand still opens, then 0.0 opens — which
  contradicts the pad-separation table above. Either the measurement is wrong, or
  the command is not reaching the joint, or something re-opens it afterwards.
- **2.x uses the same `actions.py` and the same metadata and works.** If this USD
  were simply reversed, 2.x should fail identically. It does not. Something in
  2.x must compensate, or the 2.x asset differs, or the inversion is not the real
  story. **This is the single most important unresolved question.**
- **`body_pos_w` is not fully trustworthy here.** It disagrees with the calibrated
  analytical FK by 0.66 ± 0.20 m, configuration-dependent (see §5). The pad
  separation above is a *relative* measure between two symmetric bodies, so it
  should be more robust than absolute positions — but it is the same data source.

---

## 4. What was ruled out (empirically, ~17 candidates)

Each was tested, not assumed:

- **Task config / terminations.** Our fork added `success` and
  `insertive_fell_too_low`; canonical `UW-Lab/UWLab` `upstream/main` has neither.
  Removing them changed nothing — job 39881076 ran 2,173 iterations to eoe 0.432,
  identical to baseline.
- **gSDE exploration.** *This was a real bug and is fixed.* 2.x runs rsl-rl
  `main` @959ccbc where `sample_weights()` is called only in `__init__`,
  `get_noise()` is never called, and sampling goes through the marginal
  `Normal(mean, sqrt(phi²σ²+ε))` — independent per step and per env. rsl-rl 5.x
  implements true gSDE (weights frozen per rollout; lag-1 autocorrelation +0.9996
  vs ~0). `LegacyGsdeDistribution` (`uwlab_rl/rsl_rl/distributions.py`) reproduces
  2.x exactly; verified in sim to 0.008 on normalized per-action std. **Necessary
  but not sufficient** — it did not break the plateau on its own.
- **Datasets.** Reset states and `partial_assemblies.pt` match the HuggingFace
  cloud versions statistically; quaternion conventions differ exactly as expected
  with matching magnitudes.
- **Observations.** No degenerate policy terms; normalized |obs| 0.767 vs 0.798
  expected for standard normal. Outlier-inflation of the normalizer: ratio 0.987
  (refuted).
- **Entropy-gradient plumbing.** Reaches `log_std` with correct sign, total
  −4.2e−02, identical to stock gSDE and to a plain Gaussian. 2.x uses the same
  (64,7) `log_std` shape.
- **Episode-length confound.** `dense_success_reward` and `ee_asset_distance` are
  identical across versions at matched iterations, so episodes are the same length.
- **isaacsim physics version.** Downgraded `isaacsim-extscache-physics` 6.0.1.0 →
  6.0.0.0, retested, restored. Peg fell at 98–99% on **both**. Not the cause.
- Also: `task_id` assignment balance (~0.25 each), success-rate attribution order,
  success thresholds (read from metadata, no fallback), metadata
  `assembled_offset` quat convention (a real latent defect — `(1,0,0,0)` is
  identity in wxyz but 180°X in xyzw — but A/B'd as **non-causal** here because
  it conjugates out of both `||p_rel||` and `|e_x|+|e_y|`), and the OSC settle
  (inherent to a PD controller with no gravity compensation; unchanged from 2.x).

---

## 5. Open anomalies (unexplained)

1. **Calibrated FK vs PhysX disagree by 0.66 ± 0.20 m** for `wrist_3`,
   configuration-dependent, per-sample. Yet the recorded peg sits 0.182 ± 0.016 m
   from the *calibrated-FK* gripper — tight and consistent with the dataset. Both
   cannot be right. `compute_jacobian_analytical` feeds only the OSC
   (`task_space_actions.py:154`), which mixes it with `body_pos_w`-derived task
   error — so if the frames disagree, the OSC maps forces through a mismatched
   Jacobian.
2. **Grasp-state generation is ~300× slower than when the datasets were made.**
   Aug 28: 10,149 validated states in ~80 min (~2/sec). Now: 165 s/state. Not
   explained by the isaacsim upgrade (tested). The Aug 28 datasets are *not* cloud
   downloads (10,149 states / 38,345,619 bytes vs cloud 10,005 / 37,918,485).
3. **`Table` and `UR5MetalSupport` show 0 collider prims** in the loaded stage,
   and adding `collision_props` changed nothing. May be an artifact of traversing
   only loaded prims (geometry is behind USD references) — unresolved.

---

## 6. Files changed this session (all uncommitted)

| file | change | confidence |
|---|---|---|
| `uwlab_rl/rsl_rl/distributions.py` (new) | `LegacyGsdeDistribution` | **high** — verified vs 2.x source |
| `omnireset/.../agents/rsl_rl_cfg.py` | point `class_name` at it | high |
| `omnireset/.../rl_state_cfg.py` | canonical terminations (drop `success`, `insertive_fell_too_low`) | high (matches upstream) but **no measured benefit** |
| `omnireset/mdp/events.py` | `_reset_state_quat_to_xyzw` | **high** — validated |
| `omnireset/mdp/events.py` | `_open_gripper` linkage + mirroring | **low — not working, consider reverting** |
| `uwlab_assets/.../ur5e_robotiq_gripper/actions.py` | swapped open/close | **low — not working, consider reverting** |
| `scripts_v2/tools/diagnostics/*` (12 new) | diagnostics, see below | — |

Reverted during the session and **should stay reverted**: a change to `R_180Z` in
`ur5e_robotiq_gripper/kinematics.py` (`diag(-1,1,-1)`). It was based on comparing
`FK(mean(q))` against `mean(peg)` — invalid, FK is nonlinear. Per-sample, the
original `diag(-1,-1,1)` gives a tight constant peg-to-gripper offset
(0.182 m, std 0.017) and the "fix" gave 0.565 m, std 0.187.

### Diagnostics (`scripts_v2/tools/diagnostics/`)

- `peg_retention_height.py` — **most useful.** Frame-independent: does the object
  stay up after a grasped reset? Run this on 2.x.
- `gripper_direction.py` — joint angle → pad separation sweep, no object.
- `gui_grasp_demo.py` — holds ONE reset state with the gripper commanded closed
  and a camera on the hand. Use `--viz kit`. (`visualize_reset_states.py` reloads
  every 0.1 s and deliberately *opens* a closed gripper, so nothing ever falls
  there — which is why the resets looked fine in it.)
- `grasp_retention.py`, `drop_predictor.py`, `obs_audit.py`,
  `obs_outlier_impact.py`, `audit_exploration_sim.py`, `reset_success_audit.py`,
  `lift_test.py`, `gripper_mimic_test.py`, `gripper_grasp_test.py`.

---

## 7. What to do next, in order

1. **Run `peg_retention_height.py` on a 2.x machine.** Two minutes, and it is the
   highest-information experiment available:
   ```
   python scripts_v2/tools/diagnostics/peg_retention_height.py --num_envs 256 --steps 40 \
       env.scene.insertive_object=peg env.scene.receptive_object=peghole
   ```
   Holds the object on 2.x → grasping is a 3.0 regression, and §3 is the right
   area. Drops it on 2.x too → grasped resets were never physically real in either
   version, and the plateau is elsewhere.
2. **Run `gripper_direction.py` on 2.x.** Settles whether `finger_joint` is
   reversed on this USD or whether the 3.0 reading is an artifact. This directly
   resolves the contradiction in §3.
3. **Decide whether to revert the two gripper changes.** They are not producing
   correct behaviour and may be actively wrong. The quat fix (§2) should be kept
   regardless.
4. **Regeneration order, once grasping is genuinely fixed:** grasps →
   reset states (they replay `gripper_joint_positions` from `grasps.pt`,
   `events.py` ~664/771) → partial assemblies are unaffected
   (`PartialAssembliesActionsCfg` is `pass`; no gripper action, no robot state in
   the file).
5. Re-run training only after the retention numbers are good. Env-side bugs of
   this size make training comparisons uninterpretable.

---

## 8. How much to trust this document

I raised and retracted **seven** causal hypotheses during this session:
gripper open/close inversion (retracted, then partially revived — see §3), a
"0% success at reset" reading (my probe bug), the motion deficit as a *cause*
(it is a consequence — at iterations 0–40 the 3.0 policy moves *more* than 2.x
and has *higher* early `success_reward`), a t3 pre-learning anomaly (artifact of
comparing against the wrong reference run), the `R_180Z` kinematics fix
(reverted), the metadata `assembled_offset` convention (real but non-causal), and
the isaacsim physics upgrade (falsified by test).

Recurring failure modes in my own analysis, listed so they can be checked for:

- comparing `FK(mean(q))` to `mean(peg)` across a **nonlinear** function
- using a distance between **body origins** as a proxy for gripper aperture when
  the linkage moves those origins non-monotonically with the pads
- reading **Fabric-stale USD transforms** as live state
- a hydra override silently discarded because a `variants` dict replaces the whole
  cfg object in `__post_init__`
- measuring a **kinematic** body and concluding it "held"

**The measurements in this document are reproducible; the causal claims built on
them have repeatedly not survived.** Bug #1 (§2) is the one I would stake
something on — it was independently spotted visually and the fix flips every
asset from inverted to upright. Treat §3 as an open lead, not a diagnosis.

One process note: the actual root cause of the inverted scene was **visible on
screen the entire time** and was found by looking at the GUI, after a night of
numeric measurement failed to surface it. Look at the thing early.

---

## 9. Update 2026-09-10 — grasp sampler fixed; §3 resolved (gripper was NOT reversed)

**Result:** `record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192
--num_grasps 1000 --headless env.scene.object=peg` now yields **3,935 / 8,192 = 48.0%**
successful grasps in 13 s, statistically identical to the 2.x dataset
(3,868 grasps, `finger_joint` 0.481 ± 0.041 vs 0.481 ± 0.042, `|relative_position|`
0.1642 vs 0.1641). Peg is held motionless through gravity-on and the random pokes.

**Root cause of the sampler failure:** the two §3 "fixes" themselves. On this USD
`finger_joint = 0` is OPEN and `0.785` is CLOSED — exactly what `actions.py` at HEAD,
the metadata (`finger_open_joint_angle: 0.0`), the real 2F-85, and the 2.x dataset
(q ≈ 0.48 while holding a 30 mm peg) all said. The §3 "pad separation" sweep measured
the distance between `inner_finger` *body origins* (at the knuckle end), which grows as
the fingers close — the §8 failure mode ("body origins as a proxy for aperture"). With
the swap in place the sampler (1) "opened" to `hi = 0.873` = fully pinched, (2) teleported
those pinched fingertips *into* the object (→ kicks with gravity off, and the linkage
explosions: joints with ±inf limits spin to −6 rad), then (3) "closed" to 0.0 = opened
the hand, dropping the object when gravity came on.

Verified by rendering, not by body positions (`scripts_v2/tools/diagnostics/`):

- `gripper_close_on_peg.py --open_q 0.0 --close_q 0.785` — kinematic peg between the
  pads: finger stalls cleanly at **q = 0.521**, all five passive joints track, |jv| = 0,
  rendered pads flush on both sides of the peg. With `--open_q 0.8 --close_q 0.0` the
  linkage tears apart (|jv| ~ 1e6 rad/s).
- `grasp_sampler_track.py` — the sampler env stepped with the sampler's own constant
  `close` action, printing peg z / drop % / finger q / passive q per step. Before revert:
  100 % dropped, success 0.0 %. After: 47.5 % held, success 47.5 %.
- `gripper_render.py`, `root_frame_probe.py` — headless camera renders and a root-frame
  probe (root-pose writes on the fixed-base gripper DO land; the fingers extend along
  root-link +x as the sampler assumes).

**Reverted** (git): `ur5e_robotiq_gripper/actions.py` (open/close swap) and
`_open_gripper` in `omnireset/mdp/events.py` (mirroring + 6-joint linkage write) — both
back to HEAD semantics, `.torch` accessors kept. **Kept:** `_reset_state_quat_to_xyzw`,
`LegacyGsdeDistribution`, canonical terminations.

Still true and still harmless-looking, but worth knowing: PhysX logs "needs a finite
limit set to be used by the mimic joint feature" for the four ±inf-limit joints, and a
negative mass on both outer knuckles. The linkage nevertheless tracks `finger_joint`
(verified with the `open` command: passive joints follow to ±0.785), so this is not what
broke grasping — but it is what makes the linkage explode instead of resisting when
fingertips are forced into an object, and it is the same reversed-joint asset issue that
blocks Newton.

**Next:** regenerate the reset-state datasets from a fresh 3.0 `grasps.pt` (check the
`relative_orientation` convention consumed by `reset_end_effector_from_grasp_dataset`,
events.py ~679: the 2.x file stores wxyz, a 3.0 file stores xyzw), re-run
`peg_retention_height.py` on the regenerated states, then retrain.

### 9a. Reset-dataset quaternion marker (2026-09-10, later the same day)

`TorchDatasetFileHandler.flush` now stamps a top-level `quat_convention: "xyzw"` into every
file it writes, and `MultiResetManager` converts wxyz→xyzw **only** when that key is absent
(default = legacy 2.x file). The per-reset unconditional `_reset_state_quat_to_xyzw` calls in
`_reset_to` are gone; conversion happens once at load and is logged
(`[MultiResetManager] <file>: N states, quat_convention=...`).

Caveat found while doing this: the Aug 28 local `Datasets/OmniReset` files (now in
`Datasets/OmniReset_backup_2x_20260910_174004/`) were **already xyzw** — they were recorded
on 3.0, not 2.x (their table quat `(0,0,-0.707,0.707)` is the config's −90° yaw only as
xyzw). `scripts_v2/tools/diagnostics/reset_upright_probe.py` shows that applying the §2
conversion to them puts the table on its side and the peghole tilted, while the freshly
recorded stamped file resets every asset upright. So §2's "fix" was correct for the
HuggingFace 2.x datasets (the task configs' default `dataset_dir`) but wrong for those
local files. Do not feed the backup files through the loader; use the regenerated ones.

Regeneration on 3.0 (`scripts_v2/tools/regen_peg_datasets_isaaclab3.sh`, logs in
`logs/regen_20260910_174004/`): partial assemblies 364 poses; grasps 3,919/8,192 (47.8%);
Reaching 10,272 states at 79.5% in 3 min; grasped variants ~5 states/s (low yield is
normal — the 2.x-era timestamps imply the same rate).

---

## 10. 2026-09-11 — ROOT CAUSE of the t0/t1 plateau: rsl_rl 5.2 gSDE forward drops activations

**Not the environment.** The 2.x PPO expert `peg_state_rl_expert_seed0.pt`, run as a plain
MLP inside the 3.0 env (`scripts_v2/tools/diagnostics/expert_xenv_probe.py`), scores
t0 93.1 / t1 89.8 / t2 81.5 / t3 94.2 % — the same as in the 2.x env (90.5 / 85.7 / 81.5 /
88.1). Physics, observations, actions, resets and the regenerated datasets are all fine.
Also verified at parity between 2.x and 3.0: scripted grasp acquisition on t1 states
(`acquire_probe.py`, ~25 % hold in both), OSC step response (`osc_probe.py`), PPO
hyper-parameters, and the exploration std trajectory (0.9 -> 2.9 in both runs).

**The learner.** In the rsl_rl `feature/manipulation` fork (5.2.0), `MLP.__init__` appends one
shared activation-module instance after every hidden layer. `MLPModel.forward` takes the gSDE
branch whenever the distribution has `set_features` and walks
`children = list(self.mlp.children())` — but `nn.Module.children()` de-duplicates modules by
identity, so `named_children()` is `['0','1','2','4','6','8']` and the walk computes
`Linear -> ELU -> Linear -> Linear -> Linear -> Linear`: one nonlinearity instead of four.
That network is what every 3.0 gSDE run rolled out AND optimised. Measured with the loaded
2.x expert (`scripts/reinforcement_learning/rsl_rl/actor_parity_probe.py`): correct forward
`self.mlp(latent)` matches the expert to 0.0000; the gSDE walk gives mean |a| 96.5 vs 4.8
(max diff 630). 55–78 % of hidden pre-activations are negative, so the dropped ELUs are not
cosmetic. The plateaued `model_700` policy never opens the gripper (<1 % of steps) — a
one-hidden-layer policy learns descend-and-insert (t2/t3) but not open→reach→close→lift.

Side effect: `_ExportMLPModel` / ONNX export use the full `self.mlp`, so exported 3.0 gSDE
policies are NOT the trained function.

**Fix:** `rsl_rl/models/mlp_model.py`, gSDE branch: `children = list(self.mlp)` (iterate the
Sequential, which keeps duplicates). Applied to `env_isaaclab3` site-packages on starfish and
as an overlay at `tillicum:/gpfs/scrubbed/profjat/uwlab3/rsl_rl_patched` (sbatch passes
`PYTHONPATH=/workspace/uwlab/rsl_rl_patched` and prints `[INFO] gsde fix : True`). Needs to
land in UW-Lab/rsl_rl `feature/manipulation` and the container image. After the patch the
runner's inference policy equals the configured network (parity probe).

**Runs:** Tillicum job 287616 (`gsde_mlp_fix`, 14336/rank from scratch, patched) and a
starfish sanity finetune from the converted 2.x expert (`expert_seed0_rslrl52.pt`, made by
`/tmp/jtran_convert.py`; note `runner.load(..., load_cfg=...)` needs explicit
`"actor": True, "critic": True` or nothing is loaded).

Also learned: t1 (`ObjectRestingEEGrasped`) states have finger q ≈ 0.78 (fully shut, not
pinching) in BOTH the 2.x cloud and regenerated datasets — t1 is a re-grasp task, not a
hold-and-lift task. The 64K-material cap at 16384 envs/rank is unchanged (job 287490 hung).

**Confirmed from scratch (2026-09-11 08:22):** Tillicum job 287624 (`gsde_mlp_fix`, wandb
`b1092vqk`, 14336 envs/rank x 4) at it 369: eoe 0.553, t0 0.026, t1 0.637, t2 0.877,
t3 0.793 — matches the 2.x reference at it 375 (0.558 / 0.013 / 0.509 / 0.890 / 0.830) and
clears the old 3.0 ceiling (t1 <= 0.15, t0 = 0). The 3.0 port is at parity with 2.x.

### 10a. How the gSDE fix is cemented (no upstream commit, no image rebuild)

* `docker/cluster/patches/apply_rsl_rl_gsde_fix.py` — idempotent applier/checker for an
  rsl_rl package dir (`--check` exits 1 if unpatched).
* `docker/cluster/submit_{tillicum,hyak}_omnireset*.sh` — `ensure_rsl_rl_patched()` builds
  `$UWLAB_DIR/rsl_rl_patched/` from the `.sif` on first use, applies the patch, prepends it to
  `PYTHONPATH` inside the container (together with the four `uwlab*` source dirs, which the
  override would otherwise drop), and **aborts the job** unless the in-container tripwire
  prints `[INFO] gsde fix : True`. Verified on Tillicum (fresh build + cached path) and on a
  klone compute node (login nodes cannot run apptainer).
* starfish `env_isaaclab3`: site-packages patched; re-run the applier after any reinstall.
* Rebuilding the image or bumping the rsl_rl pin makes the overlay redundant but harmless
  (the tripwire still guards).

---

## 11. 2026-09-11 — Robotiq USD fix for Newton (sandboxed, opt-in)

`tools/fix_robotiq_usd.py` writes patched copies of both calibrated USDs to `usd/patched/`:
body0/body1 (+localPos/localRot) swapped on `left/right_inner_finger_knuckle_joint` (the
"reversed joints" Newton rejects) and `MassAPI` (12.7 g, inertia 2e-6) on both outer knuckles
(no collider -> PhysX/Newton fell back to a sphere with mass -1). Everything else is kept as
authored — in particular the `PhysxMimicJointAPI`s and the +-inf limits.

What was tried and rejected on PhysX (gripper_close_on_peg / lift_probe / acquire_probe):
* finite limits (activates the mimics): grasp hold 86% -> 50%, acquisition 25% -> 9-16% — the
  zero mimic offsets fight the calibrated joint frames; corrected gearing on
  `right_inner_knuckle_joint` (authored -1 contradicts the measured four-bar) does not rescue it.
* stripping the mimic APIs: the loop closure collapses entirely, even though PhysX logs the
  mimics as rejected. Their presence changes joint-graph parsing; leave them.

The swap negates those two joint angles, so datasets recorded with the original USD must be
converted: `tools/convert_datasets_for_patched_gripper.py --src Datasets/OmniReset --dst
Datasets/OmniReset_patched` (negates the two columns in `joint_position/velocity` and in
`grasps.pt`, stamps `robot_joint_convention: swapped_ifk`). `MultiResetManager` and
`reset_end_effector_from_grasp_dataset` raise on any asset/dataset mismatch.

Enable with `UWLAB_ROBOT_ASSETS_DIR=$HOME/research/UWLab/usd/patched` (asset cfg reads it;
`ROBOT_ASSETS_PATCHED` flag) and `...dataset_dir=./Datasets/OmniReset_patched`.

Validation (PhysX, patched asset + converted datasets vs baseline): lift hold 22/83 % vs 24/86,
scripted acquisition 18-22 % vs 22-25, 0 % abnormal, 0 PhysX errors/warnings, mismatch guard
raises. Newton: `ModelBuilder.add_usd` fails on the originals and succeeds on both patched files
(9 bodies/11 joints; 16 bodies/18 joints). Not yet tested: Newton dynamics with the (kept)
mimic constraints — expect to revisit gearing/offsets there.

## 12. 2026-09-11 — first Newton (MuJoCo-Warp) run of the full task: status

Task `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0`
(`Ur5eRobotiq2f85RelCartesianOSCEvalNewtonCfg`): `NewtonCfg(MJWarpSolverCfg(njmax=2000,
nconmax=1000, implicitfast), NewtonCollisionPipelineCfg, num_substeps=4, margin 5 mm)`, critic
group dropped, `joint_pos` obs pinned to the 12 PhysX-order joints, dataset resets mapped by name
(`_map_dataset_joints`, `DATASET_ROBOT_JOINT_NAMES`). Asset: `usd/patched_newton`
(`fix_robotiq_usd.py --mimic strip_inner --gravcomp`). Eval driver:
`scripts_v2/tools/diagnostics/policy_eval_video.py` (loads 2.x or 5.2 checkpoints as a plain MLP).

Works: import, scene build, CUDA-graph capture, resets from the converted datasets, stepping.
Found and fixed on the way:
* Newton exposes 14 robot joints (adds the two loop-closing `*_inner_finger_joint`s, different
  order) -> obs/reset bridge above.
* Newton ignores `physxRigidBody:disableGravity`; the effort-controlled arm free-fell. Fixed by
  authoring `mjc:gravcomp=1` on every robot body (MuJoCo gravity compensation).
* PhysX enforces `physxJoint:maxJointVelocity`; Newton does not, and without that clamp the OSC
  arm oscillated at 15-35 rad/s and NaN'd within 20 steps. Stand-in: arm implicit-actuator
  damping (`UWLAB_NEWTON_ARM_DAMPING`, default 10) + 4 substeps -> holds still.
* MuJoCo `nefc overflow` at 64 envs -> njmax/nconmax raised.

Still broken (in priority order):
1. **Contacts**: Reaching resets (peg resting on the table) blow one MuJoCo world up on the first
   step (arm joint velocities 1e12 with constant positions). Peg/peghole ship as SDF meshes; the
   Newton collision pipeline needs convex/CoACD approximations and tuning.
2. **OSC effort path**: with actuator damping > 0 the OSC's `set_joint_effort_target` produces
   ~0 applied torque and the arm no longer tracks commands; need to check how Newton composes
   PD + effort targets, or retune the OSC gains without the damping stand-in.
3. **Gripper four-bar**: with only the `right_outer_knuckle` mimic kept (the one PhysX actually
   enforced), Newton's synthesized CONNECT loop constraints are compliant (finger reaches 0.46 of
   0.785, inner joints lag 0.2 rad). Stripping all mimics explodes; keeping all fights. Needs
   mimic gearing/offsets recomputed for the calibrated joint frames or stiffer loop constraints.
4. **Rendering**: Newton syncs transforms to Fabric only with the Kit visualizer; `--viz kit`
   cannot be combined with `--headless` here, so RTX cameras render black over SSH. Use
   `--viz newton`/rerun/viser or a display session for videos.

Result of the requested 4-env expert eval under Newton: 0/8 (all four reset types), video
frames black -- not meaningful yet; the PhysX-side parity numbers are the reference.

## 13. 2026-09-11 — Newton item 1 (contacts) worked: result

The "Reaching resets explode" symptom was NOT a peg/table contact: 2-3/64 envs blew up on a
full-range gripper swing in free air (start 0 -> close, or 0.78 -> open), i.e. the loop-closure
four-bar under Newton (item 3). Fix: model the gripper as a pure tree with mimic couplings, the
standard MuJoCo way — `tools/fix_robotiq_usd.py --mimic fix --limits finite --loop strip
--gravcomp` -> `usd/patched_newton_tree` (loop joints deleted, five mimics with the PhysX-measured
gearings; right_inner_knuckle -1 -> +1). Free-air tracking is now consistent on all passive joints
(finger 0.72 of 0.785 commanded, passives 0.60-0.71; the residual is MuJoCo equality softness) and
0/64 explosions in either direction.

Object colliders: `tools/fix_object_usd_for_newton.py` writes peg.usd with `convexHull` and
peg_hole.usd with a baked 16-piece CoACD decomposition (one convexHull collider prim per piece;
Newton's own convexDecomposition path silently fell back to a single hull, which would seal the
hole). `mdp.utils.apply_local_object_assets(env_cfg)` redirects the scene objects to
`UWLAB_ROBOT_ASSETS_DIR` copies (call after hydra composition; probes/eval do).
MuJoCo-native contacts (`UWLAB_NEWTON_MJ_CONTACTS=1`) are unusable here (NaN in >90 % of envs).

Where it stands (Newton, tree gripper, convex objects, `usd/patched_newton_tree`,
`Datasets/OmniReset_patched`):
* peg rests on the table (mesh) fine; convex-convex contacts exist (a peg seated in the peghole
  does not fall through) but are badly resolved — seated pegs get shoved 5-11 cm sideways.
* closing on a kinematic peg: pad-origin gap goes to 15-24 mm vs the 60 mm stall on PhysX -> the
  pads push into the peg; grasped resets: hold@lift 7 % (PhysX 83 %).
* dataset grasps written into the tree gripper + contact -> mimic joints drift far from ±q
  (l_ik 2.1 rad); the mimic constraints are too soft to carry contact loads.
Next for contacts: stiffen the mimic equality (MuJoCo solref/solimp via `mjc:*` joint attributes or
Newton mimic params), thicker pad colliders / `NewtonShapeCfg.margin`, and re-derive the reset
gripper joint values from the finger angle (q_ref -> mimic) instead of writing PhysX-recorded
passives.

## 14. 2026-09-11 — Newton item 2 (OSC) discriminating tests: result

Setup: `usd/patched_newton_tree` now also `--zero_root` (all bodies rigidly re-based so the
root body's authored xform is identity: Newton keeps the authored base_link 180-deg yaw under
the articulation root, PhysX overrides it with the root-pose write — before this, the whole arm
was mirrored relative to the datasets and pads sat ~1 m from a "grasped" peg; every earlier
Newton grasp number is void). OSC and the critic velocity obs now use the base_link BODY pose
instead of `root_*_w` (no-op on PhysX). `UWLAB_NEWTON_SUBSTEPS`, `UWLAB_NEWTON_ARM_DAMPING`,
`UWLAB_OSC_DEBUG=1` (prints the OSC's internal terms) added.

Findings (probe `/tmp/jtran_osc_newton.py` on starfish; grasped resets, unit +x command):
* Effort targets ARE applied additively on Newton (`applied_torque` only shows the actuator PD
  part). With joint damping 10 the OSC moves the EE 0.027 m / 12 steps (PhysX 0.038 / 10).
* Substeps 1/3/8/16 make no difference to the damping-0 blow-up (25-49 rad/s within the first
  policy step) — the OSC torque is recomputed at 120 Hz regardless of substeps.
* Not the dataset joint write, not the gripper couplings (default reset, gripper held open,
  zero command: still 36 rad/s). Per-link inertia, armature (0) and friction (0) match PhysX.
* OSC debug trace: substep-1 torques are ~1 N·m and sane; wrist joints (I ~ 1e-4 kg m^2) then
  oscillate with sign flips growing every substep. Explicit task-space D-term at 120 Hz on that
  inertia: kd*dt/I ~ 300 >> 2. PhysX is saved by its joint-velocity clamp (velocity_limit_sim /
  physxJoint:maxJointVelocity, 1.6-3.1 rad/s) which MuJoCo does not have.
* Implicit joint damping (MuJoCo implicitfast) is unconditionally stable for holds: damping 10,
  zero command, Reaching resets: 0/32 explosions. But commanded motion is still erratic on
  Reaching resets (osc_probe |d| ~0.2 m with per-env std ~0.2, identical at damping 10 and 30,
  also with OSC kd zeroed and damping 1-3) while +x from grasped resets is smooth — unresolved;
  likely a subset of envs blowing up under commands. Needs per-env tracing under commands.
Recommendation stands: emulate the velocity clamp in the action term (saturating damping once
|qd| > limit) and/or move the D-term to implicit joint damping, then retune against
`osc_probe.py` (PhysX ref 0.038 m / 10 steps, direction-consistent, small per-env std).

## 15. Newton: rendering solved, OSC instability root-caused to gripper mass (2026-09-12)

### 15a. Headless rendering without RTX (also the H100 RGB path)
IsaacLab 3.0 already ships a non-RTX tiled camera: `CameraCfg(..., renderer_cfg=NewtonWarpRendererCfg())`
(`isaaclab_newton.renderers`) routes the camera through Newton's Warp raytracer
(`newton.sensors.SensorTiledCamera`; pure CUDA, no RT cores, no Kit/Fabric sync). It works
`--headless` on the Newton backend and produces proper frames (arm, gripper, peg, hole, table).
`scripts_v2/tools/diagnostics/policy_eval_video.py --renderer newton` uses it; the first visual of
Newton (`/tmp/jtran_newton_warp_smoke.mp4`) showed the resets are placed correctly (gripper on the
peg in all 4 envs at t=0) and the blow-up happens in the first 5 policy steps.
Outputs available: color/hdr_color/depth/normal/albedo/instance-seg. This is the path for RGB jobs on
H100/H200 (Tillicum): `TiledCameraCfg` is deprecated in favour of `CameraCfg` + `renderer_cfg`.

### 15b. The OSC "instability" is a mass discrepancy, not a missing velocity clamp
Same reset, similar torque, one physics step (dt 1/120):
  PhysX  wrist_1 tau=-0.27 -> dqd=-0.015 rad/s ;  Newton tau=-0.25 -> dqd=-0.17 (11x)
  PhysX  shoulder_pan tau=0.9 -> 0.006        ;  Newton 0.9 -> 0.031 (5x)
Unit-torque probe (`/tmp/jtran_inertia_step.py`) + mass dump (`/tmp/jtran_mass_dump.py`, with
`env.events.randomize_robot_mass=null` etc. so values are nominal):
  - Arm links: identical (authored MassAPI honoured by both).
  - Gripper: PhysX robotiq_base_link 1.338 kg (3 colliders incl. D415 mount+cable) vs Newton 0.788;
    outer knuckles PhysX **1.0 kg, I=0.004** (no collider, no MassAPI on the cloud asset -> PhysX
    fallback; the recipe has always trained with this ~3.4 kg gripper) vs Newton 0.013 (our patch);
    fingers / inner knuckles ~2.7x heavier in PhysX (density x different volume estimate).
  - Effective wrist inertia on Newton was 3-5x lower (wrist_3 0.0056 vs 0.0166) with much stronger
    wrist coupling -> the PhysX-tuned explicit D-term is unstable there. The +-20 % per-link scatter
    seen with randomization on is just `randomize_robot_mass` (0.7-1.3 scale) with different draws.
Fix: `tools/fix_robotiq_usd.py --mass_from <physx nominal dump.json>` authors mass / centerOfMass /
diagonalInertia / principalAxes (eigendecomposition of the full tensor) on every rigid body ->
`usd/patched_newton_mass` (built on top of `patched_newton_tree`, Props copied). Newton honours all
of it: M diag wrist = 0.13/0.073/0.017 = PhysX; wrist M_eff 0.039/0.031/0.020 vs PhysX
0.033/0.037/0.017.
The velocity-clamp emulation (`RelCartesianOSCAction._apply_velocity_clamp`, auto on Newton,
`UWLAB_OSC_VEL_CLAMP=0/1`) was implemented first and did NOT help on the light gripper
(osc_probe |d| 0.34, std 0.19) - it was chasing a symptom. Keep it available; it is cheap.
Open at the time of writing: a zero-torque drift (elbow ~0.09 rad/s per step) on the heavy asset,
under test with gravity off to see whether `mjc:gravcomp` is incomplete.

### 15c. Result: arm dynamics and OSC step response now at PhysX parity
Fixed-pose unit-torque probe (`--pose 0,-1.2,1.5,-1.9,-1.57,0 --finger 0.4`), M_eff per joint
(PhysX / Newton `patched_newton_mass`): pan 1.74/1.68, lift 1.69/1.70, elbow 0.338/0.338,
w1 0.040/0.039, w2 0.047/0.043, w3 0.021/0.017. (Light asset `patched_newton_tree`: w1 0.024,
w3 0.0063 - the old 3-5x gap.) Caveat for probes: a single step right after a hard state write
with the fingers exactly on their limit (q=0) shows a gravity-dependent transient on the
shoulder/elbow response; it is gone after 4 steps or with the fingers mid-range.
`osc_probe.py` on Newton, heavy asset, `UWLAB_NEWTON_ARM_DAMPING=0` (PhysX-identical actuator):
  clamp off: +x 0.038 +y 0.039 +z 0.036 -z 0.037 (PhysX 0.038), std 0.00-0.04 (a few envs wobble on +z/rx/rz)
  clamp on : same means, std <= 0.002 on every command, rz 0.166 rad vs PhysX 0.161.
So: mass parity is the fix; the velocity-clamp emulation removes the residual outliers and stays
default-on for Newton (`UWLAB_OSC_VEL_CLAMP`). `UWLAB_NEWTON_ARM_DAMPING` should now be 0 (the
viscous stand-in is no longer needed and would deviate from PhysX).
Asset to use for Newton from now on: `UWLAB_ROBOT_ASSETS_DIR=usd/patched_newton_mass` with
`Datasets/OmniReset_patched` (swapped_ifk datasets).

### 15d. Remaining blocker: pad-peg contact under Newton (2026-09-13)
After the mass fix, the 2.x expert still scores 0/512 on Newton (0/146 even on
ObjectPartiallyAssembledEEGrasped). Video (`policy_eval_video.py --renderer newton`) shows the arm
behaving sanely but every peg being ejected from the gripper within the first policy step.

Found and fixed on the way: `_map_dataset_joints` short-circuited on equal joint COUNT (12) and
assumed dataset order == articulation order. Newton orders the gripper joints by tree traversal
(left branch first), so dataset gripper values were written to the wrong joints and the mimics then
snapped the linkage open (r_inner_knuckle -> 1.03 rad, pad gap 0.066 -> 0.113). Now always
scattered by name. Linkage is consistent through holds (linkbreak 0 %).

What is measured now (`lift_probe.py`, 128 envs, ObjectAnywhereEEGrasped; PhysX ref: held@hold
99.2 %, held@lift 91.3 %, finger q stays 0.53 = blocked by the peg):
  Newton default (margin 0.005, ke 2.5e3/kd 100, pyramidal, impratio 1): hold 15 %, lift 0 %,
    finger closes 0.53 -> 0.61 with the peg between the pads, peg falls within ~5 steps.
  margin 0: finger closes to the 0.785 limit -> the pads pass completely through the peg.
  /tmp/jtran_contact_probe.py: both pads DO register a contact with the peg (peg-pad = 2 for steps
    1-3), then the peg creeps down out of the pads. Pad mu = 100 (fingertip PhysicsMaterial), peg mu
    0.36-1.8, condim 3. nconmax 1000 vs 50000: identical (not a contact-budget starvation).
  Tried, none hold: stiffer contacts (ke 4e4 / kd 400 -> solref 5 ms) -> NaNs; elliptic cone +
    impratio 10 (with/without margin) -> 5 % lift, some linkbreak; pyramidal impratio 10 -> finger
    stops at 0.527 like PhysX but the peg is still not held (14.6 % hold); MuJoCo-native contacts
    (`UWLAB_NEWTON_MJ_CONTACTS=1`) -> 63-71 % abnormal/NaN, elliptic variant crashed.
  Mimic equality solref 0.02 -> 0.005 / 0.002 (`UWLAB_NEWTON_EQ_SOLREF` hook in lift_probe): fixes
    linkbreak, does not change hold.
Reading: the dataset grasps hold the peg at the very tip of the pads (see PhysX render
/tmp/jtran_grasp/rtx_t3.png). PhysX's hard contacts tolerate that; Newton's soft contacts let the
pads sink ~7 mm into the peg and the converging pad tips squirt it out; friction creep does the rest.
Knobs now in the Newton cfg (env vars): UWLAB_NEWTON_CONTACT_KE/KD, _MARGIN, _CONE, _IMPRATIO,
_NCONMAX, _NJMAX, _MJ_CONTACTS, _SUBSTEPS, _ARM_DAMPING (default 0), UWLAB_OSC_VEL_CLAMP.
Next candidates (untested): (1) port `gripper_close_on_peg.py` to Newton to test a centred grasp
without the datasets; (2) replace the thin pad convex hull with a box collider of the same extents
(convex-convex normals on thin plates are the likely culprit); (3) per-shape solimp on pads/peg
(`mjc:solimp`, high d0/dwidth -> near-rigid) instead of global ke/kd; (4) regenerate Grasps/reset
datasets under Newton so the grasps are Newton-consistent (deeper pinch) - this is the option that
does not require contact parity, but it changes the task distribution.
Newton camera quirk: `set_world_poses_from_view` only takes effect after `env.reset()` or
`cam.reset()` (works in policy_eval_video.py because the pose is set before the first reset).

### 15e. THE grasp blocker: Newton drops root-pose writes on fixed-base articulations (2026-09-13)
Method: force the same dataset state on both backends (`/tmp/jtran_obs_diff.py`, randomization off)
and diff obs / link poses. Gripper bodies relative to robotiq_base_link: identical to <0.5 mm and
<0.004 quat. But every robot link in WORLD is offset by (-2.6, +26.3, -4.0) mm on Newton: PhysX
base_link = the dataset's root pose (0.7526, -0.7763, 0.004), Newton base_link = env origin exactly,
while `root_pos_w` reports the written pose on both. The datasets randomize the robot base per state
(x/y/z +-1 cm, y offset -3.9 cm), so on Newton the gripper was ~2.7 cm off the peg at every
"grasped" reset -> peg squeezed out within one policy step -> policy opens (grip +10) and fumbles.
Every contact-parameter experiment in §15d was measuring this, not the contact model.
Mechanism: `write_root_link_pose_to_sim` writes `joint_X_p` (Newton keeps a fixed base's pose in the
root joint's parent transform); MuJoCo-Warp has its own copy of body positions that is only
re-uploaded by `notify_model_changed(JOINT_PROPERTIES)` (`_update_joint_properties`, "mocap body
transforms first (fixed-root bodies have no MuJoCo joints)"). Nothing in IsaacLab triggers that, so
the base snaps back to the spawn pose at the next step. Floating-base robots are unaffected
(joint_q). This bites ANY fixed-base robot whose reset writes a root pose (datasets, base
randomization, multi-robot layouts).
Fix (sandbox): `_notify_newton_fixed_base_moved(articulation)` in `mdp/events.py`, called right after
the root-pose write in `MultiResetManager._reset_to` -> `NewtonManager.add_model_change(
SolverNotifyFlags.JOINT_PROPERTIES)` (applied at the next physics step). No-op on PhysX. Upstream
this into isaaclab_newton `Articulation.write_root_link_pose_to_sim_*` for fixed bases.
Also verified along the way: `joint_pos` policy obs now sign-corrected for the swapped ifk joints
(`joint_pos_signed`, Newton cfg) - correct but was not the cause; box pad / box peg colliders make
no difference (assets `usd/patched_newton_box`, `usd/patched_newton_pegbox` can be deleted).

### 15f. Contact settings that hold the peg (on the corrected geometry) - first Newton successes
With the fixed-base root pose actually applied (§15e), the contact matrix finally separates:
  lift_probe (128 envs, ObjectAnywhereEEGrasped; PhysX ref hold 99 % / lift 91 %):
    default (ke 2.5e3/kd 100, pyramidal, impratio 1, margin 0.005) : hold 14 %  lift 0 %
    margin 0 only                                                  : hold 15 %  lift 0 %
    stiff contacts only (ke 4e4 / kd 400 -> solref 5 ms)          : hold 11 %  lift 0 %
    elliptic cone + impratio 10                                    : hold 95 %  lift 35 %  linkbreak 92 %
    stiff + elliptic + impratio 10 + margin 0                      : hold 94 %  lift 75 %  linkbreak 88 %
  Friction-constraint stiffness (impratio) is the decisive knob; contact stiffness helps on top.
  The residual is the tree gripper's mimic equalities yielding under load (linkbreak, pad gap
  0.062 -> 0.13); `set_newton_equality_solref` (startup event, `UWLAB_NEWTON_EQ_TIMECONST`,
  default 0.005 s) removed linkbreak completely in the earlier test.
Newton task defaults are now: ke 4e4, kd 400, elliptic, impratio 10, margin 0, eq timeconst 5 ms,
arm damping 0, velocity-clamp emulation on. All overridable by env var.
Expert (2.x PhysX-trained, `expert_seed0_rslrl52.pt`), 64 envs x 2 on Newton with these defaults:
  ObjectAnywhereEEAnywhere 1/30, ObjectRestingEEGrasped 0/36, ObjectAnywhereEEGrasped 0/29,
  ObjectPartiallyAssembledEEGrasped 12/33   (was 0/512 before §15e).
Contact-detail probe (old defaults): pad-peg contacts existed but carried |F| = 0 with normals
~45 deg off the pad face - the peg was being squirted, not held.
Known nuisance: one lift_probe launch died with `malloc(): unaligned tcache chunk detected`
during Kit startup (heap corruption, not reproducible on the next launch).

### 15g. Newton training from the expert: local smoke + Tillicum launch (2026-09-13)
Training task `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Newton-v0` (TrainNewtonCfg: Stage-1
recipe + `_apply_newton_overrides(keep_critic=True)`). Critic kept at the PhysX layout (204 dims):
`get_material_properties`/`get_mass` have Newton paths (per-collider mu/restitution from the Newton
model; `num_shapes` tiles robot->16, receptive->1) so the expert's critic loads.
Robustness: `abnormal_robot_state` now also terminates non-finite joint states (NaN compares False,
so blow-ups never terminated before); `_NanSafeRslRlVecEnvWrapper` in train_expert_init.py zeroes
non-finite rewards/obs. Measured NaN rate: 0-1 events per 1024 envs x 120 steps (rare solver
blow-up on a grasped reset); substeps 8 / eq timeconst 0.01 / softer contacts made no difference.
train_expert_init.py now calls `apply_local_object_assets(env_cfg)` (convexified peg/peghole under
UWLAB_ROBOT_ASSETS_DIR) - the eval scripts always did, the trainer did not.
Local smoke (starfish, 2048 envs, 6 iters): expert loads (actor+critic), 11.3 s/iter, 7.8 GB GPU,
task_3 success 0.22 -> 0.50, task_1 0 -> 0.08 in 6 iterations.
Tillicum: image has newton 1.2.1 / mujoco_warp 3.8.0.3 / isaaclab_newton 0.13.6 = starfish.
`docker/cluster/submit_tillicum_omnireset_newton.sh` (tripwires: gSDE overlay, asset/dataset dirs,
fix markers in events.py / __init__.py / terminations.py / train_expert_init.py) ->
job 293860, run `newton_expert_resume`, 16384 envs/rank x 4. Kit-startup `malloc(): unaligned
tcache chunk` flake seen ~1/10 launches locally: resubmit if the job aborts before iteration 0.

### 15h. Tillicum launches: startup flake + self-resubmission
293860: rank 0 died during scene creation (torchrun reported -9 after the other ranks' 600 s NCCL
timeout); 293897: rank 0 SIGABRT at ~90 s with `malloc(): unaligned tcache chunk detected`, right
after "Registered backend 'newton' for factory FrameView" = inside Newton's USD import. The same
abort hits ~1/10 single-process launches on starfish (lift_probe, nan probe), always at that phase,
never once training runs. Not memory (14 GB RSS; nodes have 2 TB, per-GPU cap 240 GB -> --mem 960G).
Mitigation: `submit_tillicum_omnireset_newton.sh` resubmits itself (ATTEMPT/MAX_ATTEMPTS=6) when the
apptainer exec exits non-zero before any "Learning iteration" line; RUN_NAME/NUM_ENVS/RESUME_PATH are
forwarded. 293900 is the first job with the retry. Root-causing the heap corruption (Newton import_usd
/ pxr threading?) is open; a workaround worth trying is OMP_NUM_THREADS=1 / PXR_WORK_THREAD_LIMIT=1
for the import phase.

### 15i. First Newton fine-tune (293900) collapsed; relaunch with critic warm-up (293936)
293900 (`newton_expert_resume`, 16384x4, PPO cfg as-is: lr 1e-4 adaptive-KL): task_3 0.44 at it 0,
all four tasks non-zero by it 5-6 (task_0 0.018, task_1 0.047, task_2 0.046, task_3 0.43), then a
monotone collapse: it 12-13 task_0-2 = 0.000, task_3 0.15, abnormal_robot termination rate
0.0015 -> 0.018 with its -100 penalty dominating the mean reward. Cancelled at it 13.
Reading: PPO with a mismatched critic + adaptive LR (rsl_rl multiplies lr by 1.5 whenever KL is
small) wrecks the expert faster than it adapts, and the drifting policy drives more Newton blow-ups.
Relaunch 293936 (`newton_expert_ft`): `train_expert_init.py --critic_warmup_iters 8` (actor param
group lr=0 for 8 iterations - requires_grad=False is rejected by rsl_rl's actor-grad-norm logging),
`agent.algorithm.schedule=fixed agent.algorithm.learning_rate=3e-5`. Submit-script knobs:
CRITIC_WARMUP, PPO_LR, PPO_SCHEDULE. The trainer now prints `[step-time]` (rank 0, mean env.step
over 32 steps) to pin down the throughput gap below.
Throughput mystery (open): single-GPU H200 profiles give 1.4 s/env.step at 16384 envs (=> ~34 s of
collection per iteration), yet 293900's collection took 310-410 s (10x). Not physics state (free
space 655 ms vs in-hole 541 ms per step at 4096; nefc 23-36 per world), not solver iterations
(converges in 1), not the clamp/notify/reset code (profiled), GPUs at 100 %, each rank 1 core at
100 %. Suspects left: the 4-rank process (distributed obs-normalizer all-reduce every step, lock-step
with the slowest rank), or the training-time state distribution (random-ep-len + gSDE noise) hitting
something superlinear. `[step-time]` in 293936 will say whether env.step itself is slow in-training.
Newton tuning knobs also added: UWLAB_NEWTON_ITERS / _LS_ITERS / _CUDA_GRAPH.

### 15j. Throughput mystery solved: MuJoCo-Warp warm start survives IsaacLab resets
Single-rank trainer at 16384 envs ran 7-10 s/env.step and got slower every iteration, while the
same env stepped from clean resets ran 1.4 s. Cause: after a blow-up the world's
`mjw_data.qacc_warmstart` (and qacc) hold huge/NaN values; IsaacLab's reset writes only q/qd, so
the Newton solver starts every substep from garbage in that world, runs to the iteration cap and
keeps re-diverging. Cost therefore grows with the cumulative number of blow-ups, and those worlds
feed garbage transitions to PPO - which is also the likely mechanism behind 293900's collapse.
Fix: `_notify_newton_fixed_base_moved` now zeroes `qacc_warmstart` / `qacc` rows of the reset
worlds (hand-off §15e hook, events.py). Result: 1.9-2.9 s/env.step at 16384 (physics 1.7-2.7 s,
action 0.1 s, resets <=0.13 s, obs/reward 0.01 s), solver niter max 1-3, zero non-finite worlds;
4-rank 293938 at ~112 s/iteration (was 310-410). The trainer's `[step-time]` line now reports
per-phase timing, solver niter stats and the non-finite-warm-start world count every 32 steps.
Upstream candidate: isaaclab_newton MJWarp manager should reset per-world solver state on env reset
(it only does so for Kamino via `_world_reset_mask`).
Remaining physics cost grows mildly with scene messiness (contact volume); levers measured at 4096:
substeps 2 = 1.3x, pyramidal cone = 1.6x (but loses the grasp). Buffer sizes under test (293940).

### 15k. 293938 -> 293963: solver-stuck termination
293938 (warm-up 8 + lr 3e-5 fixed, warm-start clearing) was stable for 18 iterations: task_3
0.44-0.50, tasks 0-2 at 0.000 (no collapse, no improvement yet), abnormal_robot 0.8-1.3 %/episode.
But 1-3 worlds per 32-step window sat at the solver iteration cap (jammed after a blow-up without
tripping the joint-velocity check), and since every world waits for the slowest, step time went
1.9 -> 3-7 s. New termination `newton_solver_stuck` (terminations.py; Newton task cfg, opt-out
`UWLAB_NEWTON_STUCK_TERM=0`): terminate any env whose world hit `opt.iterations` this step, so the
reset hook clears its warm start. Relaunched as 293963 (`newton_expert_ft2`), same recipe otherwise.
Profile 293940 (16384 envs, in-hole): nconmax 256/njmax 512 vs 1000/2000 = 1.05x (not worth it);
substeps 2 = 1.3x (kept 4 for now).

### 15l. Why the expert cannot insert on Newton: the peg rotates inside the grasp
Method: identical dataset states on both backends (`/tmp/jtran_obs_diff.py`, `/tmp/jtran_seat_probe.py`,
`/tmp/jtran_spin_probe.py`; note `env_cfg.seed` alone does NOT give identical samples across
backends - RNG consumption differs - force `_reset_to` on fixed indices).
- Seating physics matches: scripted close+push-down from PartiallyAssembled states gives success
  0.45 (PhysX) vs 0.44 (Newton); the CoACD base-plate hull top is 0.7 mm above the true hole floor
  (peg z 0.0155 vs 0.0148), inside the 2.5 mm threshold. Peg friction (0.1-0.6) irrelevant.
- Obs/action diff on identical states: positions match to mm at t=1, but the expert's actions
  diverge immediately because `insertive_asset_pose` (peg orientation in the wrist frame, axis-angle)
  differs. Some rows are the harmless axis-angle sign flip near 180 deg; the rest are real: the peg
  rotates in the grasp on Newton (env 5: 35 deg in one step).
- Spin probe (hold gripper closed 20 steps, no arm motion): PhysX p50 0.1 deg, p90 0.5 deg;
  Newton p50 5 deg, p90 46 deg, 33-38 % of grasps rotate > 10 deg (held: 95 %). Same under +x push.
- Tried on Newton: mimic equalities stiffened (0.005 s), condim 4 + torsional friction 0.02 / 0.1
  (`mjc:condim` via /tmp/author_condim.py -> usd/patched_newton_v2; torsional set at runtime by the
  `set_newton_torsional_friction` startup event, `UWLAB_NEWTON_TORSIONAL`): p90 37-46 deg, no change.
  Driving the five passive joints with the gripper PD (`BinaryJointPositionMimicActionCfg`,
  `UWLAB_NEWTON_MIMIC_DRIVE=1` default, actuator extended to all six joints) fixes holding
  (lift_probe held@lift 0 % -> 80.5 %, PhysX 91 %) but not the rotation (p90 46 deg).
Reading: Newton's convex narrowphase returns one intermittent contact point per pad; PhysX's
multi-point manifold on the flat pad faces gives a true two-face clamp. Being tested last:
box-box pad/peg colliders (`usd/patched_newton_box`, primitive pair -> manifold).
Training consequence: with the peg's in-grasp orientation drifting, the expert's insertion never
completes (task_0-2 ~0-2 %), and PPO cannot learn its way out at 3e-5 (293963: flat at task_3
~0.5 for 40 iterations). Fix the grasp rigidity first; relaunch afterwards.
Box-box result: `usd/patched_newton_box` (Cube colliders for pads + peg) is worse (held 34 %,
rotation p50 80 deg) - the injected boxes are not placed like the hulls; abandon that asset.

### 15m. Morning status (2026-09-13 ~07:30)
Running: Tillicum 293963 `newton_expert_ft2` (16384x4, critic warm-up 8, lr 3e-5 fixed,
warm-start clearing, stuck-world termination), ~100-115 s/iter, stable, FLAT: task_3 ~0.50,
task_0-2 ~0-1 %, abnormal 0.6 %, stuck 0.3 %. It will not recover the expert without the grasp
fix (§15l); left running so the curve is visible on wandb - cancel if the GPU-hours matter.
Everything below is on starfish (working tree, uncommitted), mirrored to Tillicum
/gpfs/scrubbed/profjat/uwlab3 and to the local repo (hand-off, fixer, submit scripts, trainer).
What is fixed and verified on Newton: rendering (Warp tiled camera), gripper mass parity, root
pose of fixed-base articulations, dataset joint order, joint-sign obs, solver warm start on reset,
stuck-world termination, contact settings (elliptic/impratio 10/stiff), mimic-driven passive joints,
critic obs layout, NaN-safe training, self-resubmitting Tillicum job, per-step timing/solver stats.
Open: grasp rigidity (peg rotates in the two-pad pinch; PhysX manifold vs Newton single contact).
Candidates: (1) MuJoCo-native contacts for the pad/peg pair only (multiccd manifolds) if the NaN
issue there can be tamed on the corrected geometry; (2) approximate the pad face with 2-4 small
sphere/capsule colliders (several contact points per pad); (3) regenerate grasp datasets under
Newton so grasps are Newton-stable, then train from scratch instead of resuming the PhysX expert.

### 15n. NaN in per-term reward logs (ee_asset_distance / dense_success_reward / joint_vel)
Seen on wandb for 293963 in 86/480 iterations. Mechanism: a blown-up world is terminated by
`abnormal_robot` (non-finite check) but the reward of that step is computed from the non-finite
state before the reset; the three state-based terms return NaN and `RewardManager` averages the
per-term episode sums as-is. Training is unaffected (the trainer wrapper zeroes the total reward);
only the Episode_Reward/* means are poisoned. Fix (Newton cfg override): every function-type reward
term is wrapped with `torch.nan_to_num(..., 0)` via functools.wraps (signature check intact).
Applies from the next launch; 293963 keeps logging NaN for those terms until then.

### 15o. ft2 cancelled; algorithm deviations reverted; Newton controls launched (2026-09-13 15:xx)
wandb comparison (b1092vqk = PhysX control from scratch, dnhhwgnu = ft2 resume-from-expert):
the PhysX control also sits at eoe ~0.15 / t3 ~0.6 / t0-2 = 0 for ~250 iterations, with entropy
growing 7.9 -> 18 and adaptive lr climbing to ~1e-3, before breaking through at it 300-400 (all
tasks ~0.97 by it 700). ft2 kept the expert's residual t0-2 (0.02-0.08) only until it ~30, then
sat at t3 ~0.5 with entropy decaying 19 -> 12 under the fixed 3e-5 lr. The "no early abnormal
spike" and "lower entropy" in ft2 are the resume + warm-up + fixed lr, not Newton. Decision: any
algorithm-side deviation from the omnireset recipe is out; levers are env/assets/solver only.
Reverted: submit script defaults (CRITIC_WARMUP=0, no agent.* overrides). Kept, env-side and
opt-out: OSC velocity-clamp emulation (`UWLAB_OSC_VEL_CLAMP`, PhysX enforces velocity_limit_sim
in-solver; MuJoCo has no equivalent; measured effect = removes OSC step-response outliers only),
newton_solver_stuck termination, NaN-safe reward/obs sanitising, non-finite abnormal_robot.
Resume/fine-tune runs now continue the checkpoint iteration counter (`train_expert_init.py`
load_cfg iteration=True; `--restart_iteration` to opt out) so the wandb x-axis continues.
Launched (stock train.py, 14336x4, same as the PhysX control): 294502 newton_control_clamp1,
294503 newton_control_clamp0 (`FROM_SCRATCH=1 UWLAB_OSC_VEL_CLAMP=<1|0>`). Judge after it ~400.
Throughput: Newton 25-40k fps vs PhysX 85-97k fps at this scale (2.5-3.5x slower per env-step).

## 15p. From-scratch Newton controls: velocity-clamp on vs off (2026-09-13/14)

Jobs: Tillicum 294853 `newton_control_clamp1` (wandb 8dha4e7a) and 294854 `newton_control_clamp0`
(wandb phcj5cq7). Both `FROM_SCRATCH=1`, stock PPO recipe, 14336 envs x 4 GPU, patched_newton_mass
assets, OmniReset_patched dataset, convexified peghole (hydra `_apply_local_object_assets_if_configured`,
§15o). Reference: PhysX control b1092vqk (`2026-09-11_06-11-51_gsde_mlp_fix`).

Startup gotcha: the first submission (294814/294815) died on the submit script's own tripwire after
the asset call moved from `train_expert_init.py` into the hydra wrapper; marker list fixed.

| it  | PhysX eoe | clamp0 eoe | clamp1 eoe | PhysX H | clamp0 H | clamp1 H | abnormal (P/c0/c1) |
|-----|-----------|------------|------------|---------|----------|----------|--------------------|
| 125 | 0.15 | 0.14 | 0.13 | 14.7 | 13.7 | 11.7 | 0.3% / 1.0% / 0.04% |
| 300 | 0.26 | 0.14 | 0.14 | 16.0 | 16.0 | 13.6 | 0.7% / 1.6% / 0.13% |
| 400 | 0.62 | 0.16 | 0.15 | 16.7 | 16.1 | 13.6 | 1.0% / 1.9% / 0.1%  |
| 500 | 0.67 | 0.26 | 0.16 | 16.8 | 15.8 | 13.7 | |
| 600 | 0.77 | 0.32 | 0.24 | 17.4 | 15.8 | 13.0 | |
| 700 | 0.94 | 0.35 | 0.36 | 17.5 | 16.1 | 12.5 | |
| 800 | 0.96 | 0.40 (it 769) | 0.53 | 17.5 | 16.2 | 12.2 | 1.1% / 2.0% / 0.08% |

Per-task at the last read: clamp1 (it 832) t0 0.25 / t1 0.63 / t2 0.67 / t3 0.83;
clamp0 (it 770) t0 0.00 / t1 0.14 / t2 0.68 / t3 0.79 (t1 flat since it ~600).

Findings
- Clamp emulation OFF reproduces the PhysX exploration profile (entropy within 0.5 nats, action-rate
  penalty -0.080 vs PhysX -0.087, abnormal-robot terminations present) and leaves the plateau first
  (it ~390 vs ~520). Clamp ON suppresses entropy by ~3 nats, halves the action-rate penalty and
  nearly eliminates abnormal terminations (0.08 %).
- But clamp OFF plateaus on the reach-from-scratch resets (task_0 = 0, task_1 ~0.14 for 180 it)
  while clamp ON keeps climbing on every reset type and overtakes on eoe by it ~700. Neither is a
  collapse: clamp0 entropy stays ~16, LR normal.
- Reward-term breakdown (it 350): shaping terms (ee_asset_distance, dense_success) match PhysX to
  a few %, so the policies get equally close/aligned; the gap is entirely `success_reward`. clamp0
  pays an extra -0.03 in abnormal_robot penalty (2 % of episodes) — PhysX's native joint-velocity
  clamp holds that at ~1 %, the emulation over-corrects to 0.1 %.
- Both Newton runs learn the same curve as PhysX, ~2x slower in iterations (PhysX 0.96 at it 800;
  best Newton 0.53). Per-iteration wall time is also ~2x (45-60 s vs ~25 s), so ~4x in wall-clock.
  Consistent with the grasp-rigidity gap (§15l/§15m) being a rate penalty rather than a wall.

Decision (pending user): neither run matched the "clamp0 tracks PhysX" rule, so both were left
running. Recommendation: keep clamp1 as the Newton baseline (better asymptote so far), keep the
clamp emulation ON by default but revisit its aggressiveness (it zeroes cross-axis motion in the
OSC step probe; a softer clamp that reproduces PhysX's ~1 % abnormal rate rather than 0.1 % may
combine clamp0's exploration with clamp1's reach learning). Physics next step: regenerate grasp
datasets under Newton (§15l candidate 3).
