# Porting OmniReset to IsaacLab 3.0

Working notes from migrating `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0`
from IsaacLab 2.3.2 to 3.0.0-beta2.patch1 (`env_isaaclab3`).

**Status:** the PhysX port is at parity with 2.x. The 2.x expert scores 93 % on
ObjectAnywhereEEAnywhere resets in the 3.0 env, and a from-scratch 3.0 run on
Tillicum (job 287624) reproduces the 2.x learning curve — but only with the
rsl_rl gSDE patch in [silent failure 7](#7-rsl_rl-52-gsde-trains-a-one-nonlinearity-network).
The Newton (MuJoCo-Warp) backend also trains from scratch with the stock recipe;
that work is documented separately in `NEWTON_HANDOFF.md` (distilled) and
`ISAACLAB_3_GRASP_HANDOFF.md` §9–§15p (chronological).

The headline lesson: the expensive part of this migration was not the crashes.
It was the handful of changes that do not crash and are silently wrong.

---

## Environment

|                | 2.3.2 (`env_uwlab`)          | 3.0-beta2.patch1 (`env_isaaclab3`)          |
| -------------- | ---------------------------- | ------------------------------------------- |
| IsaacLab       | `2f91d7d`                    | `ffff603e`                                  |
| Isaac Sim      | 5.1.0                        | 6.0.1.0                                     |
| Python         | 3.11                         | 3.12                                        |
| torch          | 2.7.0+cu128                  | 2.11.0+cu128 (cu130 in the Hyak container)  |
| rsl-rl-lib     | 3.1.2 API (`959ccbc4`)       | 5.2.0 (`92f01d71`, `feature/manipulation`)  |
| wandb          | unpinned                     | `>=0.19.6,<0.20`                            |

Two things the table cannot express:

* `rsl-rl-lib 5.2.0` must carry the gSDE `list(self.mlp)` patch
  (`docker/cluster/patches/apply_rsl_rl_gsde_fix.py`; `--check` verifies). Without
  it every gSDE policy is a one-nonlinearity network — see silent failure 7.
* `wandb` must stay `<0.20` until `rsl_rl/utils/wandb_utils.py` drops
  `Settings(start_method="thread")` (loud failure below). Verify with
  `python -c "import wandb; wandb.Settings(start_method='thread')"` — the
  `uwlab_rl` pin does not stop a later `pip install` from upgrading it.

`isaacsim` 6.0.x ships wheels for Python 3.12 **only** — it cannot be installed
into a 3.11 env, which is why this needed a separate conda environment rather
than an in-place upgrade.

`env_uwlab` is deliberately **not** kept in sync. The `_isaaclab/IsaacLab`
checkout is shared by the whole repo and is now on the beta commit, and
`uwlab_rl/setup.py` has a single global rsl-rl pin. Rebuild `env_uwlab` against
these pins if you need it back.

---

## Silent failures

These produce a program that runs and trains. Nothing raises. They are the ones
worth internalizing.

### 1. Quaternion convention flipped: `(w,x,y,z)` → `(x,y,z,w)`

The big one. IsaacLab 3.0 changed quaternion element order across config,
sim data, and `isaaclab.utils.math`. `AssetBaseCfg.InitialStateCfg.rot` now
defaults to `(0,0,0,1)` and is documented `(x, y, z, w)`.

A 2.x literal reads as a completely different rotation:

| literal                    | meant (wxyz) | now reads as (xyzw) |
| -------------------------- | ------------ | ------------------- |
| `(1.0, 0.0, 0.0, 0.0)`     | identity     | **180° about X**    |
| `(0.707, 0.0, 0.0, -0.707)`| −90° about Z | **−90° about X**    |

IsaacLab left a diagnostic behind for exactly this — set
`WARN_ON_TORCH_QUATF_ACCESS=1` and every `.torch` read of a `quatf` array warns,
which is the fastest way to find call sites that still assume 2.x order.

Three distinct sub-cases, and only two of them are bugs:

- **Data → 3.0 math utils.** *Already correct.* Both sides moved to xyzw
  together, so it stays internally consistent. Do not "fix" these.
- **Hand-built quaternions.** Broken. `task_space_actions.py` built the OSC
  delta as `cat([cos(half), axis*sin(half)])` — scalar first — then passed it to
  `quat_mul`. Every orientation command composed the wrong rotation, silently.
  Now scalar-last.
- **Config literals.** Broken. 20 converted across `rl_state_cfg`,
  `reset_states_cfg`, `partial_assemblies_cfg`, `grasp_sampling_cfg`, and the
  UR5e asset. `Offset.quat` in `assembly_keypoints.py` also defaulted to wxyz
  identity, i.e. a 180° X flip on every default-constructed offset.

One lucky escape: the only metadata-sourced offset the state task uses is
`gripper_offset = [0.5, 0.5, 0.5, 0.5]`, which is symmetric and therefore
identical under both conventions.

**Not yet audited** (~15 literals): `factory_extension`, and the `xarm_leap`,
`leap`, `tycho`, `xarm_uf_gripper` assets. Those tasks will spawn silently
rotated until someone goes through them.

### 2. Recorded reset-state datasets carry no convention marker

`recorders.py` writes `root_state_w[:, 3:7]`. The slice indices did not change,
so the recorder kept working across the migration — what changed is the element
order *inside* the slice.

- Datasets recorded under 2.x hold **wxyz**
- Datasets recorded now hold **xyzw**
- `info.yaml` stores only the two USD paths — nothing distinguishes them

Mixing them is silent. Recommend stamping `quat_convention: xyzw` into
`info.yaml` before regenerating anything.

Note the existing `reset_state_datasets/ObjectAnywhereEEAnywhere/*.pt` is a
129-byte git-lfs pointer; `git lfs pull` to materialize it.

### 3. `ensure_cuda_torch()` silently downgraded torch

`uwlab.sh` runs `ensure_cuda_torch` a second time *after* the IsaacLab installs,
and it force-reinstalls a hardcoded version. It was clobbering the torch 2.10
that pip had correctly resolved for `isaacsim-core` back down to 2.7.0. pip
reports this only as a dependency-conflict warning at the end of a long install,
which is easy to scroll past. Non-ARM pin is now 2.10.0 / 0.25.0.

### 4. A git pin can be "changed" without being installed

Both `vendor/leggedrobotics` and `feature/manipulation` report version `5.2.0`,
so `pip install` saw the version already satisfied and **skipped the reinstall**.
The pin in `setup.py` was correct while the environment ran different code.

```bash
# the only reliable check
python -c "import importlib.metadata as m; \
  print(m.distribution('rsl-rl-lib').read_text('direct_url.json'))"
# and the fix
pip install --force-reinstall --no-deps "rsl-rl-lib @ git+...@<sha>"
```

### 5. The wrong rsl-rl branch passes the easy tests

`vendor/leggedrobotics` is a clean mirror of upstream v5.2.0: right API, **none**
of UWLab's additions (gsde: 0 files, DAgger: 0 files). Cartpole trained fine on
it. OmniReset needs both. Use `feature/manipulation` — same API, plus the
UW-Lab features.

### 6. pytorch3d was skipped without comment

The wheel table in `uwlab_tasks/setup.py` had cp310/cp311 entries only, guarded
by `if py in wheel_by_py`. On cp312 the guard just... skipped, and the missing
dependency surfaced much later as an import error. There is no cp312 wheel built
against torch 2.10 (newest is pt2.8.0), so the import is now lazy — only
point-cloud tasks need it, and they raise a clear error pointing at the table.

### 7. rsl_rl 5.2 gSDE trains a one-nonlinearity network

The cause of the 3.0 OmniReset plateau (task_0 pinned at 0, task_1 capped ~0.13
while task_2/3 learned), and the most expensive item in this list. In
`feature/manipulation`, `MLP.__init__` reuses one activation module instance and
`MLPModel.forward`'s gSDE branch walks `list(self.mlp.children())`. `children()`
de-duplicates by identity, so the network that is trained and rolled out is
Linear-ELU-Linear-Linear-Linear-Linear. Exported policies use the full
`self.mlp` and are therefore a *different function* than the one trained.

Found by loading the 2.x expert into the 5.2 runner and comparing its forward
against a hand-built MLP (mean |a| 96 vs 4.8). Fix: `children = list(self.mlp)`
in `rsl_rl/models/mlp_model.py`. Applied to `env_isaaclab3` site-packages on
starfish (re-run the applier after any reinstall) and overlaid in the cluster
jobs via `PYTHONPATH=/workspace/uwlab/rsl_rl_patched` (submit scripts print
`[INFO] gsde fix : True`). Not yet upstreamed to `UW-Lab/rsl_rl`; the `.sif`
still ships the broken copy. Any gSDE checkpoint trained before the fix is a
one-hidden-layer policy — do not fine-tune from it.

Two related traps: `OnPolicyRunner.load(path, load_cfg=...)` with a partial dict
loads *nothing* unless `"actor": True, "critic": True` are present (silent), and
when a learner underperforms while a known-good policy scores fine in the env,
suspect the model forward before the simulator (`actor_parity_probe.py` in
`scripts_v2/tools/diagnostics/`). Details: `ISAACLAB_3_GRASP_HANDOFF.md` §10.

---

## Loud failures

These crash immediately. Cheap to fix once you know the cause.

**`ModuleNotFoundError: isaacsim.core.utils`** — behaves differently across
6.0.0 and 6.0.1, which matters:

* Under **6.0.0** the extension ships under `isaacsim/exts/` but is not enabled,
  so enabling it via the Kit extension manager put it back on `sys.path`. That
  hook is still in `uwlab_tasks/__init__.py`.
* Under **6.0.1** enabling reports success but no longer joins `sys.path`, so the
  workaround is dead. The helpers moved to `isaacsim.core.experimental.utils`,
  which still provides `create_bbox_cache` / `compute_obb` / `get_obb_corners` —
  only `compute_obb` changed signature (`compute_obb(prim, *, bbox_cache=...)`).
  Other sites map to `isaaclab.sim.utils.legacy` (prims) and
  `isaaclab.sim.SimulationContext`.

**Package split, 4 → 13.** `uwlab.sh` still installed only `isaaclab`,
`isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl`. 3.0 splits core into 13
packages plus `newton` / `rl` / `visualizer` extra-feature installs. Missing
`isaaclab_physx` alone breaks imports even when the backend is Newton.

**`quat_inv() expected Tensor, found ProxyArray`** — 3.0 returns warp-backed
`ProxyArray` from `.data.*`. Most torch ops pass through a deprecation bridge,
but `torch.jit.script`'d math utils reject anything that is not a real Tensor,
and `wp.array.view()` is a dtype reinterpret with a different signature than
`Tensor.view()`. Use `.torch`. (~100 sites.)

**`KeyError: 'class_name'`** — *not* a version mismatch, despite matching the
symptom in the older pin commit. `isaaclab_rl`'s compat shim dispatches on
`type(cfg.policy) is RslRlPpoActorCriticCfg` — an **exact** type check.
`RslRlFancyActorCriticCfg` subclassed it, so it printed the "inferring actor /
critic" warnings and then fell through every branch, leaving `actor` MISSING.
The subclass was obsolete anyway: 3.0's base class now defines both
`state_dependent_std` and `noise_std_type`. It is now an alias — do not make it
a subclass again.

**`ValueError: Unknown standard deviation type: gsde`** — in rsl-rl ≥ 4.0 gSDE
is not a `std_type`, it is its own distribution class (`GsdeDistribution`)
selected via `distribution_cfg.class_name`. The legacy `policy` path maps
`noise_std_type` onto `GaussianDistribution.std_type`, which only accepts
`scalar` / `log`, so gSDE **cannot** survive that path. The agent config now
declares explicit `actor` / `critic` model configs.

**`ValidationError: start_method — Extra inputs are not permitted`** — a bug in
the pinned rsl_rl fork, not UWLab. `rsl_rl/utils/wandb_utils.py` builds
`wandb.Settings(start_method="thread")`; wandb removed that field and its
Settings model forbids extras. Worked around with `wandb<0.20`; the real fix is
dropping the kwarg upstream in `UW-Lab/rsl_rl`. This fires at `runner.learn()`,
i.e. after a full ~90s startup.

**`PermissionError: /tmp/Assets/...`** — machine-specific, not IsaacLab.
IsaacLab's asset cache defaults to `tempfile.gettempdir()`, and `/tmp/Assets` on
`starfish` is owned by another user. `env_isaaclab3` now sets `TMPDIR` via a
conda activate hook.

---

## Reproducing the run

```bash
conda activate env_isaaclab3          # sets TMPDIR via activate.d hook
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
    --num_envs 4096 --logger wandb --headless \
    env.scene.insertive_object=fbleg \
    env.scene.receptive_object=fbtabletop
```

Sanity checks on the output:

- `Total steps: 131072` = 4096 envs × 32 steps — confirms env count
- `grad_norm/actor_var` present — confirms `GsdeDistribution` is active
- `logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/<ts>/` has a growing
  `events.out.tfevents`, `model_*.pt`, `params/{agent,env}.yaml`, `git/UWLab.diff`

wandb needs `WANDB_API_KEY`; without it use `WANDB_MODE=offline`.

The IsaacLab pin is overridable without editing the script:

```bash
UWLAB_ISAACLAB_COMMIT=<sha> ./uwlab.sh -i
```

---

## Open items

1. **Audit the remaining ~15 quaternion literals** in `factory_extension` and
   the `xarm_leap` / `leap` / `tycho` / `xarm_uf_gripper` assets. Those tasks
   spawn silently rotated until someone goes through them.
2. **Stamp the convention into reset-dataset `info.yaml`.** The OmniReset
   datasets in use (`Datasets/OmniReset`, `Datasets/OmniReset_patched`) were
   regenerated under 3.0 and hold xyzw, but nothing in the files says so.
3. **Upstream the rsl_rl fixes** to `UW-Lab/rsl_rl` (gSDE `list(self.mlp)`,
   drop `Settings(start_method="thread")` in `wandb_utils.py`), rebuild the
   `.sif`, lift the `wandb<0.20` ceiling, and drop the `rsl_rl_patched` overlay
   from the submit scripts.
4. **Deprecation debt.** `RigidBodyPropertiesCfg`, `ArticulationRootPropertiesCfg`,
   `RigidBodyMaterialCfg`, and `ViewerCfg` still work via shims that are
   scheduled for removal in 4.0/5.0.
5. **`uwlab.sh` still installs the old 4-package list.** A clean rebuild will
   hit `ModuleNotFoundError: isaaclab_physx` — the 13 core packages and the
   `newton` / `rl` / `visualizer` extras were installed by hand into
   `env_isaaclab3`. Either enumerate them in `uwlab.sh` or delegate that step to
   the vendored `isaaclab.sh -i`.
6. **Nothing is committed.** The port lives on `fix/omnireset-task-parity` on
   starfish as ~100 modified/untracked files; the Tillicum tree is an rsync of it.
7. **Newton.** Trains from scratch with the stock recipe but ~2× slower per
   iteration and ~2× more iterations than PhysX; the grasp-rigidity gap and the
   rest of the Newton to-do list are in `NEWTON_HANDOFF.md` §5.

---

## Blockers found while trying to open a GUI

Both visualizer backends now work; the two blockers below are kept because the
first one is also what blocked Newton as a *physics* backend and its fix is
load-bearing for every Newton run.

### `--viz newton`: the Robotiq USD has reversed joints

```
ValueError: Reversed joints are not supported:
  /World/envs/env_0/Robot/left_inner_finger_knuckle_joint,
  /World/envs/env_0/Robot/right_inner_finger_knuckle_joint.
Ensure that the joint parent body is defined as physics:body0 and the child
as physics:body1 in the joint prim.
```

Two joints in the calibrated Robotiq 2F85 USD have `physics:body0` / `body1`
swapped. **PhysX tolerates this — Newton's USD parser does not.** That is why
every headless PhysX run in this migration worked while Newton fails.

This is not merely a rendering problem. `newton._src.utils.import_usd.parse_usd`
is the same code path used when Newton is the *physics* backend, so **the Newton
port is blocked on this asset**, not just the viewer. The failure surfaces
indirectly as `RuntimeError: Model must be set before calling
set_visible_worlds()`, which is only the downstream effect of the model never
being built.

The asset is remote
(`Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd`
under `UWLAB_CLOUD_ASSETS_DIR`). **Resolved without re-publishing:**
`tools/fix_robotiq_usd.py` writes patched copies (body0/body1 swapped on the two
joints, which flips their sign, plus explicit MassAPI on every gripper link) and
`UWLAB_ROBOT_ASSETS_DIR=<patched root>` redirects the robot cfg to them. Datasets
recorded with the original gripper must be converted
(`tools/convert_datasets_for_patched_gripper.py`; `reset_from_reset_states`
refuses to mix them). `usd/patched` is the PhysX-parity variant,
`usd/patched_newton_mass` the one Newton training uses. See
`ISAACLAB_3_GRASP_HANDOFF.md` §11 and `NEWTON_HANDOFF.md` §1.

### `--viz kit`: Isaac Sim version is behind the pin

`apps/isaaclab.python.kit` depends on `isaacsim.core.experimental.primdata`,
which does not ship in `isaacsim==6.0.0.0` (only `materials`, `objects`,
`prims`, `utils` are present). Kit fails dependency resolution and exits.
IsaacLab 3.0-beta2.patch1 pins `isaacsim[all,extscache]==6.0.1.0`; this
environment was built on 6.0.0.0. The headless experience does not pull that
extension, which is why only the GUI path is affected.

### Both GUI blockers are now resolved

`--viz kit` works. The Isaac Sim 6.0.0.0 -> 6.0.1.0 upgrade supplied the missing
`isaacsim.core.experimental.primdata`, and the `isaacsim.core.utils` imports were
ported properly (below). `--viz newton` works with the patched USDs above.
Headless rendering without RTX also works, on either backend, via
`CameraCfg(renderer_cfg=NewtonWarpRendererCfg())` (camera pose writes take
effect only after `env.reset()`/`cam.reset()`).

Two traps in that upgrade:

* **`torch` must be forced to the cu128 build.** `isaacsim-core 6.0.1.0` pins
  `torch==2.11.0`, and the default PyPI wheel for 2.11.0 is built against
  **CUDA 13.0**. This machine's driver is 12.6, so that wheel silently yields
  `torch.cuda.is_available() == False` -- the GPU disappears entirely. Install
  `torch==2.11.0+cu128` from the PyTorch cu128 index, and note pip will *skip*
  the swap because `2.11.0` already looks satisfied: `--force-reinstall` is
  required. (Same trap as the rsl-rl pin.)
* **`isaacsim.core.utils` was not deleted -- it moved.** Under 6.0.0 you could
  reach it by enabling the Kit extension; under 6.0.1 the extension still
  "enables" successfully but no longer lands on `sys.path`, so that workaround is
  dead. The helpers now live in `isaacsim.core.experimental.utils`, which has
  `create_bbox_cache` / `compute_obb` / `get_obb_corners` with one signature
  change: `compute_obb(prim, *, bbox_cache=...)` rather than
  `compute_obb(cache, prim)`. Other call sites map to `isaaclab.sim.utils.legacy`
  (prims) and `isaaclab.sim.SimulationContext`. While porting, note
  `mesh_converter.py` imported `isaacsim.coreutils.extensions` -- a typo missing
  a dot, so it had never resolved on any version.

---

## Cluster deployment status

**Training runs on both Hyak (klone, L40S) and Tillicum (H200).** The 4-GPU
distributed version needs `NCCL_CUMEM_ENABLE=0` (see below); every sbatch
script sets it.

Tillicum is where current runs go. Differences from klone, all encoded in
`docker/cluster/submit_tillicum_omnireset.sh` (PhysX) and
`submit_tillicum_omnireset_newton.sh` (Newton), with `.env.tillicum`:
`--account=weirdlab --partition=gpu-h200`, no `--constraint` (nodes have no
features; any constraint is unschedulable), filesystem root `/gpfs`
(`/gpfs/scrubbed/profjat/uwlab3` = rsync of the starfish tree,
`/gpfs/scrubbed/profjat/sif/uwlab_isaaclab3_cu130.sif`), node-local `/tmp` is a
3.5 TB NVMe, driver 610 / CUDA 13 → the cu130 image. Jobs request 4 GPUs /
960 G (~240 G per rank at 14336 envs) / 24 h; logs in
`/gpfs/scrubbed/profjat/uwlab3/logs/slurm-<job>.out`, checkpoints under
`logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/<timestamp>_<RUN_NAME>/`. Both
scripts build the `rsl_rl_patched` overlay at startup and refuse to run if any
of the port's fix markers is missing from the staged tree.

### Hyak (klone)

First run: job `39296918`, via `docker/cluster/submit_hyak_omnireset_1gpu.sh`,
4096 envs, ~6.1-6.4k steps/s, using the reset datasets generated here and
rsynced up. Logs land in
`/gscratch/weirdlab/profjat/uwlab/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/`.
(That run predates the gSDE fix and its checkpoints are one-hidden-layer policies.)

### How the environment got there

`cluster_interface.sh` builds its image with
`apptainer build ... docker-daemon://`, which needs a working local Docker
daemon -- unavailable on `starfish`. Apptainer will bootstrap from `docker://`
(a public registry) with no Docker at all, which is what
`docker/cluster/uwlab_isaaclab3.def` does. That is also the right answer for this
cluster rather than merely a workaround: `/gscratch/weirdlab` sits at ~97% of its
17M inode quota with ~451k files of headroom for the whole lab, and a
conda-style install of this stack is ~266k files. A `.sif` is **one** file.

Build it locally, ship it, submit:

```bash
apptainer build --fakeroot uwlab_isaaclab3.sif docker/cluster/uwlab_isaaclab3.def
rsync -a uwlab_isaaclab3.sif klone-login:/gscratch/weirdlab/profjat/sif/
ssh klone-login 'cd /gscratch/weirdlab/profjat/uwlab && sbatch docker/cluster/submit_hyak_omnireset_1gpu.sh'
```

### Everything that had to be fixed to get a job running

Each of these failed *only* in the containerized batch context -- none reproduce
in a local interactive run:

1. **EULA prompt.** Isaac Sim asks `Do you accept the EULA? (Yes/No):` on first
   run. Batch jobs have no stdin, so every rank exits 1 within ~19s. Normally
   cached in `$HOME` after one interactive accept, which `--containall` hides.
   Fix: `--env OMNI_KIT_ACCEPT_EULA=YES`.
2. **`--writable-tmpfs` is a ~64 MiB RAM overlay.** Isaac Lab's logging fills it
   during asset init; `logging.flush()` throws `Errno 28`, then `omni.datastore`
   spins in garbage collection. The job stays `RUNNING` at 0% GPU forever --
   it looks alive in `squeue` but never iterates. Fix: a directory overlay on
   node-local disk (`--overlay`).
3. **`--env HOME=...` is silently ignored.** Apptainer prints
   `Overriding HOME environment variable with APPTAINERENV_HOME is not permitted`
   and carries on. Combined with no `--pwd`, the run's `log_dir` resolved into
   the small private tmpfs and `dump_yaml()` died writing `env.yaml`. Fix:
   `--home` and `--pwd`. (Both `/tmp` and `/gscratch` had plenty of space --
   neither was the filesystem being written to.)
4. **PhysX caps materials at 64K per scene.** Every collision shape in every env
   gets one. `num_buckets` does not help -- it limits the distinct property
   *values* sampled, which are written into already-allocated materials. The
   measured ceiling for UR5e+Robotiq+peg+peghole is **14336 envs per process**
   (`docker/cluster/submit_tillicum_omnireset.sh` defaults to it); `--num_envs`
   is per process under `--distributed`, so 4 ranks give 57344 envs.
5. **torch CUDA build must match the *cluster* driver, not the workstation.**
   `starfish` runs driver 12.6 and needs the cu128 wheel; Hyak's L40S nodes run
   580.178.04 / CUDA 13.0 and want the cu130 build that isaacsim pulls by
   default. Carrying the starfish-specific cu128 override into the cluster image
   shipped NCCL `cuda12.9` against a CUDA 13.0 driver.

Useful cluster facts learned along the way: `SLURM_TMPDIR` is **not set** on
klone, so `${SLURM_TMPDIR:-/tmp}` resolves to `/tmp`, which on a compute node is
a 2.8 TB local NVMe. Accounts pair with partitions as
`--account=gpu-l40s-weirdlab --partition=gpu-l40s`.

### The 4-GPU blocker — SOLVED: `NCCL_CUMEM_ENABLE=0`

Symptom: all ranks build their sim, print `Synchronizing parameters for rank
0..N`, then die with `SIGSEGV` and no Python traceback.

Cause: Isaac Sim's warp/PhysX layer allocates through the CUDA virtual-memory
APIs, which collide with NCCL's `cuMem` path. Setting `NCCL_CUMEM_ENABLE=0`
fixes it.

Why it took so long to find — every component passes **in isolation**:

* single-GPU Isaac Sim runs, trains and logs;
* a 4-rank NCCL `all_reduce` in the same container with no Isaac Sim passes;
* per-rank GPU binding is correct (`cuda:0..N` via `app_launcher.local_rank`);
* `CUDA_VISIBLE_DEVICES` and `torch.cuda.device_count()` are right.

Only the *combination* of Isaac Sim and NCCL in one process fails, so no
component-level test reproduces it. Ruled out by experiment before finding the
real cause: `/dev/shm` size, NCCL P2P/IB, and the cu128-vs-cu130 torch build.

Two debugging lessons worth keeping:

* **Reproduce off-cluster.** The same failure occurs on a 2-GPU workstation with
  no container and no Slurm. That turned a 5-minute Slurm queue cycle into a
  2-minute local loop and is what made the fix findable at all.
* **Verify NCCL with the launcher the job actually uses** (`torch.distributed.run`).
  An `mp.spawn` probe fails here even when NCCL is healthy, which briefly made a
  *correct* fix look wrong.

Instrumenting `broadcast_parameters` to print each tensor was useful for ruling
out a shape/order desync: both ranks enqueue all 30 tensors identically, so the
crash is in the communicator, not the call sequence.

### Two environment fixes applied along the way

* **`TMPDIR`.** Isaac Lab caches downloaded assets under
  `tempfile.gettempdir()`. On `starfish`, `/tmp/Assets` and `/tmp/datasets` are
  both owned by another user (`iggy:iggy 775`, and we are not in that group), so
  the mkdir fails. `env_isaaclab3` now sets `TMPDIR` via a conda `activate.d`
  hook -- which only takes effect on `conda activate`, so a shell already inside
  the env must reactivate.
* **Shadowed `warp`.** `newton` requires `warp-lang>=1.13,<1.14`, but Kit ships
  its own `omni.warp.core-1.12.0` whose `warp/` directory shadowed part of the
  import tree, producing
  `ImportError: cannot import name 'get_deprecated_api' from 'warp._src.utils'`.
  Repointed that directory to a symlink at the environment's `site-packages/warp`
  (original kept alongside as `warp.bak-1.12.0`). This mirrors what Isaac Lab's
  own installer does via `_PREBUNDLE_REPOINT_PACKAGES` -- `warp` is on that list,
  but the installer only scans `pip_prebundle` directories and this copy lives
  directly in an `extscache` extension.
