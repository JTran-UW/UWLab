# OmniReset on Newton (MuJoCo-Warp) — Hand-off

Status as of 2026-09-14. Companion to `ISAACLAB_3_MIGRATION.md` (2.x → 3.0 port) and
`ISAACLAB_3_GRASP_HANDOFF.md` §13–§15p (the chronological lab notebook; this document is the
distilled version). Everything below lives on `starfish:~/research/UWLab`, branch
`fix/omnireset-task-parity` (uncommitted), mirrored to Tillicum at `/gpfs/scrubbed/profjat/uwlab3`.

## 0. Headline

- OmniReset (UR5e + Robotiq 2F-85 peg insertion, state obs) **trains from scratch on Newton with
  the stock PhysX recipe** (4 × H200, 14336 envs/GPU, unmodified PPO cfg). Tillicum job 294853 /
  wandb `8dha4e7a` (`2026-09-13_16-38-22_newton_control_clamp1`) is the best run: end-of-episode
  success **0.83 at iteration 1146** (task_0 0.63 / task_1 0.86 / task_2 0.86 / task_3 0.91) and
  still climbing. Checkpoints every 100 iterations in
  `/gpfs/scrubbed/profjat/uwlab3/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-09-13_16-38-22_newton_control_clamp1/`.
- Versus the PhysX control (wandb `b1092vqk`, `2026-09-11_06-11-51_gsde_mlp_fix`): same learning
  curve, **~2× slower in iterations** (PhysX 0.96 at it 800; Newton 0.53 at it 800, 0.81 at 1124)
  and **~2× slower per iteration** (45–65 s vs ~25 s), so ~4× in wall-clock.
- The PhysX-trained expert transfers to Newton only partially (12/33 PartiallyAssembled resets after
  all fixes; fine-tuning it plateaued at task_3 ≈ 0.5). Training from scratch is the working path.
- Non-RTX rendering works (Warp raytracer), so RGB jobs can run on H100/H200.

## 1. What was necessary (in dependency order)

Each item was a hard blocker; the file that carries the fix is listed so it can be found after a
rebase. Env-var knobs are read in `_apply_newton_overrides` in
`source/uwlab_tasks/.../omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py` unless stated.

| # | Blocker | Root cause | Fix |
|---|---------|------------|-----|
| 1 | Newton USD importer rejects the calibrated Robotiq gripper ("reversed" joints) | `left/right_inner_finger_knuckle_joint` authored with body1 as parent; PhysX tolerated it | `tools/fix_robotiq_usd.py` swaps body0/body1 (flips joint sign) → `usd/patched*`; datasets recorded with the original sign must be converted (`tools/convert_datasets_for_patched_gripper.py` → `Datasets/OmniReset_patched`). Selected at runtime by `UWLAB_ROBOT_ASSETS_DIR`. |
| 2 | OSC controller "unstable" on Newton (gripper flies) | Newton derived 1.06 kg for the gripper vs PhysX ~3.4 kg: outer knuckles have no collider, PhysX falls back to 1.0 kg / I = 0.004 per link, Newton uses the authored (tiny) values | `fix_robotiq_usd.py --mass_from tools/newton_probes/mass_physx_nominal.json` writes explicit MassAPI on every gripper link → **`usd/patched_newton_mass`** (the asset to use). Unit-torque effective-mass now matches PhysX on all joints. |
| 3 | Dataset joint values landed on the wrong gripper joints | Newton orders joints by tree traversal, PhysX by USD order; `_map_dataset_joints` short-circuited on equal joint *count* | `mdp/events.py::_map_dataset_joints` scatters by name unless the order is identical. |
| 4 | Every reset put the gripper 2.7 cm off the peg (expert 0/512) | Newton drops `write_root_pose_to_sim` on fixed-base articulations: the pose lives in `model.joint_X_p` of the root joint and MuJoCo-Warp re-uploads it only on `notify_model_changed(JOINT_PROPERTIES)`; datasets randomize the base per state | `mdp/events.py::_notify_newton_fixed_base_moved`: write `joint_X_p` rows for the per-env root joints + `NewtonManager.add_model_change(SolverNotifyFlags.JOINT_PROPERTIES)`. All links match PhysX to < 0.5 mm. |
| 5 | Training 10× slower over time, then PPO collapse | `mjw_data.qacc_warmstart` survives IsaacLab resets: a world that blew up stays at the solver iteration cap forever and emits garbage transitions | Same reset hook zeroes `qacc_warmstart`/`qacc` rows for reset worlds. Plus `newton_solver_stuck` termination (`mdp/terminations.py`, reads `solver_niter` vs `opt.iterations`, `UWLAB_NEWTON_STUCK_TERM=1`). |
| 6 | Pegs squeezed out of the pinch / pads pass through the peg | Newton's default soft pyramidal contacts + convex narrowphase | Newton solver/contact defaults in the task cfg: elliptic cone, impratio 10, contact ke 4e4 / kd 400 (solref 5 ms), margin 0, mimic-equality timeconst 5 ms (`set_newton_equality_solref` startup event), `iterations` 100 / `ls_iterations` 50, 4 substeps, CUDA graph on, arm joint damping 0. lift_probe 75 % (PhysX 91 %). |
| 7 | Peg drops during lift | Passive linkage joints not driven | `BinaryJointPositionMimicActionCfg` (`mdp/actions/`), gears {r_ok +1, r_ik −1, l_ik +1, r_ifk +1, l_ifk +1}, `UWLAB_NEWTON_MIMIC_DRIVE=1` (default). Hold 97 % / lift 80 %. |
| 8 | Peghole sealed under Newton (task_3 = 0 in training) | `physics:approximation = sdf` is not mapped by Newton's importer; its convexDecomposition path silently falls back to one hull | `tools/fix_object_usd_for_newton.py` bakes a CoACD decomposition into the object USDs (under `usd/patched_newton_mass/Props`); `apply_local_object_assets` redirects `insertive_object`/`receptive_object` to them. **Must run after hydra composition** (variants replace the whole object cfg) → `uwlab_tasks/utils/hydra.py::_apply_local_object_assets_if_configured`, called in both `hydra_task_config` and `hydra_task_compose`. Log line to look for: `[INFO] local object assets redirected: ['insertive_object', 'receptive_object']`. |
| 9 | NaN rewards / rsl_rl NaN abort | Rare exploded worlds produce non-finite states | NaN-safe reward variants in `mdp/rewards.py` (`ProgressContextNanSafe`, `ee_asset_distance_tanh_nan_safe`, `dense_success_reward_nan_safe`, `joint_vel_l2_clamped_nan_safe`) — importable functions, because hydra re-resolves callables by `module:qualname` and undoes in-place wrapping; `abnormal_robot_state` also fires on non-finite joint velocity (`uwlab/envs/mdp/terminations.py`). |
| 10 | Critic obs layout differs (material properties, mass) | Newton exposes per-shape, PhysX per-body | `mdp/observations.py` Newton paths (`get_material_properties` tiles `num_shapes`, `get_mass`, `_newton_collider_shape_ids`); training cfg keeps the PhysX critic layout so PhysX experts can be loaded. `joint_pos_signed` obs. |
| 11 | rsl_rl 5.2 gSDE trains a 1-nonlinearity net (affects PhysX too) | `MLPModel.forward` walks `mlp.children()` which dedups the shared ELU | `list(self.mlp)`; patched in starfish `env_isaaclab3` site-packages; on Tillicum the submit script builds `rsl_rl_patched/` overlay via `docker/cluster/patches/apply_rsl_rl_gsde_fix.py` (`[INFO] gsde fix : True`). |
| 12 | Kit/Newton USD-import heap corruption at startup (~1/10 launches) | `malloc(): unaligned tcache chunk` in the importer | Submit script self-resubmits when the abort signature appears before iteration 0 (only that signature; other failures are not retried). |
| 13 | No RTX on H100 | — | `CameraCfg(renderer_cfg=NewtonWarpRendererCfg())` (isaaclab_newton) renders headless without RTX. Camera pose writes take effect only after `env.reset()`/`cam.reset()`. |

Things that were **tried and reverted** (do not re-try without new evidence): OSC velocity-clamp
emulation OFF as default (§4 below), critic warm-up / fixed LR fine-tuning (deviates from the recipe
and collapsed anyway), condim 4 + torsional friction, box pad colliders, MuJoCo-native contacts
(NaN), larger nconmax, stiffer mimic equalities. The code and asset variants for these
(`UWLAB_NEWTON_TORSIONAL`, `UWLAB_NEWTON_MJ_CONTACTS`, `fix_robotiq_usd.py --pad_box`,
`--critic_warmup_iters` / `CRITIC_WARMUP` / `PPO_SCHEDULE` / `PPO_LR`, `usd/patched_newton{,_v2,_box,
_pegbox,_tree}`) were removed on 2026-09-14; the experiment matrices live in
`ISAACLAB_3_GRASP_HANDOFF.md` §15d/§15l if they need to be revisited.

## 2. Reproduce on starfish (single GPU, sanity + eval)

```bash
ssh starfish; cd ~/research/UWLab
PY=~/miniconda3/envs/env_isaaclab3/bin/python          # IsaacLab 6.1.14 / isaaclab_newton 0.13.6 / newton 1.2.1 / mujoco_warp 3.8.0.3
export UWLAB_ROBOT_ASSETS_DIR=$PWD/usd/patched_newton_mass  # mandatory on Newton
DS=$PWD/Datasets/OmniReset_patched                         # mandatory on Newton (sign-flipped gripper joints)
export OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/tmp/$USER          # as in ISAACLAB_3_MIGRATION.md
```

Tasks (registered in `config/ur5e_robotiq_2f85/__init__.py`):
- `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Newton-v0` — training (`Ur5eRobotiq2f85RelCartesianOSCTrainNewtonCfg`)
- `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0` — eval (`...EvalNewtonCfg`)

Always pass `env.events.reset_from_reset_states.params.dataset_dir=$DS`.

Smoke test the training env (3 steps, prints the asset-redirect line and Newton settings):
```bash
$PY tools/newton_probes/train_newton_dryrun.py --headless --num_envs 64 \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Newton-v0 \
  env.events.reset_from_reset_states.params.dataset_dir=$DS
```

Eval + video of a checkpoint (Warp renderer, no RTX):
```bash
$PY scripts_v2/tools/diagnostics/policy_eval_video.py --headless --enable_cameras --renderer newton \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0 \
  --checkpoint checkpoints_newton/<model>.pt --num_envs 1 --episodes 24 \
  --reset_types ObjectPartiallyAssembledEEGrasped --dataset_dir $DS --video_out videos/<name>.mp4
```
(`--reset_types` accepts a comma list of the four reset types; `--num_envs 4` gives a tiled grid.)

Physics probes (all in `tools/newton_probes/`, copied from `/tmp` on 2026-09-14; run with the same
env vars): `osc_newton.py` (OSC step response, expects ~0.038 m per 10 unit steps, std ≤ 0.002),
`mass_dump.py` (per-link mass/inertia vs the PhysX nominal JSON), `root_probe.py` / `world_probe.py`
(fixed-base pose after reset), `contact_probe.py`, `seat_probe.py`, `spin_probe.py` (peg rotation
in grasp), `obs_diff.py` (obs parity vs PhysX on identical states), `nan_probe.py`, `prof.py`
(per-phase step timing). PhysX references for the grasp probes: hold 99 % / lift 91 % / finger q
stays 0.53; identical-state comparisons need forced `_reset_to` on fixed indices because the same
seed draws different samples on the two backends.

Rebuilding the assets from the cloud cache (only if `usd/patched_newton_mass` is lost):
```bash
$PY tools/fix_robotiq_usd.py --src <hf cache root with Robots/...> --dst usd/patched_newton_mass \
    --mass_from tools/newton_probes/mass_physx_nominal.json
$PY tools/fix_object_usd_for_newton.py --src <hf cache root with Props/...> --dst usd/patched_newton_mass
$PY tools/convert_datasets_for_patched_gripper.py --src Datasets/OmniReset --dst Datasets/OmniReset_patched
```

## 3. Reproduce on Tillicum (4 × H200 training — the run that produced the headline)

```bash
ssh tillicum; cd /gpfs/scrubbed/profjat/uwlab3      # rsync target of starfish:~/research/UWLab
FROM_SCRATCH=1 NUM_ENVS=14336 UWLAB_OSC_VEL_CLAMP=1 RUN_NAME=newton_control_clamp1 \
  sbatch docker/cluster/submit_tillicum_omnireset_newton.sh
```
- Image `/gpfs/scrubbed/profjat/sif/uwlab_isaaclab3_cu130.sif` (same package versions as starfish).
- The script runs the standard recipe: `torch.distributed.run --nproc_per_node 4 train.py --task
  ...-State-Newton-v0 --num_envs $NUM_ENVS --logger wandb --headless --distributed
  env.scene.insertive_object=peg env.scene.receptive_object=peghole
  env.events.reset_from_reset_states.params.dataset_dir=/workspace/uwlab/Datasets/OmniReset_patched`
  with `UWLAB_ROBOT_ASSETS_DIR=/workspace/uwlab/usd/patched_newton_mass`.
- Without `FROM_SCRATCH=1` it runs `scripts/reinforcement_learning/rsl_rl/train_expert_init.py`
  resuming actor+critic from `RESUME_PATH` (default `expert_seed0_rslrl52.pt`); the wandb step
  continues from the checkpoint's iteration (`--restart_iteration` to reset it).
- Memory: 960 G requested (240 G/GPU observed at 14336 envs). 24 h wall time; resubmit with
  `RESUME_PATH=<model_N.pt>` to continue.
- Startup tripwires abort with `[FATAL] fix marker ... missing` if the staged tree lacks any of the
  fixes in §1 (edit the marker list in the script when moving code). Watch for `[INFO] gsde fix :
  True`, `[INFO] osc clamp : 1`, `[INFO] local object assets redirected: [...]` in
  `logs/slurm-<job>.out`.
- Throughput: 45–65 s/iteration at 14336 × 4 envs (PhysX: ~25 s). ~85 % of step time is the
  MuJoCo-Warp solver (see `tools/newton_probes/prof.py`, hand-off §15h).

## 4. Results

From-scratch controls, stock recipe (hand-off §15p; wandb project
`profjat-university-of-washington/isaaclab`, comparison scripts `tools/newton_probes/wandb_*.py`):

| it   | PhysX `b1092vqk` | Newton clamp ON `8dha4e7a` | Newton clamp OFF `phcj5cq7` |
|------|------|------|------|
| 300  | 0.26 | 0.14 | 0.14 |
| 400  | 0.62 | 0.15 | 0.16 |
| 600  | 0.77 | 0.24 | 0.32 |
| 800  | 0.96 | 0.53 | 0.40 |
| 1000 | —    | 0.74 | 0.42 (killed at 1057) |
| 1146 | —    | **0.83** | — |

- **Velocity-clamp emulation** (`UWLAB_OSC_VEL_CLAMP`, default ON on Newton,
  `mdp/actions/task_space_actions.py::_apply_velocity_clamp`): OFF reproduces the PhysX exploration
  profile (entropy within 0.5 nats, action-rate penalty −0.080 vs −0.087, ~2 % abnormal-robot
  terminations vs PhysX ~1 %) and leaves the plateau ~130 iterations earlier, but never learns the
  reach-from-scratch resets (task_0 = 0, task_1 ≈ 0.14 for 300 iterations). ON has ~3 nats lower
  entropy and 0.1 % abnormal terminations but keeps improving on every reset type. Keep ON.
- Shaping rewards (`ee_asset_distance`, `dense_success`) match PhysX within a few %; the whole
  reward gap is `success_reward`, i.e. the insertion itself.
- Expert transfer / fine-tune (hand-off §15e–§15k): PhysX expert on Newton 12/33 on
  PartiallyAssembled resets; `newton_expert_ft2` (wandb `dnhhwgnu`) plateaued at task_3 ≈ 0.5 and
  was killed; checkpoint `checkpoints_newton/newton_expert_ft2_model_400.pt`, video
  `videos/newton_ft2_model400_partially_assembled_1env_24ep_20260913.mp4` (local machine).

## 5. Not yet done

Physics / parity
1. **Grasp rigidity** (hand-off §15l/§15m) — on identical states the peg rotates inside the two-pad
   pinch (hold 20 steps: p90 46°, 33–38 % > 10°; PhysX p90 0.5°). Newton's convex narrowphase gives
   one intermittent contact per pad vs PhysX's manifold. Untried candidates: per-pair MuJoCo-native
   contacts for pad↔peg only, multi-sphere pad colliders, or regenerate the grasp datasets under
   Newton (`scripts_v2/tools/record_grasps.py` etc. have not been run on Newton). This is the most
   likely cause of the 2× iteration penalty.
2. lift_probe 75–80 % vs PhysX 91 %; final-insertion seating occasionally fails (friction
   max-combination suspect, hand-off §15k).
3. Velocity clamp is all-or-nothing: the emulation over-corrects (0.1 % abnormal terminations vs
   PhysX 1 %), no emulation under-corrects (2 %). A softer clamp targeting PhysX's rate might
   combine clamp-OFF exploration with clamp-ON reach learning.
4. Abnormal-robot terminations at 0.1 % (clamp ON) means episodes that PhysX would terminate are
   running to time-out; check whether that changes the success-rate semantics.

Throughput
5. Solver is ~85 % of step time; `iterations` 100 / `ls_iterations` 50 / 4 substeps were chosen for
   stability, not speed. No sweep was done. `UWLAB_NEWTON_ITERS`, `_LS_ITERS`, `_SUBSTEPS`,
   `_NCONMAX`, `_NJMAX` are the knobs.
6. Startup heap-corruption flake in the USD importer is worked around (resubmit), not fixed.

Training / tooling
7. The clamp-ON run (294853) is still training; nobody has pulled a checkpoint ≥ model_900 to
   starfish for a 1-env eval/video yet (`scp tillicum:/gpfs/scrubbed/profjat/uwlab3/logs/rsl_rl/
   ur5e_robotiq_2f85_omnireset_agent/2026-09-13_16-38-22_newton_control_clamp1/model_*.pt
   checkpoints_newton/`), then §2 eval command.
8. Multi-seed: one Newton seed only. The PhysX control is also one seed.
9. Long-horizon comparison past it ~1100 (does Newton reach PhysX's 0.96 asymptote?).
10. Eval-time parity of the *trained Newton policy on PhysX* (sim2sim in the other direction) —
    not attempted.
11. RGB training on Newton: the Warp renderer works for eval videos; a tiled-camera RGB training
    task on Newton has not been configured or benchmarked.
12. Upstream hygiene: nothing is committed (96 modified/untracked files on the branch); the gSDE
    fix should go to UW-Lab/rsl_rl and the .sif rebuilt; the submit script's fix marker list is a
    maintenance hazard. The 2026-09-14 cleanup on starfish has NOT been synced to Tillicum
    (`/gpfs/scrubbed/profjat/uwlab3`) because job 294853 is still running out of that tree — rsync
    after it finishes (or before the next submit).
13. Fine-tuning from the PhysX expert never recovered expert performance under Newton; if that
    path matters (e.g. for RGB distillation), it needs the grasp-rigidity fix first.

## 6. Pointers

- Chronology and every experiment matrix: `ISAACLAB_3_GRASP_HANDOFF.md` §13 (USD fix), §14
  (datasets), §15 (rendering), §15a–§15o (mass parity, root-pose bug, contact matrix, warm-start,
  stuck termination, grasp rigidity, NaN rewards, wandb comparisons), §15p (clamp controls).
- Newton cfg + all env knobs with defaults: `rl_state_cfg.py::_apply_newton_overrides` (search
  `UWLAB_NEWTON_`).
- Tillicum submit: `docker/cluster/submit_tillicum_omnireset_newton.sh`; single-GPU helpers
  `docker/cluster/train1_tillicum_newton.sh`, `docker/cluster/prof_tillicum_newton.sh`.
- Local machine (not a code location): `videos/newton_*` eval videos only.
