# Sim2Sim Gap Finetuning — Handoff (2026-08-20)

FastSAC (holosoma) finetuning of a state-based UR5e peg-insertion expert under injected sim2sim gaps
(dynamics + observation), swept on Hyak ckpt. This doc is the full context for continuing the work.

## Task

One base task is used for EVERY gap run; gaps are injected purely via hydra CLI overrides.

- **Task ID:** `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Sparse-No-Privileged-Obs-v0`
- **Env cfg:** `Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSparseNoPrivilegedObsCfg` in
  `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py`
- **Registration:** `.../config/ur5e_robotiq_2f85/__init__.py` (agent entry `Base_FastSACRunnerCfg`)
- **Scene objects (per run):** `env.scene.insertive_object=peg env.scene.receptive_object=peghole`
- **Actions:** `Ur5eRobotiq2f85RelativeOSCAction` (relative Cartesian OSC)
- **Observations:** `ObservationsNoPrivilegedObsCfg` — policy AND critic see the same 6 terms
  (`prev_actions, joint_pos, end_effector_pose, insertive_asset_pose, receptive_asset_pose,
  insertive_asset_in_receptive_asset_frame`), all poses as pos(3)+axis-angle(3) in
  wrist/receptive frames, 5-frame history, policy `enable_corruption=True`, critic False,
  critic has NO privileged terms (matches the recorded expert buffer's `n_critic_obs`).
- **Rewards:** `RewardsScaledSparseCfg` (sparse; no dense shaping):
  `action_magnitude` −1e-4, `action_rate` −1e-3, `joint_vel` −1e-3 (arm joints),
  `abnormal_robot` −10.0, `progress_context` 0.1 (ProgressContext; reset managers read
  `func.success` off it), `success_reward` 1.0.
- **Terminations:** `TerminationsCfg` — `time_out`, `abnormal_robot`, `first_episode_termination`.
  **NO success termination** (solved episodes run to horizon). Episode 16 s, decimation 12, sim dt 1/120.
- **Resets (finetune):** narrowed to reaching-only via CLI:
  `env.events.reset_from_reset_states.params.reset_types=[ObjectAnywhereEEAnywhere]` and
  `...params.probs=[1.0]` (base task is a 4-path mixture at 0.25 each).

## Gap knobs (added 2026-08-20, all on branch `experimental`, uncommitted)

### Dynamics: `dynamics_gap` event term
- Class `apply_dynamics_gap` at the end of `.../omnireset/mdp/events.py`; wired as the LAST term of
  `BaseEventCfg` in `rl_state_cfg.py`, `mode="reset"` → re-pins after the DR terms every reset, so it
  overrides whatever DR sampled. Delegates to the stock isaaclab randomizers with `(v,v)` ranges.
- Knobs (params, float sentinel `-1.0` = disabled; any negative disables):
  `peg_mass`, `socket_mass` (abs kg), `table_mass_scale`, `robot_mass_scale`,
  `peg_friction`, `socket_friction`, `table_friction`, `robot_friction` (pins static=dynamic, restitution 0),
  `gripper_stiffness_scale`, `gripper_damping_scale`.
- Training DR reference (static friction): peg 1.0–2.0, socket 0.2–0.6, robot 0.3–1.2; peg mass DR 0.02–0.2 kg.
- Example: `env.events.dynamics_gap.params.peg_friction=2.2`

### Observation: gap-capable peg-pose terms
- Function `target_asset_pose_in_root_asset_frame_with_gap` in `.../omnireset/mdp/observations.py`;
  used by the two POLICY peg terms (`insertive_asset_pose`, `insertive_asset_in_receptive_asset_frame`)
  in both `ObservationsCfg` and `ObservationsNoPrivilegedObsCfg`. Critic terms stay clean.
- Knobs per term: `pos_noise_std` / `rot_noise_std` (zero-mean gaussian per step; default 0.0 = off),
  `pos_bias` / `rot_bias` (FIXED constant additive offset; default `[0.0,0.0,0.0]`).
  Rotation knobs require axis-angle repr (already the case).
- Example (applied to BOTH peg terms in the bias sweep):
  `'env.observations.policy.insertive_asset_pose.params.pos_bias=[0.01,0.01,0.01]'`

### Hydra override gotchas (cost us the first sweep)
- `update_class_from_dict` type-checks against the CURRENT value: float→None default rejected
  (hence the sentinels), **int→float rejected too — always pass decimals** (`peg_mass=1.0`, not `1`).
- Scalar→list rejected: biases MUST be given as 3-lists. Flat lists replace wholesale (any length OK).
- Failed overrides kill the job ~3 min in as COMPLETED exit 0; error is in the SLURM `.err`
  ("Incorrect type under namespace ...").
- Test overrides locally via `cfg.to_dict()` → modify → `cfg.from_dict()` (the hydra path), not by
  setting params in Python.

## Common training recipe (identical across all runs)

Warm-start from a full-state expert, RLPD-style mix with a recorded expert buffer, 1 collect env.

- **Expert checkpoint (`--resume_path`):**
  `checkpoints/Full-State-Expert-N-Step-3-Rewards-Scaling-Sparse-No-Priv-Ratio005/model_0214000.pt`
  (on Hyak: `/mmfs1/gscratch/weirdlab/jtran/uwlab_latest/checkpoints/<same>/model_0214000.pt`)
- **Expert replay buffer (`--expert_transitions`):** recorded with 1 env, n-step 3, sparse rewards,
  no-priv critic obs, full 4-path resets w/ DR:
  local `expert_rb/full_resets_peghole_anywhere_w_dr_n_step_3_sparse_no_priv_1env.pt` (7.14 GB) =
  Hyak `/mmfs1/gscratch/weirdlab/jtran/uwlab_latest/expert_rb/full_resets_peghole_anywhere_w_dr_n_step_3_sparse_no_priv_1env_20260816_015201.pt`
- **`--expert_ratio 0.75`**, `--num_envs 1024`
- **Agent HPs:** `handle_truncations=true target_entropy_ratio=-1.0 num_updates=10.0
  update_interval=160 batch_size=3072 buffer_size=1000000 critic_learning_rate=3e-5
  actor_learning_rate=5e-6 alpha_learning_rate=3e-5 tau=0.01 num_steps=3 num_collect_envs=1
  learning_starts=160 policy_frequency=4 reset_optimizers=true num_learning_iterations=280000`
  (i.e. finetune regime: LRs ÷10 vs expert training, tau 0.01, optimizers reset, collect from 1 env)
- **Trainer:** `scripts/reinforcement_learning/holosoma/train.py`, `--headless --logger wandb`

## Cluster submission (Hyak / klone)

```bash
UWLAB_TIMESTAMP_SNAPSHOT=1 PARTITION=ckpt ACCOUNT=ckpt-weirdlab GPUS_PER_NODE=1 \
./docker/cluster/cluster_interface.sh job base <train.py args + hydra overrides>
```
- Config via ENV VARS, not flags. 1 GPU per job. Code (incl. the uncommitted gap knobs) rsyncs from
  the local working tree to a per-job `uwlab_<TS>` snapshot; data paths point at `uwlab_latest/`
  (reachable in-container via the blanket `/mmfs1` bind). `expert_rb/` and `checkpoints/` are
  dockerignored — sync data files manually to `uwlab_latest/`.
- Logs: `/mmfs1/gscratch/scrubbed/jtran/slurm_logs/uwlab-dist-<datetime>-<jobid>.{out,err}`.
- Verify a submission with `scontrol write batch_script <jobid> -`.
- ckpt is preemptible w/ auto-requeue; a requeued job RESTARTS from the expert checkpoint.
  True cancel: `scontrol update JobId=<id> Requeue=0 && scancel <id>` (plain scancel resurrects).
- **Pip race:** simultaneous job starts race in the shared `holosoma-deps` (death ~2 min, exit 0,
  "Could not install packages ... warp"). Stagger submissions ~4 min; resubmit victims; do NOT rm the deps dir.

## Runs (wandb names `Finetune-Anywhere-<sweep>-<value>`, submitted 2026-08-20 evening)

| Sweep | Value | Job | Status EOD |
|---|---|---|---|
| Peg-Mass (dynamics_gap.peg_mass) | 0.4 / 0.5 / 0.6 / 0.7 | 38713114/116/179/185 | ALL CANCELLED by user (after ~1 h) |
| Friction (peg/socket/robot pinned) | 2.2/0.7/1.3 | 38715604 | CANCELLED by user (late evening) |
| Friction | 2.5/0.8/1.5, 3.0/1.0/1.8, 3.5/1.2/2.2 | 38715272/273/277 | CANCELLED by user |
| Peg-Bias (pos_bias on both peg terms) | 0.01 | 38720427 | sole survivor, on non-preemptible gpu-l40 (ACCOUNT=gpu-l40-weirdlab) 2026-08-21; earlier instances 38715755 (ckpt) and 38719831 (gpu-l40s) cancelled. NOTE: weirdlab gscratch quota filled by 91 stale uwlab_<TS> snapshots — cleaned 2026-08-21; delete old snapshots if quota errors reappear |
| Peg-Bias | 0.001 / 0.0001 | 38716425 / 38716581 | CANCELLED by user (2026-08-21) |
| Peg-Bias | 1.0 / 0.1 | 38716259 / 38716045 | cancelled before start (replaced by 0.001/0.0001) |

| Peg-Noise (pos_noise_std on both peg terms, per-step gaussian) | 0.001 / 0.01 | 38721065 / 38721358 | RUNNING on ckpt, verified past env setup 2026-08-21 (0.1 level dropped by user) |

All surviving runs verified past "Completed setting up the environment" with zero config errors.

## Pre-existing alternative (unused by these sweeps)
Hardcoded `Dynamics-Gap` task IDs (`TrainEventPegMassGapCfg` etc., startup-pinned peg 0.5 kg) still
exist and are unchanged; the sweeps use the CLI-knob mechanism instead.
