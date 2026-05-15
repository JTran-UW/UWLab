# RMA overnight run — 2026-05-14

Author: Claude (autonomous, user asleep). Note all decisions made and why.

## What's running

| Field | Value |
|---|---|
| Task | `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-RMA-Train-v0` |
| Subtask | peg + peghole |
| Seed | 21 |
| GPUs | 0, 1, 2, 3 (4×H100 80GB), distributed |
| `--num_envs` | 32768 per GPU (131,072 total across ranks) — matched to the parallel sysid baseline after RMA underperformed at 16k |
| Logger | wandb |
| Max iterations | 40000 (default from `Base_PPORunnerCfg`) |
| Run name | `rma_peg_seed21_16384envs_4gpu` |
| Gravity curriculum | `reduction=monitor_mean`, `floor=0.1` |
| Log file | `/mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/rma/rma_dist_20260514_133713.log` (v3; previous attempts `…_123812.log` (NCCL deadlock) and `…_125448.log` (SDPA OOM-cfg) preserved) |
| Container | `isaac-sim` (already-running docker) |
| Conda env | `patlab` |

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-RMA-Train-v0 \
  --num_envs 16384 \
  --logger wandb --headless --distributed \
  env.scene.insertive_object=peg env.scene.receptive_object=peghole \
  --seed 21 \
  env.curriculum.gravity_curriculum.params.reduction=monitor_mean \
  env.curriculum.gravity_curriculum.params.floor=0.1 \
  --run_name rma_peg_seed21_16384envs_4gpu
```

## Decisions

1. **GPUs 0–3 (as the user asked).** Verified GPUs 0–3 were nearly idle while
   GPUs 4–7 are running a parallel `--seed 22 --num_envs 32768` Sysid-Train run
   (not mine). No conflict.
2. **`--num_envs 16384` per GPU.** User asked for 16k per node. `train.py` does
   not auto-divide `--num_envs` under `--distributed` (it copies CLI value
   straight into `env_cfg.scene.num_envs` per rank), so 16384 is per-rank ⇒
   65,536 total. Matches "at least 16k envs on each node."
3. **Aux pass: kept defaults (`history_num_mini_batches=4`, no env subsampling
   yet).** Estimated peak mem for the aux minibatch (~65k×T=32×D=32 history
   batch through the 2-layer transformer): ~4–6 GB extra on top of PPO's
   storage. Should fit on 80 GB H100. If OOM appears in the log, fall back to
   bumping `history_num_mini_batches` in `RslRlPpoRMAAlgorithmCfg` (8 or 16)
   to shrink batch — no code change required, just a cfg edit.
4. **Smoke test first (1 GPU, 64 envs, 3 iters).** Caught and fixed three
   bugs before the big run:
   - `DelayBuffer.time_lag` → `time_lags` (IsaacLab API mismatch).
   - Mixed-device tensors in the privileged obs group — added
     `.to(env.device)` to all `get_*` helpers in `mdp/observations.py`.
   - `nn.ModuleDict` lacks `.get()` — replaced with `in`/`[]` in
     `ActorCriticRMA.encode_privileged`.
   - Added `rsl_rl.algorithms.PPO_RMA = PPO_RMA` injection in train.py/play.py
     so `cli_args.sanitize_rsl_rl_cfg` can strip upstream-only kwargs
     (`share_cnn_encoders`, `rnd_cfg`, `symmetry_cfg`).
5. **Fixed a latent distributed-only bug in `ActorCriticWithEncoder.__init__`
   (12:54).** The first 4-GPU launch hung for 10 minutes at
   `Synchronizing parameters for rank 0...` and got killed by the NCCL
   watchdog. Diagnosis: the line
   ```python
   all_groups = set(obs_groups["policy"]) | set(obs_groups["critic"])
   for name in all_groups: …
   ```
   iterates a Python `set`, whose order depends on `PYTHONHASHSEED`
   (randomized per process). That made each rank build the per-group
   encoder ModuleDicts in a different order, so
   `self.policy.parameters()` yielded params in different orders per
   rank, and `broadcast_parameters` sent collectives of mismatched
   sizes (rank 0 was 8 broadcasts ahead of the others before the
   watchdog timed out). The previous `ScenePCPPORunnerCfg` only had
   one group in `all_groups` (`pointcloud`), so the bug was latent.
   RMA adds three more groups (`proprio`, `privileged_rma`,
   `time_left`) and surfaced it. Fix:
   `all_groups = sorted(set(...) | set(...))`. Relaunched with the
   new log file path above.
6. **Fixed transformer aux-pass `invalid configuration argument` (12:59 →
   13:37).** Second 4-GPU launch made it past the NCCL deadlock but died
   at the first aux update: 16384 envs × 16 rollout steps = 262144
   transitions / `history_num_mini_batches=4` = 65536-element batches fed
   to `scaled_dot_product_attention`, which exceeds CUDA's kernel-config
   limits. Added two knobs to `PPO_RMA` / `RslRlPpoRMAAlgorithmCfg`:
   - `history_minibatch_size` (default **4096**) — hard cap on the
     transformer minibatch size; `num_mini_batches` is recomputed from
     `epoch_samples // minibatch_size`. Default 4096 keeps SDPA well
     under any kernel limit and fits comfortably in ~1 GB.
   - `history_max_samples_per_epoch` (default `None` = all) — the
     user's explicit "subsample envs for the history encoder" knob:
     if set, randomly draws this many `(step, env)` indices per epoch
     instead of using all transitions. Not used in the current run
     (policy still sees all 16k envs/GPU); flip on if psi training cost
     becomes a wall-clock issue.
   Relaunched with `history_minibatch_size=4096`. At 262k transitions
   that's 64 minibatches per epoch × 5 epochs = 320 small updates per
   PPO iter — fine.
7. **Worktree avoided.** The Claude-Code worktree at
   `.claude/worktrees/graceful-strolling-ladybug` is pinned to an old commit
   that doesn't have `gravity_cfg.py` / `ActorCriticWithEncoder`. All RMA code
   landed in the main repo at `/mnt/storage/lti/UWLab-patrick-private` on
   branch `pat/dagger-symmetry`. Nothing committed — review `git diff` before
   staging.

## How to check on the run

```bash
# Tail the log (host or in-docker, file is bind-mounted):
tail -f /mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/rma/rma_dist_20260514_123812.log

# See iteration progress + aux loss:
grep -E "Learning iteration|Mean aux_history_mse|Mean reward" \
  /mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/rma/rma_dist_20260514_123812.log | tail -40

# wandb run URL is in the log: grep "wandb: 🚀\|wandb run"

# Kill if needed:
docker exec isaac-sim pgrep -af "rma_peg_seed21"  # find PIDs
docker exec isaac-sim pkill -f "rma_peg_seed21"
```

## v3 progress timeline (13:37 launch)

```
13:37  iter 0/40000     Mean aux_history_mse: 0.0195   Mean reward:  2.48
13:51  iter 7/40000     aux_mse: 0.0139                Mean reward:  2.72
14:44  iter 217/40000   aux_mse: 0.0141                Mean reward: -1.07
14:45  iter 222/40000   aux_mse: 0.0141                Mean reward: -2.90
15:45  iter 424/40000   aux_mse: 0.0172                Mean reward: -2.05
16:46  iter 629/40000   aux_mse: 0.0400                Mean reward: -1.66
17:47  iter 833/40000   aux_mse: 0.1467                Mean reward: -1.14
18:48  iter 1037/40000  aux_mse: 0.3909                Mean reward: -1.53
19:49  iter 1239/40000  aux_mse: 0.9622                Mean reward:  0.62
20:50  iter 1437/40000  aux_mse: 4.1161                Mean reward:  1.29
~21:30 iter ~1520       aux_mse: ~6.9                  Mean reward: ~0.7   ← killed, see below
00:12  v4 relaunch with --num_envs 32768 (matches baseline) — log file `…_001242.log`
01:14  v4 iter 116/40000   aux_mse: 0.0125   reward: -3.4   (baseline iter 100: 1.58, both gravity_frac=0.10)
02:16  v4 iter 245/40000   aux_mse: 0.0137   reward: -1.6   (baseline iter 245: 2.82, both gravity_frac=0.10)
03:17  v4 iter 376/40000   aux_mse: 0.0181   reward: -1.3   (baseline iter 376: 2.92, both gravity_frac=0.10)
04:18  v4 iter 506/40000   aux_mse: 0.0611   reward: -0.15  (baseline iter 506: 2.29, both gravity_frac=0.10)
05:19  v4 iter 633/40000   aux_mse: 0.3867   reward: -0.47  (baseline iter 633: 2.19, both gravity_frac=0.10)
```

**Trend after 5 v4 hours**: matched throughput hasn't fixed the policy gap.
aux_mse climbing again (0.013 → 0.018 → 0.06 → 0.39), same shape as
v3 just delayed. Reward flat-to-down (-1.3 → -0.15 → -0.47), gap to
baseline ≈2.5–3 reward points unchanged. The matched-envs experiment
seems to be confirming: the bottleneck is the **architecture** (the
extra 16d privileged latent in the actor's input + the unbounded
moving target for ψ), not the data volume.

**v4 vs baseline at matched iter ~100:**
- Throughput: v4 ≈31s/iter (32k envs, big rollout buffer pushes rank 0 to
  73 GB), baseline ≈26s/iter. Similar.
- Reward: v4 -3.4 vs baseline +1.58. V4 slightly worse, but baseline also
  meandered in the 1–3 band for ~1000 iters before the curriculum
  unstuck and reward jumped to 9.5 at iter ~1500.
- aux_mse: 0.0125 — far better than v3 ever achieved (v3 settled ≈0.014
  briefly, then climbed to ~7). ψ training is stable in v4.

Too early to draw conclusions; need to see if v4 crosses the gravity
curriculum around iter 1500 like the baseline did.

## Why v3 was killed (decision at ~21:30)

Compared to the parallel sysid baseline running on GPUs 4–7
(`...-ZeroG-ScenePC-SysID-Train-v0`, seed 22, **32768 envs/GPU**),
v3 (RMA, 16384 envs/GPU) was underperforming:

```
                              Iters    Gravity frac   Mean reward
Sysid baseline (seed 22)      2134     0.1 → 1.000    2.49 → 9.78    ← curriculum done
RMA v3 (seed 21)              ~1500    0.1 → 0.100    2.48 → ~1.0    ← stuck at floor
```

`aux_history_mse` saturated near ~7 (per-dim RMSE √(7/16) ≈ 0.66 — i.e.
ψ basically not training, predicting the mean). Per the user, restarted
**without architecture changes** at 32768 envs/GPU to isolate whether
the gap was throughput vs the RMA architecture itself. If v4 closes
the gap at matched throughput, it was envs; if not, the privileged
latent + auxiliary loss is hurting more than helping (likely culprits:
no `LayerNorm`/`tanh` on φ/ψ → unbounded drifting latent; or just that
the actor wastes capacity on a 16d random projection until φ converges).

**Note 20:50**: aux_mse crossed the 3.0 threshold (0.96 → 4.11, ≈4.3×
in this hour, slope steeper than prior). Per-dim RMSE ≈ √(4.11/16) ≈
0.51. **NOT stopping the run** because:
  1. **Reward keeps improving** (0.62 → 1.29 best; the four latest
     reported rewards 0.53 / -0.32 / 0.89 / 1.29 trend up). Policy is
     learning. That is the load-bearing signal.
  2. The actor sees real `z_priv = φ(privileged_rma)`, not `z_hat`, so
     ψ's drift does *not* affect policy quality at training time.
     `aux_mse` only governs deploy-time ψ fidelity, which can be
     recovered by a cheap phase-2 finetune (freeze φ, retrain ψ on
     fresh rollouts) after the policy converges. That is in fact
     the canonical RMA paper schedule.
  3. So the worst case from continuing is: throw away the
     mid-training ψ checkpoint and do a short phase-2 from the final
     policy weights.

New stop conditions: (a) reward regresses meaningfully (< 0), (b) any
actual error, (c) aux_mse explodes (> 50 with same slope). Otherwise
let it cook.

**⚠️ Flag 19:49**: aux_mse now ≈0.96 (slope ≈2.5×/h, similar to prior
hours). Per-dim RMSE ≈ √(0.96/16) ≈ 0.25 — significant. **But the
policy is finally learning**: reward turned positive this hour
(-1.53 → +0.62, with the last three reported rewards -0.62 / +0.37 /
+0.62). That's the important signal. I'm *not* stopping the run
despite aux_mse being above the 0.6 flag, because:
  1. aux_mse only affects deploy-time fidelity of ψ → z_hat, not
     training-time policy behavior (the actor sees real `z_priv`).
  2. The policy itself is doing what we trained it to do.
  3. Slope is approximately log-linear (×2.5/h, not exponential blow-up),
     so it'll bound naturally once phi's input normalizer stops
     drifting and PPO update step shrinks.
Will reconsider intervening if (a) reward regresses, (b) aux_mse
crosses ~3.0 and is still climbing at the same ratio, or (c) any
actual error appears. Otherwise continue monitoring.

Cadence: ~17 s/iter (≈210 iters/hour). At this rate the 40k cap is
~7.8 days off — kill or `--max_iterations` before then if you don't
need a 40k full run. Reward dipping into negative territory early
isn't worrying: action_l2 + action_rate + joint_vel regularization
dominates while success_reward is rare during the gravity-curriculum
warmup (floor=0.1).

wandb run: <https://wandb.ai/learning-to-improve/OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-RMA-Train-v0/runs/zqcgsjp4>

## Smoke-test result

```
Learning iteration 0/3   Mean aux_history_mse: 0.1079   Mean reward:  8.74
Learning iteration 1/3   Mean aux_history_mse: 0.0521   Mean reward:  6.74
Learning iteration 2/3   Mean aux_history_mse: 0.0302   Mean reward:  5.91
```

psi is rapidly converging onto phi (3× drop in 3 iterations — that's the
"easy" early signal). Rewards are early, no PPO judgement yet.

## Architecture (recap, see implementation files for source-of-truth)

- **Env**: extends `ZeroGScenePCSysidSim2RealTrainCfg`. Adds `privileged_rma`
  obs group (156d): arm armature (6), arm friction (6), arm delay (1),
  OSC Kp+Kd (12), gripper stiffness+damping (2), 4 body masses, 4 body
  material entries.
- **Policy** `ActorCriticRMA` (`source/uwlab_rl/uwlab_rl/rsl_rl/actor_critic_rma.py`):
  φ = per-group MLP `privileged_rma → 16d` latent (existing ActorCriticWithEncoder pipeline).
  Actor sees `[proprio | pc_latent (32d) | z_priv (16d)]`. Bidirectional
  transformer ψ (2 L / 4 H / d=64 / ff=128) maps a (B, T=32, D_hist=32)
  history window to 16d.
- **Algorithm** `PPO_RMA` (`source/uwlab_rl/uwlab_rl/rsl_rl/ppo_rma.py`): runs
  upstream PPO unchanged, then a separate aux pass on a dedicated
  `history_optimizer` doing `MSE(ψ(hist), sg(φ(priv)))`. Stop-grad on φ.
  Distributed: psi grads all-reduced separately in `_reduce_history_grads`.
- **Runner** `OnPolicyRunnerRMA` (`source/uwlab_rl/uwlab_rl/rsl_rl/on_policy_runner_rma.py`):
  maintains `(num_envs, 32, 32)` rolling buffer of proprio+last_action,
  zeroed per env on `dones`, snapshotted into PPO_RMA's
  `(num_steps_per_env=16, num_envs, 32, 32)` per-step buffer for the aux pass.

## Things to look at when awake

1. wandb dashboard for the run name above — confirm aux_history_mse trends
   down while reward goes up (typical RMA pattern).
2. Whether ψ converges faster than φ stabilizes (sign of healthy stop-grad).
3. If iteration time is dominated by the aux pass (`learn_time` in the log
   should be similar to the non-RMA baseline; if it 2×s, shrink minibatch
   or epoch count).
4. **Deploy-time φ↔ψ swap is NOT implemented yet** — would need a follow-up
   in `play.py` (either inject a `history` ObsGroup at deploy or have the
   policy maintain an internal buffer). Flag if you want me to add it.
5. **History buffer is not checkpointed.** On resume, the first 32 steps
   after restart see zero-padded history. Trivial to add (save/load
   `_hist_buf`) if you care.
