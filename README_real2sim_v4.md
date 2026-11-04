# Real2Sim v4: RGB FlashSAC pretraining for the real UR5e

Pixel FlashSAC (`FlashSACRGB`) on OmniReset peg insertion, trained in Newton (MuJoCo-Warp) and rendered with the Warp
renderer. The sim cameras and their look are matched to the real robot's side and wrist RealSenses:

- **Cameras**: side and wrist poses fitted to real frames.
- **Look**: per-camera exposure and white balance, matched peg and hole textures, and lights.
- **Randomization**: all of the above are randomized, plus camera-mount jitter.

Physics uses the max-sysid arm and OSC. `vision_state` is the 19-D layout used on the real robot:
previous actions (7) | arm joint positions (6) | end-effector pose (6).

This page covers only what is new in v4. Setup, the expert-buffer format, and evaluation are as in
[README_vision_pretraining.md](README_vision_pretraining.md).

## Tasks

Prefix: `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-WristSide-Sysid-Aux-RGB-`

| Task suffix | Use |
|---|---|
| `Real2SimV4-ArmJoints-VisRand4s-v0` | **Train.** All visual randomization redrawn every 4 s. |
| `Real2SimV4-ArmJoints-VisRandPerStep-v0` | **Collect the expert buffer.** Visuals redrawn every step. |
| `Real2SimV4-ArmJoints-v0` | Visuals drawn once per episode (eval / export). |
| `Real2SimV4-v0` | Older 17-D `vision_state` (all joints); kept for old checkpoints. |

The agent entry point is the default `flashsac_cfg_entry_point`. It runs 4096 envs at batch 1024 x 16 updates per
iteration, with a 1.31M replay buffer and an aux state-prediction head.

## Setup (container)

We run everything in `uwlab_isaaclab31_cu130.sif`, which contains Isaac Lab 3 and Newton 1.5.2.
`scripts/tools/run_in_sif.sh` binds this checkout at `/workspace/uwlab` and keeps a writable overlay in
`.sif_overlay/`.

```bash
export SIF=/path/to/uwlab_isaaclab31_cu130.sif
bash scripts/tools/run_in_sif.sh --setup     # once: installs the setup.py pins (rsl-rl-lib, rsl-rl-flashsac)
R="bash scripts/tools/run_in_sif.sh"          # $R <script> <args...> runs a script inside the container
```

Outside the container, install the FlashSAC extras: `pip install -e source/uwlab_rl[flashsac]`. Then run the same
scripts with `python`.

Every command below takes these overrides. **Rewards are sparse**: the two dense shaping terms are switched off, which
leaves success, the action penalties and the safety penalty.

```bash
COMMON="--visualizer none presets=newton_mjwarp,newton_renderer env.sim.physics.solver_cfg.njmax=400 \
  env.scene.insertive_object=peg env.scene.receptive_object=peghole \
  env.rewards.ee_asset_distance=null env.rewards.dense_success_reward=null"
T=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-WristSide-Sysid-Aux-RGB-Real2SimV4-ArmJoints
# the recipe: aux state head, policy EMA, per-camera encoder width 0.55, cosine LR 3e-4 -> 3e-6 (needed to load checkpoints too)
AGENT="agent.algorithm.aux_head=mlp agent.algorithm.aux_weight=0.1 agent.algorithm.policy_ema=1e-4 \
  agent.algorithm.encoder_num_cameras=2 agent.algorithm.encoder_width=0.55 agent.algorithm.learning_rate_end=3e-6"
```

## 1. Collect the expert buffer

You need a state-based rsl_rl PPO expert for OmniReset on Newton. We used `newton_symmetry_ppo_finetuned_model2500.pt`,
a symmetry-SGS expert. It acts from its own `policy` observation group. The collector records the vision observation
groups (`image`, `vision_state`, `aux_state`).

1024 envs x 320 steps gives 325k transitions, a ~16 GB file. It takes about 15 minutes on one GPU.

```bash
$R scripts/tools/collect_flashsac_expert_buffer.py --task ${T}-VisRandPerStep-v0 \
  --checkpoint /workspace/uwlab/checkpoints/<state_ppo_expert>.pt --expert_agent rsl_rl_cfg_entry_point \
  --fs_agent flashsac_cfg_entry_point --obs_dtype image=uint8 --num_envs 1024 --num_steps 320 --buffer_device cpu \
  --enable_cameras --output /workspace/uwlab/experts/real2sim_v4_armjoints_visrandperstep_sparse.pt $COMMON
```

Check the printed `[expert] ... success` before training. It should be about 98%.

## 2. Train

```bash
$R scripts/reinforcement_learning/flashsac/train.py --task ${T}-VisRand4s-v0 --num_envs 4096 --seed 1 \
  --enable_cameras --max_iterations 20000 --run_name real2sim_v4_armjoints_visrand4s_sparse_s1 $COMMON \
  env.events.reset_from_reset_states.params.probs=[1,1,1,1] \
  agent.algorithm.expert_buffer_path=/workspace/uwlab/experts/real2sim_v4_armjoints_visrandperstep_sparse.pt $AGENT
```

- Add `--logger wandb --log_project_name <project>` for wandb.
- Checkpoints go to `logs/flashsac/<experiment>/<timestamp>_<run_name>/model_<it>.pt` every 400 iterations.
- To resume, add `--resume_path <model_*.pt>`. The run continues for the remaining iterations, but the replay buffer
  is not saved in checkpoints, so it starts empty again.
- Reference run, on one H200: 0.90 training success at 30M env steps (4.2 h), 0.97 by about iteration 15k. A full
  20k iterations takes about 12 h.
- **Optional: the real UR5e's OSC error clip.** Add `env.actions.arm.pose_error_clip=[0.05,0.3]` (5 cm / 0.3 rad per
  component, the real controller's default). Collect the expert buffer with the same clip, using an expert that works
  under it. The plain model_2500 expert only reaches 56% with the clip on.

## 3. Evaluate and export

```bash
# video, one episode per env, back to back
$R scripts/reinforcement_learning/flashsac/eval_video.py --task ${T}-v0 --agent flashsac_cfg_entry_point \
  --checkpoint <model_*.pt> --out_prefix videos/real2sim_v4 --num_envs 10 $COMMON $AGENT \
  agent.algorithm.expert_buffer_path= agent.algorithm.replay_buffer_size=1000

# TorchScript policy + spec.json + parity_rollout.npz for the real robot (--use_ema: the EMA policy)
$R scripts/tools/export_flashsac_rgb_policy.py --task ${T}-v0 --checkpoint <model_*.pt> [--use_ema] \
  --out_dir exports/real2sim_v4 --num_envs 16 --steps 160 --save_envs 4 $COMMON $AGENT \
  agent.algorithm.expert_buffer_path= agent.algorithm.replay_buffer_size=1000
```

## Notes

- **Use one GPU.** Multi-GPU FlashSAC-RGB needs `_sync_running_stats` in `rsl-rl-flashsac`. The commit pinned in
  `source/uwlab_rl/setup.py` does not have it yet, so a multi-GPU run will crash at the end of its first update.
- **Friction randomization on Newton with cameras** is fixed here by `randomize_rigid_body_material` in
  `omnireset/mdp/events.py`. Isaac Lab's Newton version writes to the wrong shapes when visual meshes are loaded. Runs
  from an older tree trained at fixed friction (peg 0.5, hole 0.5, fingers 1.0).
- **GPU memory.** An H200 holds the 1.31M replay buffer plus the expert buffer. On smaller GPUs, pass
  `--buffer_device cpu` to `train.py`.
