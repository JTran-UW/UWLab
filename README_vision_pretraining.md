# Vision FlashSAC pretraining on OmniReset (pixel and frozen DINOv3)

How to train the UR5e + Robotiq 2F-85 peg-insertion policy from cameras with FlashSAC, from scratch, on Newton
(MuJoCo-Warp) physics with the Warp renderer. Two encoders:

- **Pixel**: FlashSAC-RGB's own conv encoder on 64x64 grayscale cameras (`image`, stored as uint8).
- **Frozen DINOv3**: ViT-B/16 features of 128x128 RGB cameras (CLS + 2x2 pooled patch grid per camera, `dino`,
  stored as fp16); only the actor/critic trunks train.

Both also see `vision_state` (proprioception: previous actions, joint positions, end-effector pose; 3-frame history for
DINO). The critic sees the same inputs (no privileged state). 5% of every batch comes from an expert replay buffer
collected by a state-based symmetry PPO expert.

## 1. Setup

Isaac Lab 3.0 (Isaac Sim 6.1, Newton 1.5) with this repo installed, plus the FlashSAC extras:

```bash
pip install -e source/uwlab_rl[flashsac]    # pins rsl-rl-lib (UW-Lab uw-v5.4.2) and rsl-rl-flashsac (Holiday-Robot)
pip install timm==1.0.20                     # DINOv3 only
```

On the clusters we run the same tree inside the `uwlab_newton_isaacsim61.sif` container (Hyak / Tillicum
`.../sif/`), with the repo bound at `/workspace/uwlab`, `PYTHONPATH` set to the `source/*` packages, and
`/opt/venv/bin/python`. Reset-state datasets and USD assets download from Hugging Face on first use.

You need two files the scripts do not fetch:

- **The PPO expert** used to collect expert demos: `newton_symmetry_ppo_expert_model500.pt`, the state-based
  symmetry-invariant rsl_rl PPO expert trained on Newton. Its policy observation group is the vision tasks' `policy`
  group. It is in the released checkpoint zip (`experts/`).
- **DINOv3 weights** (DINOv3 only). The timm repo is public, no token or request form:

  ```python
  from huggingface_hub import hf_hub_download
  for f in ("model.safetensors", "config.json"):
      hf_hub_download("timm/vit_base_patch16_dinov3.lvd1689m", f, token=False,
                      local_dir="checkpoints/dinov3/vit_base_patch16_dinov3.lvd1689m")
  ```

## 2. Tasks and camera sets

All tasks use 4096 envs, a 160-step horizon (16 s at 10 Hz) and the four OmniReset reset types
(AnywhereEEAnywhere, RestingEEGrasped, AnywhereEEGrasped, PartiallyAssembled). Prefix every task with
`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-`.

| Cameras | Pixel task | DINOv3 task | Extra override |
|---|---|---|---|
| Zoomed front + wrist (baseline) | `Vision-FlashSAC-v0` | `Vision-FlashSAC-Dino3-v0` | `env.scene.front_camera.spawn.focal_length=26.4` |
| Side + wrist | `Vision-FlashSAC-WristSide-v0` | `Vision-FlashSAC-Dino3-WristSide-v0` | none |
| Unzoomed front + side + wrist | `Vision-FlashSAC-WristSideFront-v0` | `Vision-FlashSAC-Dino3-WristSideFront-v0` | none (front keeps its 13.2 mm focal length) |

The front-camera zoom is a CLI override, so pass `focal_length=26.4` only for the baseline. Agent configs come from
`agents/flashsac_cfg.py`. For pixel, use `OmniReset_FlashSACRGB4096RunnerCfg`, which is batch 1024 x 16 updates per
iteration and a 1.31M buffer. It is the `flashsac_cfg_entry_point` of the camera-set tasks and
`flashsac_rgb_4096_cfg_entry_point` of `Vision-FlashSAC-v0`. For DINOv3, use `OmniReset_FlashSACDinoRunnerCfg`, the
same recipe with an fp16 buffer.

Common Hydra arguments for every command below:

```bash
COMMON="--visualizer none presets=newton_mjwarp,newton_renderer env.sim.physics.solver_cfg.njmax=400 \
  env.scene.insertive_object=peg env.scene.receptive_object=peghole"
DINO="env.observations.dino.features.params.model_path=$PWD/checkpoints/dinov3/vit_base_patch16_dinov3.lvd1689m \
  env.observations.dino.features.params.pool_grid=2"
```

(`DINO3_MODEL_DIR` defaults to `/workspace/uwlab/checkpoints/dinov3`, the container layout. Override `model_path`
anywhere else.)

## 3. Collect the expert buffer (once per task / camera set)

The expert acts from its own state observation group. The collector records the vision agent's observation groups at
the agent's action bounds. 1024 envs x 320 steps gives 325k transitions, about 10 minutes on one GPU.

```bash
# pixel (baseline shown). For the camera-set tasks, swap --task, drop the focal override and use
# --fs_agent flashsac_cfg_entry_point (their default entry is the same 4096-env recipe)
python scripts/tools/collect_flashsac_expert_buffer.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-v0 \
  --checkpoint experts/newton_symmetry_ppo_expert_model500.pt --expert_agent rsl_rl_cfg_entry_point \
  --fs_agent flashsac_rgb_4096_cfg_entry_point --obs_dtype image=uint8 --num_envs 1024 --num_steps 320 --enable_cameras \
  --output experts/omnireset_newton_symppo_flashsac_rgb.pt $COMMON env.scene.front_camera.spawn.focal_length=26.4

# DINOv3
python scripts/tools/collect_flashsac_expert_buffer.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-Dino3-v0 \
  --checkpoint experts/newton_symmetry_ppo_expert_model500.pt --expert_agent rsl_rl_cfg_entry_point \
  --fs_agent flashsac_cfg_entry_point --obs_dtype float16 --num_envs 1024 --num_steps 320 --enable_cameras \
  --output experts/omnireset_newton_symppo_flashsac_dino3_vitb16_g2.pt $COMMON $DINO env.scene.front_camera.spawn.focal_length=26.4
```

Check the printed `[expert] ... success` before training. It should be about 99%, with about 1% of action components
clipped. A buffer collected on one task or camera set cannot be used for another: its observation groups and shapes
must match.

## 4. Train

```bash
# pixel
python scripts/reinforcement_learning/flashsac/train.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-v0 \
  --agent flashsac_rgb_4096_cfg_entry_point --num_envs 4096 --seed 1 --enable_cameras --logger wandb --log_project_name uwlab_vision_rl \
  --run_name fsac_rgb_front_wrist_s1 $COMMON env.scene.front_camera.spawn.focal_length=26.4 \
  env.events.reset_from_reset_states.params.probs=[1,1,1,1] \
  agent.algorithm.expert_buffer_path=$PWD/experts/omnireset_newton_symppo_flashsac_rgb.pt

# DINOv3
python scripts/reinforcement_learning/flashsac/train.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-Dino3-v0 \
  --num_envs 4096 --seed 1 --enable_cameras --logger wandb --log_project_name uwlab_vision_rl \
  --run_name fsac_dino3_vitb16_g2_front_wrist_s1 $COMMON $DINO env.scene.front_camera.spawn.focal_length=26.4 \
  agent.algorithm.expert_buffer_path=$PWD/experts/omnireset_newton_symppo_flashsac_dino3_vitb16_g2.pt
```

For the pixel camera-set tasks, drop `--agent` (their default entry point is the same recipe) and the focal override.
For the DINOv3 camera-set tasks, swap `--task` and drop the focal override.

Checkpoints go to `logs/flashsac/<experiment>/<timestamp>_<run_name>/model_<it>.pt` every 400 iterations. One
iteration is one 4096-env step plus 16 updates. Pick checkpoints by the logged training success; the curves plateau
long before `max_iterations`.

**GPU memory at 4096 envs:**

| Encoder | 2 cameras | 3 cameras |
|---|---|---|
| Pixel | 26 GB | 35 GB |
| DINOv3 | 40 GB | 55 GB |

An L40 (48 GB) fits pixel. Use an H200 for DINOv3, especially with three cameras.

## 5. What to expect

All runs were seed 1 and single-GPU. Success is the mean of the four per-reset-type training success rates.

| Encoder | Cameras | GPU | s / iteration | 90% | 95% | Plateau |
|---|---|---|---|---|---|---|
| Pixel | Zoomed front + wrist | L40 | 1.0 | 12M steps / 1.0 h | 12M / 1.1 h | 0.99 |
| Pixel | Side + wrist | H200 | 1.0 | 10M / 0.8 h | 12M / 0.9 h | 0.99 |
| Pixel | Unzoomed front + side + wrist | H200 | 1.0 | 11M / 0.9 h | 12M / 0.9 h | 0.99 |
| DINOv3 | Zoomed front + wrist | H200 | 1.6 | 12M / 1.5 h | 14M / 1.7 h | 0.98 |
| DINOv3 | Side + wrist | H200 | 1.6 | 11M / 1.3 h | 12M / 1.4 h | 0.99 |
| DINOv3 | Unzoomed front + side + wrist | H200 | 2.0 | 16M / 2.3 h | 18M / 2.5 h | 0.98 |

## 6. Evaluate

```bash
# per-reset success over 8 horizons (deterministic actor)
python scripts/reinforcement_learning/flashsac/eval_success.py --task <task> --agent <same agent entry point> \
  --checkpoint <model_*.pt> --out results.json --num_envs 128 --rounds 8 $COMMON [$DINO] [focal override]

# video: one episode per env, played back to back (workspace view + the policy's own camera column);
# --no_video writes only the per-episode results .json (success, time to success)
python scripts/reinforcement_learning/flashsac/eval_video.py --task <task> --agent <agent entry point> \
  --checkpoint <model_*.pt> --out_prefix videos/run --num_envs 10 $COMMON [$DINO] [focal override]
```

Replace `newton_renderer` with `isaacsim_rtx` for RTX rendering. That needs a workstation GPU with Vulkan and driver
580 or newer; the clusters have no Vulkan.

## 7. Notes and pitfalls

- **Newton solver blow-ups.** Rare MuJoCo-Warp NaNs used to leave an env stuck: non-finite and terminated every step for
  the rest of the run, which slowed iterations up to 4x and degraded training. Since commit `8701e960`,
  `abnormal_robot_state_or_blowup` zeroes the blown-up worlds' solver data before reset, and the NaN guard
  (`uwlab_rl.env_guards`, Newton only) zeroes the one bad transition. Grep the log for `[env_guard] zeroed`. A few
  isolated events are normal; one per iteration means a stuck env.
- **Hydra cannot override `None`-typed cfg fields** with a string (for example `agent.algorithm.buffer_device`). Use the
  CLI flags: `--buffer_device cpu` on `train.py` and on the collector.
- **Replay buffer on CPU** (`--buffer_device cpu`) frees GPU memory for small GPUs. On machines that refuse pinned host
  allocations of 4 GB or more, set `FLASHSAC_NO_PIN=1`.
- **Action bounds** are per dimension at the expert's p99 (`EXPERT_P99_ACTION_BOUNDS`). The max-sysid tasks use
  `SYSID_EXPERT_P99_ACTION_BOUNDS`. When finetuning a nominal checkpoint onto them, pass `--reset_action_scaling`.
- **Related tasks in the same files:**
  - max-sysid training without a curriculum: `Vision-FlashSAC-Finetune-v0`, `-Dino3-Finetune-v0`,
    `-WristSide-Sysid-v0`;
  - RTX visual randomization: `-WristSide-Sysid-VisRand-v0` redraws every 4 s; `-VisRandEp-v0` redraws once per
    episode per env;
  - pixel and DINO OOD eval tasks: `-OOD-Play-v0`.
