# RTX look on the Newton renderer

The RTX RealCam tasks (`...-VisRand-ArmJoints-RealCam-...-GlobalDome7k[-NoDLAA]-v0`) need Isaac RTX and therefore an
RT-capable GPU. The two tasks here render the same look with the UWLab Newton renderer, on the Warp BVH or through
OptiX ([README_optix_renderer.md](README_optix_renderer.md)), so it also runs on H200 nodes.

Prefix: `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-WristSide-Sysid-Aux-RGB-`

| Task suffix | Use |
|---|---|
| `RtxLook-ArmJoints-v0` | The look without visual randomization. |
| `RtxLook-ArmJoints-VisRand4s-v0` | Twin of the RTX training task: appearance, sky and cameras redrawn on reset and every 4 s. |

Both keep the 19-D ArmJoints `vision_state` and the calibrated side and wrist cameras of the RealCam tasks.

## Run

As any Newton vision task, with the Newton renderer preset. The RTX runs use 96x96 cameras:

```bash
UWLAB_OPTIX_DEPS=r570 UWLAB_NEWTON_RENDER_BACKEND=optix bash scripts/tools/run_in_sif.sh \
    scripts/tools/render_grid.py --num_envs 64 --steps 1 --out outputs/rtx_look \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-FlashSAC-WristSide-Sysid-Aux-RGB-RtxLook-ArmJoints-VisRand4s-v0 \
    --visualizer none presets=newton_mjwarp,newton_renderer env.sim.physics.solver_cfg.njmax=400 \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole \
    env.scene.side_camera.width=96 env.scene.side_camera.height=96 \
    env.scene.wrist_camera.width=96 env.scene.wrist_camera.height=96
```

`render_grid.py` writes a contact sheet of every env's policy images. Without the two OptiX variables the same task
renders on the Warp BVH (3.5-5.6x slower per frame on an RTX 4090). The first run downloads the sky map, and for the
randomized task the 150 textures and 100 HDRIs of the RTX task (`resources/*_subset*.yaml`), and reduces them to
render size once (about 90 s; cached in `~/.cache/uwlab/render_bank`).

## What RTX does, and what is matched

Read from the container (`apps/isaaclab.python.headless.rendering.kit`, `isaaclab_physx/renderers`) and fitted to the
RTX session's dome sweep (`UWLab-vision/outputs/ladder_eval/gdome1k`):

| | RTX GlobalDome7k | Here |
|---|---|---|
| Renderer | `RealTimePathTracing`, up to 3 bounces, denoised | One-bounce path lighting, 2 lighting rays per pixel, filtered |
| Light | White dome 7000 plus sky dome 1000 (`kloofendal_43d_clear_puresky`) | The same two, in units of 1000; sky values clamped at 5, which removes the sun |
| Tone mapping | ACES at about unit exposure on intensity / 1000 | ACES, exposure +0.45 stops |
| Anti-aliasing | Renderer default (DLAA off) | 4 primary rays per pixel on a fixed pattern and a 0.5 px Gaussian |
| Materials | USD `OmniPBR` / `UsdPreviewSurface` | The same values (Newton imports them), OmniPBR layering |
| Cameras | `apply_real_camera_calibration` | The same |

The sun of the sky map does not show in the RTX frames (a sky-only render lights the robot like the sky without its
sun would). With the 7k dome the sky adds about a tenth of the light.

Randomization of `RtxLook-ArmJoints-VisRand4s-v0` (`mdp/rtx_look.py:RandomizeRtxLook`), per env:

- **Materials**, one draw for each of the wrist mount, the two inner fingers, peg, hole, table mat and three curtains,
  as `randomize_visual_appearance_multiple_meshes`: with probability 0.5 a texture of the 150-texture subset, projected
  cubically at 0.7-5 repeats per metre and multiplied by a tint in [0, 1]^3, else a solid colour in [0, 1]^3;
  specular level 0-1; roughness / metallic 0.2-1 / 0-0.8 on the wrist mount and fingers, 0-1 / 0-1 on peg, hole and
  curtains, 0.3-0.9 / 0-0.3 on the table mat. Drawn once at the start and then every 4 s, not on reset, as on RTX. On RTX a solid colour stays multiplied by the tint of the
  surface's last textured draw (the randomizer never resets `diffuse_tint`); that is reproduced.
- **Sky**: a map of the 100-HDRI subset, intensity 1000-4000, any rotation.
- **Cameras**: side +-2 cm / +-2 deg and focal 25.05-29.05, wrist +-5 mm / +-2 deg and focal 27.03-29.03.

RTX has one sky for all envs and redraws every env's materials whenever any env's 4 s timer fires; here both are per
env. In the RTX session's replay data the redraws are about 4 s apart in all envs at once.

## Results

RTX 4090, 2026-11-18. The RTX references are the RTX session's own renders (`UWLab-vision/outputs/ladder_eval`).

**No randomization, same scenes.** RTX: `gdome1k/g7000` (seed 7, 1024 envs, front camera of the collection task).
Here: `rtx_match.py` with the same seed and env count. Mean absolute difference 11.6 of 255 (median
8.5, signed +1.2). Median colour per surface:

| Surface | RTX | Here |
|---|---|---|
| Robot links (albedo 0.82) | 245, 245, 245 | 242, 242, 242 |
| Robot joints (albedo 0.28) | 225, 225, 225 | 223, 224, 225 |
| Robot caps (blue) | 243, 247, 248 | 241, 246, 248 |
| Robot black parts (albedo 0.03) | 121, 120, 120 | 149, 151, 153 |
| Peg and hole (green) | 240, 248, 237 | 238, 248, 236 |
| Base plate | 204, 203, 185 | 191, 193, 173 |
| Table mat (black, rough) | 93, 94, 93 | 86, 87, 89 |
| Back curtain (black) | 64, 65, 63 | 82, 83, 84 |
| Right curtain (black) | 26, 27, 27 | 24, 24, 24 |
| Table frame (teal metal) | 2, 36, 49 | 6, 45, 59 |

**Randomized policy images** (96x96, before image augmentation). RTX: the raw replay frames of the current run
(`nodlaa_aug2_run_audit`, 12 episodes, 480 frames). Here: expert rollouts on the twin
(64 episodes, 2560 frames).

| | RTX side | Here side | RTX wrist | Here wrist |
|---|---|---|---|---|
| Mean level | 174 | 162 | 142 | 138 |
| 10th / 50th / 90th percentile | 100 / 184 / 231 | 75 / 171 / 231 | 39 / 148 / 227 | 44 / 141 / 220 |
| Saturation | 0.36 | 0.36 | 0.26 | 0.27 |
| Laplacian | 12.4 | 13.1 | 12.3 | 11.2 |

The gripper as the wrist camera sees it has a median level of 110 on RTX and 124 here. The
RTX sample holds about 48 material draws; two RTX samples of that size differ by ten to twenty levels
in the mean (side view 162 and 184 in the `abl/DomeOnly` batch and series).

**Speed**, 1024 envs, 96x96, per camera (side / wrist):

| | Side | Wrist |
|---|---|---|
| Render, Warp BVH | 212 ms | 327 ms |
| Render, OptiX | 60 ms | 58 ms |
| OptiX render + filter + tone map in the env | 70 ms | 68 ms |
| The same for the Real2Sim v4 look | 23 ms | 22 ms |

An env step takes 658 ms against 567 ms with the Real2Sim v4 look. Not measured on an H200.

**Against RTX**, per 1024-env step at 96x96:

| Where | RTX task | This twin |
|---|---|---|
| Hyak L40, in training (job 43302208, collection time from its log) | 0.66 s typical, 1.32 s mean | not measured |
| This workstation's 4090, `bench_env_step.py` (random actions, 160 steps) | 3.40 s | 0.66 s |
| Start-up to the first step on this workstation | 29 min | 4 min |

On Hyak about four RTX steps in ten redraw every env's materials and sky and take 2.2 to 2.3 s; the twin's redraw
costs 3 ms. The local RTX number comes from the native Isaac install, which is CPU-bound here (GPU 3 to 25 % busy)
and far slower than the training container, so the Hyak row is the one to compare against.

**Checks.** `test_newton_path_lighting.py`: 13 checks, `PATH_LIGHTING_OK`. `test_newton_optix_render.py`:
17 checks, `OPTIX_PARITY_OK`. `optix_ab.py` on the randomized twin: `OPTIX_AB_OK` (the backends differ only by
sampling noise). The state expert succeeds in 31/32 episodes on the randomized twin
and 32/32 without randomization. Renders: `~/research/real2sim_renders/optix_rtx_look/`.

## Differences from RTX

- **Dark glossy plastic is too light.** Robot black parts 151 against 120, back curtain
  83 against 65; under randomization the gripper reads 124 against 110.
- **Large matt surfaces are slightly dark**: table mat 87 against 94, base plate 193
  against 203; the darkest tenth of the randomized side view sits at 75 against 100.
- **Bounce.** One bounce at weight 0.3 stands in for RTX's three. A higher weight brightens the scene towards RTX
  and lightens the gripper further; 0.3 is the compromise.
- **Sky and redraws.** RTX has one sky for all envs and redraws every env at once; here both are per env.
- **Suns** are clamped out of every sky map. RTX shows no sun with the default sky; the other maps were not checked.
- **No exact RTX reference for the policy cameras** without randomization. It needs
  `record_norand_compare.py --task <RTX task> --num_envs 1024 --seed 7 --record_cams side_camera,wrist_camera` in the
  training container (the local Isaac install renders dome lights dimmer, so not on this workstation).

Use only renders of the current task as references for the randomized look. The `DomeOnly` ablation has the same
lights and the same reflection, ambient-occlusion and denoiser settings, but keeps frame generation on
(`enable_dlssg`, the base default), and its wrist-view gripper is much darker (median 37 against 106). The
exact-state reference `gdome1k` was rendered with frame generation off and DLAA on.

## How it works

`UWLabNewtonWarpRendererCfg(path_samples=1, ...)` turns on path lighting; `source/uwlab/uwlab/render/README.md`
(section Path lighting) describes the kernel and the filter. Settings of the look, all on the camera's renderer cfg or
the `rtx_look` event:

| Setting | Value | Meaning |
|---|---|---|
| `path_samples` | 1 | Lighting directions per pixel and lobe (2 rays per pixel). 2 lowers the remaining noise for twice the lighting rays. |
| `path_aa` | 4 | Primary rays per pixel. |
| `path_bounce` | 0.3 | Weight of the bounced light (RTX shows less inter-reflection than one full bounce gives). |
| `path_sky_clamp` | 5.0 | Clamp of the sky map's radiance. |
| `path_filter_step` | 6 | Spacing [px] of the second filter pass. |
| `tone_blur` | 0.5 | Gaussian [px] before tone mapping. |
| `events.rtx_look.params.dome` | 7.0 | White dome, RTX intensity / 1000. |
| `events.rtx_look.params.exposure` | 0.45 | Tone-mapping exposure [stops]. |

## Verify

```bash
# analytic scenes, about a minute; prints PATH_LIGHTING_OK
PYTHONPATH=_optix_deps/r570:source/uwlab python source/uwlab/test/test_newton_path_lighting.py

# Warp vs OptiX on the task, with timing; prints OPTIX_AB_OK
python scripts/tools/optix_ab.py --task <RtxLook task> --num_envs 1024 --rounds 2 --steps 20 --bench 20 \
    --out outputs/optix_ab --visualizer none presets=newton_mjwarp,newton_renderer <task overrides>
```

`scripts/tools/rtx_match.py` reproduces the scenes of the RTX session's `record_norand_compare.py` (fixed seed, zero
actions): `--save_states` at the reference's env count and seed, then `--load_states` in a small env. The reset
states depend on the seed and the env count only, not on the renderer, so its images show the same robot poses and
objects as the RTX reference.

| File | Role |
|---|---|
| `source/uwlab/uwlab/render/kernel.py`, `postprocess.py` | Path lighting and its filter |
| `source/uwlab/uwlab/render/bank.py` | Texture and environment-map banks (disk cache) |
| `source/uwlab_tasks/.../omnireset/mdp/rtx_look.py` | Dome and appearance randomization |
| `source/uwlab_tasks/.../ur5e_robotiq_2f85/rl_vision_cfg.py` | `add_rtx_look`, the two task cfgs |
| `scripts/tools/rtx_match.py`, `render_grid.py` | Reference scenes; contact sheets |
| `source/uwlab/test/test_newton_path_lighting.py` | Analytic tests |
