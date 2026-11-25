# OptiX backend for the Newton renderer

`UWLabNewtonWarpRenderer` can trace shapes with NVIDIA OptiX instead of Warp's BVH. It is a port of Newton's
experimental OptiX camera backend (newton-physics/newton #5285 and #5334, November 2026) to the Newton 1.5.2 /
Isaac Lab 3 stack used here. Everything except the shape traversal is unchanged: the same render kernel, materials,
lights, textures, tone mapping and outputs, so images match the Warp backend (see [Parity](#parity)).

OptiX needs no RT cores and no Vulkan, so it runs on the H200 nodes that cannot run the RTX renderer.

| File | Role |
|---|---|
| `source/uwlab/uwlab/render/optix.py` | Acceleration structure build / refit, and the OptiX trace and occlusion functions |
| `source/uwlab/uwlab/render/kernel.py`, `context.py` | Backend switch in the render kernel and the launch |
| `source/uwlab/uwlab/utils/newton_warp_renderer*.py` | `backend` option of the Isaac Lab renderer |
| `scripts/tools/build_optix_deps.sh` | Builds the Python packages the backend needs |
| `source/uwlab/test/test_newton_optix_render.py` | Warp vs OptiX parity on a synthetic scene (no Isaac Sim) |
| `scripts/tools/optix_ab.py` | Warp vs OptiX parity and timing on a camera task |

## Setup

The backend needs `warp-optix`, which needs Warp 1.17 or newer and Python 3.12. The container ships Warp 1.16, so the
packages go into a directory that is put first on `PYTHONPATH`. **Warp 1.19 then replaces Warp 1.16 for the whole
process**, simulation included; the first run recompiles every Warp kernel.

```bash
bash scripts/tools/build_optix_deps.sh r570 /path/to/python3.12    # writes _optix_deps/r570 (about 5 minutes)
```

| Flavor | Contents | NVIDIA driver |
|---|---|---|
| `stock` | PyPI `warp-lang 1.19.0`, `warp-optix 0.3.0`, `pyoptix 9.1.1` (OptiX SDK 9.1) | R590 or newer |
| `r570` | the same, with `pyoptix` and the `warp-optix` bridge recompiled against the OptiX 9.0 headers | R570 or newer |

Use `r570` unless every machine has a new driver; it also runs on new drivers (tested on R580 and R620). The stock
wheels fail on an older driver with `ImportError: Unsupported ABI version`. Drivers as of 2026-11-17: Tillicum H200
nodes 620.70, Hyak 580.196, the RTX 4090 workstation 580.178.

The driver's `libnvoptix.so.1` and `libnvidia-rtcore.so.*` must be visible in the container; `apptainer --nv` binds
them.

In the SIF:

```bash
UWLAB_OPTIX_DEPS=r570 UWLAB_NEWTON_RENDER_BACKEND=optix bash scripts/tools/run_in_sif.sh <script> <args...>
```

Without the SIF, put `_optix_deps/<flavor>` first on `PYTHONPATH` yourself.

## Enable

Either set the environment variable, which applies to every `UWLabNewtonWarpRenderer` camera:

```bash
export UWLAB_NEWTON_RENDER_BACKEND=optix
```

or set it per camera (Hydra):

```bash
env.scene.side_camera.renderer_cfg.backend=optix env.scene.wrist_camera.renderer_cfg.backend=optix
```

`backend` is a field of `UWLabNewtonWarpRendererCfg`, next to two tuning options:

| Option | Default | Meaning |
|---|---|---|
| `backend` | `"warp"` | `"warp"` or `"optix"` |
| `optix_extent` | `None` | Half-width [m] of the cube about each env's centre that is traced. Default: twice the half-width of the env's bounds when the structure is built (2.0 m for the peg tasks). Shapes that leave it are hidden. |
| `optix_rebuild_interval` | `0` | Rebuild the instance structure every N refits instead of refitting it. |

The backend works with and without `enable_patches`. Only the Real2Sim tasks use `UWLabNewtonWarpRendererCfg` today;
the other vision tasks get Isaac Lab's `NewtonWarpRendererCfg` from the `newton_renderer` preset and have to switch
their cameras to `UWLabNewtonWarpRendererCfg` first. Without patches that renderer draws the same image as Isaac
Lab's (bit-identical on `...-Aux-RGB-WarpRand-v0`; `scripts/tools/render_check.py --uwlab_renderer` does the swap).

If OptiX cannot start (packages missing, driver too old, CPU device), the renderer raises at start-up. It never falls
back to Warp silently.

## Verify

```bash
# synthetic scene, about one minute; prints OPTIX_PARITY_OK
PYTHONPATH=_optix_deps/r570:source/uwlab python source/uwlab/test/test_newton_optix_render.py

# a camera task: renders the same states through both backends, compares them and times them; prints OPTIX_AB_OK
python scripts/tools/optix_ab.py --task <task> --num_envs 1024 --rounds 3 --bench 30 --out outputs/optix_ab \
    --visualizer none presets=newton_mjwarp,newton_renderer <task overrides>
```

`optix_ab.py --diagnose` lists the shapes the differing pixels are on. Both tools have negative controls: the synthetic
test checks that a stale acceleration structure and a wrong ray offset are detected, and `optix_ab.py --expect_fail`
skips the OptiX refit and passes only if parity then breaks.

## Parity

Measured with `optix_ab.py` on `...-Real2SimV4-ArmJoints-v0` (64x64 side and wrist cameras, PBR, IBL, textures,
two shadow-casting lights), same simulation state through both backends, after 100 and 200 random-action steps:

| Machine | Envs | Pixels on a different shape | Mean colour difference (0-255) | Pixels off by more than 2 levels |
|---|---|---|---|---|
| H200 (Tillicum) | 4096 | 0.04 % side, 0.02 % wrist | 0.03 | 0.06 % side, 0.11 % wrist |
| RTX 4090 | 1024 | 0.04 % side, 0.02 % wrist | 0.02-0.03 | 0.05 % side, 0.08 % wrist |

The differing pixels are silhouette pixels, where two float32 traversals pick different shapes, and pixels on shadow
boundaries. In the synthetic test no pixel on the same shape differs by more than 3 levels, and depth agrees within
0.6 mm.

End to end, with the simulation also on Warp 1.19: the PPO state expert on
`...-Real2SimV4-ArmJoints-VisRandPerStep-v0` (1024 envs, 2048 episodes, H200) reaches 98.34 % success with the OptiX
backend and 98.34 % with the unmodified stack (Warp 1.16, Warp BVH).

The Warp backend itself is unchanged: on identical states every output is bitwise identical to the parent commit's.
(Whole-env runs are not bit-reproducible: two identical Warp runs differ by up to 3 levels on about 0.1 % of pixels.)

## Speed

One frame of one camera, acceleration refit included, on the task above:

| GPU | Envs | Camera | Warp | OptiX | Speed-up |
|---|---|---|---|---|---|
| H200 | 4096 | side | 292.3 ms | 154.0 ms | 1.9x |
| H200 | 4096 | wrist | 423.4 ms | 177.6 ms | 2.4x |
| H200 | 1024 | side | 74.8 ms | 39.3 ms | 1.9x |
| H200 | 1024 | wrist | 108.1 ms | 44.9 ms | 2.4x |
| RTX 4090 | 1024 | side | 39.4 ms | 9.1 ms | 4.3x |
| RTX 4090 | 1024 | wrist | 59.2 ms | 8.2 ms | 7.2x |

In FlashSAC-RGB training (`...-Real2SimV4-ArmJoints-VisRand4s-v0`, 4096 envs, one H200, the recipe of
[README_real2sim_v4.md](README_real2sim_v4.md)), two 120-iteration runs from the same tree, averaged over iterations
40-119:

| Backend | Collection | Learning | Iteration |
|---|---|---|---|
| Warp (Warp 1.16) | 1.11 s | 0.57 s | 1.69 s |
| OptiX (Warp 1.19) | 0.72 s | 0.57 s | 1.29 s |

Warp's collection time was still falling in that window. A 700-iteration OptiX run stays at 0.71-0.80 s collection
and about 1.3 s per iteration through iteration 699. The 20k-iteration Warp run of 2026-11-04 (same recipe plus the OSC
error clip) sat at 1.37-1.45 s collection and 1.94-2.04 s per iteration from iteration 500 on. At iteration 700 the two
runs report the same training success (0.157 and 0.156) and aux loss (0.61 and 0.63).

Where the time goes (H200, 1024 envs, side camera): 9.5 ms without shadows, 39.6 ms with the two shadow rays per
pixel. The H200 traverses in software; the 4090 has RT cores. Refitting takes 0.35 ms at 4096 envs, and a rebuilt
instance structure is no faster than the refit one after 200 steps.

Two settings matter and are already set:

- Rays that cannot reach the box around the global shapes (here the ground plane mesh) skip the second structure.
  This is worth 1.6x on the H200.
- The pipeline is compiled with a 255-register cap. Upstream's 80 makes the shadow path spill (38 ms instead of 9 ms
  on the 4090). The cap has no effect on the H200.

## Differences from the Warp backend

- **No CUDA-graph capture.** OptiX launches cannot be recorded, so the camera task runs eagerly. Physics stepping is
  still captured.
- **Shapes far outside their env are hidden.** Each env is traced inside a cube of half-width `optix_extent` about its
  centre. A shape whose bounding box leaves the cube, or whose transform is not finite, is not rendered, and a
  warning is logged. The Warp backend would still draw it.
- **Cones no longer shadow themselves.** Newton 1.5.2's shadow test counts the small negative distance its cone routine
  returns for a ray leaving the surface as a hit, so a lit cone renders in shadow under Warp. OptiX rejects the hit.
  The peg tasks have no cone shapes.
- **Back-face culling** (`enable_backface_culling=True`, off by default): OptiX skips back faces and continues to what
  lies behind. Warp drops the whole mesh for a ray whose nearest hit on it is a back face.
- **Not supported:** particles, deformable surfaces and Gaussian splats. The renderer raises if the model has any.
- **Convex hulls** are not rendered, as with the Warp backend on Newton 1.5.2.
- **Small batches.** A launch plus refit has a fixed cost of about 2 ms per camera frame (RTX 4090), so the speed-up
  shrinks with the env count: 1.4-2.2x at 64 envs.

## How it works

One OptiX instance per shape, in a single instance structure for all envs. Meshes are triangle structures shared by
all their instances, boxes are a triangle cube, and spheres, capsules, cylinders, cones and ellipsoids are a unit cube
with an intersection program that calls Newton's analytic routines. Planes are intersected analytically outside
OptiX. Every step the shape transforms are recomputed on the GPU and the instance structure is refit.

Envs are kept apart by space, not by ray masks: each env's shapes are moved to their own cell of a grid, and each
ray is moved to its env's cell and clipped to it. The cells are packed about each env's own centre. Upstream packs
about the world origin, which would spread 4096 Isaac Lab envs over kilometres and cost float32 precision. Global
shapes (world -1) live in a second structure at their true coordinates.

The render kernel is compiled as the OptiX ray-generation program by `warp-optix`, so the shading code is the Warp
kernel's.
