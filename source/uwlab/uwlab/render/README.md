# Newton Warp extensions

This optional path targets **Newton 1.5.2 / Warp 1.16**. `context.py` and
`kernel.py` own the render launch and shading loop; Newton still owns geometry,
BVH queries, mesh import, and albedo sampling. The installed Newton package is
unchanged. The adapted launch/shading code retains Newton's Apache-2.0 notice
and license. All files ship through the existing `source/` rsync.

The extensions stay off by default until the plan's full OmniReset MIG speed
and frame gate is measured. They do not require Vulkan, RTX, or RT cores.

## Enable

```bash
export UWLAB_NEWTON_WARP_PATCHES=1
export UWLAB_NEWTON_WARP_SHADOWS=1
export UWLAB_NEWTON_WARP_SPP=4
export UWLAB_NEWTON_WARP_TONE_MAPPER=aces
export UWLAB_NEWTON_WARP_EXPOSURE_RANGE=-1,1
# Optional additional cost:
export UWLAB_NEWTON_WARP_PBR=1
export UWLAB_NEWTON_WARP_REFLECTIONS=1
export UWLAB_NEWTON_WARP_SHADOW_SAMPLES=4
# Sample the HDRI as a shadow-casting dome light (requires an environment map):
export UWLAB_NEWTON_WARP_DOME=1
export UWLAB_NEWTON_WARP_DOME_SAMPLES=1
# Stable precomputed image-based lighting (alternative to sampled dome):
export UWLAB_NEWTON_WARP_IBL=1
```

The camera's `UWLabNewtonWarpRendererCfg` exposes all settings for Hydra/config
edits. For example, `env.scene.side_camera.renderer_cfg.shadow_floor=0.15`.
Apply matching scene settings to each camera if they should share a renderer.

| Setting | Behavior |
| --- | --- |
| `ambient_sky`, `ambient_ground`, `ambient_intensity` | Linear RGB hemispheric ambient; defaults match stock Newton |
| `spot_inner`, `spot_outer` | Cone cosine limits, with `outer < inner` |
| `distance_attenuation`, `shadow_floor` | Inverse quadratic point-light falloff and blocked-ray visibility |
| `samples_per_pixel` | Center sample plus deterministic jittered color samples, 1–64 |
| `shadow_samples` | Shadow/light-position samples per light, 1–64 |
| `per_world_intrinsics` | Read each world's full K, including skew, directly in the pixel kernel |
| `per_world_lights` | Independent lights, colors, intensities and radii, including partial resets |
| `tone_mapper`, `exposure_range` | `none`, `linear`, `reinhard`, or ACES filmic approximation; exposure in stops |
| `aperture_radius`, `focus_distance` | Thin lens radius and focal-plane distance in metres |
| `shutter_fraction` | Camera-motion blur between previous/current rendered poses, 0–1 |
| `enable_pbr`, `enable_reflections` | Cook–Torrance GGX and one secondary glossy ray per color sample |
| `environment_paths` | Equirectangular HDRI bank, sampled for background and ambient |
| `enable_dome_light`, `dome_samples` | Brightness-weighted HDRI samples at each surface; optional visibility rays when shadows are enabled |
| `enable_precomputed_ibl` | Precomputed diffuse irradiance and roughness-filtered GGX HDRI reflections; no per-pixel dome rays |
| `material_overrides` | Shape-label regex to scalar material and texture-map settings |

Color/HDR samples are averaged in linear light. Normals, albedo, segmentation,
and depth describe the unjittered center sample. Exposure/tone mapping changes
RGBA only; requested HDR remains raw scene-linear radiance. Planar depth uses
the same world-specific K as ray generation. PPISP and the local tone mapper
are mutually exclusive to prevent applying two camera responses.

Per-world K is evaluated in the kernel instead of allocating a
`worlds × cameras × H × W` ray table, saving about 1.8 GiB for 1,024 worlds at
320×240. Input/output arrays remain allocated throughout rendering, including
CUDA graph capture. Reset events update array contents, preserving pointers.

## Lights, materials and environments

`RandomizeNewtonLights` samples relative to each resetting world's origin when
patches are enabled. Stock shared lights still update only on a full reset.
Set `UWLAB_NEWTON_WARP_MIN_POINT_LIGHTS=1` and
`UWLAB_NEWTON_WARP_MAX_POINT_LIGHTS=1` to force two active lights for benchmarking.
The lower-level `renderer.set_lights` accepts optional `env_ids`, `types`
(0 spot, 1 directional, 2 point), `cast_shadow`, `colors`, `intensities` and
`radii`. Directional radii are angular perturbations; point radii are metres.

Mesh metallic/roughness values come from Newton's USD importer. The material
array also accepts explicit overrides before the first render:

```python
camera.renderer_cfg.material_overrides = {
    r"/Peg/": {
        "metallic": 0.8,
        "roughness": 0.3,
        "specular": 0.5,
        "roughness_texture": "/path/to/roughness.png",  # linear red channel
        "normal_texture": "/path/to/normal_gl.png",    # OpenGL tangent normal
        "normal_strength": 0.5,
        "emissive": (0.0, 0.0, 0.0),                 # linear RGB
        # "emissive_texture": "/path/to/emissive.png", # sRGB multiplier
    },
}
```

UV meshes use a tangent basis from triangle positions/UVs, including mirrored
UV handedness. Meshes without UVs use the existing cubic projection.
Texture banks and material maps load before graph capture; scalar material
values can subsequently be changed in place with `context.set_materials`.

Set `environment_paths=("/path/sky.npy", ...)` or the path-separated
`UWLAB_NEWTON_WARP_ENVIRONMENTS` variable. Float `.npy` files must hold finite,
nonnegative linear RGB(A), H×W×3/4. HDR/EXR use an available imageio decoder;
no decoder is downloaded at runtime. LDR images are converted from sRGB.
Reset DR selects a map, rotation and intensity separately per world.
Sampled dome lighting requires a loaded environment map. It replaces the
normal-direction ambient approximation, uses each world's HDRI selection and
rotation, and can light a scene with all direct lights disabled. A compact
128x64 luminance-weighted distribution is built for each HDRI before graph
capture. One sample can still be noisy, especially on glossy metals or HDRIs
with small bright regions. This first implementation does not denoise the output.
Set `dome_samples` higher for less noise at higher render cost. When
`enable_shadows` is false, dome samples do not trace visibility rays.

Precomputed IBL is an alternative to sampled dome lighting and requires PBR and
an environment map. At map load, it builds a cosine-convolved diffuse map,
six GGX-filtered reflection maps and a BRDF lookup table. The runtime shader
uses filtered texture lookups, so its output is stable at one sample per pixel.
The maps are shared across worlds; per-world HDRI selection, rotation and
intensity still apply. Precomputed IBL does not account for scene geometry
blocking the sky, so contact and self-shadowing require a separate technique.
One shadow-casting directional light can be used alongside IBL for robot shape
and contact shadows. In the peg benchmark, the clearest tested compromise used
one key light at intensity 2.0, HDRI intensity 0.25, +0.5 camera exposure stop,
and no roughness map on the table/insertive object. The benchmark case is
`ibl_hybrid_key2_exposure05` in `scripts/tools/benchmark_newton_features.py`.

## Verification and limits

```bash
WARP_CACHE_PATH=/tmp/uwlab-warp-cache PYTHONPATH=source/uwlab \
  /home/yandabao/miniforge3/envs/patlab31/bin/python \
  source/uwlab/test/test_newton_warp_render.py
# Add UWLAB_RENDER_TEST_DEVICE=cuda:0 for CUDA and graph-capture checks.
# CUDA compiler check without a GPU:
WARP_CACHE_PATH=/tmp/uwlab-warp-cache PYTHONPATH=source/uwlab \
  /home/yandabao/miniforge3/envs/patlab31/bin/python \
  scripts/tools/compile_newton_warp_cuda.py --arch 90
```

The tests render real BVH queries on small multi-world scenes: stock Lambert
parity, independent zoom/light buffers, sample averaging, material maps, HDRI,
exposure/HDR preservation, depth conversion, and the IsaacLab wrapper buffers.

Use `scripts/tools/probe_newton_rgb.py --visualizer none --headless` for the
actual task frame/throughput gate. It saves the first four worlds' camera PNGs
and timing JSON. `--save_state` / `--load_state` reproduce scene states between
runs; disable camera-pose DR for a fixed-camera visual comparison. The synthetic
unit scenes do not establish OmniReset throughput or sim-to-real quality.

Camera-motion blur is implemented; moving-object/deforming-geometry blur is
not implemented. The older HDRI ambient samples the normal direction rather
than a prefiltered diffuse convolution. Reflections are one stochastic specular bounce
with direct/ambient shading at the secondary hit, with no denoiser. These are
experimental rendering features and retain visible sampling noise at low spp.

The metallic/roughness BRDF uses GGX distribution, Smith visibility and Schlick
Fresnel; see the [Khronos material reference](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material).

## Matched real-to-sim training environment

`Ur5eRobotiq2f85Real2SimNewtonEnvCfg` in
`uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.real2sim_cfg`
can be passed directly to `isaaclab.envs.ManagerBasedRLEnv(cfg=cfg)`. It needs the
Newton 1.5.2 / IsaacLab 3.1 stack (`patlab31` locally). There is no task registration
or agent configuration. The inherited peg rewards, actions, RGB/proprioceptive
observations and reset datasets support algorithm-specific wrappers.

Each camera explicitly selects `UWLabNewtonWarpRendererCfg(enable_patches=True)`.
Backend finalization preserves this explicit renderer configuration. This training
preset uses GGX materials, shadows, ACES and precomputed studio IBL.
Traced reflections and sampled dome lighting remain disabled.
Textures are preloaded before the first render, and resets update persistent device
buffers only for the selected worlds.

`cfg.events.matched_appearance.params` controls additive linear-RGB jitter (default
±0.035, ±0.015 on green objects), roughness jitter (±0.08), metallic jitter (±0.04),
texture probability (0.5), key-light position jitter (±0.12 m per axis) and light
strength multipliers (0.8–1.2). The key is a point light near (0.7, -0.4, 1.0) in
each world's frame, with a directional fill. Light colors vary ±0.04; the fill
direction varies ±0.1 per component. Existing camera pose/focal and physical
randomizations are inherited. The texture bank consists of generated near-white
mat grain, cloth weave and print-line multipliers; it deliberately preserves the
matched base colors. These are synthetic similar-surface variants, not scans of
the real materials.

The standalone capture scripts remain separate. This env uses live robot poses and
the training scene geometry, not the frozen capture's curtain folds or cable curves.
Run `scripts/tools/smoke_real2sim_env.py` on GPU 0 to check live steps, all three
cameras and partial-reset isolation.

### Camera response and surface realism

The matched training preset now preloads normal/roughness maps for mat grain,
cloth, printed plastic and brushed metal, plus three linear synthetic studio
environment maps (neutral/warm/cool). `scripts/tools/generate_real2sim_maps.py`
regenerates the bundled maps deterministically. They approximate studio lighting
and surface structure; they are not measured scans or calibrated HDR captures.
Precomputed diffuse irradiance, roughness-filtered specular maps and a BRDF LUT
are built once before rendering, with no secondary scene-reflection rays.

The reset event samples exposure in ±0.2 stops, linear white-balance gains in
1±0.04, display gamma in [0.95,1.05], read-noise standard deviation in [0,0.003]
and signal-dependent noise coefficient in [0,0.006]. Noise is a lightweight
display-space approximation rather than a physical raw-sensor pipeline. It is
seeded per world and changes each render. White balance and exposure precede
ACES; gamma/noise follow sRGB conversion. IBL strength varies in [0.65,1.0],
with a neutral/warm/cool map choice and rotation in ±0.035 turns. These settings
are event parameters; renderer features remain individually configurable.

Partial resets preserve untouched worlds' camera/IBL state and all buffer
pointers. The renderer retains the original tone-mapping path when
`enable_camera_response=False`; PPISP and this tone-mapping path are mutually
exclusive. Raw HDR output remains unchanged by camera postprocessing.

For a fixed-camera before/after capture, run `real2sim_newton_render.py` with
`--settings scripts/tools/real2sim_newton_realism_settings.json`. It enables all
four features at nominal response values and mild noise, using the same saved
scene and camera matrices as the original comparisons.
