# RTX real2sim training env: scope (2026-10-29, not started)

Yanda's RTX real2sim look exists only as a static capture (`newton-flow-distill`: `scripts/tools/real2sim_render.py` +
`scripts/tools/real2sim_settings.json`): one frozen, path-traced env-0 stage. The Newton-Warp version is trainable and
ported here as `...-WristSide-Sysid-Aux-RGB-Real2Sim-v0` (`add_real2sim_appearance`, ac0fa455 look).

## Capture components -> training env

| Component | Capture | Training version | Effort / risk |
|---|---|---|---|
| Scene materials (12 groups: curtains, peg, hole, tabletop, table metal, UR5 support, mount, finger meshes) | UsdPreviewSurface (color + roughness) bound on env 0 | startup event binding the same materials on every env's prims | low |
| Robot materials (LinkGrey / JointGrey / URBlue) + smoothed arm normals | rewrites robot materials and mesh normals, robot made non-instanceable | edit the shared robot prototype once (no de-instancing memory cost) | low-medium |
| Bounded randomization | none (one frame) | existing RTX colour / roughness / texture events with narrow ranges around the matched values, plus Yanda's synthetic textures and surface maps | medium |
| Lighting: dome intensity 230 + warm tint, 0.14 m RectLight softbox at 250000 near (0.7, -0.4, 1.0) | one light in env 0's frame | per-env lights (1024 lights; spill onto neighbour envs 1.5 m apart, render cost) or one global DistantLight (cheap, identical shadows, no per-env jitter) | medium-high, needs an experiment |
| Curtain folds | 100x100 subdivided wave meshes replace the box curtains | ~30M verts at 1024 envs is too heavy: one instanced fold mesh, or coarser | medium |
| Cables | one spline | skip at first | low |
| Render mode | path traced (4 spp / 256 total) | real-time RTX + DLAA (path tracing at 1024 envs is far too slow) | look gap, validate separately |
| Cameras | policy cameras use the training poses; extra reference cameras only | nothing | none |

## Risks
1. Real-time vs path-traced: the training look will differ (soft shadows, indirect light); retune dome / key
   intensities against the real photos in real-time mode.
2. Per-env RTX lights are not env-bounded: brightness can scale with env count (cf. the robot dome-light bug) and add cost.
3. Memory: RTX at 1024 envs x 64 px is already ~20 GB on a 4090 with the texture set; lights / fold meshes add to it.
4. RTX only renders on workstations (local 4090, ur5e 2x4090), not on cluster GPUs.

## Plan
1. Static look (~half day): materials, robot materials, dome + one global key light, smoothed arm. Render the training
   env at the capture state, compare to the real photos and Yanda's RTX capture, retune intensities for real-time RTX.
2. Bounded randomization (~half day): per-env colour / roughness jitter, his textures + surface maps, key-light
   direction / intensity jitter (global or per-env, from the light experiment).
3. Benchmark at 1024 envs (64 px, and 128 px for DINO): memory and speed.
4. Only if a visible gap remains: instanced curtain folds, cables.
5. Then the usual pipeline: RTX expert buffer + finetune, so a Warp real2sim pretrain finetunes into the matching RTX look.
