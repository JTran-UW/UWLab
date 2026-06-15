# Sim2Real Point Cloud Augmentation — Tabletop Manipulation

**Setup:** Single fixed **Intel RealSense D455**, used only as a **stereo IR camera pair** — depth comes from **FoundationStereo** (Wen et al., NVIDIA, CVPR 2025) inferenced on the rectified pair, *not* from the D455's onboard block-matching ASIC. Relatively unoccluded tabletop manipulation. Goal: make synthetic (CAD-derived) point clouds match the FoundationStereo output, so a policy trained in sim transfers.

**Why this flips the noise model:** hardware block-matching produces *sparse, holey, quantized, high-frequency-noisy* depth. A learned iterative matcher with monocular priors produces the opposite — **dense, smooth, low per-point noise** — but introduces **edge bleeding** and **hallucinated geometry**. So model the latter, not the former.

**Facts that drive the noise model here:**
- **Dense, gap-filled output.** FoundationStereo fills textureless regions via monocular priors → **do NOT model textureless/low-texture dropout**; surfaces that the D455 ASIC would hole out come back filled (sometimes wrongly).
- **Continuous disparity.** Learned float output → **no 1/32-px hardware quantization step.** Drop it.
- **Edge over-smoothing / foreground–background bleed.** Iterative cost-volume + monocular priors smear depth across object boundaries — the dominant geometric artifact. Manifests as a thin skirt of points bridging object edge → table.
- **Hallucination in ambiguous regions.** Occluded, repetitive, reflective, or transparent areas get plausible-but-wrong depth (transparent objects are a stated model weakness). Shows up as smooth phantom surfaces, not holes.
- **Residual triangulation error still applies.** Depth is still bounded by the **95 mm baseline** + input resolution, so axial error still grows ~quadratically with range — but the constant is smaller and the noise is **low-frequency/biased** (gently bent planes) rather than per-point jitter.
- **IR-pair caveat:** FoundationStereo is trained on photorealistic RGB-ish data; the D455 projector's **dot pattern is out-of-distribution**. Most setups **turn the emitter OFF** and feed clean IR (or RGB-rectified) pairs. Decide this *before* generating sim data — it changes the input distribution.

**Core insight:** For a single fixed camera, occlusion is *deterministic* (you always see only front/top surfaces; object backs are always missing). The dominant sim2real gap is therefore **depth sensor noise**, not occlusion diversity. Budget effort accordingly.

---

## Pipeline (order matters)

```
CAD point cloud
  → 1. HPR from real camera pose (self-occlusion)
  → 2. edge bleed (boundary skirt) + low-freq surface bias  ← the big one for learned stereo
  → 3. light random dropout
  → partial, realistic cloud
```

HPR runs first because it defines which surface is visible; noise and dropout then act only on the surviving points.

---

## 1. Occlusion — Hidden Point Removal (HPR)

**Technique:** Katz–Tal–Basri HPR operator (Katz et al., *Direct Visibility of Point Sets*, SIGGRAPH 2007). Spherical inversion + convex hull; approximates visibility from a viewpoint with no meshing or ray tracing.

**Library:** Open3D, built in.

```python
import numpy as np

def occlude_hpr(pcd, camera_center, radius_scale=1000.0):
    diameter = np.linalg.norm(
        np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0)
    )
    radius = diameter * radius_scale
    _, visible_idx = pcd.hidden_point_removal(camera_center, radius)
    return pcd.select_by_index(visible_idx)
```

**Key decisions for fixed single camera:**
- **Run HPR from the *real* camera pose, not random viewpoints.** Random spherical sampling is for category-level pretraining; here it injects a domain gap we don't have.
- **Occlusion diversity comes from object pose randomization**, which the sim already does — not from moving the camera.
- `radius` is the only knob: too small over-prunes concave regions; a generously large radius preserves fine concave detail. Tune once against real frames.

---

## 2. Boundary Bleed + Surface Bias — highest-value addition

Clean CAD clouds don't have the two artifacts that dominate learned-stereo output. Per-point Gaussian noise is *not* the main gap here — FoundationStereo's per-point noise is small. Model these instead:

**(a) Edge bleed — the dominant geometric artifact.** At object boundaries the network interpolates depth between foreground and background, producing a "skirt" of points that bridge the object edge down to the table. To simulate: find boundary points (grazing-angle / depth-discontinuity), and *add* interpolated points along the ray between object depth and the surface behind it.

```python
def edge_bleed(points, normals, camera_center,
               bg_depth_fn, n_skirt=3, frac=0.4):
    view = points - camera_center
    z = np.linalg.norm(view, axis=1, keepdims=True)
    dirs = view / (z + 1e-9)
    grazing = np.abs((normals * (view / z)).sum(1)) < 0.3   # boundary points
    edge = points[grazing]
    edge_dirs = dirs[grazing]
    z_obj = z[grazing]
    z_bg = bg_depth_fn(edge)                  # depth of surface behind (e.g. table)
    skirts = []
    for a in np.linspace(0.2, 0.9, n_skirt):  # interpolate obj->bg
        zi = (1 - a) * z_obj + a * z_bg[:, None]
        skirts.append(camera_center + edge_dirs * zi)
    keep = int(frac * len(edge))
    extra = np.concatenate(skirts)[np.random.choice(len(np.concatenate(skirts)), keep, False)]
    return np.vstack([points, extra])
```

**(b) Low-frequency surface bias.** Learned stereo bends flat surfaces gently and gets the *global* scale/offset slightly wrong, rather than adding white noise. Model as a smooth field, not per-point jitter:

```python
def surface_bias(points, camera_center,
                 axial_scale=0.005, bend=0.003):
    rays = points - camera_center
    z = np.linalg.norm(rays, axis=1, keepdims=True)
    dirs = rays / (z + 1e-9)
    bias = axial_scale * z                     # small systematic scale error
    bias += bend * (z - z.mean())**2 / (z.mean()**2 + 1e-9)  # gentle plane bend
    pts = camera_center + dirs * (z + bias)
    pts += np.random.randn(*pts.shape) * 0.0005   # tiny residual jitter only
    return pts
```

**Calibration:** the residual triangulation floor still scales with z²/baseline (95 mm), but for learned stereo the *bias* terms dominate. Don't fit these from the datasheet — fit them by comparing FoundationStereo output on a few real flat-target captures against ground truth (a checkerboard plane works). The bend/scale you measure there is what to inject.

**Reference:** Wen et al., *FoundationStereo: Zero-Shot Stereo Matching*, CVPR 2025 (arXiv:2501.09898) — note the stated weakness on transparent objects, which feeds §3.

---

## 3. Dropout — minimal

With a simple scene (no transparent/reflective/repetitive surfaces) there's nothing for the network to hallucinate into, so skip phantom geometry entirely. What remains is light:

- **Light random dropout** — mild regularization (keep ~90–95%).
- **Occlusion-border thinning** *(optional)* — disparity right at a left/right occlusion boundary is unreliable; thin those narrow bands if you see it in real captures. Often negligible for a sparse tabletop.

**Do NOT model** (hardware-stereo artifacts FoundationStereo removes, or irrelevant here): textureless/specular dropout, disparity quantization, MinZ holes, region/angle removal, phantom surfaces.

```python
def light_dropout(points, keep=0.92):
    return points[np.random.rand(len(points)) < keep]
```

---

## Priority Ranking

1. **Edge bleed** (§2a) — the dominant FoundationStereo artifact; boundary skirts are the most recognizable sim2real tell.
3. **HPR self-occlusion** (§1) — necessary and well-justified.
4. **Surface bias** (§2b) — gentle plane bend / scale error; low effort once measured.
5. **Light dropout** (§3) — minor regularization; little else to do for a simple scene.

## Validation (do before tuning blind)

Run FoundationStereo on a few real tabletop captures and compare the resulting clouds against sim clouds at matched object poses. For a *learned* matcher the dominant visible discrepancy is almost always **boundary bleed and over-smoothed fine geometry**, not per-point noise — so eyeball the object edges and thin features first. A flat checkerboard target gives you the surface-bias (bend/scale) numbers for §2b directly. Cheaper than tuning in the dark.

## Resolved: Depth Source

**FoundationStereo on the D455 IR stereo pair** (not the onboard ASIC). Implications baked into §2/§3:
- Noise model is **edge bleed + low-frequency surface bias**, *not* disparity quantization or textureless dropout.
- **Dropout/holes mostly removed**; with a simple non-reflective scene there's no ambiguous region to hallucinate into, so §3 is just light random dropout.
- **Decide the emitter state (on/off) before generating sim data** — the projector dot pattern is OOD for the model; most use it off. Whatever you pick, keep it consistent between calibration captures and deployment.
- Residual triangulation floor still set by the **95 mm baseline** + input resolution; pin `FOCAL_PX` to the resolution you actually feed the model.

## References

- Katz, Tal, Basri. *Direct Visibility of Point Sets.* SIGGRAPH 2007. (HPR operator)
- Wen et al. *FoundationStereo: Zero-Shot Stereo Matching.* CVPR 2025. arXiv:2501.09898. (learned stereo; transparent-object weakness)
- Intel RealSense D455 / D400-series datasheet. (95 mm baseline; IR stereo pair geometry)
- Open3D `hidden_point_removal` documentation.