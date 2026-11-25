# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Path lighting of the UWLab Newton renderer on analytic scenes (no Isaac Sim needed).

    PYTHONPATH=<optix deps>:source/uwlab python source/uwlab/test/test_newton_path_lighting.py

An open floor under a uniform dome must receive exactly the dome (every secondary ray escapes); a roof over it must
darken it and a roof moved away must not (negative control). The interleaved samples must average to the plain
random estimator, the filter must remove their pattern without moving the mean or leaking across a depth edge, and
both backends must agree.
"""

import numpy as np
import sys

import newton
import warp as wp
from newton._src.sensors.warp_raytrace.types import ClearData

from uwlab.render.context import RenderContext
from uwlab.render.postprocess import filter_path_lighting, resolve_path_lighting
from uwlab.render.types import RenderConfig

WIDTH, HEIGHT, FOCAL = 96, 96, 110.0
DOME = 7.0
BASE = (0.5, 0.4, 0.3)
WORLDS = 2
SPACING = 20.0


def build_model(roof_height):
    """A grey floor per world, world 1 with a black roof at ``roof_height`` over its middle."""
    main = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for w in range(WORLDS):
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), hx=3.0, hy=3.0, hz=0.05, color=BASE)
        if w == 1:
            b.add_shape_box(
                -1,
                xform=wp.transform((0.0, 0.0, roof_height), wp.quat_identity()),
                hx=1.0,
                hy=1.0,
                hz=0.02,
                color=(0.0, 0.0, 0.0),
            )
        main.add_world(b, xform=wp.transform((w * SPACING, 0.0, 0.0), wp.quat_identity()))
    return main.finalize(device="cuda:0")


def cameras(device):
    def look_at(eye, target):
        forward = np.asarray(target, dtype=np.float64) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        q = wp.quat_from_matrix(wp.mat33(*np.stack([right, up, -forward], axis=1).reshape(-1)))
        return [*eye, q[0], q[1], q[2], q[3]]

    transforms = [look_at(np.array([w * SPACING + 0.9, -0.7, 0.3]), np.array([w * SPACING, 0.0, 0.0])) for w in range(WORLDS)]
    transforms = wp.array(np.array(transforms, dtype=np.float32)[None], dtype=wp.transformf, device=device)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    d = np.stack([(x + 0.5 - WIDTH / 2) / FOCAL, (HEIGHT / 2 - y - 0.5) / FOCAL, -np.ones_like(x)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    rays = np.zeros((1, HEIGHT, WIDTH, 2, 3), dtype=np.float32)
    rays[0, :, :, 1] = d
    intrinsics = np.tile(np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]], dtype=np.float32), (WORLDS, 1, 1))
    return transforms, wp.array(rays, dtype=wp.vec3f, device=device), wp.array(intrinsics, dtype=wp.mat33f, device=device)


class Renderer:
    def __init__(self, model, backend):
        self.model = model
        config = RenderConfig(
            enable_shadows=False, enable_textures=False, enable_pbr=True, max_distance=50.0, path_samples=1, path_outputs=True
        )
        self.context = RenderContext(world_count=model.world_count, device=model.device, config=config, backend=backend)
        self.context.init_from_model(model, load_textures=False)
        wp.to_torch(self.context.environment_ambient)[:] = DOME
        self.context.lights_active.fill_(False)  # the dome is the only light
        self.transforms, self.rays, self.context.camera_intrinsics = cameras(model.device)
        shape = (model.world_count, 1, HEIGHT, WIDTH)
        self.out = {name: wp.zeros(shape, dtype=wp.vec3f, device=model.device) for name in ("hdr_color_image", "light_image", "gloss_image", "tint_image", "normal_image")}
        for name in ("depth_image", "cover_image"):
            self.out[name] = wp.zeros(shape, dtype=wp.float32, device=model.device)

    def render(self, state, seed=0, paths=1):
        self.context.frame_seed, self.context.path_samples = seed, paths
        if self.context.backend == "warp":
            self.model.bvh_refit_shapes(state)
        self.context.render(self.model, state, camera_transforms=self.transforms, camera_rays=self.rays, clear_data=ClearData(clear_color=0xFF000000), **self.out)
        wp.synchronize_device(self.model.device)
        return {k.split("_")[0]: v.numpy()[:, 0].copy() for k, v in self.out.items()}

    def filtered(self, step=6):
        """The renderer's filter passes on the current buffers; returns the filtered light and the final HDR."""
        o = self.out
        scratch = [wp.zeros_like(o["light_image"]) for _ in range(4)]
        guides = [o["normal_image"], o["depth_image"]]
        shape = o["light_image"].shape
        launch = lambda kernel, inputs: wp.launch(kernel, dim=shape, inputs=inputs, device=self.model.device)
        launch(filter_path_lighting, [o["light_image"], o["gloss_image"], *guides, 1, 0, scratch[0], scratch[1]])
        launch(filter_path_lighting, [scratch[0], scratch[1], *guides, 0, 0, scratch[2], scratch[3]])
        launch(filter_path_lighting, [scratch[2], scratch[3], *guides, 1, step, scratch[0], scratch[1]])
        light = wp.zeros_like(o["light_image"])
        launch(filter_path_lighting, [scratch[0], scratch[1], *guides, 0, step, light, scratch[2]])
        hdr = wp.clone(o["hdr_color_image"])
        launch(resolve_path_lighting, [scratch[0], scratch[1], *guides, 0, step, o["tint_image"], o["cover_image"], hdr])
        wp.synchronize_device(self.model.device)
        return light.numpy()[:, 0], hdr.numpy()[:, 0]


def run():
    """Run every check, print one line each, and return the names of the failed ones."""
    wp.config.quiet = True
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
        if not ok:
            failures.append(name)

    model = build_model(roof_height=0.8)
    state = model.state()
    r = Renderer(model, "warp")
    a = r.render(state)
    floor = a["depth"] > 0.0
    open_light, roofed_light = a["light"][0][floor[0]], a["light"][1][floor[1]]

    # an open floor: every secondary ray escapes, so the diffuse light is the dome itself
    deviation = float(np.abs(open_light - DOME).max())
    check("open floor receives exactly the dome", deviation < 1.0e-4, f"max |light - dome| = {deviation:.2e}")
    tint = a["tint"][0][floor[0]]
    base = np.array([c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in BASE])  # shape colours are sRGB
    in_range = bool((tint <= base + 1e-5).all() and (tint >= 0.5 * base).all())
    check("tint is the base colour under the coat", in_range, f"mean {tint.mean(0).round(4)} of base {base.round(4)}")
    gloss = a["gloss"][0][floor[0]].mean()
    check("dielectric gloss is a few percent of the dome", 0.01 * DOME < gloss < 0.25 * DOME, f"mean {gloss:.3f}")
    check("unlit radiance is left for the resolve pass", float(np.abs(a["hdr"]).max()) < 1.0e-6)

    # a roof: most of the dome is blocked, and the black roof bounces nothing back
    ratio = float(roofed_light.mean() / DOME)
    check("roof darkens the floor", 0.1 < ratio < 0.8, f"mean light / dome = {ratio:.3f}")
    far = build_model(roof_height=500.0)
    b = Renderer(far, "warp").render(far.state())
    recovered = float(np.abs(b["light"][1][b["depth"][1] > 0.0] - DOME).max())
    check("negative control: a roof far away changes nothing", recovered < 1.0e-4, f"max |light - dome| = {recovered:.2e}")

    # interleaved samples are an unbiased estimate: their mean matches plain random sampling
    reference = np.mean([r.render(state, seed=s, paths=4)["light"][1] for s in range(1, 33)], axis=0)
    r.render(state)
    bias = float(abs(roofed_light.mean() - reference[floor[1]].mean()) / reference[floor[1]].mean())
    check("interleaved samples match random sampling in the mean", bias < 0.04, f"relative difference {bias:.4f}")

    # the filter: removes the sampling pattern, keeps the mean, and equals the dome where nothing is blocked
    light, hdr = r.filtered()
    raw_error = float(np.abs(a["light"][1] - reference)[floor[1]].mean())
    filtered_error = float(np.abs(light[1] - reference)[floor[1]].mean())
    check("filter approaches the converged light", filtered_error < 0.3 * raw_error, f"error {raw_error:.3f} -> {filtered_error:.3f}")
    drift = float(abs(light[1][floor[1]].mean() - roofed_light.mean()) / roofed_light.mean())
    check("filter keeps the mean", drift < 0.04, f"relative change {drift:.4f}")
    flat = float(np.abs(light[0][floor[0]] - DOME).max())
    check("filter leaves a uniform field alone", flat < 1.0e-3, f"max deviation {flat:.2e}")
    expected = (a["tint"][0] * DOME + a["gloss"][0])[floor[0]].mean()
    resolved = float(abs(hdr[0][floor[0]].mean() - expected) / expected)
    check("resolved radiance is tint * light + gloss in the mean", resolved < 0.03, f"relative error {resolved:.4f}")
    leak = float(np.abs(hdr[0][~floor[0]]).max()) if (~floor[0]).any() else 0.0
    check("no light leaks onto the background", (~floor[0]).any() and leak < 1.0e-6, f"max {leak:.2e}")

    try:
        import warp_optix  # noqa: F401
    except ImportError:
        print("[SKIP] backend parity: warp_optix is not installed", flush=True)
    else:
        o = Renderer(model, "optix").render(state)
        worst = max(float(np.abs(a[k] - o[k]).mean()) for k in ("light", "gloss", "tint"))
        differing = float((np.abs(a["light"] - o["light"]).max(-1) > 0.05).mean())
        check("OptiX matches Warp", worst < 0.02 and differing < 0.01, f"mean abs diff {worst:.2e}, pixels differing {differing:.4f}")

    print("PATH_LIGHTING_" + ("OK" if not failures else "FAILED: " + ", ".join(failures)), flush=True)
    return failures


def test_path_lighting():
    import pytest

    if not wp.is_cuda_available():
        pytest.skip("the render kernels are tested on a CUDA device")
    assert run() == []


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
