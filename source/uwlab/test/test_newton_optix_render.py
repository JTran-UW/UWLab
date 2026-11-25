# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OptiX vs Warp parity for the UWLab Newton renderer on a small multi-world scene (no Isaac Sim needed).

    PYTHONPATH=<optix deps>:source/uwlab python source/uwlab/test/test_newton_optix_render.py [--out DIR]

Renders the same states with ``backend="warp"`` and ``backend="optix"`` and compares every output. The
negative controls must fail: an acceleration that was not refit after the bodies moved, and a world
whose cell offset is wrong.
"""

import argparse
import numpy as np
import sys
import time

import newton
import warp as wp
from newton._src.sensors.warp_raytrace.types import ClearData

from uwlab.render.context import RenderContext
from uwlab.render.types import RenderConfig

WIDTH, HEIGHT, FOCAL = 96, 72, 90.0
NO_HIT = 0xFFFFFFFF


def _icosphere(subdivisions=2):
    t = (1.0 + 5.0**0.5) / 2.0
    v = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    f = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    v = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(subdivisions):
        cache, out = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = out
    return np.array(v, dtype=np.float32), np.array(f, dtype=np.int32).reshape(-1)


def _cube_mesh():
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)
    f = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    return v, np.array(f, dtype=np.int32)


def build_model(world_count=5, spacing=37.0, seed=0, cones=True):
    """Worlds far apart (as Isaac Lab lays them out), every shape type, a global plane and a global mesh.

    ``cones=False`` leaves the cones out: Newton 1.5.2's Warp shadow test counts the small negative distance
    its cone routine returns for a ray leaving the surface as a hit, so lit cones shadow themselves there.
    OptiX rejects it, so scenes with cones are only compared without shadows.
    """
    rng = np.random.default_rng(seed)
    sphere_v, sphere_f = _icosphere()
    smooth = newton.Mesh(sphere_v, sphere_f, normals=sphere_v.copy())
    flat = newton.Mesh(*_cube_mesh())
    cube_v, cube_f = _cube_mesh()
    # q = (0, 0, -0.707, 0.707) takes local +x to world -y, so the 3 m (scaled) offset is undone by the shape pose
    offset = newton.Mesh(
        cube_v * np.array([1.0, 0.07, 1.0], dtype=np.float32) + np.array([10.0, 0.0, 0.0], dtype=np.float32), cube_f
    )
    main = newton.ModelBuilder(up_axis=newton.Axis.Z)
    main.add_ground_plane(color=(0.55, 0.5, 0.45))
    # global (world -1) mesh next to world 1
    main.add_shape_mesh(
        -1,
        xform=wp.transform((spacing + 0.9, 0.9, 0.3), wp.quat_identity()),
        mesh=flat,
        scale=(0.15, 0.15, 0.3),
        color=(0.9, 0.2, 0.2),
    )
    for w in range(world_count):
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)

        def body(x, y, z):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            p = np.array([x, y, z]) + rng.uniform(-0.05, 0.05, 3)
            return b.add_body(xform=wp.transform(tuple(p), wp.quat(*q)))

        def color():
            return tuple(rng.uniform(0.2, 1.0, 3))

        b.add_shape_box(
            -1, xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), hx=0.8, hy=0.6, hz=0.05, color=color()
        )
        # Authored with a 3-digit quaternion (|q| = 0.99985) and vertices metres from the mesh origin, like
        # the OmniReset table: the Warp traversal renders it scaled by 1/|q|^2 and OptiX must do the same.
        b.add_shape_mesh(
            -1,
            xform=wp.transform((-0.3, 2.55, 0.2), wp.quat(0.0, 0.0, -0.707, 0.707)),
            mesh=offset,
            scale=(0.3, 0.3, 0.1),
            color=color(),
            label="unnormalised",
        )
        b.add_shape_sphere(body(-0.5, -0.3, 0.35), radius=0.15, color=color())
        b.add_shape_capsule(body(-0.2, 0.25, 0.4), radius=0.08, half_height=0.18, color=color())
        b.add_shape_cylinder(body(0.15, -0.3, 0.4), radius=0.1, half_height=0.15, color=color())
        cone = (body(0.45, 0.3, 0.4), color())
        if cones:
            b.add_shape_cone(cone[0], radius=0.12, half_height=0.16, color=cone[1])
        b.add_shape_ellipsoid(body(0.5, -0.25, 0.45), rx=0.16, ry=0.09, rz=0.12, color=color())
        b.add_shape_box(body(-0.55, 0.3, 0.35), hx=0.1, hy=0.07, hz=0.12, color=color())
        b.add_shape_mesh(body(0.0, 0.0, 0.45), mesh=smooth, scale=(0.14, 0.14, 0.14), color=color())
        b.add_shape_mesh(body(0.2, 0.1, 0.75), mesh=flat, scale=(0.12, 0.05, 0.08), color=color())
        # Newton 1.5.2's render kernel never hits convex hulls; OptiX must leave them out too
        b.add_shape_convex_hull(body(-0.1, -0.45, 0.6), mesh=flat, scale=(0.1, 0.1, 0.1), color=color(), label="hull")
        if w == 0:
            # only in world 0: must never show up in another world's image
            b.add_shape_mesh(
                -1,
                xform=wp.transform((0.0, 0.0, 1.1), wp.quat_identity()),
                mesh=smooth,
                scale=(0.25, 0.25, 0.1),
                color=(1.0, 0.0, 1.0),
            )
        row, col = divmod(w, 3)
        main.add_world(b, xform=wp.transform((col * spacing, row * spacing, 0.0), wp.quat_identity()))
    model = main.finalize(device="cuda:0")
    origins = np.array([[(w % 3) * spacing, (w // 3) * spacing, 0.0] for w in range(world_count)], dtype=np.float32)
    return model, origins


def look_at(eye, target):
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    q = wp.quat_from_matrix(wp.mat33(*np.stack([right, up, -forward], axis=1).reshape(-1)))
    return [*eye, q[0], q[1], q[2], q[3]]


def cameras(origins, device):
    transforms = [look_at(o + np.array([1.5, -1.3, 1.2]), o + np.array([0.0, 0.0, 0.4])) for o in origins]
    transforms = wp.array(np.array(transforms, dtype=np.float32)[None], dtype=wp.transformf, device=device)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    d = np.stack([(x + 0.5 - WIDTH / 2) / FOCAL, (HEIGHT / 2 - y - 0.5) / FOCAL, -np.ones_like(x)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    rays = np.zeros((1, HEIGHT, WIDTH, 2, 3), dtype=np.float32)
    rays[0, :, :, 1] = d
    return transforms, wp.array(rays, dtype=wp.vec3f, device=device)


class Renderer:
    def __init__(self, model, backend, **config):
        params = dict(enable_shadows=True, enable_textures=False, enable_backface_culling=False, max_distance=50.0)
        params.update(config)
        self.model = model
        self.config = RenderConfig(**params)
        self.context = RenderContext(
            world_count=model.world_count, device=model.device, config=self.config, backend=backend
        )
        self.context.init_from_model(model, load_textures=False)
        shape = (model.world_count, 1, HEIGHT, WIDTH)
        self.out = dict(
            color_image=wp.zeros(shape, dtype=wp.uint32, device=model.device),
            depth_image=wp.zeros(shape, dtype=wp.float32, device=model.device),
            shape_index_image=wp.zeros(shape, dtype=wp.uint32, device=model.device),
            normal_image=wp.zeros(shape, dtype=wp.vec3f, device=model.device),
            albedo_image=wp.zeros(shape, dtype=wp.uint32, device=model.device),
        )

    def render(self, state, transforms, rays):
        if self.context.backend == "warp":
            self.model.bvh_refit_shapes(state)
        self.context.render(
            self.model,
            state,
            camera_transforms=transforms,
            camera_rays=rays,
            clear_data=ClearData(clear_color=0xFF664422),
            **self.out,
        )
        wp.synchronize_device(self.model.device)
        out = {k: v.numpy() for k, v in self.out.items()}
        for k in ("color_image", "albedo_image"):
            out[k] = out[k].view(np.uint8).reshape(*out[k].shape, 4)[..., :3]
        return out


def compare(a, b):
    """Parity metrics of two output sets."""
    same = a["shape_index_image"] == b["shape_index_image"]
    hit = same & (a["shape_index_image"] != NO_HIT)
    color = np.abs(a["color_image"].astype(np.int32) - b["color_image"].astype(np.int32))
    return dict(
        shape_mismatch=1.0 - float(same.mean()),
        depth_max=float(np.abs(a["depth_image"] - b["depth_image"])[hit].max()) if hit.any() else 0.0,
        normal_max=float(np.abs(a["normal_image"] - b["normal_image"])[hit].max()) if hit.any() else 0.0,
        color_mean=float(color.mean()),
        color_same_shape_gt3=float((color.max(-1)[same] > 3).mean()),
        albedo_same_shape_max=int(
            np.abs(a["albedo_image"].astype(np.int32) - b["albedo_image"].astype(np.int32))[same].max()
        ),
        hit_fraction=float((a["shape_index_image"] != NO_HIT).mean()),
    )


def parity_ok(m):
    # Silhouette pixels may flip between two float32 traversals, and a shadow edge with them; everything
    # else must agree closely.
    return (
        m["shape_mismatch"] < 2.0e-3
        and m["depth_max"] < 2.0e-3
        and m["normal_max"] < 2.0e-2
        and m["color_mean"] < 0.25
        and m["color_same_shape_gt3"] < 3.0e-4
        and m["albedo_same_shape_max"] <= 1
    )


def move_bodies(model, state, seed):
    rng = np.random.default_rng(seed)
    q = state.body_q.numpy().copy()
    q[:, :3] += rng.uniform(-0.12, 0.12, (len(q), 3)).astype(np.float32)
    rot = rng.normal(size=(len(q), 4)).astype(np.float32)
    q[:, 3:] = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    state.body_q.assign(q)


def save(out_dir, name, images):
    from PIL import Image

    tiles = [np.concatenate(list(images["color_image"][:, 0]), axis=1)]
    depth = images["depth_image"][:, 0]
    depth = (255 * np.clip(depth / max(float(depth.max()), 1e-6), 0, 1)).astype(np.uint8)
    tiles.append(np.repeat(np.concatenate(list(depth), axis=1)[..., None], 3, axis=2))
    normal = (255 * (0.5 * images["normal_image"][:, 0] + 0.5)).clip(0, 255).astype(np.uint8)
    tiles.append(np.concatenate(list(normal), axis=1))
    Image.fromarray(np.concatenate(tiles, axis=0)).save(f"{out_dir}/{name}.png")


def run(out_dir=""):
    """Run every check, print one line each, and return the names of the failed ones."""
    wp.config.quiet = True
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
        if not ok:
            failures.append(name)

    def fmt(m):
        return " ".join(f"{k}={v:.3g}" for k, v in m.items())

    model, origins = build_model(cones=False)
    state = model.state()
    transforms, rays = cameras(origins, model.device)
    print(f"warp {wp.__version__} newton {newton.__version__} | {model.world_count} worlds, {model.shape_count} shapes")

    for label, config in (
        ("shadows", {}),
        ("no_shadows", dict(enable_shadows=False)),
        ("culling", dict(enable_backface_culling=True)),
        ("pixel_order", dict(render_order=0)),
        ("no_global_world", dict(enable_global_world=False)),
    ):
        warp_r, optix_r = Renderer(model, "warp", **config), Renderer(model, "optix", **config)
        t0 = time.time()
        a = warp_r.render(state, transforms, rays)
        t1 = time.time()
        b = optix_r.render(state, transforms, rays)
        m = compare(a, b)
        check(
            f"parity[{label}]",
            parity_ok(m) and m["hit_fraction"] > 0.1,
            fmt(m) + f" (first render {t1 - t0:.1f}s / {time.time() - t1:.1f}s)",
        )
        if out_dir and label == "shadows":
            save(out_dir, "warp", a)
            save(out_dir, "optix", b)

    # every primitive type, cones included (no shadows: see build_model)
    cone_model, cone_origins = build_model(cones=True)
    cone_cameras = cameras(cone_origins, cone_model.device)
    m = compare(*(
        Renderer(cone_model, backend, enable_shadows=False).render(cone_model.state(), *cone_cameras)
        for backend in ("warp", "optix")
    ))
    check("parity[all primitives, no shadows]", parity_ok(m), fmt(m))

    # a shape with an unnormalised quaternion sits where the Warp backend puts it (0.03 % scale of a 3 m offset)
    slab = [i for i, label in enumerate(model.shape_label) if "unnormalised" in str(label)]
    a, b = (Renderer(model, backend).render(state, transforms, rays) for backend in ("warp", "optix"))
    on_slab = np.isin(a["shape_index_image"], slab) & (a["shape_index_image"] == b["shape_index_image"])
    slab_depth = float(np.abs(a["depth_image"] - b["depth_image"])[on_slab].max()) if on_slab.any() else float("nan")
    check(
        "unnormalised quaternion matches the Warp convention",
        on_slab.sum() > 500 and slab_depth < 2.0e-4,
        f"pixels={int(on_slab.sum())} depth_max={slab_depth:.3g}",
    )

    hulls = [i for i, label in enumerate(model.shape_label) if "hull" in str(label)]
    check(
        "convex hulls are drawn by neither backend",
        len(hulls) == model.world_count
        and not np.isin(a["shape_index_image"], hulls).any()
        and not np.isin(b["shape_index_image"], hulls).any(),
    )

    # the unpatched renderer path: stock Newton sensor (Warp) vs the UWLab kernel with the stock look (OptiX)
    stock = newton.sensors.SensorTiledCamera(
        model,
        default_render_config=newton.sensors.SensorTiledCamera.RenderConfig(
            enable_shadows=True, enable_textures=False, enable_backface_culling=False, max_distance=50.0
        ),
        load_textures=False,
    )
    stock.utils.create_default_light(enable_shadows=True)
    stock_look = Renderer(
        model,
        "optix",
        per_world_lights=False,
        per_world_intrinsics=False,
        render_order=stock.default_render_config.render_order,
    )
    model.bvh_refit_shapes(state)
    stock.update(state, transforms, rays, clear_data=ClearData(clear_color=0xFF664422), **stock_look.out)
    wp.synchronize_device(model.device)
    a = {k: v.numpy() for k, v in stock_look.out.items()}
    for k in ("color_image", "albedo_image"):
        a[k] = a[k].view(np.uint8).reshape(*a[k].shape, 4)[..., :3]
    m = compare(a, stock_look.render(state, transforms, rays))
    check("parity[stock Newton sensor vs OptiX stock look]", parity_ok(m), fmt(m))

    warp_r, optix_r = Renderer(model, "warp"), Renderer(model, "optix")
    reference = warp_r.render(state, transforms, rays)
    shape_world = model.shape_world.numpy()
    # world 0's exclusive (magenta) shape must not leak into other worlds
    exclusive = max(s for s in range(model.shape_count) if shape_world[s] == 0)
    seen = optix_r.render(state, transforms, rays)["shape_index_image"]
    check("world isolation", bool((seen[0] == exclusive).any()) and not bool((seen[1:] == exclusive).any()))
    # the global mesh is visible from world 1 only through the global structure
    global_mesh = max(s for s in range(model.shape_count) if shape_world[s] < 0)
    pixels = int((seen[1] == global_mesh).sum())
    check(
        "global mesh visible",
        pixels > 0 and pixels == int((reference["shape_index_image"][1] == global_mesh).sum()),
        f"pixels={pixels}",
    )

    # moving bodies: refit keeps parity over several steps
    worst = None
    for step in range(5):
        move_bodies(model, state, 100 + step)
        m = compare(warp_r.render(state, transforms, rays), optix_r.render(state, transforms, rays))
        if worst is None or m["shape_mismatch"] > worst["shape_mismatch"]:
            worst = m
        if not parity_ok(m):
            break
    check("parity after motion (5 refits)", parity_ok(worst), fmt(worst))

    # NEGATIVE CONTROL 1: without a refit the OptiX image is stale and parity must fail
    move_bodies(model, state, 999)
    a = warp_r.render(state, transforms, rays)
    shapes = optix_r.context.optix_shapes(model, state, refit=False)
    optix_r.context.optix_shapes = lambda model, state, refit=True: shapes  # skip the refit once
    stale = compare(a, optix_r.render(state, transforms, rays))
    del optix_r.context.optix_shapes
    check("negative control: stale acceleration is detected", not parity_ok(stale), fmt(stale))
    fresh = compare(a, optix_r.render(state, transforms, rays))
    check("parity restored by refit", parity_ok(fresh), fmt(fresh))

    # NEGATIVE CONTROL 2: rays offset with a wrong world centre miss that world's geometry and parity must fail
    centers = shapes.world_centers.numpy().copy()
    broken = centers.copy()
    broken[2] += np.array([0.05, 0.0, 0.0], dtype=np.float32)
    shapes.world_centers.assign(broken)
    optix_r.context.optix_shapes = lambda model, state, refit=True: shapes  # keep the instances where they are
    wrong = compare(a, optix_r.render(state, transforms, rays))
    del optix_r.context.optix_shapes
    shapes.world_centers.assign(centers)
    check("negative control: wrong ray offset is detected", not parity_ok(wrong), fmt(wrong))

    # robustness: a non-finite body and a body far outside its cell are hidden, nothing else changes
    q = state.body_q.numpy().copy()
    backup = q.copy()
    q[0, :3] = np.nan
    q[9, :3] += 500.0
    state.body_q.assign(q)
    shapes.out_of_cell.zero_()
    hidden = optix_r.render(state, transforms, rays)
    flagged = int(shapes.out_of_cell.numpy()[0])
    state.body_q.assign(backup)
    restored = compare(a, optix_r.render(state, transforms, rays))
    check(
        "non-finite / out-of-cell bodies are hidden and flagged",
        flagged == 1 and parity_ok(restored),
        f"flag={flagged} " + fmt(restored),
    )
    worlds_untouched = compare({k: v[2:] for k, v in a.items()}, {k: v[2:] for k, v in hidden.items()})
    check("other worlds unaffected by a blown-up body", parity_ok(worlds_untouched), fmt(worlds_untouched))

    print("OPTIX_PARITY_" + ("OK" if not failures else "FAILED: " + ", ".join(failures)), flush=True)
    return failures


def test_optix_matches_warp():
    import pytest

    pytest.importorskip("warp_optix")
    if not wp.is_cuda_available():
        pytest.skip("the OptiX backend needs a CUDA device")
    assert run() == []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="", help="directory for image dumps")
    sys.exit(1 if run(parser.parse_args().out) else 0)
