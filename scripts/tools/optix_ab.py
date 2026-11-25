# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp vs OptiX A/B on a camera task: render the *same* simulation states through both shape-traversal backends of
the UWLab Newton renderer, compare every output and time both.

The env itself runs with whatever backend its cameras are configured for; the comparison calls the cameras' render
context directly, so both images come from one state. Needs warp-optix on the path (see README_optix_renderer.md).
"""

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rounds", type=int, default=3, help="Number of states to compare.")
parser.add_argument("--steps", type=int, default=10, help="Env steps (random actions) before each comparison.")
parser.add_argument("--bench", type=int, default=0, help="Timed renders per backend and camera (0: skip).")
parser.add_argument("--out", type=str, required=True, help="Output directory (PNG strips and results.json).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--expect_fail", action="store_true", help="Negative control: skip the OptiX refit; parity must fail."
)
parser.add_argument("--diagnose", action="store_true", help="Break the differing pixels down by shape and cause.")
parser.add_argument("--dump", action="store_true", help="Save both backends' raw outputs of the first round as .npz.")
parser.add_argument("--bench_variants", action="store_true", help="Also time reduced render configs (cost breakdown).")
parser.add_argument(
    "--optix_max_registers", type=int, nargs="*", default=[], help="Also time OptiX compiled with these register caps."
)
parser.add_argument(
    "--bench_build",
    action="store_true",
    help="Also time OptiX after rebuilding the refit instance structure (refit quality check).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import os
import torch

import isaaclab_tasks  # noqa: F401
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton._src.sensors.warp_raytrace.types import ClearData
from PIL import Image

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

NO_HIT = 0xFFFFFFFF
STATUS = {"failures": None}
OUTPUTS = dict(
    color_image=wp.uint32,
    hdr_color_image=wp.vec3f,
    depth_image=wp.float32,
    shape_index_image=wp.uint32,
    normal_image=wp.vec3f,
    albedo_image=wp.uint32,
)


PATH_OUTPUTS = dict(light_image=wp.vec3f, gloss_image=wp.vec3f, tint_image=wp.vec3f, cover_image=wp.float32)


def _buffers(model, rd, names):
    """Output buffers; a camera with path lighting also gets the lighting buffers its render kernel writes."""
    shape = (model.world_count, 1, rd.height, rd.width)
    types = {name: OUTPUTS[name] for name in names}
    if hasattr(rd, "uw_path"):
        types.update(PATH_OUTPUTS)
    return {name: wp.zeros(shape, dtype=dtype, device=model.device) for name, dtype in types.items()}


def _render(renderer, rd, backend, out, refit=True, config=None):
    """One render of the camera ``rd`` through ``backend`` into ``out``, with the camera's own config."""
    ctx, model, state = renderer._context, NewtonManager.get_model(), NewtonManager.get_state_0()
    config = config or renderer.newton_sensor.default_render_config
    previous, ctx.backend = ctx.backend, backend
    try:
        if renderer._patches:
            ctx.camera_intrinsics, ctx.previous_camera_transforms = rd._intrinsics, rd._previous_transforms
        if backend == "warp":
            model.bvh_refit_shapes(state)
            ctx.render(
                model,
                state,
                camera_transforms=rd.camera_transforms,
                camera_rays=rd.camera_rays,
                clear_data=ClearData(clear_color=rd.clear_color),
                config=config,
                **out,
            )
        else:
            if not refit:  # negative control: trace the acceleration structure of an earlier state
                shapes = ctx.optix_shapes(model, state, refit=False)
                ctx.optix_shapes = lambda model, state, refit=True: shapes
            try:
                ctx.render(
                    model,
                    state,
                    camera_transforms=rd.camera_transforms,
                    camera_rays=rd.camera_rays,
                    clear_data=ClearData(clear_color=rd.clear_color),
                    config=config,
                    **out,
                )
            finally:
                ctx.__dict__.pop("optix_shapes", None)
    finally:
        ctx.backend = previous


def _tone_map(renderer, rd, out):
    """The renderer's post-render steps (path-lighting filter, then tone mapping with the same exposure / response /
    noise frame for both backends)."""
    if not renderer._patches or renderer._tone_mapper is None:
        return
    if "light_image" in out:
        path = {k: out[f"{k}_image"] for k in ("light", "gloss", "tint", "cover")}
        path.update({f"{k}_scratch": wp.zeros_like(out[f"{k}_image"]) for k in ("light", "gloss")})
        path["guides"] = (out["normal_image"], out["depth_image"])
        renderer.resolve_path_lighting(path, out["hdr_color_image"])
    inputs = [out["hdr_color_image"], out["color_image"], getattr(rd, "uw_exposure", renderer._exposure)]
    if renderer.cfg.enable_camera_response:
        inputs += [getattr(rd, "uw_white_balance", renderer._white_balance), renderer._camera_response, 12345]
    wp.launch(renderer._tone_mapper, dim=out["color_image"].shape, inputs=inputs, device=renderer._context.device)


def _numpy(out):
    res = {k: v.numpy()[:, 0] for k, v in out.items()}
    for k in ("color_image", "albedo_image"):
        res[k] = res[k].view(np.uint8).reshape(*res[k].shape, 4)[..., :3]
    return res


def _compare(a, b):
    same = a["shape_index_image"] == b["shape_index_image"]
    hit = same & (a["shape_index_image"] != NO_HIT)
    color = np.abs(a["color_image"].astype(np.int32) - b["color_image"].astype(np.int32))
    hdr = np.abs(a["hdr_color_image"] - b["hdr_color_image"])
    return dict(
        shape_mismatch=1.0 - float(same.mean()),
        hit_fraction=float((a["shape_index_image"] != NO_HIT).mean()),
        depth_max_mm=1e3 * float(np.abs(a["depth_image"] - b["depth_image"])[hit].max()) if hit.any() else 0.0,
        normal_max=float(np.abs(a["normal_image"] - b["normal_image"])[hit].max()) if hit.any() else 0.0,
        color_mean=float(color.mean()),
        color_p999=float(np.quantile(color.max(-1), 0.999)),
        color_gt2_fraction=float((color.max(-1) > 2).mean()),
        color_same_shape_gt2_fraction=float((color.max(-1)[same] > 2).mean()),
        hdr_mean=float(hdr.mean()),
        albedo_same_shape_gt2_fraction=float(
            (np.abs(a["albedo_image"].astype(np.int32) - b["albedo_image"].astype(np.int32)).max(-1)[same] > 2).mean()
        ),
        image_mean=float(a["color_image"].mean()),
        image_std=float(a["color_image"].std()),
        **(
            dict(light_mean_diff=float(np.abs(a["light_image"] - b["light_image"]).mean()), light_mean=float(a["light_image"].mean()))
            if "light_image" in a
            else {}
        ),
    )


def _diagnose(model, a, b, origins):
    """Which shapes the differing pixels are on, and whether the two backends hit the same surface there."""
    labels = model.shape_label
    bad_all = np.abs(a["color_image"].astype(np.int32) - b["color_image"].astype(np.int32)).max(-1) > 2
    per_world = bad_all.reshape(len(bad_all), -1).mean(1)
    distance = np.linalg.norm(origins, axis=1)
    order = np.argsort(distance)
    quarter = max(len(order) // 4, 1)
    dd = np.abs(a["depth_image"] - b["depth_image"]).reshape(len(bad_all), -1)
    print(
        f"[optix_ab]   differing-pixel fraction, nearest quarter of envs (<= {distance[order[quarter - 1]]:.1f} m from"
        f" the origin): {per_world[order[:quarter]].mean():.4f}, farthest quarter (>= {distance[order[-quarter]]:.1f}"
        f" m): {per_world[order[-quarter:]].mean():.4f}; median |depth diff| [um] nearest"
        f" {1e6 * np.median(dd[order[:quarter]]):.1f} farthest {1e6 * np.median(dd[order[-quarter:]]):.1f}",
        flush=True,
    )
    si = a["shape_index_image"]
    same = (si == b["shape_index_image"]) & (si != NO_HIT)
    bad = same & (np.abs(a["color_image"].astype(np.int32) - b["color_image"].astype(np.int32)).max(-1) > 2)
    dot = (a["normal_image"] * b["normal_image"]).sum(-1)
    depth = np.abs(a["depth_image"] - b["depth_image"])
    print(
        f"[optix_ab]   differing same-shape pixels: {int(bad.sum())} of {int(same.sum())}; of those normal flipped "
        f"(dot<0): {float((dot[bad] < 0).mean()):.3f}, same depth (<0.1 mm): {float((depth[bad] < 1e-4).mean()):.3f}, "
        f"same normal (dot>0.999): {float((dot[bad] > 0.999).mean()):.3f}",
        flush=True,
    )
    ids, counts = np.unique(si[bad], return_counts=True)
    totals = {int(i): int((si[same] == i).sum()) for i in ids}
    by_name = {}
    for i, c in zip(ids, counts):
        name = "/".join(str(labels[int(i)]).split("/")[4:])  # drop /World/envs/env_N
        entry = by_name.setdefault(name, [0, 0])
        entry[0] += int(c)
        entry[1] += totals[int(i)]
    for name, (c, total) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:14]:
        print(f"[optix_ab]     {c:6d} / {total:6d} px  {name}", flush=True)


def _parity_ok(m):
    # Silhouette pixels may flip between two float32 traversals (and their shadows, or a texel, with them); the
    # rest must agree. Fractions, not maxima: a maximum over millions of pixels only measures the rarest flip.
    if "light_mean" in m:
        # Path lighting is a Monte Carlo estimate whose filter spreads every flipped secondary ray over its
        # footprint: the images must agree statistically, not pixel by pixel.
        return (
            m["shape_mismatch"] < 3.0e-3
            and m["color_mean"] < 1.0
            and m["light_mean_diff"] < 0.02 * m["light_mean"]
            and m["albedo_same_shape_gt2_fraction"] < 1.0e-3
            and m["image_std"] > 5.0
        )
    return (
        m["shape_mismatch"] < 3.0e-3
        and m["color_mean"] < 0.3
        and m["color_same_shape_gt2_fraction"] < 3.0e-3
        and m["albedo_same_shape_gt2_fraction"] < 1.0e-3
        and m["image_std"] > 5.0
    )


def _time(fn, count):
    device = NewtonManager.get_model().device
    for _ in range(3):
        fn()
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(count):
        fn()
    wp.synchronize_device(device)
    return (time.perf_counter() - start) / count


def _bench(renderer, rd, backend, names, count, config=None):
    """Seconds per frame of one camera: acceleration refit + render, as the env does every step."""
    out = _buffers(NewtonManager.get_model(), rd, names)
    return _time(lambda: _render(renderer, rd, backend, out, config=config), count)


VARIANTS = {
    "no_shadows": dict(enable_shadows=False),
    "no_shadows_no_textures": dict(enable_shadows=False, enable_textures=False),
    "lambert": dict(enable_pbr=False, enable_precomputed_ibl=False, enable_reflections=False),
    "lambert_no_shadows_no_textures": dict(
        enable_pbr=False,
        enable_precomputed_ibl=False,
        enable_reflections=False,
        enable_shadows=False,
        enable_textures=False,
    ),
}


def _bench_refit(backend, count):
    """Seconds per acceleration refit alone."""
    from uwlab.render import optix as optix_backend

    model, state = NewtonManager.get_model(), NewtonManager.get_state_0()
    if backend == "warp":
        return _time(lambda: model.bvh_refit_shapes(state), count)
    return _time(lambda: optix_backend.refit_shapes(model, state, model._uwlab_optix_shapes), count)


@hydra_task_config(args_cli.task, "flashsac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    os.makedirs(args_cli.out, exist_ok=True)
    cameras = {name: s for name, s in env.scene.sensors.items() if "rgb" in getattr(s.data, "output", {})}
    model = NewtonManager.get_model()
    print(
        f"[optix_ab] warp {wp.__version__} | {model.world_count} worlds, {model.shape_count} shapes | cameras "
        f"{dict((n, (s._render_data.height, s._render_data.width)) for n, s in cameras.items())}",
        flush=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(args_cli.seed)
    results, failures = [], []
    for round_index in range(args_cli.rounds):
        for _ in range(args_cli.steps):
            action = 2.0 * torch.rand(env.num_envs, env.action_manager.total_action_dim, generator=generator) - 1.0
            env.step(action.to(env.device))
        for name, sensor in cameras.items():
            renderer, rd = sensor._renderer, sensor._render_data
            warp_out, optix_out = _buffers(model, rd, OUTPUTS), _buffers(model, rd, OUTPUTS)
            _render(renderer, rd, "warp", warp_out)
            stale = args_cli.expect_fail and round_index > 0
            _render(renderer, rd, "optix", optix_out, refit=not stale)
            _tone_map(renderer, rd, warp_out)
            _tone_map(renderer, rd, optix_out)
            wp.synchronize_device(model.device)
            a, b = _numpy(warp_out), _numpy(optix_out)
            metrics = _compare(a, b)
            ok = _parity_ok(metrics)
            if stale:
                ok = not ok  # the negative control passes only when parity breaks
            results.append(dict(round=round_index, camera=name, ok=ok, stale=stale, **metrics))
            if not ok:
                failures.append(f"{name}@{round_index}")
            print(
                f"[optix_ab] round {round_index} {name} {'stale ' if stale else ''}{'PASS' if ok else 'FAIL'} "
                + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()),
                flush=True,
            )
            if args_cli.diagnose:
                _diagnose(model, a, b, env.scene.env_origins.cpu().numpy())
            if args_cli.dump and round_index == 0:
                np.savez_compressed(
                    f"{args_cli.out}/{name}_dump.npz",
                    **{f"warp_{k}": v for k, v in a.items()},
                    **{f"optix_{k}": v for k, v in b.items()},
                    camera_transforms=rd.camera_transforms.numpy(),
                    labels=np.array([str(x) for x in model.shape_label]),
                    shape_world=model.shape_world.numpy(),
                    shape_scale=model.shape_scale.numpy(),
                    shape_type=model.shape_type.numpy(),
                    env_origins=env.scene.env_origins.cpu().numpy(),
                    shape_transforms=model.bvh_shape_world_transforms.numpy(),
                    optix_shape_transforms=model._uwlab_optix_shapes.shape_world_transforms.numpy(),
                    intrinsics=rd._intrinsics.numpy() if getattr(rd, "_intrinsics", None) is not None else np.zeros(0),
                )
                # geometry of the shapes whose depth differs most, for an offline float64 ground truth
                same = (a["shape_index_image"] == b["shape_index_image"]) & (a["shape_index_image"] != NO_HIT)
                depth_diff = np.where(same, np.abs(a["depth_image"] - b["depth_image"]), 0.0)
                worst = {}
                for shape in np.unique(a["shape_index_image"][0][same[0]]):
                    worst[int(shape)] = float(np.median(depth_diff[0][a["shape_index_image"][0] == shape]))
                meshes = {}
                for shape in sorted(worst, key=worst.get, reverse=True)[:3]:
                    source = model.shape_source[shape]
                    if hasattr(source, "vertices"):
                        meshes[f"mesh{shape}_vertices"] = np.asarray(source.vertices)
                        meshes[f"mesh{shape}_indices"] = np.asarray(source.indices)
                np.savez_compressed(f"{args_cli.out}/{name}_meshes.npz", **meshes)
            count = min(4, model.world_count)
            diff = np.clip(10 * np.abs(a["color_image"].astype(np.int32) - b["color_image"].astype(np.int32)), 0, 255)
            rows = [
                np.concatenate(list(x[:count]), axis=1)
                for x in (a["color_image"], b["color_image"], diff.astype(np.uint8))
            ]
            Image.fromarray(np.concatenate(rows, axis=0)).save(f"{args_cli.out}/{name}_round{round_index}.png")
    shapes = getattr(model, "_uwlab_optix_shapes", None)
    summary = dict(
        task=args_cli.task,
        num_envs=env.num_envs,
        warp=wp.__version__,
        gpu=wp.get_device(model.device).name,
        shape_count=model.shape_count,
        comparisons=results,
        failures=failures,
    )
    if shapes is not None:
        summary["optix"] = dict(
            instances=int(shapes.instance_shapes.shape[0]),
            global_instances=int(shapes.global_instance_shapes.shape[0]),
            geometries=len(shapes.geometry),
            extent=shapes.extent,
            grid=shapes.grid,
            out_of_cell=int(shapes.out_of_cell.numpy()[0]),
        )
        summary["optix"]["global_shapes"] = [
            str(model.shape_label[int(i)]) for i in shapes.global_instance_shapes.numpy()
        ]
        print(f"[optix_ab] optix {summary['optix']}", flush=True)
    if args_cli.bench:
        refit = {backend: _bench_refit(backend, args_cli.bench) for backend in ("warp", "optix")}
        summary["bench"] = dict(warp_refit_ms=1e3 * refit["warp"], optix_refit_ms=1e3 * refit["optix"])
        print(
            f"[optix_ab] bench refit only x {env.num_envs} envs: warp BVH {1e3 * refit['warp']:.2f} ms, "
            f"optix {1e3 * refit['optix']:.2f} ms",
            flush=True,
        )
        production = ("color_image", "hdr_color_image")
        for name, sensor in cameras.items():
            renderer, rd = sensor._renderer, sensor._render_data
            names = production if renderer._patches and renderer._tone_mapper is not None else ("color_image",)
            timing = {backend: _bench(renderer, rd, backend, names, args_cli.bench) for backend in ("warp", "optix")}
            # depth only: one primary ray per pixel and no shading, i.e. traversal (plus the refit)
            depth = {
                backend: _bench(renderer, rd, backend, ("depth_image",), args_cli.bench)
                for backend in ("warp", "optix")
            }
            summary["bench"][name] = dict(
                resolution=[rd.height, rd.width],
                warp_ms=1e3 * timing["warp"],
                optix_ms=1e3 * timing["optix"],
                warp_env_fps=env.num_envs / timing["warp"],
                optix_env_fps=env.num_envs / timing["optix"],
                speedup=timing["warp"] / timing["optix"],
                warp_depth_only_ms=1e3 * depth["warp"],
                optix_depth_only_ms=1e3 * depth["optix"],
            )
            print(
                f"[optix_ab] bench {name} {rd.height}x{rd.width} x {env.num_envs} envs: warp "
                f"{1e3 * timing['warp']:.2f} ms, optix {1e3 * timing['optix']:.2f} ms "
                f"({timing['warp'] / timing['optix']:.2f}x) | depth only: warp {1e3 * depth['warp']:.2f} ms, optix "
                f"{1e3 * depth['optix']:.2f} ms ({depth['warp'] / depth['optix']:.2f}x)",
                flush=True,
            )
    if args_cli.bench and (args_cli.bench_variants or args_cli.optix_max_registers):
        import dataclasses

        from uwlab.render import optix as optix_backend

        name, sensor = next(iter(cameras.items()))
        renderer, rd = sensor._renderer, sensor._render_data
        base = renderer.newton_sensor.default_render_config
        names = (
            ("color_image", "hdr_color_image")
            if renderer._patches and renderer._tone_mapper is not None
            else ("color_image",)
        )
        summary["bench_variants"] = {}
        for label, overrides in (VARIANTS.items() if args_cli.bench_variants else ()):
            config = dataclasses.replace(base, **overrides)
            timing = {
                backend: _bench(renderer, rd, backend, names, args_cli.bench, config) for backend in ("warp", "optix")
            }
            summary["bench_variants"][label] = {k: 1e3 * v for k, v in timing.items()}
            print(
                f"[optix_ab] variant {name} {label}: warp {1e3 * timing['warp']:.2f} ms, optix "
                f"{1e3 * timing['optix']:.2f} ms ({timing['warp'] / timing['optix']:.2f}x)",
                flush=True,
            )
        default_cap = optix_backend._MAX_REGISTERS
        for cap in args_cli.optix_max_registers:
            optix_backend._MAX_REGISTERS = cap
            shapes.pipelines.clear()  # pipelines are compiled with the cap
            elapsed = _bench(renderer, rd, "optix", names, args_cli.bench)
            summary["bench_variants"][f"optix_max_registers_{cap}"] = 1e3 * elapsed
            print(f"[optix_ab] optix max_registers={cap} {name}: {1e3 * elapsed:.2f} ms", flush=True)
        optix_backend._MAX_REGISTERS = default_cap
        shapes.pipelines.clear()
    if args_cli.bench and args_cli.bench_build:
        from uwlab.render import optix as optix_backend

        name, sensor = next(iter(cameras.items()))
        renderer, rd = sensor._renderer, sensor._render_data
        names = (
            ("color_image", "hdr_color_image")
            if renderer._patches and renderer._tone_mapper is not None
            else ("color_image",)
        )
        state = NewtonManager.get_state_0()
        result = {}
        result["refit_only_ms"] = 1e3 * _bench(renderer, rd, "optix", names, args_cli.bench)
        start = time.perf_counter()
        optix_backend.refit_shapes(model, state, model._uwlab_optix_shapes, rebuild=True)
        wp.synchronize_device(model.device)
        result["rebuild_cost_ms"] = 1e3 * (time.perf_counter() - start)
        result["after_rebuild_ms"] = 1e3 * _bench(renderer, rd, "optix", names, args_cli.bench)
        summary["bench_build"] = result
        print(f"[optix_ab] build {name}: " + " ".join(f"{k}={v:.2f}" for k, v in result.items()), flush=True)
    with open(f"{args_cli.out}/results.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("[optix_ab] " + ("OPTIX_AB_OK" if not failures else "OPTIX_AB_FAILED " + ",".join(failures)), flush=True)
    STATUS["failures"] = failures
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0 if STATUS["failures"] == [] else 1)
