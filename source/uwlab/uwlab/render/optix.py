# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) 2026 The UW Lab Project Developers
# SPDX-License-Identifier: Apache-2.0
"""OptiX shape traversal for the UWLab Newton renderer.

Port of Newton's experimental OptiX camera backend (newton-physics/newton #5285 and #5334:
``geometry/optix.py``, ``render/render_optix.py``, ``Model._optix_build_shapes``) to the pinned
Newton 1.5.2, whose ``Model`` has no OptiX support. Only the shape traversal changes; the render
kernel, shading, textures and outputs stay those of ``kernel.py``.

Changes from upstream: each world is packed about its own centre (Isaac Lab spreads the worlds out,
which would cost float32 precision), global shapes get their own structure that rays skip when they
cannot reach it, non-finite transforms are hidden, and instances follow the Warp traversal's
treatment of unnormalised quaternions.

Needs ``warp-optix`` (Warp >= 1.17) and an OptiX-capable driver; import this module lazily.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

import warp as wp
import warp_optix as wo
from newton import ShapeFlags
from newton._src.geometry import GeoType, raycast
from newton._src.geometry.bvh import (
    compute_enabled_shapes,
    compute_shape_bvh_bounds,
    compute_shape_local_bounds,
    compute_shape_world_transforms,
)
from newton._src.sensors.warp_raytrace import raytrace

from .types import SceneData

# Raygen register cap (RTX hardware only; ignored without RT cores). Upstream uses 80 for its kernel; this
# kernel's shadow path then spills heavily: 38 ms/frame at 80, 12.5 ms at the OptiX default, 8.9 ms at 255
# (1024 envs, 64x64, RTX 4090).
_MAX_REGISTERS = 255
# Distance between cell centres in units of the cell half-width. Rays are clipped to their own cell and
# shapes outside it are hidden, so any value above 2 keeps the worlds apart; smaller keeps coordinates small.
_CELL_SPACING = 2.25
_SBT_TRIANGLES = 0
_SBT_PRIMITIVES = 1


class UnsupportedError(RuntimeError):
    """OptiX cannot trace this model on its device."""


@dataclass
class OptixShapes:
    """OptiX instance acceleration over a model's shapes, with one cell per world.

    Each world's shapes are moved to a cube of half-width :attr:`extent` about that world's centre,
    the cubes are spaced ``2.25 * extent`` apart on a grid, and rays are clipped to their world's
    cube. Global (world -1) shapes keep their coordinates in a second structure. Planes are
    intersected analytically.
    """

    id: int
    """Traversable handle of :attr:`accel`, or 0 without per-world instances."""
    accel: wo.Accel | None
    global_id: int
    """Traversable handle of :attr:`global_accel`, or 0 without global instances."""
    global_accel: wo.Accel | None
    geometry: list
    """Triangle and primitive acceleration structures shared by the instances."""
    shape_types: wp.array
    """Render shape types (heightfields as meshes), shape ``[shape_count]``."""
    shape_world_transforms: wp.array
    """World-space shape transforms, shape ``[shape_count]``."""
    shape_box_centers: wp.array
    shape_box_halves: wp.array
    """Box of each shape in the frame of its acceleration structure (mesh bounds; the unit cube for
    primitives), shape ``[shape_count]``."""
    shape_traversables: wp.array
    """Acceleration structure instanced by each shape, shape ``[shape_count]``."""
    shape_cells: wp.array
    """Cell of each shape, or -1 for global shapes, shape ``[shape_count]``."""
    instance_shapes: wp.array
    instances: wp.array
    global_instance_shapes: wp.array
    global_instances: wp.array
    planes: tuple
    """Global planes."""
    world_planes: wp.array
    """Planes of each world, padded with -1, shape ``[world_count, max planes per world]``."""
    world_planes_flat: wp.array
    world_centers: wp.array
    """Centre each world is packed about [m], shape ``[world_count]``."""
    grid: int
    extent: float
    """Half-width of each world's cell [m]."""
    world_count: int
    out_of_cell: wp.array
    """Set to 1 on the device when a refit hides a shape that left its cell or is not finite, shape ``[1]``."""
    global_bounds: tuple | None = None
    """World-space ``(lower, upper)`` box of the global instances when none of them can move; rays that
    miss it skip the global structure."""
    pipelines: dict = field(default_factory=dict)
    refit_count: int = 0

    @property
    def has_instances(self) -> bool:
        return self.instance_shapes.shape[0] > 0

    @property
    def has_global_instances(self) -> bool:
        return self.global_instance_shapes.shape[0] > 0


@wp.func
def world_offset(world: int, grid: int, extent: float) -> wp.vec3:
    center = float(grid - 1) * 0.5
    cell = wp.vec3(float(world % grid), float((world // grid) % grid), float(world // (grid * grid)))
    return (cell - wp.vec3(center)) * (wp.static(_CELL_SPACING) * extent)


@wp.func
def ray_interval(origin: wp.vec3, direction: wp.vec3, extent: float, max_distance: float) -> wp.vec2:
    """Clip traversal to one cell, including rays whose origins lie outside it."""
    near = float(0.0)
    far = max_distance
    for axis in range(3):
        if direction[axis] != 0.0:
            a = (-extent - origin[axis]) / direction[axis]
            b = (extent - origin[axis]) / direction[axis]
            near = wp.max(near, wp.min(a, b))
            far = wp.min(far, wp.max(a, b))
        elif wp.abs(origin[axis]) > extent:
            far = -1.0
    return wp.vec2(near, far)


@wp.func
def ray_reaches_box(origin: wp.vec3, direction: wp.vec3, lower: wp.vec3, upper: wp.vec3, max_distance: float) -> bool:
    """Whether the ray segment ``[0, max_distance]`` overlaps an axis-aligned box."""
    near = float(0.0)
    far = max_distance
    for axis in range(3):
        if direction[axis] != 0.0:
            a = (lower[axis] - origin[axis]) / direction[axis]
            b = (upper[axis] - origin[axis]) / direction[axis]
            near = wp.max(near, wp.min(a, b))
            far = wp.min(far, wp.max(a, b))
        elif origin[axis] < lower[axis] or origin[axis] > upper[axis]:
            return False
    return near <= far


@wp.func
def primitive_half_extents(kind: int, size: wp.vec3) -> wp.vec3:
    """Half extents that scale the shared unit cube to an analytic shape's bounds (or a mesh's scale)."""
    if kind == GeoType.SPHERE:
        return wp.vec3(size[0])
    if kind == GeoType.CAPSULE:
        return wp.vec3(size[0], size[0], size[0] + size[1])
    if kind == GeoType.CYLINDER or kind == GeoType.CONE:
        return wp.vec3(size[0], size[0], size[1])
    if kind == GeoType.ELLIPSOID:
        return wp.vec3(wp.abs(size[0]), wp.abs(size[1]), wp.abs(size[2]))
    return size


@wp.func
def ray_intersect_primitive_local(size: wp.vec3, kind: int, origin: wp.vec3, direction: wp.vec3) -> float:
    """Distance along a shape-local ray to an analytic primitive, or -1."""
    t = float(-1.0)
    normal = wp.vec3(0.0)
    if kind == GeoType.SPHERE:
        t, normal = raycast.ray_intersect_sphere(origin, direction, size[0])
    elif kind == GeoType.CAPSULE:
        t, normal = raycast.ray_intersect_capsule(origin, direction, size[0], size[1])
    elif kind == GeoType.CYLINDER:
        t, normal = raycast.ray_intersect_cylinder(origin, direction, size[0], size[1])
    elif kind == GeoType.CONE:
        t, normal = raycast.ray_intersect_cone(origin, direction, size[0], size[1])
    elif kind == GeoType.ELLIPSOID:
        t, normal = raycast.ray_intersect_ellipsoid(origin, direction, size)
    elif kind == GeoType.BOX:
        t, normal = raycast.ray_intersect_box(origin, direction, size)
    return t


@wp.kernel(enable_backward=False)
def compute_shape_instances(
    grid: wp.int32,
    extent: wp.float32,
    instance_shapes: wp.array[wp.int32],
    shape_cells: wp.array[wp.int32],
    world_centers: wp.array[wp.vec3],
    shape_types: wp.array[wp.int32],
    shape_sizes: wp.array[wp.vec3],
    shape_box_centers: wp.array[wp.vec3],
    shape_box_halves: wp.array[wp.vec3],
    shape_transforms: wp.array[wp.transformf],
    shape_traversables: wp.array[wp.uint64],
    out_instances: wp.array[wo.Instance],
    out_of_cell: wp.array[wp.int32],
):
    i = wp.tid()
    si = instance_shapes[i]
    kind = shape_types[si]
    half = primitive_half_extents(kind, shape_sizes[si])
    transform = shape_transforms[si]
    # The Warp traversal maps rays to the shape frame with the conjugate quaternion, which only inverts
    # the rotation of a unit quaternion. Invert that map instead of using the rotation matrix, so shapes
    # authored with an unnormalised quaternion land where the Warp backend renders them.
    to_local = wp.quat_to_matrix(wp.quat_inverse(wp.transform_get_rotation(transform)))
    rotation = wp.inverse(to_local) @ wp.diag(half)
    center = wp.transform_get_translation(transform)
    cell = shape_cells[si]
    mask = int(255)
    if cell >= 0:
        center = center - world_centers[cell]
        # Hide shapes that left their cell rather than let them reach another world's.
        box_center = rotation @ shape_box_centers[si] + center
        box_half = shape_box_halves[si]
        for axis in range(3):
            reach = wp.abs(box_center[axis])
            for j in range(3):
                reach += wp.abs(rotation[axis, j]) * box_half[j]
            if reach > extent:
                mask = 0
        center = center + world_offset(cell, grid, extent)
    if not (wp.isfinite(center) and wp.isfinite(rotation)):
        # A blown-up body must not reach the acceleration build.
        mask = 0
        rotation = wp.identity(n=3, dtype=float)
        center = wp.vec3(0.0)
        if cell >= 0:
            center = world_offset(cell, grid, extent)
    if mask == 0:
        out_of_cell[0] = 1
    sbt = int(wp.static(_SBT_PRIMITIVES))
    flags = int(wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT)))
    if kind == GeoType.MESH:
        sbt = wp.static(_SBT_TRIANGLES)
    elif kind == GeoType.BOX:
        # Boxes are triangles that keep the analytic box's behaviour: inside faces stay visible with culling.
        sbt = wp.static(_SBT_TRIANGLES)
        flags = wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT | wo.InstanceFlags.DISABLE_TRIANGLE_FACE_CULLING))
    out_instances[i] = wo.make_instance(
        rotation, center, shape_traversables[si], wp.uint32(si), wp.uint32(sbt), wp.uint32(mask), wp.uint32(flags)
    )


def is_available(device=None) -> tuple[bool, str]:
    """Whether an OptiX context can be created on ``device``, and why not."""
    try:
        device = wp.get_device(device)
        if not device.is_cuda:
            return False, "the OptiX backend requires a CUDA device"
        wo.get_context(device)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _launch_instances(model, shapes: OptixShapes, instance_shapes, instances) -> None:
    wp.launch(
        compute_shape_instances,
        dim=instance_shapes.shape[0],
        inputs=[
            shapes.grid,
            shapes.extent,
            instance_shapes,
            shapes.shape_cells,
            shapes.world_centers,
            shapes.shape_types,
            model.shape_scale,
            shapes.shape_box_centers,
            shapes.shape_box_halves,
            shapes.shape_world_transforms,
            shapes.shape_traversables,
        ],
        outputs=[instances, shapes.out_of_cell],
        device=model.device,
    )


def _build_geometry(model, traced: np.ndarray, types: np.ndarray):
    """Triangle structures of the meshes used by ``traced``, a triangle cube for boxes, and the other
    primitives' unit cube."""
    device = model.device
    sources = model.shape_source_ptr.numpy()
    geometry, mesh_ids = [], {}
    used = set(sources[traced[types[traced] == int(GeoType.MESH)]].tolist())
    for mesh in [*model._mesh_keep_alive, *model.heightfield_meshes]:
        if mesh.id in used and mesh.id not in mesh_ids:
            geometry.append(wo.Accel([wo.triangles(mesh.points, mesh.indices)], compact=True))
            mesh_ids[mesh.id] = geometry[-1].id
    missing = used - set(mesh_ids)
    if missing:
        raise UnsupportedError(f"{len(missing)} mesh shapes have no Warp mesh on the model")
    corners = wp.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=wp.vec3, device=device)
    faces = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    box = wo.Accel([wo.triangles(corners, wp.array(faces, dtype=wp.int32, device=device))], compact=True)
    bounds = wp.array([[-1.001] * 3 + [1.001] * 3], dtype=float, device=device)
    primitive = wo.Accel([wo.aabbs(bounds=bounds)], compact=True)
    geometry += [box, primitive]
    traversables = np.zeros(model.shape_count, dtype=np.uint64)
    for si in traced:
        kind = int(types[si])
        traversables[si] = (
            mesh_ids[int(sources[si])]
            if kind == int(GeoType.MESH)
            else box.id if kind == int(GeoType.BOX) else primitive.id
        )
    return geometry, wp.array(traversables, dtype=wp.uint64, device=device)


def build_shapes(model, state, shape_types: wp.array, *, extent: float | None = None) -> OptixShapes:
    """Build the OptiX acceleration of ``model``'s shapes at ``state``.

    Traces the shapes of the model's Warp shape BVH (the visible ones when it has none) so both
    backends render the same set. ``shape_types`` are the render types (heightfields as meshes).
    ``extent`` is the half-width [m] of the region about each world's centre that its shapes may
    occupy; it defaults to the largest half-width of a world's build-time bounds, doubled, and at
    least 0.5 m.
    """
    device = model.device
    if not device.is_cuda:
        raise UnsupportedError("the OptiX backend requires a CUDA device")
    try:
        wo.get_context(device)
    except Exception as exc:
        raise UnsupportedError(f"OptiX is not supported on {device}: {exc}") from exc
    if model.shape_count >= 1 << 24:
        raise UnsupportedError("OptiX shape instance IDs must fit in 24 bits")
    if extent is not None and not extent > 0.0:
        raise ValueError(f"extent must be positive, got {extent}")
    shape_count, world_count = model.shape_count, model.world_count

    if model.bvh_shape_enabled is not None:
        enabled = model.bvh_shape_enabled.numpy()[: model.bvh_shape_count_enabled].astype(np.int32)
    else:
        enabled_wp = wp.empty(shape_count, dtype=wp.uint32, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            compute_enabled_shapes,
            dim=shape_count,
            inputs=[model.shape_type, model.shape_flags, int(ShapeFlags.VISIBLE), enabled_wp, count],
            device=device,
        )
        enabled = enabled_wp.numpy()[: int(count.numpy()[0])].astype(np.int32)
    enabled = np.sort(enabled)
    types = shape_types.numpy().astype(np.int32) if shape_count else np.empty(0, dtype=np.int32)
    if np.any(types[enabled] == int(GeoType.GAUSSIAN)):
        raise UnsupportedError("the OptiX backend does not support Gaussian shapes")
    # Newton 1.5.2 never hits convex hulls in the render kernel; leave them out as well.
    primitives = (GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE, GeoType.ELLIPSOID, GeoType.BOX)
    traced = enabled[np.isin(types[enabled], [int(GeoType.MESH), *(int(k) for k in primitives)])]
    planes = enabled[types[enabled] == int(GeoType.PLANE)]

    transforms = wp.empty(shape_count, dtype=wp.transformf, device=device)
    wp.launch(
        compute_shape_world_transforms,
        dim=shape_count,
        inputs=[state.body_q, model.shape_body, model.shape_transform, transforms],
        device=device,
    )
    # World-space bounds size the cell that holds each world.
    local_bounds = wp.empty((shape_count, 2), dtype=wp.vec3f, device=device)
    wp.launch(
        compute_shape_local_bounds,
        dim=shape_count,
        inputs=[model.shape_type, model.shape_source_ptr, model.gaussians_data, local_bounds],
        device=device,
    )
    count = len(traced)
    lowers = wp.zeros(count, dtype=wp.vec3f, device=device)
    uppers = wp.zeros(count, dtype=wp.vec3f, device=device)
    groups = wp.zeros(count, dtype=wp.int32, device=device)
    if count:
        wp.launch(
            compute_shape_bvh_bounds,
            dim=count,
            inputs=[
                count,
                world_count + 1,
                model.shape_world,
                wp.array(traced.astype(np.uint32), dtype=wp.uint32, device=device),
                model.shape_type,
                model.shape_scale,
                transforms,
                local_bounds,
                lowers,
                uppers,
                groups,
            ],
            device=device,
        )
    lowers, uppers = lowers.numpy().reshape(-1, 3), uppers.numpy().reshape(-1, 3)
    shape_world = model.shape_world.numpy() if shape_count else np.empty(0, dtype=np.int32)
    worlds = shape_world[traced]
    finite = np.isfinite(lowers).all(axis=1) & np.isfinite(uppers).all(axis=1)
    # Boxes in the frame the instance transform maps from: mesh bounds, or the primitives' unit cube.
    box_centers = np.zeros((shape_count, 3), dtype=np.float32)
    box_halves = np.ones((shape_count, 3), dtype=np.float32)
    meshes = traced[types[traced] == int(GeoType.MESH)]
    mesh_bounds = local_bounds.numpy()[meshes].astype(np.float64).reshape(-1, 2, 3)
    box_centers[meshes] = 0.5 * (mesh_bounds[:, 0] + mesh_bounds[:, 1])
    box_halves[meshes] = 0.5 * (mesh_bounds[:, 1] - mesh_bounds[:, 0])

    # Each world is packed about the centre of its build-time bounds.
    centers = np.zeros((world_count, 3), dtype=np.float32)
    reach = 0.0
    local = finite & (worlds >= 0)
    if local.any():
        low = np.full((world_count, 3), np.inf)
        high = np.full((world_count, 3), -np.inf)
        np.minimum.at(low, worlds[local], lowers[local])
        np.maximum.at(high, worlds[local], uppers[local])
        occupied = np.isfinite(low).all(axis=1)
        centers[occupied] = (0.5 * (low[occupied] + high[occupied])).astype(np.float32)
        reach = float(np.maximum(high[occupied] - centers[occupied], centers[occupied] - low[occupied]).max())
    # Twice the build-time bounds leaves the shapes room to move.
    extent = float(extent or max(2.0 * reach, 0.5))

    world_plane_shapes = planes[shape_world[planes] >= 0]
    world_plane_shapes = world_plane_shapes[np.argsort(shape_world[world_plane_shapes], kind="stable")]
    plane_worlds = shape_world[world_plane_shapes]
    world_planes = np.full((world_count, np.bincount(plane_worlds).max(initial=0)), -1, dtype=np.int32)
    world_planes[plane_worlds, np.arange(len(plane_worlds)) - np.searchsorted(plane_worlds, plane_worlds)] = (
        world_plane_shapes
    )
    grid = max(int(np.ceil(world_count ** (1.0 / 3.0))), 1)
    while grid**3 < world_count:
        grid += 1

    cells = np.full(shape_count, -1, dtype=np.int32)
    cells[traced] = worlds
    instance_shapes = traced[worlds >= 0]
    global_shapes = traced[worlds < 0]
    # Global shapes fixed to the world never move, so rays can be tested against their box first.
    global_bounds = None
    if len(global_shapes) and (model.shape_body.numpy()[global_shapes] < 0).all() and finite[worlds < 0].all():
        low, high = lowers[worlds < 0].min(axis=0), uppers[worlds < 0].max(axis=0)
        pad = np.maximum(1.0e-4 * (high - low), 1.0e-3)
        global_bounds = (tuple(float(x) for x in low - pad), tuple(float(x) for x in high + pad))
    geometry, traversables = _build_geometry(model, traced, types)
    world_planes_wp = wp.array(world_planes, dtype=wp.int32, device=device)
    shapes = OptixShapes(
        id=0,
        accel=None,
        global_id=0,
        global_accel=None,
        geometry=geometry,
        shape_types=shape_types,
        shape_world_transforms=transforms,
        shape_box_centers=wp.array(box_centers, dtype=wp.vec3, device=device),
        shape_box_halves=wp.array(box_halves, dtype=wp.vec3, device=device),
        shape_traversables=traversables,
        shape_cells=wp.array(cells, dtype=wp.int32, device=device),
        instance_shapes=wp.array(instance_shapes, dtype=wp.int32, device=device),
        instances=wp.empty(len(instance_shapes), dtype=wo.Instance, device=device),
        global_instance_shapes=wp.array(global_shapes, dtype=wp.int32, device=device),
        global_instances=wp.empty(len(global_shapes), dtype=wo.Instance, device=device),
        planes=tuple(int(si) for si in planes[shape_world[planes] < 0]),
        world_planes=world_planes_wp,
        world_planes_flat=world_planes_wp.flatten(),
        world_centers=wp.array(centers, dtype=wp.vec3, device=device),
        grid=grid,
        extent=extent,
        world_count=world_count,
        out_of_cell=wp.zeros(1, dtype=wp.int32, device=device),
        global_bounds=global_bounds,
    )
    # A fast-trace build or periodic rebuilds measured no faster than refitting this one (4096 envs, H200).
    flags = wo.BuildFlags.ALLOW_UPDATE | wo.BuildFlags.PREFER_FAST_BUILD
    if shapes.has_instances:
        _launch_instances(model, shapes, shapes.instance_shapes, shapes.instances)
        shapes.accel = wo.Accel([wo.instances(shapes.instances, shapes.geometry)], flags=flags)
        shapes.id = shapes.accel.id
    if shapes.has_global_instances:
        _launch_instances(model, shapes, shapes.global_instance_shapes, shapes.global_instances)
        shapes.global_accel = wo.Accel([wo.instances(shapes.global_instances, shapes.geometry)], flags=flags)
        shapes.global_id = shapes.global_accel.id
    return shapes


def refit_shapes(model, state, shapes: OptixShapes, *, rebuild: bool = False) -> None:
    """Move the instances to ``state``; ``rebuild`` rebuilds the instance structures instead of refitting them."""
    if model.device.stream.is_capturing:
        raise RuntimeError("the OptiX backend does not support CUDA graph capture")
    if model.shape_count:
        wp.launch(
            compute_shape_world_transforms,
            dim=model.shape_count,
            inputs=[state.body_q, model.shape_body, model.shape_transform, shapes.shape_world_transforms],
            device=model.device,
        )
    for accel, instance_shapes, instances in (
        (shapes.accel, shapes.instance_shapes, shapes.instances),
        (shapes.global_accel, shapes.global_instance_shapes, shapes.global_instances),
    ):
        if accel is not None:
            _launch_instances(model, shapes, instance_shapes, instances)
            if rebuild:
                accel.rebuild()
            else:
                accel.refit()
    shapes.refit_count += 1


def create_pipeline(shapes: OptixShapes, kernel: wp.Kernel) -> wo.Pipeline:
    """Compile the render ``kernel`` as the ray generation program of a pipeline over ``shapes``."""

    @wp.kernel(module="unique", enable_backward=False)
    def intersect(scene: SceneData):
        # The instance scales the unit cube to the shape, so scaling the object-space ray by the
        # half extents gives the shape-local ray with the same distance.
        si = int(wp.optix_get_instance_id())
        kind = scene.shape_types[si]
        size = scene.shape_sizes[si]
        half = primitive_half_extents(kind, size)
        distance = ray_intersect_primitive_local(
            size,
            kind,
            wp.cw_mul(wp.optix_get_object_ray_origin(), half),
            wp.cw_mul(wp.optix_get_object_ray_direction(), half),
        )
        if distance >= wp.optix_get_ray_tmin():
            wp.optix_report_intersection(distance, wp.uint32(0), wp.uint32(0), wp.uint32(0))

    return wo.Pipeline(
        kernel,
        device=shapes.shape_types.device,
        sbt=wo.SbtLayout(hit_groups={"triangles": wo.HitGroup(), "primitives": wo.HitGroup(intersection=intersect)}),
        max_registers=_MAX_REGISTERS,
        module_options={"fast_math": kernel.module.options["fast_math"]},
        graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
    )


def _global_box(shapes: OptixShapes):
    if shapes.global_bounds is None:
        return False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return True, shapes.global_bounds[0], shapes.global_bounds[1]


def _check_state(state) -> None:
    if state.has_particles or state.num_gaussians > 0:
        raise UnsupportedError("the OptiX backend does not support particles, deformable surfaces or Gaussians")


def create_trace_function(config, state, shapes: OptixShapes) -> wp.Function:
    """Closest hit of a world-space ray, as ``trace(scene, world, origin, direction, forward)``.

    Reads :attr:`SceneData.optix_id`, ``optix_global_id``, ``optix_world_planes`` and
    ``optix_world_centers``; ``scene.shape_transforms`` must be :attr:`OptixShapes.shape_world_transforms`.
    """
    _check_state(state)
    grid, extent = shapes.grid, shapes.extent
    planes_per_world = shapes.world_planes.shape[1]
    cull = bool(config.enable_backface_culling)
    ray_flags = int(wo.RayFlags.CULL_BACK_FACING_TRIANGLES) if cull else 0
    cull_global, global_lower, global_upper = _global_box(shapes)

    @wp.func
    def plane_hit(
        hit: raytrace.ClosestHit, si: int, scene: SceneData, origin: wp.vec3f, direction: wp.vec3f
    ) -> raytrace.ClosestHit:
        distance, normal = raycast.ray_intersect_shape(
            scene.shape_transforms[si], scene.shape_sizes[si], GeoType.PLANE, origin, direction, wp.static(cull)
        )
        if distance >= 0.0 and distance < hit.distance:
            hit.distance = distance
            hit.normal = normal
            hit.shape_index = wp.uint32(si)
            hit.bary_u = 0.0
            hit.bary_v = 0.0
            hit.face_idx = -1
        return hit

    @wp.func
    def instance_hit(
        hit: raytrace.ClosestHit,
        handle: wp.uint64,
        packed_origin: wp.vec3f,
        near: float,
        far: float,
        scene: SceneData,
        origin: wp.vec3f,
        direction: wp.vec3f,
    ) -> raytrace.ClosestHit:
        query = wo.accel_query_ray(handle, packed_origin, direction, far, near, 255, wp.static(ray_flags))
        if query.result:
            if query.t < hit.distance:
                si = int(query.instance_id)
                hit.distance = query.t
                hit.shape_index = wp.uint32(si)
                if scene.shape_types[si] == GeoType.MESH:
                    mesh = scene.shape_source_ptr[si]
                    face = int(query.primitive_index)
                    # Warp's barycentrics weight the first two vertices, OptiX's the last two.
                    u = 1.0 - query.u - query.v
                    v = query.u
                    normal_local = wp.vec3f(0.0)
                    smooth = bool(False)
                    mesh_data_id = scene.shape_mesh_data_ids[si]
                    if mesh_data_id > -1:
                        normals = scene.mesh_data[mesh_data_id].normals
                        if normals.shape[0] > 0:
                            smooth = True
                            n0 = wp.mesh_get_index(mesh, face * 3 + 0)
                            n1 = wp.mesh_get_index(mesh, face * 3 + 1)
                            n2 = wp.mesh_get_index(mesh, face * 3 + 2)
                            normal_local = normals[n0] * u + normals[n1] * v + normals[n2] * (1.0 - u - v)
                    if not smooth:
                        normal_local = wp.mesh_eval_face_normal(mesh, face)
                    normal_local = raycast.safe_div_vec3(normal_local, scene.shape_sizes[si])
                    hit.normal = wp.normalize(wp.transform_vector(scene.shape_transforms[si], normal_local))
                    hit.bary_u = u
                    hit.bary_v = v
                    hit.face_idx = face
                else:
                    _distance, normal = raycast.ray_intersect_shape(
                        scene.shape_transforms[si],
                        scene.shape_sizes[si],
                        scene.shape_types[si],
                        origin,
                        direction,
                        wp.static(cull),
                    )
                    hit.normal = normal
                    hit.bary_u = 0.0
                    hit.bary_v = 0.0
                    hit.face_idx = -1
        return hit

    @wp.func
    def trace(
        scene: SceneData, world: int, origin: wp.vec3f, direction: wp.vec3f, forward: wp.vec3f
    ) -> raytrace.ClosestHit:
        hit = raytrace.ClosestHit()
        hit.distance = wp.static(float(config.max_distance))
        hit.shape_index = raytrace.NO_HIT_SHAPE_ID
        hit.color = wp.vec3f(0.0)
        # Planes first: a nearer hit shortens the traversals.
        if wp.static(bool(config.enable_global_world)):
            for p in range(wp.static(len(shapes.planes))):
                hit = plane_hit(hit, wp.static(shapes.planes[p]), scene, origin, direction)
        for p in range(wp.static(planes_per_world)):
            si = scene.optix_world_planes[world * wp.static(planes_per_world) + p]
            if si >= 0:
                hit = plane_hit(hit, si, scene, origin, direction)
        if wp.static(shapes.has_instances):
            local_origin = origin - scene.optix_world_centers[world]
            interval = ray_interval(local_origin, direction, wp.static(extent), hit.distance)
            if interval[1] >= interval[0]:
                hit = instance_hit(
                    hit,
                    scene.optix_id,
                    local_origin + world_offset(world, wp.static(grid), wp.static(extent)),
                    interval[0],
                    interval[1],
                    scene,
                    origin,
                    direction,
                )
        if wp.static(bool(config.enable_global_world and shapes.has_global_instances)):
            if wp.static(not cull_global) or ray_reaches_box(
                origin, direction, wp.static(wp.vec3(*global_lower)), wp.static(wp.vec3(*global_upper)), hit.distance
            ):
                hit = instance_hit(hit, scene.optix_global_id, origin, 0.0, hit.distance, scene, origin, direction)
        return hit

    return trace


def create_occluded_function(config, state, shapes: OptixShapes) -> wp.Function:
    """Whether anything blocks a world-space ray, as ``occluded(scene, world, origin, direction, max_distance)``."""
    _check_state(state)
    grid, extent = shapes.grid, shapes.extent
    planes_per_world = shapes.world_planes.shape[1]
    cull = bool(config.enable_backface_culling)
    cull_global, global_lower, global_upper = _global_box(shapes)

    @wp.func
    def plane_blocks(si: int, scene: SceneData, origin: wp.vec3f, direction: wp.vec3f, max_distance: float) -> bool:
        distance, _normal = raycast.ray_intersect_shape_no_normal(
            scene.shape_transforms[si], scene.shape_sizes[si], GeoType.PLANE, origin, direction, wp.static(cull)
        )
        return distance > -1.0 and distance < max_distance

    @wp.func
    def occluded(scene: SceneData, world: int, origin: wp.vec3f, direction: wp.vec3f, max_distance: float) -> bool:
        if wp.static(bool(config.enable_global_world)):
            for p in range(wp.static(len(shapes.planes))):
                if plane_blocks(wp.static(shapes.planes[p]), scene, origin, direction, max_distance):
                    return True
        for p in range(wp.static(planes_per_world)):
            si = scene.optix_world_planes[world * wp.static(planes_per_world) + p]
            if si >= 0:
                if plane_blocks(si, scene, origin, direction, max_distance):
                    return True
        if wp.static(shapes.has_instances):
            local_origin = origin - scene.optix_world_centers[world]
            interval = ray_interval(local_origin, direction, wp.static(extent), max_distance)
            if interval[1] >= interval[0]:
                if wo.accel_query_ray_anyhit(
                    scene.optix_id,
                    local_origin + world_offset(world, wp.static(grid), wp.static(extent)),
                    direction,
                    interval[1],
                    interval[0],
                    255,
                ):
                    return True
        if wp.static(bool(config.enable_global_world and shapes.has_global_instances)):
            if wp.static(not cull_global) or ray_reaches_box(
                origin, direction, wp.static(wp.vec3(*global_lower)), wp.static(wp.vec3(*global_upper)), max_distance
            ):
                if wo.accel_query_ray_anyhit(scene.optix_global_id, origin, direction, max_distance, 0.0, 255):
                    return True
        return False

    return occluded
