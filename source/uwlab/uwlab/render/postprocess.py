# SPDX-License-Identifier: BSD-3-Clause
"""HDR camera response and per-world planar-depth conversion."""

import math

import warp as wp
from newton._src.sensors.warp_raytrace.tiling import pack_rgba_to_uint32
from newton._src.utils.color import linear_to_srgb_wp


def _create_hdr_reader(blur):
    """Reads the HDR image, through a 3x3 Gaussian of ``blur`` pixels when positive (the softness of a
    reconstruction filter: one sample per pixel is sharper than a temporally anti-aliased image)."""
    side = math.exp(-0.5 / (blur * blur)) if blur > 0.0 else 0.0

    @wp.func
    def read(hdr: wp.array4d[wp.vec3f], world: int, camera: int, y: int, x: int):
        if wp.static(blur <= 0.0):
            return hdr[world, camera, y, x]
        total = wp.vec3f(0.0)
        weight = float(0.0)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                qy = y + dy
                qx = x + dx
                if qy >= 0 and qy < hdr.shape[2] and qx >= 0 and qx < hdr.shape[3]:
                    w = float(1.0)
                    if dy != 0:
                        w *= wp.static(side)
                    if dx != 0:
                        w *= wp.static(side)
                    total += hdr[world, camera, qy, qx] * w
                    weight += w
        return total / weight

    return read


def create_tone_mapper(mode, blur=0.0):
    if mode not in ("linear", "reinhard", "aces"):
        raise ValueError("tone mapping must be linear, reinhard, or aces")
    read = _create_hdr_reader(blur)

    @wp.kernel(enable_backward=False, module="unique")
    def apply(hdr: wp.array4d[wp.vec3f], rgba: wp.array4d[wp.uint32], exposure: wp.array[wp.float32]):
        world, camera, y, x = wp.tid()
        c = read(hdr, world, camera, y, x) * wp.pow(2.0, exposure[world])
        for i in range(3):
            v = wp.max(c[i], 0.0)
            if wp.static(mode == "reinhard"):
                v = v / (1.0 + v)
            elif wp.static(mode == "aces"):
                v = wp.clamp(v * (2.51 * v + 0.03) / (v * (2.43 * v + 0.59) + 0.14), 0.0, 1.0)
            c[i] = wp.clamp(v, 0.0, 1.0)
        rgba[world, camera, y, x] = pack_rgba_to_uint32(linear_to_srgb_wp(c), 1.0)

    return apply


@wp.kernel
def planar_depth(
    ray_depth: wp.array4d[wp.float32],
    intrinsics: wp.array[wp.mat33f],
    plane_depth: wp.array4d[wp.float32],
    per_world: int,
):
    world, camera, y, x = wp.tid()
    k_index = camera * ray_depth.shape[0]
    if per_world != 0:
        k_index += world
    k = intrinsics[k_index]
    dy = (float(y) + 0.5 - k[1, 2]) / k[1, 1]
    dx = (float(x) + 0.5 - k[0, 2] - k[0, 1] * dy) / k[0, 0]
    plane_depth[world, camera, y, x] = ray_depth[world, camera, y, x] / wp.sqrt(1.0 + dx * dx + dy * dy)


def create_camera_mapper(mode: str, blur: float = 0.0):
    """Tone mapping with linear white balance and a lightweight display-noise approximation."""
    assert mode in ('linear', 'reinhard', 'aces')
    read = _create_hdr_reader(blur)

    @wp.kernel(enable_backward=False, module='unique')
    def apply(hdr: wp.array4d[wp.vec3f], rgba: wp.array4d[wp.uint32], exposure: wp.array[wp.float32],
              white_balance: wp.array[wp.vec3f], response: wp.array[wp.vec4f], frame: int):
        world, camera, y, x = wp.tid()
        c = wp.cw_mul(read(hdr, world, camera, y, x), white_balance[world]) * wp.pow(2.0, exposure[world])
        for i in range(3):
            v = wp.max(c[i], 0.0)
            if wp.static(mode == 'reinhard'):
                v = v / (1.0 + v)
            elif wp.static(mode == 'aces'):
                v = wp.clamp(v*(2.51*v+.03)/(v*(2.43*v+.59)+.14), 0.0, 1.0)
            c[i] = wp.clamp(v, 0.0, 1.0)
        c = linear_to_srgb_wp(c)
        params = response[world]
        rng = wp.rand_init(int(params[3]) + frame*7919 + world*104729, (camera*hdr.shape[2]+y)*hdr.shape[3]+x)
        for i in range(3):
            v = wp.pow(wp.max(c[i], 0.0), 1.0/params[0])
            sigma = wp.sqrt(params[1]*params[1] + params[2]*params[2]*v)
            c[i] = wp.clamp(v + sigma*wp.randn(rng), 0.0, 1.0)
        rgba[world, camera, y, x] = pack_rgba_to_uint32(c, 1.0)

    return apply


PATH_FILTER_RADIUS = 7
"""Half-width of the path-lighting tent filter: 15 taps weight every residue of the 8-pixel sampling tile equally."""


@wp.func
def _path_tap(
    light: wp.array4d[wp.vec3f],
    gloss: wp.array4d[wp.vec3f],
    normal: wp.array4d[wp.vec3f],
    depth: wp.array4d[wp.float32],
    world: int,
    camera: int,
    y: int,
    x: int,
    axis: int,
    step: int,
):
    """Filtered diffuse light and specular radiance along one image axis: the 15-tap tent (``step`` 0) or a 5-tap
    B-spline with ``step`` pixel spacing. Neighbours count by how close they lie to the centre pixel's surface:
    similar normal, and depth near the centre's local depth slope. A background-centred pixel (anti-aliased edge)
    takes the plain average of its neighbours."""
    z = depth[world, camera, y, x]
    n = normal[world, camera, y, x]
    limit = depth.shape[3]
    coord = x
    if axis == 0:
        limit = depth.shape[2]
        coord = y
    # one-sided depth slope of smaller magnitude: stays on the centre pixel's surface next to an edge
    slope = float(0.0)
    found = int(0)
    if z > 0.0:
        for side in range(2):
            q = coord + 2 * side - 1
            if q >= 0 and q < limit:
                zq = float(0.0)
                if axis == 0:
                    zq = depth[world, camera, q, x]
                else:
                    zq = depth[world, camera, y, q]
                if zq > 0.0:
                    candidate = (zq - z) * float(2 * side - 1)
                    if found == 0 or wp.abs(candidate) < wp.abs(slope):
                        slope = candidate
                    found = 1
    radius = int(wp.static(PATH_FILTER_RADIUS))
    spacing = int(1)
    if step > 0:
        radius = 2
        spacing = step
    sum_light = wp.vec3f(0.0)
    sum_gloss = wp.vec3f(0.0)
    total = float(0.0)
    wide_light = wp.vec3f(0.0)
    wide_gloss = wp.vec3f(0.0)
    wide = float(0.0)
    window = float(0.0)
    for tap in range(2 * radius + 1):
        t = tap - radius
        offset = t * spacing
        q = coord + offset
        if q >= 0 and q < limit:
            qy = y
            qx = q
            if axis == 0:
                qy = q
                qx = x
            zq = depth[world, camera, qy, qx]
            weight = float(radius + 1 - wp.abs(t))
            if step > 0:
                weight = 6.0
                if wp.abs(t) == 1:
                    weight = 4.0
                if wp.abs(t) == 2:
                    weight = 1.0
            window += weight
            if zq > 0.0:
                if z > 0.0:
                    cosine = wp.max(wp.dot(n, normal[world, camera, qy, qx]), 0.0)
                    loose = weight * (0.2 + 0.8 * cosine) * wp.exp(-wp.abs(zq - z) / (0.04 * z))
                    wide_light += light[world, camera, qy, qx] * loose
                    wide_gloss += gloss[world, camera, qy, qx] * loose
                    wide += loose
                    cosine = cosine * cosine
                    cosine = cosine * cosine
                    tolerance = 0.01 * z * wp.max(1.0, 0.25 * float(wp.abs(offset)))
                    weight *= cosine * cosine * wp.exp(-wp.abs(zq - z - slope * float(offset)) / tolerance)
                sum_light += light[world, camera, qy, qx] * weight
                sum_gloss += gloss[world, camera, qy, qx] * weight
                total += weight
    # a surface with few neighbours of its own (thin parts, creases) borrows from nearby geometry at its depth
    if window > 0.0 and wide > 0.0:
        borrow = wp.clamp(1.0 - 2.0 * total / window, 0.0, 1.0)
        sum_light += wide_light * borrow
        sum_gloss += wide_gloss * borrow
        total += wide * borrow
    if total > 1.0e-6:
        return sum_light / total, sum_gloss / total
    return light[world, camera, y, x], gloss[world, camera, y, x]


@wp.kernel(enable_backward=False)
def filter_path_lighting(
    light: wp.array4d[wp.vec3f],
    gloss: wp.array4d[wp.vec3f],
    normal: wp.array4d[wp.vec3f],
    depth: wp.array4d[wp.float32],
    axis: int,
    step: int,
    light_out: wp.array4d[wp.vec3f],
    gloss_out: wp.array4d[wp.vec3f],
):
    """One pass of the separable path-lighting filter."""
    world, camera, y, x = wp.tid()
    filtered_light, filtered_gloss = _path_tap(light, gloss, normal, depth, world, camera, y, x, axis, step)
    light_out[world, camera, y, x] = filtered_light
    gloss_out[world, camera, y, x] = filtered_gloss


@wp.kernel(enable_backward=False)
def resolve_path_lighting(
    light: wp.array4d[wp.vec3f],
    gloss: wp.array4d[wp.vec3f],
    normal: wp.array4d[wp.vec3f],
    depth: wp.array4d[wp.float32],
    axis: int,
    step: int,
    tint: wp.array4d[wp.vec3f],
    cover: wp.array4d[wp.float32],
    hdr: wp.array4d[wp.vec3f],
):
    """Last filter pass; adds the filtered lighting to the radiance the render kernel left without it. ``tint``
    carries the pixel's coverage for the diffuse light, ``cover`` applies it to the specular radiance."""
    world, camera, y, x = wp.tid()
    filtered_light, filtered_gloss = _path_tap(light, gloss, normal, depth, world, camera, y, x, axis, step)
    hdr[world, camera, y, x] = (
        hdr[world, camera, y, x]
        + wp.cw_mul(tint[world, camera, y, x], filtered_light)
        + filtered_gloss * cover[world, camera, y, x]
    )
