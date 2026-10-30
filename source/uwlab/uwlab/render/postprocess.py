# SPDX-License-Identifier: BSD-3-Clause
"""HDR camera response and per-world planar-depth conversion."""

import warp as wp
from newton._src.sensors.warp_raytrace.tiling import pack_rgba_to_uint32
from newton._src.utils.color import linear_to_srgb_wp


def create_tone_mapper(mode):
    if mode not in ("linear", "reinhard", "aces"):
        raise ValueError("tone mapping must be linear, reinhard, or aces")

    @wp.kernel(enable_backward=False, module="unique")
    def apply(hdr: wp.array4d[wp.vec3f], rgba: wp.array4d[wp.uint32], exposure: wp.array[wp.float32]):
        world, camera, y, x = wp.tid()
        c = hdr[world, camera, y, x] * wp.pow(2.0, exposure[world])
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


def create_camera_mapper(mode: str):
    """Tone mapping with linear white balance and a lightweight display-noise approximation."""
    assert mode in ('linear', 'reinhard', 'aces')

    @wp.kernel(enable_backward=False, module='unique')
    def apply(hdr: wp.array4d[wp.vec3f], rgba: wp.array4d[wp.uint32], exposure: wp.array[wp.float32],
              white_balance: wp.array[wp.vec3f], response: wp.array[wp.vec4f], frame: int):
        world, camera, y, x = wp.tid()
        c = wp.cw_mul(hdr[world, camera, y, x], white_balance[world]) * wp.pow(2.0, exposure[world])
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
