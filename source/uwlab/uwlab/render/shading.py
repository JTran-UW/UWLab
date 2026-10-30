# SPDX-License-Identifier: BSD-3-Clause
"""Linear-light GGX helpers and texture coordinates for the Newton extension."""

import warp as wp
from newton._src.geometry import GeoType
from newton._src.sensors.warp_raytrace import raytrace
from .types import ENV_SAMPLE_BINS, ENV_SAMPLE_HEIGHT, ENV_SAMPLE_WIDTH, SceneData


@wp.func
def basis(n: wp.vec3f):
    axis = wp.vec3f(0.0, 0.0, 1.0)
    if wp.abs(n[2]) > 0.99:
        axis = wp.vec3f(0.0, 1.0, 0.0)
    t = wp.normalize(wp.cross(axis, n))
    return t, wp.cross(n, t)


@wp.func
def fresnel(f0: wp.vec3f, cosine: float):
    return f0 + (wp.vec3f(1.0) - f0) * wp.pow(wp.clamp(1.0 - cosine, 0.0, 1.0), 5.0)


@wp.func
def ggx_specular(
    base: wp.vec3f, metallic: float, roughness: float, specular: float, n: wp.vec3f, v: wp.vec3f, l: wp.vec3f
):
    nv = wp.max(wp.dot(n, v), 1.0e-5)
    nl = wp.max(wp.dot(n, l), 0.0)
    h = wp.normalize(v + l)
    nh = wp.max(wp.dot(n, h), 0.0)
    vh = wp.max(wp.dot(v, h), 0.0)
    alpha = wp.max(roughness * roughness, 0.0025)
    a2 = alpha * alpha
    denom = nh * nh * (a2 - 1.0) + 1.0
    distribution = a2 / (wp.pi * denom * denom)
    gv = 2.0 * nv / (nv + wp.sqrt(a2 + (1.0 - a2) * nv * nv))
    gl = 2.0 * nl / wp.max(nl + wp.sqrt(a2 + (1.0 - a2) * nl * nl), 1.0e-6)
    f0 = wp.lerp(wp.vec3f(0.08 * specular), base, metallic)
    return fresnel(f0, vh) * (distribution * gv * gl / wp.max(4.0 * nv * nl, 1.0e-6))


@wp.func
def brdf(base: wp.vec3f, metallic: float, roughness: float, specular: float, n: wp.vec3f, v: wp.vec3f, l: wp.vec3f):
    f0 = wp.lerp(wp.vec3f(0.08 * specular), base, metallic)
    f = fresnel(f0, wp.max(wp.dot(wp.normalize(v + l), v), 0.0))
    diffuse = wp.cw_mul(wp.vec3f(1.0) - f, base) * ((1.0 - metallic) / wp.pi)
    return diffuse + ggx_specular(base, metallic, roughness, specular, n, v, l)


@wp.func
def sample_ggx(n: wp.vec3f, v: wp.vec3f, roughness: float, u: float, w: float):
    alpha = wp.max(roughness * roughness, 0.0025)
    phi = 2.0 * wp.pi * u
    cos_theta = wp.sqrt((1.0 - w) / wp.max(1.0 + (alpha * alpha - 1.0) * w, 1.0e-6))
    sin_theta = wp.sqrt(wp.max(0.0, 1.0 - cos_theta * cos_theta))
    t, b = basis(n)
    h = wp.normalize(t * (wp.cos(phi) * sin_theta) + b * (wp.sin(phi) * sin_theta) + n * cos_theta)
    l = wp.normalize(2.0 * wp.dot(v, h) * h - v)
    denom = cos_theta * cos_theta * (alpha * alpha - 1.0) + 1.0
    d = alpha * alpha / (wp.pi * denom * denom)
    pdf = d * cos_theta / wp.max(4.0 * wp.abs(wp.dot(v, h)), 1.0e-6)
    return l, pdf


@wp.func
def environment(scene: SceneData, world: int, direction: wp.vec3f, fallback: wp.vec3f):
    result = fallback
    index = scene.environment_ids[world]
    if index >= 0:
        d = wp.normalize(direction)
        u = wp.atan2(d[1], d[0]) / (2.0 * wp.pi) + 0.5 + scene.environment_rotation[world]
        v = wp.acos(wp.clamp(d[2], -1.0, 1.0)) / wp.pi
        rgba = wp.texture_sample(scene.environment_data[index].texture, wp.vec2f(u, v), dtype=wp.vec4f)
        result = wp.vec3f(rgba[0], rgba[1], rgba[2]) * scene.environment_intensity[world]
    return result


@wp.func
def sample_environment(scene: SceneData, world: int, choice: float, jitter_u: float, jitter_v: float):
    """Sample a brightness-weighted latlong bin; return world direction and solid-angle PDF."""
    base = scene.environment_ids[world] * wp.static(ENV_SAMPLE_BINS)
    low = int(0)
    high = int(wp.static(ENV_SAMPLE_BINS))
    target = wp.min(choice, 0.99999994)
    for iteration in range(13):
        middle = (low + high) / 2
        if scene.environment_cdf[base + middle] < target:
            low = middle + 1
        else:
            high = middle
    cell = wp.min(low, wp.static(ENV_SAMPLE_BINS - 1))
    row = int(float(cell) / float(wp.static(ENV_SAMPLE_WIDTH)))
    col = cell - row * wp.static(ENV_SAMPLE_WIDTH)
    u = (float(col) + jitter_u) / float(wp.static(ENV_SAMPLE_WIDTH))
    v = (float(row) + jitter_v) / float(wp.static(ENV_SAMPLE_HEIGHT))
    theta = wp.pi * v
    phi = 2.0 * wp.pi * (u - 0.5 - scene.environment_rotation[world])
    sine = wp.sin(theta)
    direction = wp.vec3f(sine * wp.cos(phi), sine * wp.sin(phi), wp.cos(theta))
    pdf = (
        scene.environment_probability[base + cell]
        * float(wp.static(ENV_SAMPLE_BINS))
        / (2.0 * wp.pi * wp.pi * wp.max(sine, 1.0e-6))
    )
    return direction, pdf


@wp.func
def normal_from_map(scene: SceneData, hit: raytrace.ClosestHit, p: wp.vec3f, value: wp.vec3f, strength: float):
    shape = int(hit.shape_index)
    n = wp.normalize(hit.normal)
    t, b = basis(n)
    data_id = scene.shape_mesh_data_ids[shape]
    transform = scene.shape_transforms[shape]
    if scene.shape_types[shape] == GeoType.MESH and data_id >= 0 and scene.mesh_data[data_id].uvs.shape[0] > 0:
        mesh_id = scene.shape_source_ptr[shape]
        i0 = wp.mesh_get_index(mesh_id, hit.face_idx * 3)
        i1 = wp.mesh_get_index(mesh_id, hit.face_idx * 3 + 1)
        i2 = wp.mesh_get_index(mesh_id, hit.face_idx * 3 + 2)
        e1 = wp.cw_mul(wp.mesh_get_point(mesh_id, i1) - wp.mesh_get_point(mesh_id, i0), scene.shape_sizes[shape])
        e2 = wp.cw_mul(wp.mesh_get_point(mesh_id, i2) - wp.mesh_get_point(mesh_id, i0), scene.shape_sizes[shape])
        uv0 = scene.mesh_data[data_id].uvs[i0]
        d1 = scene.mesh_data[data_id].uvs[i1] - uv0
        d2 = scene.mesh_data[data_id].uvs[i2] - uv0
        det = d1[0] * d2[1] - d1[1] * d2[0]
        if wp.abs(det) > 1.0e-8:
            tangent = wp.transform_vector(transform, (e1 * d2[1] - e2 * d1[1]) / det)
            bitangent = wp.transform_vector(transform, (e2 * d1[0] - e1 * d2[0]) / det)
            t = wp.normalize(tangent - n * wp.dot(n, tangent))
            b = wp.cross(n, t)
            if wp.dot(b, bitangent) < 0.0:
                b = -b
    else:
        local_n = wp.transform_vector(wp.transform_inverse(transform), n)
        a = wp.vec3f(wp.abs(local_n[0]), wp.abs(local_n[1]), wp.abs(local_n[2]))
        tangent = wp.vec3f(1.0, 0.0, 0.0)
        bitangent = wp.vec3f(0.0, 1.0, 0.0)
        if a[0] >= a[1] and a[0] >= a[2]:
            tangent = wp.vec3f(0.0, 1.0, 0.0)
            bitangent = wp.vec3f(0.0, 0.0, 1.0)
        elif a[1] >= a[2]:
            bitangent = wp.vec3f(0.0, 0.0, 1.0)
        t = wp.normalize(wp.transform_vector(transform, tangent))
        b = wp.normalize(wp.transform_vector(transform, bitangent))
    mapped = value * 2.0 - wp.vec3f(1.0)
    mapped = wp.vec3f(mapped[0] * strength, mapped[1] * strength, mapped[2])
    return wp.normalize(t * mapped[0] + b * mapped[1] + n * mapped[2])
