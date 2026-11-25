# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) 2026 The UW Lab Project Developers
# SPDX-License-Identifier: Apache-2.0
"""Extended Newton 1.5.2 render factory, derived from warp_raytrace/render.py.

The BVH queries and texture sampler remain Newton's version-pinned implementations.
No modules in the installed Newton package are patched.
"""

import warp as wp
from newton._src.geometry import GeoType
from newton._src.sensors.warp_raytrace import raytrace, textures, tiling
from newton._src.sensors.warp_raytrace.types import RenderOrder
from newton._src.utils.color import ColorSpace, srgb_to_linear_wp, linear_to_srgb_wp
from .types import IBL_ROUGHNESS_LEVELS, SUN_SLOTS, SceneData, FrameData, Outputs
from .shading import basis, brdf, ggx_specular, sample_ggx, sample_environment, environment, normal_from_map


@wp.struct
class Surface:
    base: wp.vec3f
    normal: wp.vec3f
    metallic: float
    roughness: float
    specular: float
    emissive: wp.vec3f


@wp.struct
class Shaded:
    color: wp.vec3f  # outgoing radiance
    light: wp.vec3f  # path lighting: diffuse light before the surface colour
    gloss: wp.vec3f  # path lighting: specular radiance
    tint: wp.vec3f  # path lighting: the diffuse colour that multiplies ``light``


@wp.func
def path_tint(surf: Surface, n_dot_v: float):
    """Diffuse colour under the OmniPBR coat: what multiplies the diffuse light of path lighting."""
    coat = surf.specular * (0.08 + 0.92 * wp.pow(wp.clamp(1.0 - n_dot_v, 0.0, 1.0), 5.0))
    return surf.base * ((1.0 - surf.metallic) * (1.0 - coat))


def _create_warp_trace_functions(config, state):
    trace_closest = raytrace.create_closest_hit_function(config, state)
    trace_first = raytrace.create_first_hit_function(config, state)

    @wp.func
    def trace(scene: SceneData, world: int, origin: wp.vec3f, direction: wp.vec3f, forward: wp.vec3f):
        return trace_closest(
            scene.bvh_shapes_size,
            scene.bvh_shapes_id,
            scene.bvh_shapes_group_roots,
            scene.bvh_particles_size,
            scene.bvh_particles_id,
            scene.bvh_particles_group_roots,
            world,
            wp.static(config.max_distance),
            scene.shape_enabled,
            scene.shape_types,
            scene.shape_sizes,
            scene.shape_transforms,
            scene.shape_source_ptr,
            scene.shape_mesh_data_ids,
            scene.mesh_data,
            scene.particles_position,
            scene.particles_radius,
            scene.topology_particle_mask,
            scene.triangle_mesh_id,
            scene.triangle_mesh_group_roots,
            scene.gaussians_data,
            origin,
            direction,
            forward,
        )

    @wp.func
    def occluded(scene: SceneData, world: int, origin: wp.vec3f, direction: wp.vec3f, max_distance: float):
        return trace_first(
            scene.bvh_shapes_size,
            scene.bvh_shapes_id,
            scene.bvh_shapes_group_roots,
            scene.bvh_particles_size,
            scene.bvh_particles_id,
            scene.bvh_particles_group_roots,
            world,
            scene.shape_enabled,
            scene.shape_types,
            scene.shape_sizes,
            scene.shape_transforms,
            scene.shape_source_ptr,
            scene.particles_position,
            scene.particles_radius,
            scene.topology_particle_mask,
            scene.triangle_mesh_id,
            scene.triangle_mesh_group_roots,
            origin,
            direction,
            max_distance,
        )

    return trace, occluded


def create_kernel(config, state, clear, optix=None):
    """Build the render kernel; ``optix`` (an ``OptixShapes``) swaps the Warp BVH traversal for OptiX."""
    if optix is None:
        trace, occluded = _create_warp_trace_functions(config, state)
    else:
        from . import optix as optix_backend

        trace = optix_backend.create_trace_function(config, state, optix)
        occluded = optix_backend.create_occluded_function(config, state, optix)
    sample_texture = textures.create_sample_texture_function(config)
    packed = int(clear.clear_color)
    background = tuple(((packed >> shift) & 255) / 255.0 for shift in (0, 8, 16))

    @wp.func
    def sample_map(scene: SceneData, hit: raytrace.ClosestHit, point: wp.vec3f, texture_id: int):
        i = int(hit.shape_index)
        return sample_texture(
            scene.shape_types[i],
            scene.shape_transforms[i],
            scene.texture_data,
            texture_id,
            scene.shape_source_ptr[i],
            scene.mesh_data,
            scene.shape_mesh_data_ids[i],
            point,
            hit.normal,
            hit.bary_u,
            hit.bary_v,
            hit.face_idx,
        )

    @wp.func
    def sample_albedo(scene: SceneData, hit: raytrace.ClosestHit, point: wp.vec3f, texture_id: int, uv_scale: float):
        """The shape's albedo texel: its own UV mapping, or with ``uv_scale`` > 0 a cubic projection of the shape's
        local position [m] times that scale (OmniPBR ``project_uvw``)."""
        if uv_scale <= 0.0:
            return sample_map(scene, hit, point, texture_id)
        inverse = wp.transform_inverse(scene.shape_transforms[int(hit.shape_index)])
        local = wp.transform_point(inverse, point)
        axis = wp.transform_vector(inverse, hit.normal)
        ax, ay, az = wp.abs(axis[0]), wp.abs(axis[1]), wp.abs(axis[2])
        uv = wp.vec2f(local[0], local[1])
        if ax >= ay and ax >= az:
            uv = wp.vec2f(local[1], local[2])
        elif ay >= az:
            uv = wp.vec2f(local[0], local[2])
        texel = wp.texture_sample(
            scene.texture_data[texture_id].texture, wp.vec2f(uv[0] * uv_scale, 1.0 - uv[1] * uv_scale), dtype=wp.vec4f
        )
        return wp.vec3f(texel[0], texel[1], texel[2])

    @wp.func
    def surface(scene: SceneData, hit: raytrace.ClosestHit, point: wp.vec3f):
        result = Surface()
        result.base = wp.vec3f(1.0)
        result.normal = wp.normalize(hit.normal)
        result.roughness = 0.5
        result.specular = 0.5
        if hit.shape_index < raytrace.MAX_SHAPE_ID:
            i = int(hit.shape_index)
            result.base = srgb_to_linear_wp(scene.shape_colors[i])
            material = scene.materials[i]
            result.metallic = material.metallic
            result.roughness = material.roughness
            result.specular = material.specular
            result.emissive = material.emissive
            if wp.static(config.enable_textures):
                texture_id = scene.shape_texture_ids[i]
                if texture_id >= 0:
                    texel = sample_albedo(scene, hit, point, texture_id, material.uv_scale)
                    result.base = wp.cw_mul(result.base, srgb_to_linear_wp(texel))
                if material.roughness_map >= 0:
                    result.roughness = wp.clamp(
                        result.roughness * sample_map(scene, hit, point, material.roughness_map)[0], 0.05, 1.0
                    )
                if material.emissive_map >= 0:
                    result.emissive = wp.cw_mul(
                        result.emissive, srgb_to_linear_wp(sample_map(scene, hit, point, material.emissive_map))
                    )
                if material.normal_map >= 0:
                    result.normal = normal_from_map(
                        scene, hit, point, sample_map(scene, hit, point, material.normal_map), material.normal_strength
                    )
        return result

    @wp.func
    def dome_radiance(scene: SceneData, world: int, direction: wp.vec3f):
        """The dome seen along ``direction``: the uniform part plus the environment map, clamped."""
        sky = environment(scene, world, direction, wp.vec3f(0.0))
        if wp.static(config.path_sky_clamp > 0.0):
            limit = wp.static(config.path_sky_clamp) * scene.environment_intensity[world]
            sky = wp.vec3f(wp.min(sky[0], limit), wp.min(sky[1], limit), wp.min(sky[2], limit))
        return scene.environment_ambient[world] + sky

    @wp.func
    def path_radiance(scene: SceneData, world: int, origin: wp.vec3f, direction: wp.vec3f, u1: float, u2: float):
        """Radiance arriving along a secondary ray: the dome, or one diffuse bounce off the surface it hits, lit by
        the dome along one cosine-weighted direction (``u1``, ``u2``) if that direction is free."""
        bounce = trace(scene, world, origin, direction, direction)
        if bounce.shape_index == raytrace.NO_HIT_SHAPE_ID:
            return dome_radiance(scene, world, direction)
        normal = wp.normalize(bounce.normal)
        if wp.dot(normal, direction) > 0.0:
            normal = -normal
        tangent, bitangent = basis(normal)
        radius = wp.sqrt(u1)
        phi = 2.0 * wp.pi * u2
        onward = wp.normalize(
            tangent * (radius * wp.cos(phi)) + bitangent * (radius * wp.sin(phi)) + normal * wp.sqrt(wp.max(1.0 - u1, 0.0))
        )
        point = origin + direction * bounce.distance
        if occluded(scene, world, point + normal * 1.0e-4, onward, 1.0e8):
            return wp.vec3f(0.0)
        base = wp.vec3f(1.0)
        metallic = float(0.0)
        if bounce.shape_index < raytrace.MAX_SHAPE_ID:
            i = int(bounce.shape_index)
            base = srgb_to_linear_wp(scene.shape_colors[i])
            metallic = scene.materials[i].metallic
            if wp.static(config.enable_textures):
                texture_id = scene.shape_texture_ids[i]
                if texture_id >= 0:
                    texel = sample_albedo(scene, bounce, point, texture_id, scene.materials[i].uv_scale)
                    base = wp.cw_mul(base, srgb_to_linear_wp(texel))
        return wp.cw_mul(base, dome_radiance(scene, world, onward)) * ((1.0 - metallic) * wp.static(config.path_bounce))

    @wp.func
    def shade(
        scene: SceneData,
        world: int,
        hit: raytrace.ClosestHit,
        point: wp.vec3f,
        view: wp.vec3f,
        surf: Surface,
        seed: int,
        stratum: int,
    ):
        n = surf.normal
        shaded = Shaded()
        result = surf.emissive
        if wp.static(
            config.enable_ambient_lighting
            and not config.enable_dome_light
            and not config.enable_precomputed_ibl
            and config.path_samples == 0
        ):
            h = 0.5 * (n[2] + 1.0)
            sky = wp.vec3f(
                wp.static(config.ambient_sky[0]), wp.static(config.ambient_sky[1]), wp.static(config.ambient_sky[2])
            )
            ground = wp.vec3f(
                wp.static(config.ambient_ground[0]),
                wp.static(config.ambient_ground[1]),
                wp.static(config.ambient_ground[2]),
            )
            ambient = environment(scene, world, n, wp.lerp(ground, sky, h)) * wp.static(config.ambient_intensity)
            diffuse_weight = float(1.0)
            if wp.static(config.enable_pbr):
                diffuse_weight = 1.0 - surf.metallic
            result += wp.cw_mul(surf.base, ambient) * diffuse_weight
        if wp.static(config.enable_precomputed_ibl):
            map_id = scene.environment_ids[world]
            if map_id >= 0:
                rotation = scene.environment_rotation[world]
                n_map = wp.quat_rotate_inv(scene.environment_orientation[world], n)
                normal_u = wp.atan2(n_map[1], n_map[0]) / (2.0 * wp.pi) + 0.5 + rotation
                normal_v = wp.acos(wp.clamp(n_map[2], -1.0, 1.0)) / wp.pi
                irradiance = wp.texture_sample(
                    scene.ibl_diffuse_data[map_id].texture,
                    wp.vec2f(normal_u, normal_v),
                    dtype=wp.vec4f,
                )
                f0 = wp.lerp(wp.vec3f(0.08 * surf.specular), surf.base, surf.metallic)
                nv = wp.clamp(wp.dot(n, view), 0.0, 1.0)
                fresnel_view = f0 + (wp.vec3f(1.0) - f0) * wp.pow(1.0 - nv, 5.0)
                diffuse_weight = (wp.vec3f(1.0) - fresnel_view) * (1.0 - surf.metallic)
                diffuse = (
                    wp.cw_mul(
                        wp.cw_mul(surf.base, diffuse_weight),
                        wp.vec3f(irradiance[0], irradiance[1], irradiance[2]),
                    )
                    / wp.pi
                )
                reflected = wp.quat_rotate_inv(scene.environment_orientation[world], wp.normalize(2.0 * wp.dot(n, view) * n - view))
                reflected_u = wp.atan2(reflected[1], reflected[0]) / (2.0 * wp.pi) + 0.5 + rotation
                reflected_v = wp.acos(wp.clamp(reflected[2], -1.0, 1.0)) / wp.pi
                level = wp.clamp(surf.roughness, 0.0, 1.0) * float(wp.static(IBL_ROUGHNESS_LEVELS - 1))
                lower = int(wp.floor(level))
                upper = wp.min(lower + 1, wp.static(IBL_ROUGHNESS_LEVELS - 1))
                lower_rgb = wp.texture_sample(
                    scene.ibl_specular_data[map_id * wp.static(IBL_ROUGHNESS_LEVELS) + lower].texture,
                    wp.vec2f(reflected_u, reflected_v),
                    dtype=wp.vec4f,
                )
                upper_rgb = wp.texture_sample(
                    scene.ibl_specular_data[map_id * wp.static(IBL_ROUGHNESS_LEVELS) + upper].texture,
                    wp.vec2f(reflected_u, reflected_v),
                    dtype=wp.vec4f,
                )
                prefiltered = wp.lerp(
                    wp.vec3f(lower_rgb[0], lower_rgb[1], lower_rgb[2]),
                    wp.vec3f(upper_rgb[0], upper_rgb[1], upper_rgb[2]),
                    level - float(lower),
                )
                dfg = wp.texture_sample(
                    scene.ibl_brdf_data[0].texture,
                    wp.vec2f(nv, surf.roughness),
                    dtype=wp.vec4f,
                )
                result += (
                    diffuse + wp.cw_mul(prefiltered, f0 * dfg[0] + wp.vec3f(dfg[1]))
                ) * scene.environment_intensity[world]
        rng = wp.rand_init(seed)
        if wp.static(config.path_samples > 0):
            # One-bounce path lighting under the uniform dome (environment_ambient) plus the HDRI, with the OmniPBR
            # layering: diffuse under a Schlick-weighted GGX coat (0.08 * specular at normal incidence), blended
            # with a base-tinted GGX lobe by metallic.
            nv = wp.max(wp.dot(n, view), 1.0e-4)
            diffuse_weight = path_tint(surf, nv)
            alpha = wp.max(surf.roughness * surf.roughness, 0.0025)
            a2 = alpha * alpha
            gv = 2.0 * nv / (nv + wp.sqrt(a2 + (1.0 - a2) * nv * nv))
            origin = point + hit.normal * 1.0e-4
            tangent, bitangent = basis(n)
            diffuse_light = wp.vec3f(0.0)
            specular_light = wp.vec3f(0.0)
            sharp_light = wp.vec3f(0.0)
            view_local = wp.vec3f(wp.dot(view, tangent), wp.dot(view, bitangent), nv)
            stretched = wp.normalize(wp.vec3f(alpha * view_local[0], alpha * view_local[1], view_local[2]))
            # One trace site in a runtime-sized loop (an unrolled loop would multiply the trace call sites OptiX
            # compiles): even iterations sample the diffuse lobe, odd ones the GGX lobe. Interleaved sampling: the
            # pixels of an 8x8 tile take the 64 cells of a stratification of the sample square (``stratum``: 4x4
            # cell * 4 + 2x2 sub-cell, every 4x4 block holding all 16 cells), jittered within the cell, so the
            # 15-tap tent of ``postprocess.filter_path_lighting`` integrates every cell with equal weight. A
            # negative ``stratum`` draws plain random samples.
            for path_sample in range(2 * scene.path_samples):
                direction = n
                throughput = wp.vec3f(0.0)
                glossy = (path_sample % 2) == 1
                u1 = wp.randf(rng)
                u2 = wp.randf(rng)
                if stratum >= 0:
                    cell = ((stratum // 4) + (path_sample // 2) * 5) % 16
                    sub = stratum % 4
                    if glossy:
                        cell = (cell * 7 + 3) % 16
                    u1 = (float(cell % 4) + (float(sub % 2) + u1) * 0.5) * 0.25
                    u2 = (float(cell // 4) + (float(sub // 2) + u2) * 0.5) * 0.25
                if glossy:
                    # visible-normal sampling (spherical caps): the weight is the layer tint times G1(l)
                    phi = 2.0 * wp.pi * u1
                    z = (1.0 - u2) * (1.0 + stretched[2]) - stretched[2]
                    sine = wp.sqrt(wp.clamp(1.0 - z * z, 0.0, 1.0))
                    cap = wp.vec3f(sine * wp.cos(phi), sine * wp.sin(phi), z) + stretched
                    m = wp.normalize(wp.vec3f(alpha * cap[0], alpha * cap[1], wp.max(cap[2], 1.0e-6)))
                    h = wp.normalize(tangent * m[0] + bitangent * m[1] + n * m[2])
                    vh = wp.dot(view, h)
                    l = wp.normalize(2.0 * vh * h - view)
                    nl = wp.dot(n, l)
                    if nl > 0.0 and vh > 0.0 and wp.dot(l, hit.normal) > 0.0:
                        gl = 2.0 * nl / (nl + wp.sqrt(a2 + (1.0 - a2) * nl * nl))
                        curve = 0.08 + 0.92 * wp.pow(wp.clamp(1.0 - vh, 0.0, 1.0), 5.0)
                        layer = (1.0 - surf.metallic) * surf.specular * curve
                        throughput = (wp.vec3f(layer) + surf.base * surf.metallic) * gl
                        direction = l
                else:
                    phi = 2.0 * wp.pi * u2
                    radius = wp.sqrt(u1)
                    direction = wp.normalize(
                        tangent * (radius * wp.cos(phi))
                        + bitangent * (radius * wp.sin(phi))
                        + n * wp.sqrt(wp.max(1.0 - u1, 0.0))
                    )
                    if wp.dot(direction, hit.normal) > 0.0:
                        throughput = wp.vec3f(1.0)
                if throughput[0] + throughput[1] + throughput[2] > 0.0:
                    radiance = wp.cw_mul(
                        throughput, path_radiance(scene, world, origin, direction, wp.randf(rng), wp.randf(rng))
                    )
                    if glossy:
                        if surf.roughness < 0.05:
                            sharp_light += radiance
                        else:
                            specular_light += radiance
                    else:
                        diffuse_light += radiance
            if wp.static(config.path_sun and config.path_sky_clamp > 0.0):
                env = scene.environment_ids[world]
                if env >= 0:
                    turn = -2.0 * wp.pi * scene.environment_rotation[world]
                    for sun in range(wp.static(SUN_SLOTS)):
                        sun_color = scene.environment_sun_color[env * wp.static(SUN_SLOTS) + sun]
                        if sun_color[0] + sun_color[1] + sun_color[2] > 0.0:
                            sd = scene.environment_sun_direction[env * wp.static(SUN_SLOTS) + sun]
                            d = wp.vec3f(wp.cos(turn) * sd[0] - wp.sin(turn) * sd[1], wp.sin(turn) * sd[0] + wp.cos(turn) * sd[1], sd[2])
                            d = wp.quat_rotate(scene.environment_orientation[world], d)
                            nl = wp.dot(n, d)
                            if nl > 0.0 and wp.dot(d, hit.normal) > 0.0:
                                if not occluded(scene, world, origin, d, 1.0e8):
                                    flux = sun_color * (scene.environment_intensity[world] * nl * float(scene.path_samples))
                                    sharp_light += wp.cw_mul(diffuse_weight, flux) / wp.pi  # unfiltered: crisp shadows
                                    h = wp.normalize(view + d)
                                    nh = wp.max(wp.dot(n, h), 0.0)
                                    denom = nh * nh * (a2 - 1.0) + 1.0
                                    gl = 2.0 * nl / (nl + wp.sqrt(a2 + (1.0 - a2) * nl * nl))
                                    curve = 0.08 + 0.92 * wp.pow(wp.clamp(1.0 - wp.dot(view, h), 0.0, 1.0), 5.0)
                                    layer = (1.0 - surf.metallic) * surf.specular * curve
                                    lobe = a2 / (wp.pi * denom * denom) * gv * gl / (4.0 * nv * nl)
                                    sharp_light += wp.cw_mul(wp.vec3f(layer) + surf.base * surf.metallic, flux) * lobe
            shaded.light = diffuse_light / float(scene.path_samples)
            shaded.gloss = specular_light / float(scene.path_samples)
            shaded.tint = diffuse_weight
            # mirror-like lobes stay out of the filtered buffers; the filtered ones are added after filtering
            result += sharp_light / float(scene.path_samples)
            if wp.static(not config.path_outputs):
                result += wp.cw_mul(shaded.tint, shaded.light) + shaded.gloss
        if wp.static(config.enable_dome_light):
            dome_radiance = wp.vec3f(0.0)
            if scene.environment_ids[world] >= 0:
                for dome_sample in range(wp.static(config.dome_samples)):
                    direction, pdf = sample_environment(scene, world, wp.randf(rng), wp.randf(rng), wp.randf(rng))
                    cosine = wp.max(wp.dot(n, direction), 0.0)
                    if cosine > 0.0:
                        blocked = bool(False)
                        if wp.static(config.enable_shadows):
                            blocked = occluded(scene, world, point + hit.normal * 1.0e-4, direction, 1.0e8)
                        if not blocked:
                            response = surf.base / wp.pi
                            if wp.static(config.enable_pbr):
                                response = brdf(
                                    surf.base, surf.metallic, surf.roughness, surf.specular, n, view, direction
                                )
                            incoming = environment(scene, world, direction, wp.vec3f(0.0))
                            dome_radiance += wp.cw_mul(response, incoming) * (cosine / wp.max(pdf, 1.0e-8))
            result += dome_radiance / float(wp.static(config.dome_samples))
        for light in range(scene.light_count):
            slot = light
            if wp.static(config.per_world_lights):
                slot = world * scene.light_count + light
            if scene.light_active[slot]:
                contribution = wp.vec3f(0.0)
                for shadow_sample in range(wp.static(config.shadow_samples)):
                    position = scene.light_positions[slot]
                    orientation = scene.light_orientations[slot]
                    radius = scene.light_radii[slot]
                    if radius > 0.0:
                        z = 2.0 * wp.randf(rng) - 1.0
                        phi = 2.0 * wp.pi * wp.randf(rng)
                        r = wp.sqrt(wp.max(0.0, 1.0 - z * z))
                        jitter = wp.vec3f(r * wp.cos(phi), r * wp.sin(phi), z) * radius
                        if scene.light_type[slot] == 1:
                            orientation = wp.normalize(orientation + jitter)
                        else:
                            position += jitter
                    distance = float(1.0e8)
                    attenuation = float(1.0)
                    l = wp.normalize(-orientation)
                    if scene.light_type[slot] != 1:
                        to_light = position - point
                        distance = wp.length(to_light)
                        l = to_light / wp.max(distance, 1.0e-6)
                        attenuation = 1.0 / (1.0 + wp.static(config.distance_attenuation) * distance * distance)
                        if scene.light_type[slot] == 0:
                            cos_theta = wp.dot(-l, wp.normalize(orientation))
                            attenuation *= wp.clamp(
                                (cos_theta - wp.static(config.spot_outer))
                                / wp.static(config.spot_inner - config.spot_outer),
                                0.0,
                                1.0,
                            )
                    nl = wp.max(0.0, wp.dot(n, l))
                    if nl > 0.0:
                        visibility = float(1.0)
                        if wp.static(config.enable_shadows):
                            if scene.light_cast_shadow[slot]:
                                blocked = occluded(
                                    scene, world, point + hit.normal * 1.0e-4, l, wp.max(distance - 1.0e-3, 1.0e-4)
                                )
                                if blocked:
                                    visibility = wp.static(config.shadow_floor)
                        response = surf.base
                        if wp.static(config.enable_pbr):
                            response = brdf(surf.base, surf.metallic, surf.roughness, surf.specular, n, view, l)
                        contribution += response * (nl * attenuation * visibility)
                radiance = scene.light_colors[slot] * scene.light_intensities[slot]
                result += wp.cw_mul(contribution, radiance) / float(wp.static(config.shadow_samples))
        shaded.color = result
        return shaded

    @wp.kernel(enable_backward=False, module="unique", module_options={"fast_math": config.enable_fast_math})
    def render(scene: SceneData, frame: FrameData, out: Outputs):
        tid = wp.tid()
        world, camera, y, x = int(0), int(0), int(0), int(0)
        if wp.static(config.render_order == RenderOrder.PIXEL_PRIORITY):
            world, camera, y, x = tiling.tid_to_coord_pixel_priority(
                tid, scene.world_count, frame.camera_count, frame.width
            )
        elif wp.static(config.render_order == RenderOrder.VIEW_PRIORITY):
            world, camera, y, x = tiling.tid_to_coord_view_priority(tid, frame.camera_count, frame.width, frame.height)
        else:
            world, camera, y, x = tiling.tid_to_coord_tiled(
                tid,
                frame.camera_count,
                frame.width,
                frame.height,
                wp.static(config.tile_width),
                wp.static(config.tile_height),
            )
        if x >= frame.width or y >= frame.height:
            return
        index = (world * frame.camera_count + camera) * frame.width * frame.height + y * frame.width + x
        rng = wp.rand_init(wp.static(config.sample_seed) + frame.seed * 9781, index)
        accumulated = wp.vec3f(0.0)
        light = wp.vec3f(0.0)
        gloss = wp.vec3f(0.0)
        tint = wp.vec3f(0.0)
        cover = float(0.0)
        lit_shape = wp.uint32(raytrace.NO_HIT_SHAPE_ID)
        lit_color = wp.vec3f(0.0)
        lit_sky = float(1.0)
        for sample in range(wp.static(config.samples_per_pixel)):
            dx, dy = float(0.0), float(0.0)
            if sample > 0 or frame.seed > 0:
                dx, dy = wp.randf(rng) - 0.5, wp.randf(rng) - 0.5
            ray_origin = wp.vec3f(0.0)
            direction = frame.rays[camera, y, x, 1]
            if frame.intrinsics.shape[0] > 0:
                k_index = camera * scene.world_count
                if wp.static(config.per_world_intrinsics):
                    k_index += world
                k = frame.intrinsics[k_index]
                cy = (float(y) + 0.5 + dy - k[1, 2]) / k[1, 1]
                cx = (float(x) + 0.5 + dx - k[0, 2] - k[0, 1] * cy) / k[0, 0]
                direction = wp.normalize(wp.vec3f(cx, -cy, -1.0))
            if wp.static(config.aperture_radius > 0.0):
                if sample > 0:
                    radius = wp.sqrt(wp.randf(rng)) * wp.static(config.aperture_radius)
                    angle = 2.0 * wp.pi * wp.randf(rng)
                    ray_origin = wp.vec3f(radius * wp.cos(angle), radius * wp.sin(angle), 0.0)
                    focus = direction * (wp.static(config.focus_distance) / wp.max(-direction[2], 1.0e-6))
                    direction = wp.normalize(focus - ray_origin)
            transform = frame.transforms[camera, world]
            if wp.static(config.shutter_fraction > 0.0):
                if sample > 0:
                    t = 1.0 - wp.randf(rng) * wp.static(config.shutter_fraction)
                    previous = frame.previous_transforms[camera, world]
                    position = wp.lerp(
                        wp.transform_get_translation(previous), wp.transform_get_translation(transform), t
                    )
                    rotation = wp.quat_slerp(
                        wp.transform_get_rotation(previous), wp.transform_get_rotation(transform), t
                    )
                    transform = wp.transformf(position, rotation)
            origin_world = wp.transform_point(transform, ray_origin)
            dir_world = wp.transform_vector(transform, direction)
            forward = wp.transform_vector(transform, wp.vec3f(0.0, 0.0, -1.0))
            hit = trace(scene, world, origin_world, dir_world, forward)
            fallback = srgb_to_linear_wp(
                wp.vec3f(wp.static(background[0]), wp.static(background[1]), wp.static(background[2]))
            )
            color = environment(scene, world, dir_world, fallback)
            base = wp.vec3f(0.0)
            if hit.shape_index != raytrace.NO_HIT_SHAPE_ID:
                color = hit.color
                is_gaussian = bool(False)
                if wp.static(state.num_gaussians > 0):
                    if hit.shape_index < raytrace.MAX_SHAPE_ID:
                        is_gaussian = scene.shape_types[hit.shape_index] == GeoType.GAUSSIAN
                if not is_gaussian:
                    point = origin_world + dir_world * hit.distance
                    surf = surface(scene, hit, point)
                    base = surf.base
                    if wp.static(state.render_color or state.render_hdr_color):
                        # ordered-dither index of the pixel (4x4 cell, then 2x2 sub-cell of the 4x4 block);
                        # accumulation frames sample at random
                        stratum = 4 * (((x ^ y) & 1) * 2 + (y & 1)) + (((x >> 1) ^ (y >> 1)) & 1) * 2 + ((y >> 1) & 1)
                        stratum = (stratum + sample * 3) % 16
                        stratum = 4 * stratum + (((x >> 2) ^ (y >> 2)) & 1) * 2 + ((y >> 2) & 1)
                        if frame.seed > 0:
                            stratum = -1
                        primary = shade(
                            scene,
                            world,
                            hit,
                            point,
                            -dir_world,
                            surf,
                            index + sample * 7919 + frame.seed * 104729,
                            stratum,
                        )
                        color = primary.color
                        cover += 1.0
                        lit_shape = hit.shape_index
                        lit_color = primary.color
                        lit_sky = wp.max(1.0 + surf.normal[2], 0.25)
                        light += primary.light
                        gloss += primary.gloss
                        tint += primary.tint
                        if wp.static(config.enable_reflections):
                            reflected, pdf = sample_ggx(
                                surf.normal, -dir_world, surf.roughness, wp.randf(rng), wp.randf(rng)
                            )
                            nl = wp.max(wp.dot(surf.normal, reflected), 0.0)
                            if nl > 0.0:
                                bounce = trace(scene, world, point + hit.normal * 1.0e-4, reflected, forward)
                                incoming = environment(scene, world, reflected, fallback)
                                if bounce.shape_index != raytrace.NO_HIT_SHAPE_ID:
                                    bp = point + hit.normal * 1.0e-4 + reflected * bounce.distance
                                    bs = surface(scene, bounce, bp)
                                    secondary = shade(
                                        scene, world, bounce, bp, -reflected, bs, index + sample * 17011 + frame.seed * 104729, -1
                                    )
                                    incoming = secondary.color
                                weight = ggx_specular(
                                    surf.base,
                                    surf.metallic,
                                    surf.roughness,
                                    surf.specular,
                                    surf.normal,
                                    -dir_world,
                                    reflected,
                                ) * (nl / wp.max(pdf, 1.0e-6))
                                color += wp.cw_mul(weight, incoming)
            if sample == 0:
                depth = wp.static(float(clear.clear_depth))
                normal = wp.vec3f(
                    wp.static(clear.clear_normal[0]), wp.static(clear.clear_normal[1]), wp.static(clear.clear_normal[2])
                )
                if hit.shape_index != raytrace.NO_HIT_SHAPE_ID:
                    depth = float(hit.distance)
                    normal = wp.vec3f(hit.normal)
                if wp.static(state.render_depth):
                    out.depth[index] = depth
                if wp.static(state.render_forward_depth):
                    out.forward_depth[index] = depth * wp.max(0.0, -direction[2])
                if wp.static(state.render_normal):
                    out.normal[index] = normal
                if wp.static(state.render_shape_index):
                    shape = wp.static(wp.uint32(clear.clear_shape_index))
                    if hit.shape_index != raytrace.NO_HIT_SHAPE_ID:
                        shape = hit.shape_index
                    out.shape_index[index] = shape
                if wp.static(state.render_albedo):
                    packed_base = base
                    if wp.static(config.output_color_space == ColorSpace.SRGB):
                        packed_base = linear_to_srgb_wp(base)
                    albedo = tiling.pack_rgba_to_uint32(packed_base, 1.0)
                    if hit.shape_index == raytrace.NO_HIT_SHAPE_ID:
                        albedo = wp.static(wp.uint32(clear.clear_albedo))
                    out.albedo[index] = albedo
            accumulated += color
        samples = float(wp.static(config.samples_per_pixel))
        if wp.static(config.path_outputs):
            # Edge anti-aliasing: extra primary samples on a ring around the centre sample (mirrored on every other
            # pixel) average the surface tint, the unlit radiance and the coverage. They trace no lighting rays
            # (shading inside this loop costs the OptiX program several times its speed): each takes the pixel's
            # filtered light, scaled by the share of the sky its normal faces relative to the centre sample's.
            # One trace site in a runtime loop.
            if frame.intrinsics.shape[0] > 0:
                for aa_sample in range(scene.path_aa - 1):
                    aa_k_index = camera * scene.world_count
                    if wp.static(config.per_world_intrinsics):
                        aa_k_index += world
                    aa_k = frame.intrinsics[aa_k_index]
                    aa_dx = wp.randf(rng)
                    aa_dy = wp.randf(rng)
                    if frame.seed == 0:
                        aa_turn = (float(aa_sample) + 0.125 + 0.5 * float((x + y) & 1)) / float(scene.path_aa - 1)
                        aa_dx = 0.5 + 0.4 * wp.cos(2.0 * wp.pi * aa_turn)
                        aa_dy = 0.5 + 0.4 * wp.sin(2.0 * wp.pi * aa_turn)
                    aa_y = (float(y) + aa_dy - aa_k[1, 2]) / aa_k[1, 1]
                    aa_x = (float(x) + aa_dx - aa_k[0, 2] - aa_k[0, 1] * aa_y) / aa_k[0, 0]
                    aa_transform = frame.transforms[camera, world]
                    aa_origin = wp.transform_point(aa_transform, wp.vec3f(0.0))
                    aa_direction = wp.transform_vector(aa_transform, wp.normalize(wp.vec3f(aa_x, -aa_y, -1.0)))
                    aa_forward = wp.transform_vector(aa_transform, wp.vec3f(0.0, 0.0, -1.0))
                    aa_hit = trace(scene, world, aa_origin, aa_direction, aa_forward)
                    if aa_hit.shape_index != raytrace.NO_HIT_SHAPE_ID:
                        aa_surf = surface(scene, aa_hit, aa_origin + aa_direction * aa_hit.distance)
                        aa_tint = path_tint(aa_surf, wp.max(wp.dot(aa_surf.normal, -aa_direction), 1.0e-4))
                        cover += 1.0
                        if aa_hit.shape_index == lit_shape:
                            accumulated += lit_color
                        else:
                            accumulated += aa_surf.emissive
                        tint += aa_tint * wp.min(wp.max(1.0 + aa_surf.normal[2], 0.25) / lit_sky, 2.0)
                    else:
                        aa_background = srgb_to_linear_wp(
                            wp.vec3f(wp.static(background[0]), wp.static(background[1]), wp.static(background[2]))
                        )
                        accumulated += environment(scene, world, aa_direction, aa_background)
                    samples += 1.0
            out.light[index] = light / float(wp.static(config.samples_per_pixel))
            out.gloss[index] = gloss / float(wp.static(config.samples_per_pixel))
            out.tint[index] = tint / samples
            out.cover[index] = cover / samples
        accumulated /= samples
        if wp.static(state.render_hdr_color):
            out.hdr[index] = accumulated
        if wp.static(state.render_color):
            if wp.static(config.output_color_space == ColorSpace.SRGB):
                accumulated = linear_to_srgb_wp(accumulated)
            out.color[index] = tiling.pack_rgba_to_uint32(accumulated, 1.0)

    return render
