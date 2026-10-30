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
from .types import IBL_ROUGHNESS_LEVELS, SceneData, FrameData, Outputs
from .shading import brdf, ggx_specular, sample_ggx, sample_environment, environment, normal_from_map


@wp.struct
class Surface:
    base: wp.vec3f
    normal: wp.vec3f
    metallic: float
    roughness: float
    specular: float
    emissive: wp.vec3f


def create_kernel(config, state, clear):
    trace_closest = raytrace.create_closest_hit_function(config, state)
    trace_first = raytrace.create_first_hit_function(config, state)
    sample_texture = textures.create_sample_texture_function(config)
    packed = int(clear.clear_color)
    background = tuple(((packed >> shift) & 255) / 255.0 for shift in (0, 8, 16))

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
                    result.base = wp.cw_mul(result.base, srgb_to_linear_wp(sample_map(scene, hit, point, texture_id)))
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
    def shade(
        scene: SceneData,
        world: int,
        hit: raytrace.ClosestHit,
        point: wp.vec3f,
        view: wp.vec3f,
        surf: Surface,
        seed: int,
    ):
        n = surf.normal
        result = surf.emissive
        if wp.static(config.enable_ambient_lighting and not config.enable_dome_light and not config.enable_precomputed_ibl):
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
                normal_u = wp.atan2(n[1], n[0]) / (2.0 * wp.pi) + 0.5 + rotation
                normal_v = wp.acos(wp.clamp(n[2], -1.0, 1.0)) / wp.pi
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
                reflected = wp.normalize(2.0 * wp.dot(n, view) * n - view)
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
        if wp.static(config.enable_dome_light):
            dome_radiance = wp.vec3f(0.0)
            if scene.environment_ids[world] >= 0:
                for dome_sample in range(wp.static(config.dome_samples)):
                    direction, pdf = sample_environment(scene, world, wp.randf(rng), wp.randf(rng), wp.randf(rng))
                    cosine = wp.max(wp.dot(n, direction), 0.0)
                    if cosine > 0.0:
                        blocked = bool(False)
                        if wp.static(config.enable_shadows):
                            blocked = trace_first(
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
                                point + hit.normal * 1.0e-4,
                                direction,
                                1.0e8,
                            )
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
                                blocked = trace_first(
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
                                    point + hit.normal * 1.0e-4,
                                    l,
                                    wp.max(distance - 1.0e-3, 1.0e-4),
                                )
                                if blocked:
                                    visibility = wp.static(config.shadow_floor)
                        response = surf.base
                        if wp.static(config.enable_pbr):
                            response = brdf(surf.base, surf.metallic, surf.roughness, surf.specular, n, view, l)
                        contribution += response * (nl * attenuation * visibility)
                radiance = scene.light_colors[slot] * scene.light_intensities[slot]
                result += wp.cw_mul(contribution, radiance) / float(wp.static(config.shadow_samples))
        return result

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
        rng = wp.rand_init(wp.static(config.sample_seed), index)
        accumulated = wp.vec3f(0.0)
        for sample in range(wp.static(config.samples_per_pixel)):
            dx, dy = float(0.0), float(0.0)
            if sample > 0:
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
                        color = shade(scene, world, hit, point, -dir_world, surf, index + sample * 7919)
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
                                    incoming = shade(scene, world, bounce, bp, -reflected, bs, index + sample * 17011)
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
        accumulated /= float(wp.static(config.samples_per_pixel))
        if wp.static(state.render_hdr_color):
            out.hdr[index] = accumulated
        if wp.static(state.render_color):
            if wp.static(config.output_color_space == ColorSpace.SRGB):
                accumulated = linear_to_srgb_wp(accumulated)
            out.color[index] = tiling.pack_rgba_to_uint32(accumulated, 1.0)

    return render
