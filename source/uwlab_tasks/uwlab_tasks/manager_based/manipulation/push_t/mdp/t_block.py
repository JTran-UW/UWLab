# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compound T-block spawner.

Body frame: origin at the planar area centroid of the T (= its COM), z at the block's vertical center,
+x along the top bar, +y from the stem toward the bar (the stem hangs along -y).
"""

from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.materials import PhysicsMaterialCfg, VisualMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass
from pxr import Usd


def t_part_centers(
    bar_size: tuple[float, float], stem_size: tuple[float, float], centers: tuple[float, float] | None = None
) -> tuple[float, float]:
    """(bar_center_y, stem_center_y) in the body frame: ``centers`` if given, else the centroid frame."""
    if centers is not None:
        return centers
    bar_w, bar_h = bar_size
    stem_w, stem_l = stem_size
    bar_area, stem_area = bar_w * bar_h, stem_w * stem_l
    # stem center sits (bar_h + stem_l) / 2 below the bar center
    sep = 0.5 * (bar_h + stem_l)
    bar_y = stem_area * sep / (bar_area + stem_area)
    return bar_y, bar_y - sep


@clone
def spawn_t_block(
    prim_path: str,
    cfg: TBlockCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Xform root (rigid body + mass if configured) with two Cube children (bar, stem) sharing materials."""
    stage = get_current_stage()
    assert not stage.GetPrimAtPath(prim_path).IsValid(), f"A prim already exists at path: '{prim_path}'."
    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    geom_prim_path = prim_path + "/geometry"
    create_prim(geom_prim_path, prim_type="Xform", stage=stage)
    bar_y, stem_y = t_part_centers(cfg.bar_size, cfg.stem_size, cfg.part_centers)
    parts = {
        "bar": ((0.0, bar_y, 0.0), (cfg.bar_size[0], cfg.bar_size[1], cfg.height)),
        "stem": ((0.0, stem_y, 0.0), (cfg.stem_size[0], cfg.stem_size[1], cfg.height)),
    }
    part_paths = []
    for name, (center, size) in parts.items():
        path = f"{geom_prim_path}/{name}"
        # unit cube scaled to the part size
        create_prim(path, "Cube", translation=center, scale=size, attributes={"size": 1.0}, stage=stage)
        if cfg.collision_props is not None:
            schemas.define_collision_properties(path, cfg.collision_props, stage=stage)
        part_paths.append(path)

    if cfg.visual_material is not None:
        vis_path = cfg.visual_material_path
        if not vis_path.startswith("/"):
            vis_path = f"{geom_prim_path}/{vis_path}"
        cfg.visual_material.func(vis_path, cfg.visual_material)
        for path in part_paths:
            bind_visual_material(path, vis_path, stage=stage)

    if cfg.physics_material is not None:
        assert cfg.collision_props is not None, "physics_material needs collision_props"
        phys_path = cfg.physics_material_path
        if not phys_path.startswith("/"):
            phys_path = f"{geom_prim_path}/{phys_path}"
        if not stage.GetPrimAtPath(phys_path).IsValid():
            cfg.physics_material.func(phys_path, cfg.physics_material)
        for path in part_paths:
            bind_physics_material(path, phys_path, stage=stage)

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    return stage.GetPrimAtPath(prim_path)


@configclass
class TBlockCfg(RigidObjectSpawnerCfg):
    """T-block spawner. Leave collision/rigid/mass props as None for a visual-only marker."""

    func: Callable | str = spawn_t_block

    bar_size: tuple[float, float] = (0.16, 0.04)
    """Top bar (length along x, width along y) [m]."""
    stem_size: tuple[float, float] = (0.04, 0.12)
    """Stem (width along x, length along y) [m]."""
    height: float = 0.04
    """Block height along z [m]."""
    part_centers: tuple[float, float] | None = None
    """(bar_center_y, stem_center_y) in the body frame; None: origin at the area centroid, bar toward +y."""

    visual_material_path: str = "material"
    visual_material: VisualMaterialCfg | None = None
    # relative to the block's geometry prim: Newton (Warp renderer model) only parses each env's own subtree
    physics_material_path: str = "physics_material"
    physics_material: PhysicsMaterialCfg | None = None


T_BLOCK_SPAWN = TBlockCfg(
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=1.0,
        disable_gravity=False,
    ),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6, restitution=0.0),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.1)),
)

# thin, so it lies flat on the table under the T
T_GOAL_MARKER_SPAWN = TBlockCfg(
    height=0.002,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2), opacity=0.35),
)
