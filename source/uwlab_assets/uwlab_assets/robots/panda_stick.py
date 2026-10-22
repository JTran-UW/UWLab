# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManiSkill ``panda_stick``: 7-DoF Panda with a 10 cm stick instead of a gripper (URDF vendored in data/)."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from uwlab_assets import UWLAB_ASSETS_DATA_DIR

PANDA_STICK_URDF = os.path.join(UWLAB_ASSETS_DATA_DIR, "Robots", "panda_stick", "panda_stick.urdf")
PANDA_STICK_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
# the importer drops massless links, so the TCP (stick tip, URDF panda_hand_tcp) is panda_hand + 15 cm along z
PANDA_STICK_TCP_BODY = "panda_hand"
PANDA_STICK_TCP_OFFSET = (0.0, 0.0, 0.15)


_ARTICULATION_ROOT_SCHEMAS = ("PhysicsArticulationRootAPI", "PhysxArticulationAPI", "NewtonArticulationRootAPI")


def _move_articulation_root_to(root, link):
    """Move the articulation root schemas and their authored attributes from the root link to ``root``.

    NewtonArticulationRootAPI carries PhysicsArticulationRootAPI as a built-in, so it has to move too.
    """
    from pxr import Usd

    applied = link.GetAppliedSchemas()
    for name in _ARTICULATION_ROOT_SCHEMAS:
        if name not in applied:
            continue
        root.AddAppliedSchema(name)
        definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(name)
        for prop_name in definition.GetPropertyNames() if definition is not None else []:
            attr = link.GetAttribute(prop_name)
            if attr and attr.HasAuthoredValue():
                root.CreateAttribute(prop_name, attr.GetTypeName()).Set(attr.Get())
        link.RemoveAppliedSchema(name)


def spawn_urdf_rigid_props_every_link(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """``spawn_from_urdf``, then ``cfg.rigid_props`` on every link: the importer nests the links, and the stock
    ``apply_nested`` modifier stops at the first rigid body (only panda_link0 got ``disable_gravity``).

    The importer also puts the articulation root on panda_link0, which PhysX parses as a floating base held by
    ``root_joint`` as a maximal-coordinate constraint; the root is moved up to the spawn prim to get a fixed base.
    """
    from pxr import Usd, UsdPhysics

    prim = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    for root in sim_utils.find_matching_prims(prim_path) or [prim]:
        for child in Usd.PrimRange(root):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                sim_utils.modify_rigid_body_properties(str(child.GetPath()), cfg.rigid_props)
                if cfg.fix_base and child.HasAPI(UsdPhysics.ArticulationRootAPI):
                    _move_articulation_root_to(root, child)
    return prim


PANDA_STICK_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_urdf_rigid_props_every_link,
        asset_path=PANDA_STICK_URDF,
        fix_base=True,
        # keep panda_hand (the TCP reference) as its own body
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1e3, damping=1e2)
        ),
        # stands in for ManiSkill's balance_passive_force (gravity compensation, on by default)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={f"panda_joint{i + 1}": q for i, q in enumerate([0.0, 0.3927, 0.0, -1.9635, 0.0, 2.3562, 0.7854])},
    ),
    actuators={
        # ManiSkill PandaStick arm_stiffness / arm_damping / arm_force_limit
        "arm": ImplicitActuatorCfg(
            joint_names_expr=PANDA_STICK_ARM_JOINTS,
            stiffness=1e3,
            damping=1e2,
            effort_limit_sim=100.0,
            # SAPIEN never sets the URDF velocity limits; 100 rad/s is the PhysX default max joint velocity
            velocity_limit_sim=100.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
