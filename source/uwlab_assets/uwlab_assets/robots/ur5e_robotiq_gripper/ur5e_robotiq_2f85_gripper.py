# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5e + Robotiq 2F-85 robot.

The following configurations are available:

* :obj:`UR5E_ARTICULATION`: Base articulation (USD, init state).
* :obj:`EXPLICIT_UR5E_ROBOTIQ_2F85`: Full robot with DelayedPDActuator arm (PD delay, for sim2real finetuning).
* :obj:`IMPLICIT_UR5E_ROBOTIQ_2F85`: Full robot with ImplicitActuator arm (no motor delay, for RL training).
* :obj:`UR5E_ROBOTIQ_2F85`: Alias for ``EXPLICIT_UR5E_ROBOTIQ_2F85`` (backward compatibility).
"""

import logging

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_NEWTON_ASSETS_DIR

logger = logging.getLogger(__name__)

ROBOTIQ_2F85_DEFAULT_JOINT_POS = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_inner_knuckle_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
}


def spawn_ur5e_without_physics_materials(prim_path, cfg, translation=None, orientation=None):
    """Spawn the UR5e USD, then strip its two authored physics materials and their bindings.

    PhysX 110 allocates a PxMaterial per env per USD material (107 shared them), so the pair hits the
    hard 64K cap at 32768 envs; ``robot_material`` overwrites the finger friction at startup anyway.
    """
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    material_apis = [UsdPhysics.MaterialAPI]
    try:
        from pxr import PhysxSchema

        material_apis.append(PhysxSchema.PhysxMaterialAPI)
    except ImportError:  # kitless runtime
        pass

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)
    roots = sim_utils.find_matching_prims(prim_path) or [prim]
    removed: list[str] = []
    for root in roots:
        stage = root.GetStage()
        material_paths = {
            str(child.GetPath())
            for child in Usd.PrimRange(root)
            if any(child.HasAPI(api) for api in material_apis)
        }
        if not material_paths:
            continue
        for child in Usd.PrimRange(root):
            if not child.HasAPI(UsdShade.MaterialBindingAPI):
                continue
            binding_api = UsdShade.MaterialBindingAPI(child)
            rel = binding_api.GetDirectBindingRel("physics")
            if rel and any(str(target) in material_paths for target in rel.GetTargets()):
                binding_api.UnbindDirectBinding("physics")
        for material_path in sorted(material_paths):
            stage.RemovePrim(Sdf.Path(material_path))
            # prim specs authored in the referenced asset survive RemovePrim; deactivating prunes
            # them from composition (and from the PhysX parser) instead.
            leftover = stage.GetPrimAtPath(Sdf.Path(material_path))
            if leftover.IsValid():
                leftover.SetActive(False)
        removed.extend(sorted(material_paths))
    if removed:
        logger.info(f"Removed USD physics materials from spawned UR5e (PhysX 64K material cap): {removed}")
    return prim


UR5E_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": -1.5708,
    **ROBOTIQ_2F85_DEFAULT_JOINT_POS,
}

UR5E_VELOCITY_LIMITS = {
    "shoulder_pan_joint": 1.5708,
    "shoulder_lift_joint": 1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": 3.1415,
    "wrist_2_joint": 3.1415,
    "wrist_3_joint": 3.1415,
}

UR5E_EFFORT_LIMITS = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 28.0,
    "wrist_2_joint": 28.0,
    "wrist_3_joint": 28.0,
}

# PhysX articulation order; reset datasets and trained policies index joints in this order on every backend.
UR5E_JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
)

UR5E_BODY_ORDER = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "robotiq_base_link",
    "left_outer_knuckle",
    "right_outer_knuckle",
    "right_inner_knuckle",
    "left_inner_knuckle",
    "left_outer_finger",
    "right_outer_finger",
    "right_inner_finger",
    "left_inner_finger",
)


UR5E_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_ur5e_without_physics_materials,
        usd_path=f"{UWLAB_NEWTON_ASSETS_DIR}/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibratedNewton/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(0, 0, 0, 1), joint_pos=UR5E_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
    joint_ordering=UR5E_JOINT_ORDER,
    body_ordering=UR5E_BODY_ORDER,
)

# PhysX articulation order of the gripper-only asset; grasp datasets index joints by name, this keeps any positional
# use consistent across backends.
ROBOTIQ_2F85_JOINT_ORDER = (
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
)

ROBOTIQ_2F85_BODY_ORDER = (
    "robotiq_base_link",
    "left_outer_knuckle",
    "right_outer_knuckle",
    "right_inner_knuckle",
    "left_inner_knuckle",
    "left_outer_finger",
    "right_outer_finger",
    "right_inner_finger",
    "left_inner_finger",
)


ROBOTIQ_2F85 = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/RobotiqGripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_NEWTON_ASSETS_DIR}/Robots/UniversalRobots/2f85RobotiqGripperCalibratedNewton/robotiq_2f85_gripper_calibrated.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.1), rot=(0, 0, 0, 1), joint_pos=ROBOTIQ_2F85_DEFAULT_JOINT_POS
    ),
    actuators={
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=17,
            damping=5,
            effort_limit_sim=60,
        ),
    },
    soft_joint_pos_limit_factor=1,
    joint_ordering=ROBOTIQ_2F85_JOINT_ORDER,
    body_ordering=ROBOTIQ_2F85_BODY_ORDER,
)

EXPLICIT_UR5E_ROBOTIQ_2F85 = UR5E_ARTICULATION.copy()  # type: ignore
EXPLICIT_UR5E_ROBOTIQ_2F85.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=UR5E_EFFORT_LIMITS,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit=UR5E_VELOCITY_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "gripper": ROBOTIQ_2F85.actuators["gripper"],
}

IMPLICIT_UR5E_ROBOTIQ_2F85 = UR5E_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR5E_ROBOTIQ_2F85.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
    ),
    "gripper": ROBOTIQ_2F85.actuators["gripper"],
}
