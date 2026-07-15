# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RMA (Rapid Motor Adaptation) environment for ZeroG ScenePC Sysid Sim2Real.

Wraps ``ZeroGScenePCSysidSim2RealTrainCfg`` to additionally expose a
``privileged_rma`` observation group containing per-env dynamics randomization
state (arm sysid armature/friction, motor delay, OSC Kp/Kd, gripper actuator
gains, body masses, body material friction). The runner-side ``ActorCriticRMA``
projects this group through a small MLP (phi) to produce the privileged latent
``z`` that the actor conditions on, while a small bidirectional transformer
(psi) is trained in parallel to predict ``z`` from a window of past proprio +
actions via MSE.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .gravity_cfg import (
    ScenePCObsCfg,
    ZeroGScenePCSysidSim2RealEValCfg,
    ZeroGScenePCSysidSim2RealTrainCfg,
)

# Arm joint names match ``ZeroGGPSSysidEventCfg.randomize_arm_sysid``.
_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@configclass
class RMAObsCfg(ScenePCObsCfg):
    """ScenePC obs + a ``privileged_rma`` group exposing per-env DR state."""

    @configclass
    class PrivilegedRMACfg(ObsGroup):
        # Arm sysid (24): armature(6) + static friction(6) + dynamic friction(6)
        # + viscous friction(6) — joint_friction_coeff stacks static contributions
        # in IsaacLab's API; using it as the canonical handle.
        arm_armature = ObsTerm(
            func=task_mdp.get_joint_armature,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
        )
        arm_friction = ObsTerm(
            func=task_mdp.get_joint_friction,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
        )
        # Motor delay (1): integer lag of arm positions delay buffer.
        arm_delay = ObsTerm(
            func=task_mdp.get_actuator_delay,
            params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": "arm"},
        )
        # OSC Kp/Kd (12): kp_xyz + kp_rpy + kd_xyz + kd_rpy.
        osc_gains = ObsTerm(func=task_mdp.get_osc_gains, params={"action_name": "arm"})
        # Gripper actuator stiffness/damping (2).
        gripper_stiffness = ObsTerm(
            func=task_mdp.get_joint_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        gripper_damping = ObsTerm(
            func=task_mdp.get_joint_damping,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        # Body masses (4): one entry per articulation/rigid object.
        robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})
        insertive_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )
        receptive_mass = ObsTerm(
            func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )
        table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})
        # Material friction (static + dynamic per body, ``make_consistent=True`` so
        # values are identical across shapes per env).
        robot_material = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        insertive_material = ObsTerm(
            func=task_mdp.get_material_properties,
            params={"asset_cfg": SceneEntityCfg("insertive_object")},
        )
        receptive_material = ObsTerm(
            func=task_mdp.get_material_properties,
            params={"asset_cfg": SceneEntityCfg("receptive_object")},
        )
        table_material = ObsTerm(
            func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table")}
        )

        def __post_init__(self):
            # Privileged signal must not be noised.
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged_rma: PrivilegedRMACfg = PrivilegedRMACfg()


@configclass
class ZeroGScenePCSysidRMATrainCfg(ZeroGScenePCSysidSim2RealTrainCfg):
    """RMA training: ZeroGScenePCSysidSim2RealTrainCfg + privileged_rma obs group."""

    observations: RMAObsCfg = RMAObsCfg()


@configclass
class ZeroGScenePCSysidRMAEvalCfg(ZeroGScenePCSysidSim2RealEValCfg):
    """RMA eval variant matching the Sim2Real Eval pattern."""

    observations: RMAObsCfg = RMAObsCfg()
