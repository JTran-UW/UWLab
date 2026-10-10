# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics backend presets shared by the OmniReset UR5e + Robotiq 2F-85 configs (``presets=newton_mjwarp``)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg, preset

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR, UWLAB_NEWTON_ASSETS_DIR

from uwlab.sim.spawners.from_files import UsdFileWithCollisionApproximationCfg
from uwlab.utils.newton import UWLabNewtonCfg

from ... import mdp as task_mdp

NEWTON_CONVEX_HULL_ASSETS = ("Props/Custom/Peg/peg.usd",)
"""Assets whose convex hull is their exact shape; Newton collides them as convex hulls instead of triangle meshes."""


@configclass
class OmniResetPhysicsCfg(PresetCfg):
    """Physics backend. Each config passes its own PhysX settings as ``default``."""

    default: PhysxCfg = PhysxCfg()

    # Implicit PD damping (Euler chatters the gripper), Newton's collision pipeline (MuJoCo's convex-hull meshes make
    # the hole solid), contact buffers sized for ~72 contacts/world, and stiff contacts with an elliptic friction cone
    # (4 substeps permit the 5 ms contact time constant) so grasps hold. Mesh contacts get ~3x the peak
    # triangle pairs measured per env with the low-poly hole (172). Joint velocity limits are enforced as on
    # PhysX and the real arm (env.sim.physics.enforce_joint_velocity_limits=false disables).
    newton_mjwarp: UWLabNewtonCfg = UWLabNewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            use_mujoco_contacts=False,
            nconmax=256,
            njmax=1500,
            cone="elliptic",
            impratio=10.0,
        ),
        num_substeps=4,
        default_shape_cfg=NewtonShapeCfg(ke=4.0e4, kd=400.0),
        enforce_joint_velocity_limits=True,
        max_triangle_pairs_per_env=500,
    )


def object_spawn_cfg(usd_path: str, **kwargs):
    """:class:`~isaaclab.sim.UsdFileCfg` for an object; convex assets get a Newton convex-hull alternative.

    Newton collides ``approximation="sdf"`` meshes as triangle meshes, which gave the peg pad contacts with normals
    along the fingers.
    """
    if not usd_path.endswith(NEWTON_CONVEX_HULL_ASSETS):
        return sim_utils.UsdFileCfg(usd_path=usd_path, **kwargs)
    return preset(
        default=sim_utils.UsdFileCfg(usd_path=usd_path, **kwargs),
        newton_mjwarp=UsdFileWithCollisionApproximationCfg(
            usd_path=usd_path, collision_approximation="convexHull", **kwargs
        ),
    )


def newton_equality_solref_event():
    """Newton only: stiffen the gripper's mimic and loop-closure equalities (the default 20 ms lets the four-bar yield)."""
    return preset(
        default=None,
        newton_mjwarp=EventTerm(
            func=task_mdp.set_newton_equality_solref, mode="startup", params={"timeconst": 0.005, "dampratio": 1.0}
        ),
    )


def omnireset_dataset_dir():
    """Grasp / reset-state dataset root; Newton uses the datasets generated on Newton."""
    return preset(
        default=f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset_isaaclab3",
        newton_mjwarp=f"{UWLAB_NEWTON_ASSETS_DIR}/Datasets/OmniReset_newton",
    )


def peghole_usd_path():
    """Peg hole USD; Newton collides the 102-triangle hole (the 4.9k-triangle one costs ~45x the mesh contact pairs)."""
    return preset(
        default=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
        newton_mjwarp=f"{UWLAB_NEWTON_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
    )
