# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Patches for the Newton (MuJoCo-Warp) backend."""

from __future__ import annotations

import os

import torch

from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg

_PATCHED = False
_GRAVCOMP_PATCHED = False
_VELOCITY_LIMIT_PATCHED = False
_WRENCH_PATCHED = False
_TRIANGLE_PAIRS_PATCHED = False
_SOLVER_DATA_ARRAYS = None
_WRENCH_ASSETS: list = []


@configclass
class UWLabNewtonCfg(NewtonCfg):
    """:class:`~isaaclab_newton.physics.NewtonCfg` with UW Lab physics options."""

    enforce_joint_velocity_limits: bool = True
    """Clamp every joint velocity to its velocity limit after each solver substep.

    PhysX enforces articulation joint velocity limits inside its solver and the real robot has them; MuJoCo-Warp
    does not enforce them at all. Requires :func:`patch_joint_velocity_limits`.
    """

    max_triangle_pairs_per_env: int | None = None
    """Triangle-pair budget per env for mesh collisions; the collision pipeline gets this times the env count.

    Newton's buffer is shared by all envs, so a fixed ``collision_cfg.max_triangle_pairs`` overflows as the env count
    grows and silently drops contacts. Overrides ``collision_cfg.max_triangle_pairs`` when set. Requires
    :func:`patch_triangle_pairs_per_env`.
    """


def patch_fixed_base_root_pose_write() -> None:
    """Make root pose writes move fixed-base Newton articulations.

    ``isaaclab_newton`` writes the root pose to the state only, but a fixed base is placed by its root joint's
    ``model.joint_X_p`` (a MuJoCo mocap body), so forward kinematics restores the old pose. This also writes
    ``joint_X_p`` and notifies the solver. Idempotent; a no-op for floating-base articulations.
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        import warp as wp
        from isaaclab_newton.assets.articulation.articulation import Articulation
        from isaaclab_newton.physics import NewtonManager
        from newton import ModelFlags
    except ImportError:
        return

    from isaaclab.utils.math import quat_apply, quat_mul

    original = Articulation.write_root_link_pose_to_sim_index

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids=None, skip_forward=False):
        original(self, root_pose=root_pose, env_ids=env_ids, skip_forward=skip_forward)
        if not self.is_fixed_base:
            return
        ids = self._resolve_env_ids(env_ids)
        ids = (wp.to_torch(ids) if isinstance(ids, wp.array) else torch.as_tensor(ids)).long().to(self.device)
        pose = root_pose if isinstance(root_pose, torch.Tensor) else wp.to_torch(root_pose)
        pose = pose.reshape(-1, 7).to(self.device, torch.float32)
        model, view = NewtonManager._model, self.root_view
        # ArticulationView.set_root_transforms raises NotImplementedError for this layout
        articulations = wp.to_torch(view.articulation_ids).reshape(-1).long().to(self.device)
        root_joint = wp.to_torch(model.articulation_start).long().to(self.device)[articulations][ids]
        joint_x_p = wp.to_torch(model.joint_X_p)
        joint_x_c = wp.to_torch(model.joint_X_c)[root_joint]
        joint_x_p[root_joint, :3] = pose[:, :3] + quat_apply(pose[:, 3:7], joint_x_c[:, :3])
        joint_x_p[root_joint, 3:7] = quat_mul(pose[:, 3:7], joint_x_c[:, 3:7])
        NewtonManager.add_model_change(ModelFlags.JOINT_PROPERTIES)
        NewtonManager.invalidate_fk(env_ids=wp.from_torch(ids.to(torch.int32)), articulation_ids=view.articulation_ids)

    Articulation.write_root_link_pose_to_sim_index = write_root_link_pose_to_sim_index
    _PATCHED = True


def patch_disable_gravity_gravcomp() -> None:
    """Mirror ``disable_gravity=True`` rigid-body writes as MuJoCo gravity compensation.

    Newton has no per-body gravity switch, so bodies spawned with ``disable_gravity=True`` would feel gravity on
    MuJoCo-Warp. This also writes ``mjc:gravcomp = 1`` on the same prims, which PhysX ignores. Idempotent.
    """
    global _GRAVCOMP_PATCHED
    if _GRAVCOMP_PATCHED:
        return
    try:
        from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg
    except ImportError:
        return

    import isaaclab.sim.schemas as schemas

    original = schemas.modify_rigid_body_properties

    def modify_rigid_body_properties(prim_path, cfg, stage=None):
        result = original(prim_path, cfg, stage)
        if getattr(cfg, "disable_gravity", None):
            original(prim_path, MujocoRigidBodyPropertiesCfg(gravcomp=1.0), stage)
        return result

    schemas.modify_rigid_body_properties = modify_rigid_body_properties
    _GRAVCOMP_PATCHED = True


def patch_joint_velocity_limits() -> None:
    """Enforce ``model.joint_velocity_limit`` after every Newton solver substep when the active physics config is a
    :class:`UWLabNewtonCfg` with :attr:`~UWLabNewtonCfg.enforce_joint_velocity_limits` set. Idempotent.

    The clamp runs inside the captured step graph; MuJoCo-Warp re-reads joint velocities from the Newton state at the
    start of every substep, so the clamped values carry into the next one.
    """
    global _VELOCITY_LIMIT_PATCHED
    if _VELOCITY_LIMIT_PATCHED:
        return
    import warp as wp
    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics import NewtonManager

    @wp.kernel
    def clamp_joint_velocity(joint_qd: wp.array(dtype=wp.float32), limit: wp.array(dtype=wp.float32)):
        i = wp.tid()
        joint_qd[i] = wp.clamp(joint_qd[i], -limit[i], limit[i])

    original = NewtonManager.__dict__["_step_solver"].__func__

    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt):
        original(cls, state_0, state_1, control, contacts, substep_dt)
        model = cls._model
        if getattr(PhysicsManager._cfg, "enforce_joint_velocity_limits", False) and model.joint_dof_count:
            wp.launch(
                clamp_joint_velocity,
                dim=model.joint_dof_count,
                inputs=[state_1.joint_qd, model.joint_velocity_limit],
                device=model.device,
            )

    NewtonManager._step_solver = classmethod(_step_solver)
    _VELOCITY_LIMIT_PATCHED = True


def _reapply_external_wrenches(state) -> None:
    import warp as wp

    for asset in _WRENCH_ASSETS:
        wp.copy(asset._data._sim_bind_body_external_wrench, asset._uwlab_external_wrench)


def patch_substep_external_wrench() -> None:
    """Apply asset external wrenches on every Newton solver substep, not only the first.

    Newton assets write their composed wrench into ``State.body_f`` once per physics step, and the substep loop clears
    forces after each substep, so with ``num_substeps > 1`` the wrench acted for only the first substep. This keeps a
    copy of each asset's last written wrench and restores it through a state-force callback before every substep.
    Idempotent.
    """
    global _WRENCH_PATCHED
    if _WRENCH_PATCHED:
        return
    import warp as wp
    from isaaclab_newton.assets.articulation.articulation import Articulation
    from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection import RigidObjectCollection
    from isaaclab_newton.physics import NewtonManager

    for cls in (RigidObject, Articulation, RigidObjectCollection):
        original_init = cls._initialize_impl
        original_write = cls.write_data_to_sim

        def _initialize_impl(self, _original=original_init):
            _original(self)
            binding = self._data._sim_bind_body_external_wrench
            self._uwlab_external_wrench = wp.zeros(binding.shape, dtype=binding.dtype, device=binding.device)
            _WRENCH_ASSETS[:] = [a for a in _WRENCH_ASSETS if a is not self] + [self]
            # callbacks must exist before solver initialization to be part of the captured step graph
            NewtonManager.register_state_force_callback(_reapply_external_wrenches)

        def write_data_to_sim(self, _original=original_write):
            active = self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active
            _original(self)
            buffer = getattr(self, "_uwlab_external_wrench", None)
            if buffer is not None:
                if active:
                    wp.copy(buffer, self._data._sim_bind_body_external_wrench)
                else:
                    buffer.zero_()

        cls._initialize_impl = _initialize_impl
        cls.write_data_to_sim = write_data_to_sim
    _WRENCH_PATCHED = True


def patch_triangle_pairs_per_env() -> None:
    """Size the collision pipeline's triangle-pair buffer from :attr:`UWLabNewtonCfg.max_triangle_pairs_per_env`
    and the env count when Newton builds it. Idempotent."""
    global _TRIANGLE_PAIRS_PATCHED
    if _TRIANGLE_PAIRS_PATCHED:
        return
    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics import NewtonCollisionPipelineCfg, NewtonManager

    original = NewtonManager.__dict__["_initialize_contacts"].__func__

    def _initialize_contacts(cls) -> None:
        per_env = getattr(PhysicsManager._cfg, "max_triangle_pairs_per_env", None)
        if per_env is not None and cls._collision_pipeline is None:
            collision_cfg = cls._collision_cfg or NewtonCollisionPipelineCfg()
            NewtonManager._collision_cfg = collision_cfg.replace(
                max_triangle_pairs=int(per_env) * cls._model.world_count
            )
        original(cls)

    NewtonManager._initialize_contacts = classmethod(_initialize_contacts)
    _TRIANGLE_PAIRS_PATCHED = True


def _per_world_float_arrays(data, nworld: int, depth: int = 0) -> list[torch.Tensor]:
    import warp as wp

    out = []
    for value in vars(data).values():
        if isinstance(value, wp.array):
            scalar = getattr(value.dtype, "_wp_scalar_type_", value.dtype)
            if value.ndim >= 1 and value.shape[0] == nworld and scalar in (wp.float32, wp.float64):
                out.append(wp.to_torch(value))
        elif depth < 1 and hasattr(value, "__dict__") and "mujoco_warp" in type(value).__module__:
            out.extend(_per_world_float_arrays(value, nworld, depth + 1))
    return out


def zero_non_finite_solver_data(env_mask: torch.Tensor) -> None:
    """Zero non-finite MuJoCo-Warp data of the masked worlds (envs); a no-op off Newton/MuJoCo-Warp.

    Newton's reset clears only the warm-start and applied-force buffers. After a solver blow-up, NaNs in the other
    per-world ``mjw_data`` buffers survive the env reset and re-poison the next solve, so the world stays non-finite
    (and is terminated) on every step for the rest of the run. Call it on blown-up envs before they reset.
    """
    global _SOLVER_DATA_ARRAYS
    if os.environ.get("UWLAB_DISABLE_SOLVER_DATA_SANITIZE"):
        return
    from isaaclab_newton.physics import NewtonManager

    solver = getattr(NewtonManager, "_solver", None)
    data = getattr(solver, "mjw_data", None)
    if data is None or getattr(solver, "use_mujoco_cpu", False):
        return
    if _SOLVER_DATA_ARRAYS is None or _SOLVER_DATA_ARRAYS[0] is not data:
        _SOLVER_DATA_ARRAYS = (data, _per_world_float_arrays(data, data.nworld))
    worlds = env_mask[: data.nworld]
    for t in _SOLVER_DATA_ARRAYS[1]:
        t.masked_fill_(worlds.view(-1, *([1] * (t.dim() - 1))) & ~torch.isfinite(t), 0.0)
