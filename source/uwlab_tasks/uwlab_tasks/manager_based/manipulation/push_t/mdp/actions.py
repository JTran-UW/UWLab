# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class JointDeltaPositionAction(JointAction):
    """ManiSkill ``pd_joint_delta_pos``: target = qpos at the policy step + scale * clip(action, -1, 1).

    Unlike IsaacLab's ``RelativeJointPositionAction`` (which re-adds the delta to the measured qpos at every physics
    substep), the target is computed once per policy step and held for the whole decimation.
    """

    cfg: JointDeltaPositionActionCfg

    def __init__(self, cfg: JointDeltaPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        assert cfg.offset == 0.0 and cfg.clip is None, "JointDeltaPositionAction ignores offset / clip; leave defaults"
        self._target = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = actions.clamp(-1.0, 1.0) * self._scale
        self._target = self._asset.data.joint_pos.torch[:, self._joint_ids] + self._processed_actions

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)


@configclass
class JointDeltaPositionActionCfg(JointActionCfg):
    class_type: type = JointDeltaPositionAction
    scale: float = 0.1


class EEDeltaPoseAction(ActionTerm):
    """ManiSkill ``pd_ee_delta_pose`` (GPU path, ``root_translation:root_aligned_body_rotation``, ``use_target=False``).

    Per policy step: dpos = 0.1 * clip(a[:3], -1, 1) [m]; drot = a[3:] rescaled to norm <= 1, times ``rot_lower`` (-0.1,
    so the sign flips as in ManiSkill) [rad]; dq = (J^T J + 1e-4 I)^-1 J^T [dpos, drot] with J the EE-point Jacobian
    (linear rows first) in the robot root frame; joint targets = qpos + dq, held for the whole decimation.
    """

    cfg: EEDeltaPoseActionCfg
    _asset: Articulation

    def __init__(self, cfg: EEDeltaPoseActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._body_idx = self._asset.find_bodies(cfg.body_name)[0][0]
        # fixed-base Jacobians have no root-body row and no root-DoF columns
        assert self._asset.is_fixed_base, "EEDeltaPoseAction expects a fixed-base articulation"
        self._jacobi_body_idx = self._body_idx - 1
        self._offset = torch.tensor(cfg.body_offset, device=self.device).repeat(self.num_envs, 1)
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target = self._asset.data.joint_pos.torch[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _ee_jacobian_root(self) -> torch.Tensor:
        data = self._asset.data
        jac = data.body_link_jacobian_w.torch[:, self._jacobi_body_idx, :, :][..., self._joint_ids].clone()
        body_quat = data.body_link_quat_w.torch[:, self._body_idx]
        # linear Jacobian of the offset point: v_p = v_body + w x r  ->  J_v -= [r]_x J_w
        r = math_utils.quat_apply(body_quat, self._offset)
        jac[:, 0:3] -= torch.bmm(math_utils.skew_symmetric_matrix(r), jac[:, 3:6])
        root_rot = math_utils.matrix_from_quat(math_utils.quat_inv(data.root_link_quat_w.torch))
        jac[:, 0:3] = torch.bmm(root_rot, jac[:, 0:3])
        jac[:, 3:6] = torch.bmm(root_rot, jac[:, 3:6])
        return jac

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        pos = actions[:, :3].clamp(-1.0, 1.0) * self.cfg.pos_scale
        rot = actions[:, 3:].clone()
        norm = torch.linalg.norm(rot, dim=1, keepdim=True)
        rot = torch.where(norm > 1.0, rot / norm, rot) * self.cfg.rot_lower
        self._processed_actions = torch.cat([pos, rot], dim=-1)
        jac = self._ee_jacobian_root()
        jac_t = jac.transpose(1, 2)
        lhs = torch.bmm(jac_t, jac) + self.cfg.damping * torch.eye(jac.shape[-1], device=self.device)
        dq = torch.linalg.solve(lhs, torch.bmm(jac_t, self._processed_actions.unsqueeze(-1))).squeeze(-1)
        self._target = self._asset.data.joint_pos.torch[:, self._joint_ids] + dq

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0


@configclass
class EEDeltaPoseActionCfg(ActionTermCfg):
    class_type: type = EEDeltaPoseAction
    joint_names: list[str] = MISSING
    body_name: str = MISSING
    body_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Controlled point in the body frame (ManiSkill's ``ee_link``)."""
    pos_scale: float = 0.1
    rot_lower: float = -0.1
    """ManiSkill multiplies the norm-clipped rotation action by ``rot_lower`` (negative), flipping its sign."""
    damping: float = 1e-4
