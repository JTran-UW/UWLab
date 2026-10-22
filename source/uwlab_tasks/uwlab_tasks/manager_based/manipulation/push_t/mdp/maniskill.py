# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManiSkill PushT-v1 terms: T/goal-T area overlap success, ``compute_normalized_dense_reward``, resets."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .observations import T_CFG, ee_tip_pose_w, goal_pose, robot_root_pose_w, t_pose, wrap_angle
from .t_block import t_part_centers

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

INTERSECTION_THRESH = 0.90


def maniskill_z_euler(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """ManiSkill ``PushTEnv.quat_to_z_euler``: 2 acos(qw) with the sign fixed by qz, in [0, 2pi].

    Equals the yaw for a pure z rotation; for a slightly tilted T it also absorbs the tilt angle, as in ManiSkill.
    """
    sign = torch.where(quat_xyzw[:, 2] < 0, -1.0, 1.0)
    return 2.0 * torch.acos((quat_xyzw[:, 3] * sign).clamp(-1.0, 1.0))


def pose_errors(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(planar position error [m], |wrapped yaw error| [rad])."""
    t_pos, t_yaw, _ = t_pose(env)
    g_pos, g_yaw = goal_pose(env)
    pos_err = torch.linalg.norm((t_pos - g_pos)[:, :2], dim=-1)
    yaw_err = wrap_angle(t_yaw - g_yaw).abs()
    return pos_err, yaw_err


def _t_grid(bar_size, stem_size, spacing: float, device, centers=None) -> torch.Tensor:
    """(P, 2) cell centers of a ~``spacing`` grid covering the T in its body frame (equal area per point)."""
    bar_y, stem_y = t_part_centers(bar_size, stem_size, centers)
    pts = []
    for (w, h), cy in ((bar_size, bar_y), (stem_size, stem_y)):
        nx, ny = max(1, round(w / spacing)), max(1, round(h / spacing))
        xs = (torch.arange(nx, device=device) + 0.5) * (w / nx) - 0.5 * w
        ys = (torch.arange(ny, device=device) + 0.5) * (h / ny) - 0.5 * h + cy
        pts.append(torch.cartesian_prod(xs, ys))
    return torch.cat(pts)


def _inside_t(p: torch.Tensor, bar_size, stem_size, centers=None) -> torch.Tensor:
    """Planar points (..., 2) in a T body frame -> inside that T's footprint."""
    bar_y, stem_y = t_part_centers(bar_size, stem_size, centers)
    x, y = p[..., 0], p[..., 1]
    in_bar = (x.abs() <= 0.5 * bar_size[0]) & ((y - bar_y).abs() <= 0.5 * bar_size[1])
    in_stem = (x.abs() <= 0.5 * stem_size[0]) & ((y - stem_y).abs() <= 0.5 * stem_size[1])
    return in_bar | in_stem


def t_overlap(t_xy, t_yaw, g_xy, g_yaw, grid: torch.Tensor, bar_size, stem_size, centers=None) -> torch.Tensor:
    """Fraction of the T's ``grid`` points (T body frame) that land inside the goal T footprint. Planar."""
    # T body frame -> goal body frame: rotate by (t_yaw - g_yaw), offset R(-g_yaw) (t_xy - g_xy)
    rel = t_yaw - g_yaw
    c, s = torch.cos(rel), torch.sin(rel)
    d = t_xy - g_xy
    cg, sg = torch.cos(g_yaw), torch.sin(g_yaw)
    ox, oy = cg * d[:, 0] + sg * d[:, 1], -sg * d[:, 0] + cg * d[:, 1]
    gx, gy = grid[:, 0], grid[:, 1]
    px = c[:, None] * gx - s[:, None] * gy + ox[:, None]
    py = s[:, None] * gx + c[:, None] * gy + oy[:, None]
    return _inside_t(torch.stack([px, py], dim=-1), bar_size, stem_size, centers).float().mean(dim=-1)


def intersection_area(env: ManagerBasedRLEnv, spacing: float = 0.0025) -> torch.Tensor:
    """Area(T intersect goal T) / area(goal T) by point sampling. Cached per env step.

    Point-sampled version of ManiSkill's ``pseudo_render_intersection`` (T and goal T have the same area).
    """
    step = env.common_step_counter
    if getattr(env, "_push_t_inter_step", None) == step:
        return env._push_t_inter
    spawn = env.scene["t_block"].cfg.spawn
    if not hasattr(env, "_push_t_grid"):
        env._push_t_grid = _t_grid(spawn.bar_size, spawn.stem_size, spacing, env.device, spawn.part_centers)
    t_pos, _, t_quat = t_pose(env)
    g_pos, g_yaw = goal_pose(env)
    env._push_t_inter = t_overlap(
        t_pos[:, :2],
        maniskill_z_euler(t_quat),
        g_pos[:, :2],
        g_yaw,
        env._push_t_grid,
        spawn.bar_size,
        spawn.stem_size,
        spawn.part_centers,
    )
    env._push_t_inter_step = step
    return env._push_t_inter


def intersection_success(env: ManagerBasedRLEnv, thresh: float = INTERSECTION_THRESH) -> torch.Tensor:
    """ManiSkill PushT ``evaluate()["success"]``: overlap >= ``thresh``, instantaneous."""
    return intersection_area(env) >= thresh


def maniskill_push_t_reward(env: ManagerBasedRLEnv, thresh: float = INTERSECTION_THRESH) -> torch.Tensor:
    """ManiSkill PushT ``compute_normalized_dense_reward`` (in [0, 1]; 1 on success)."""
    t_pos, _, t_quat = t_pose(env)
    g_pos, g_yaw = goal_pose(env)
    r = (((torch.cos(maniskill_z_euler(t_quat) - g_yaw) + 1.0) / 2.0) ** 2) / 2.0
    d = torch.linalg.norm((t_pos - g_pos)[:, :2], dim=-1)
    r = r + ((1.0 - torch.tanh(5.0 * d)) ** 2) / 2.0
    tcp_w, _ = ee_tip_pose_w(env)
    e = torch.linalg.norm(t_pos + env.scene.env_origins - tcp_w, dim=-1)
    r = r + torch.sqrt(1.0 - torch.tanh(5.0 * e)) / 20.0
    r = torch.where(intersection_success(env, thresh), torch.full_like(r, 3.0), r)
    return r / 3.0


class push_t_intersection_success(ManagerTermBase):
    """Terminate on ``intersection_success``; logs final pose errors, overlap and success at episode end.

    ``terminate=False`` is ManiSkill's ``ignore_terminations`` (``partial_reset=False``): episodes always run to the
    time-out and ``Metrics/success_once`` / ``Metrics/success_at_end`` are logged.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._pos_err = torch.zeros(env.num_envs, device=env.device)
        self._yaw_err = torch.zeros(env.num_envs, device=env.device)
        self._inter = torch.zeros(env.num_envs, device=env.device)
        self._success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._success_once = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        # runs after extras["log"] is re-created in _reset_idx, so these land as end-of-episode metrics
        if "log" in self._env.extras and self._env.common_step_counter > 0:
            log = self._env.extras["log"]
            log["Metrics/final_pos_err"] = self._pos_err[ids].mean().item()
            log["Metrics/final_yaw_err_deg"] = math.degrees(self._yaw_err[ids].mean().item())
            log["Metrics/final_intersection"] = self._inter[ids].mean().item()
            log["Metrics/success"] = self._success[ids].float().mean().item()
            log["Metrics/success_once"] = self._success_once[ids].float().mean().item()
            log["Metrics/success_at_end"] = self._success[ids].float().mean().item()
        self._success[ids] = False
        self._success_once[ids] = False

    def __call__(
        self, env: ManagerBasedRLEnv, thresh: float = INTERSECTION_THRESH, terminate: bool = True
    ) -> torch.Tensor:
        self._pos_err[:], self._yaw_err[:] = pose_errors(env)
        self._inter[:] = intersection_area(env)
        self._success[:] = intersection_success(env, thresh)
        self._success_once |= self._success
        return self._success.clone() if terminate else torch.zeros_like(self._success)


def nonfinite_state(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """NaN / inf in the robot joint state or the T pose (PhysX blow-up guard)."""
    robot: Articulation = env.scene[robot_cfg.name]
    t_pos, _, t_quat = t_pose(env)
    ok = torch.isfinite(robot.data.joint_pos.torch).all(-1) & torch.isfinite(robot.data.joint_vel.torch).all(-1)
    ok &= torch.isfinite(t_pos).all(-1) & torch.isfinite(t_quat).all(-1)
    return ~ok


def reset_joints_by_gaussian_offset(
    env: ManagerBasedEnv, env_ids: torch.Tensor, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Default joint pos + N(0, std) per joint, zero velocity; joint targets set to the new pos."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids
    joint_pos = asset.data.default_joint_pos.torch[ids, asset_cfg.joint_ids].clone()
    joint_pos += std * torch.randn_like(joint_pos)
    limits = asset.data.soft_joint_pos_limits.torch[ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_pos), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
    asset.set_joint_position_target_index(target=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_t_uniform_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range: tuple[float, float],
    t_cfg: SceneEntityCfg = T_CFG,
):
    """ManiSkill PushT tee spawn: (x, y) ~ U(box) in the env frame, yaw ~ U(yaw_range), resting, zero velocity."""
    t: RigidObject = env.scene[t_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, dev = len(env_ids), env.device
    u = torch.rand(n, 3, device=dev)
    yaw = yaw_range[0] + (yaw_range[1] - yaw_range[0]) * u[:, 2]
    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = x_range[0] + (x_range[1] - x_range[0]) * u[:, 0]
    pose[:, 1] = y_range[0] + (y_range[1] - y_range[0]) * u[:, 1]
    pose[:, 2] = t.cfg.init_state.pos[2]
    pose[:, :3] += env.scene.env_origins[env_ids]
    pose[:, 5], pose[:, 6] = torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)
    t.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    t.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)


def _pose_root(env: ManagerBasedEnv, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    root_pos, root_quat = robot_root_pose_w(env)
    pos_b, quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, pos_w, quat_w)
    return torch.cat([pos_b, quat_b], dim=-1)


def ms_tcp_pose(env: ManagerBasedEnv) -> torch.Tensor:
    """ManiSkill ``tcp_pose`` (position + quaternion), in the robot root frame."""
    return _pose_root(env, *ee_tip_pose_w(env))


def ms_goal_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """ManiSkill ``goal_pos``, in the robot root frame."""
    g_pos, _ = goal_pose(env)
    root_pos, root_quat = robot_root_pose_w(env)
    return math_utils.quat_apply_inverse(root_quat, g_pos + env.scene.env_origins - root_pos)


def ms_obj_pose(env: ManagerBasedEnv, t_cfg: SceneEntityCfg = T_CFG) -> torch.Tensor:
    """ManiSkill ``obj_pose`` (tee position + quaternion), in the robot root frame."""
    t: RigidObject = env.scene[t_cfg.name]
    return _pose_root(env, t.data.root_link_pos_w.torch, t.data.root_link_quat_w.torch)


def ms_gray_image(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    """ManiSkill ``GrayscaleObservationWrapper``: round(0.299 R + 0.587 G + 0.114 B) in [0, 255], (N, 1, H, W) float."""
    rgb = env.scene[sensor_cfg.name].data.output["rgb"][..., :3].float()
    gray = torch.round(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).clamp(0.0, 255.0)
    return gray.unsqueeze(1)
