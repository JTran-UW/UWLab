# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import hashlib
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera
from pxr import UsdGeom, UsdPhysics
from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR, resolve_cloud_path

from uwlab_tasks.manager_based.manipulation.omnireset.assembly_keypoints import Offset
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils

# Per-machine cache fallback when HF doesn't have the canonical points yet.
SCENE_PC_LOCAL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "uwlab", "scene_pc_cache"
)


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset=None,
    root_asset_offset=None,
    rotation_repr: str = "quat",
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = target_asset.data.body_link_pos_w[:, target_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w[:, target_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

    if root_asset_offset is not None:
        root_pos, root_quat = root_asset_offset.combine(root_pos, root_quat)
    if target_asset_offset is not None:
        target_pos, target_quat = target_asset_offset.combine(target_pos, target_quat)

    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)

    if rotation_repr == "axis_angle":
        axis_angle = math_utils.axis_angle_from_quat(target_quat_b)
        return torch.cat([target_pos_b, axis_angle], dim=1)
    elif rotation_repr == "quat":
        return torch.cat([target_pos_b, target_quat_b], dim=1)
    else:
        raise ValueError(f"Invalid rotation_repr: {rotation_repr}. Must be one of: 'quat', 'axis_angle'")


class target_asset_pose_in_root_asset_frame_with_metadata(ManagerTermBase):
    """Get target asset pose in root asset frame with offsets automatically read from metadata.

    This is similar to target_asset_pose_in_root_asset_frame but automatically reads the
    assembled offsets from the asset USD metadata instead of requiring manual specification.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        target_asset_cfg: SceneEntityCfg = cfg.params.get("target_asset_cfg")
        root_asset_cfg: SceneEntityCfg = cfg.params.get("root_asset_cfg", SceneEntityCfg("robot"))
        target_asset_offset_metadata_key: str = cfg.params.get("target_asset_offset_metadata_key")
        root_asset_offset_metadata_key: str = cfg.params.get("root_asset_offset_metadata_key")

        self.target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
        self.root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]
        self.target_asset_cfg = target_asset_cfg
        self.root_asset_cfg = root_asset_cfg
        self.rotation_repr = cfg.params.get("rotation_repr", "quat")

        # Read root asset offset from metadata
        if root_asset_offset_metadata_key is not None:
            root_usd_path = self.root_asset.cfg.spawn.usd_path
            root_metadata = utils.read_metadata_from_usd_directory(root_usd_path)
            root_offset_data = root_metadata.get(root_asset_offset_metadata_key)
            self.root_asset_offset = Offset(pos=root_offset_data.get("pos"), quat=root_offset_data.get("quat"))
        else:
            self.root_asset_offset = None

        # Read target asset offset from metadata
        if target_asset_offset_metadata_key is not None:
            target_usd_path = self.target_asset.cfg.spawn.usd_path
            target_metadata = utils.read_metadata_from_usd_directory(target_usd_path)
            target_offset_data = target_metadata.get(target_asset_offset_metadata_key)
            self.target_asset_offset = Offset(pos=target_offset_data.get("pos"), quat=target_offset_data.get("quat"))
        else:
            self.target_asset_offset = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        target_asset_cfg: SceneEntityCfg,
        root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_asset_offset_metadata_key: str | None = None,
        root_asset_offset_metadata_key: str | None = None,
        rotation_repr: str = "quat",
    ) -> torch.Tensor:
        target_body_idx = 0 if isinstance(self.target_asset_cfg.body_ids, slice) else self.target_asset_cfg.body_ids
        root_body_idx = 0 if isinstance(self.root_asset_cfg.body_ids, slice) else self.root_asset_cfg.body_ids

        target_pos = self.target_asset.data.body_link_pos_w[:, target_body_idx].view(-1, 3)
        target_quat = self.target_asset.data.body_link_quat_w[:, target_body_idx].view(-1, 4)
        root_pos = self.root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
        root_quat = self.root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

        if self.root_asset_offset is not None:
            root_pos, root_quat = self.root_asset_offset.combine(root_pos, root_quat)
        if self.target_asset_offset is not None:
            target_pos, target_quat = self.target_asset_offset.combine(target_pos, target_quat)

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)

        if rotation_repr == "axis_angle":
            axis_angle = math_utils.axis_angle_from_quat(target_quat_b)
            return torch.cat([target_pos_b, axis_angle], dim=1)
        elif rotation_repr == "quat":
            return torch.cat([target_pos_b, target_quat_b], dim=1)
        else:
            raise ValueError(f"Invalid rotation_repr: {rotation_repr}. Must be one of: 'quat', 'axis_angle'")


def asset_link_velocity_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids

    asset_lin_vel_b, _ = math_utils.subtract_frame_transforms(
        root_asset.data.root_pos_w,
        root_asset.data.root_quat_w,
        target_asset.data.body_lin_vel_w[:, target_body_idx].view(-1, 3),
    )
    asset_ang_vel_b, _ = math_utils.subtract_frame_transforms(
        root_asset.data.root_pos_w,
        root_asset.data.root_quat_w,
        target_asset.data.body_ang_vel_w[:, target_body_idx].view(-1, 3),
    )

    return torch.cat([asset_lin_vel_b, asset_ang_vel_b], dim=1)


def get_material_properties(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_material_properties().view(env.num_envs, -1).to(env.device)


def get_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_masses().view(env.num_envs, -1).to(env.device)


def get_joint_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction_coeff.view(env.num_envs, -1).to(env.device)


def get_joint_armature(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_armature.view(env.num_envs, -1).to(env.device)


def get_joint_stiffness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness.view(env.num_envs, -1).to(env.device)


def get_joint_damping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping.view(env.num_envs, -1).to(env.device)


def get_osc_gains(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Per-env OSC Kp and Kd from a RelCartesianOSCAction term.

    Returned shape: (num_envs, 12)  ->  [Kp_xyz(3), Kp_rpy(3), Kd_xyz(3), Kd_rpy(3)].
    Reads ``_kp`` / ``_kd`` written by ``randomize_rel_cartesian_osc_gains``.
    """
    term = env.action_manager._terms[action_name]
    kp = term._kp.view(env.num_envs, -1).to(env.device)
    kd = term._kd.view(env.num_envs, -1).to(env.device)
    return torch.cat([kp, kd], dim=-1)


def get_actuator_delay(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    actuator_name: str,
) -> torch.Tensor:
    """Per-env motor delay (positions buffer lag) for a delayed actuator.

    Returns (num_envs, 1) float. Zero if the actuator has no delay buffer.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    actuator = asset.actuators[actuator_name]
    if not hasattr(actuator, "positions_delay_buffer"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    lag = actuator.positions_delay_buffer.time_lags
    return lag.view(env.num_envs, 1).to(dtype=torch.float32, device=env.device)


def joint_pos_last_n(
    env: ManagerBasedEnv,
    n: int = 6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Last ``n`` DOFs of the asset's joint positions, in articulation DOF order.

    For the UR5e + Robotiq 2F-85 the 12 DOFs are ordered arm-then-gripper, so the
    default ``n=6`` yields the six gripper joints (``finger_joint``,
    ``right_outer_knuckle``, the two inner knuckles, the two inner-finger knuckles).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, -n:]


def time_left(env) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        life_left = 1 - (env.episode_length_buf.float() / env.max_episode_length)
    else:
        life_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    return life_left.view(-1, 1)


def process_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    process_image: bool = True,
    output_size: tuple = (224, 224),
    depth_clip: tuple[float, float] = (0.01, 2.0),
    depth_noise_range: float = 0.0,
    depth_global_bias_range: float = 0.0,
    depth_global_scale_range: float = 0.0,
    depth_dropout_prob: float = 0.0,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_image_plane": replaces inf/nan with ``depth_clip[1]``,
      clips to ``depth_clip``, and normalizes to [0, 1].

    For depth images, three optional uniform-noise sources can be enabled (all expressed
    in normalized [0, 1] depth units, applied after clip+normalize and before any resize).
    Each parameter is a half-range ``r``; samples are drawn from ``U(-r, +r)``:

    - ``depth_noise_range``: per-pixel additive jitter (mimics sensor shot noise).
    - ``depth_global_bias_range``: per-frame additive offset, drawn once per env then
      broadcast across the image (mimics calibration drift / temperature bias).
    - ``depth_global_scale_range``: per-frame multiplicative gain ``1 + U(-r, +r)``,
      drawn once per env (mimics scale-factor miscalibration).

    A fourth optional pixel-dropout source independently zeros or saturates pixels
    to mimic dead pixels / no-return surfaces:

    - ``depth_dropout_prob``: Bernoulli probability that a pixel is dropped; dropped
      pixels are independently set to 0.0 or 1.0 with equal probability.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        process_image: Whether to normalize the image. Defaults to True.
        depth_clip: (min, max) meters for depth clipping/normalization. Ignored for rgb.
        depth_noise_range: Per-pixel uniform half-range in normalized depth units. 0 disables.
        depth_global_bias_range: Per-frame global additive bias half-range (normalized units). 0 disables.
        depth_global_scale_range: Per-frame global multiplicative gain half-range. 0 disables.
        depth_dropout_prob: Per-pixel probability of being saturated to 0.0 or 1.0 (50/50). 0 disables.

    Returns:
        The images produced at the last time-step
    """
    assert data_type in ("rgb", "depth", "distance_to_camera", "distance_to_image_plane"), (
        f"Unsupported data_type: {data_type}"
    )
    is_depth = data_type != "rgb"

    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type].clone()

    start_dims = torch.arange(len(images.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    current_size = (images.shape[s + 1], images.shape[s + 2])

    # Convert to float32 and normalize based on modality
    images = images.to(dtype=torch.float32)
    if is_depth:
        d_min, d_max = depth_clip
        # Replace inf/nan with d_max (no-return pixels), then clip and normalize to [0, 1]
        images = torch.nan_to_num(images, nan=d_max, posinf=d_max, neginf=d_max)
        images.clamp_(d_min, d_max).sub_(d_min).div_(d_max - d_min)
    else:
        images.div_(255.0).clamp_(0.0, 1.0)
    images = images.permute(start_dims + [s + 3, s + 1, s + 2])

    if is_depth and (depth_noise_range > 0.0 or depth_global_bias_range > 0.0 or depth_global_scale_range > 0.0):
        # Uniform samples on U(-r, +r) via (2*rand - 1) * r.
        # Global params: one draw per env, broadcast over (C, H, W).
        global_shape = images.shape[:-3] + (1, 1, 1)
        if depth_global_scale_range > 0.0:
            u = torch.rand(global_shape, device=images.device, dtype=images.dtype).mul_(2.0).sub_(1.0)
            images = images * (1.0 + depth_global_scale_range * u)
        if depth_global_bias_range > 0.0:
            u = torch.rand(global_shape, device=images.device, dtype=images.dtype).mul_(2.0).sub_(1.0)
            images = images + depth_global_bias_range * u
        if depth_noise_range > 0.0:
            u = torch.rand_like(images).mul_(2.0).sub_(1.0)
            images = images + depth_noise_range * u
        images = images.clamp(0.0, 1.0)

    if is_depth and depth_dropout_prob > 0.0:
        # Per-pixel dropout: with prob p, replace with 0 or 1 (50/50). Applied
        # last so additive/scale noise doesn't perturb the saturated pixels.
        drop_mask = torch.rand_like(images) < depth_dropout_prob
        high_mask = torch.rand_like(images) < 0.5
        images = torch.where(drop_mask, high_mask.to(images.dtype), images)

    if current_size != output_size:
        # Perform resize operation
        images = F.interpolate(images, size=output_size, mode="bilinear", antialias=True)

    # Skip normalization path: only valid for rgb (uint8 serialization)
    if not process_image:
        assert not is_depth, "process_image=False (uint8 path) is only supported for rgb"
        # Reverse the permutation
        reverse_dims = torch.argsort(torch.tensor(start_dims + [s + 3, s + 1, s + 2]))
        images = images.permute(reverse_dims.tolist())
        # Convert back to uint8 in-place
        images.mul_(255.0).clamp_(0, 255)  # Scale and clamp in-place
        images = images.to(dtype=torch.uint8)  # Type conversion (not in-place)

    # import matplotlib.pyplot as plt
    # img_0 = images[0].permute([1, 2, 0])
    # plt.imshow(img_0.cpu().numpy())
    # plt.savefig('saved_image_0.png', dpi=300, bbox_inches='tight')
    # img_1 = images[1].permute([1, 2, 0])
    # plt.imshow(img_1.cpu().numpy())
    # plt.savefig('saved_image_1.png', dpi=300, bbox_inches='tight')
    # img_2 = images[2].permute([1, 2, 0])
    # plt.imshow(img_2.cpu().numpy())
    # plt.savefig('saved_image_2.png', dpi=300, bbox_inches='tight')
    # img_3 = images[3].permute([1, 2, 0])
    # plt.imshow(img_3.cpu().numpy())
    # plt.savefig('saved_image_3.png', dpi=300, bbox_inches='tight')

    return images


def binary_force_contact(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    body_name: str = "wrist_3_link",
    force_threshold: float = 25.0,
) -> torch.Tensor:
    """Binary contact detection from force norm at a body.

    Reads body_incoming_joint_wrench_b, computes ||F|| from the force
    components (first 3), and returns 1.0 if above threshold, else 0.0.

    Args:
        env: The environment.
        asset_cfg: Scene entity config for the robot articulation.
        body_name: Name of the body to read wrench from.
        force_threshold: Force norm threshold (N) for contact detection.

    Returns:
        Tensor of shape (num_envs, 1): 1.0 if contact, 0.0 otherwise.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.body_names.index(body_name)
    wrench_b = robot.data.body_incoming_joint_wrench_b[:, body_idx, :]  # (N, 6)
    force_norm = torch.norm(wrench_b[:, :3], dim=-1)  # (N,)
    contact = (force_norm > force_threshold).float()
    return contact.unsqueeze(-1)  # (N, 1)


class MeshPointCloud(ManagerTermBase):
    """Mesh-sampled pointcloud in an arbitrary reference frame.

    Samples canonical points from the object's USD mesh at init, then at
    runtime transforms them into the reference frame specified by ``ref_cfg``.

    ``ref_cfg`` can be:
      - An Articulation with ``body_names`` (e.g. wrist link)
      - An Articulation without ``body_names`` (robot root frame)
      - A RigidObject (e.g. receptive_object)

    Falls back to legacy ``robot_cfg`` param if ``ref_cfg`` is not provided.

    Returns flattened ``[num_envs, num_points * 3]``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        ref_cfg: SceneEntityCfg = cfg.params.get("ref_cfg", cfg.params.get("robot_cfg"))
        self.num_points: int = cfg.params.get("num_points", 128)
        # Re-sample mesh points per env at episode reset. Breaks the "PC point
        # arrangement → absolute yaw" memorization channel that lets the policy
        # use a fixed-init point pattern as a yaw oracle for symmetric objects.
        # Does nothing geometrically if the mesh has no rotational symmetry, but
        # is necessary to make the symmetric-reward + cylindrical-PC argument
        # actually hold (otherwise the network can leak yaw via point identity).
        self.resample_on_reset: bool = cfg.params.get("resample_on_reset", False)

        self.object_asset: RigidObject = env.scene[object_cfg.name]
        self.ref_asset = env.scene[ref_cfg.name]

        self.ref_body_idx: int | None = None
        if ref_cfg.body_names and isinstance(self.ref_asset, Articulation):
            ref_cfg.resolve(env.scene)
            self.ref_body_idx = ref_cfg.body_ids[0]  # type: ignore

        self._prim_path_pattern = self.object_asset.cfg.prim_path.replace("{ENV_REGEX_NS}", ".*")
        self.canonical_points = utils.sample_object_point_cloud(
            num_envs=env.num_envs,
            num_points=self.num_points,
            prim_path_pattern=self._prim_path_pattern,
            device=str(env.device),
        )  # [num_envs, num_points, 3] in object local frame

        self._setup_visualization(cfg, env, object_cfg)

    def _get_ref_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pos_w, quat_w) of the reference frame, shape [N, 3] and [N, 4]."""
        if self.ref_body_idx is not None:
            return (
                self.ref_asset.data.body_pos_w[:, self.ref_body_idx],
                self.ref_asset.data.body_quat_w[:, self.ref_body_idx],
            )
        return self.ref_asset.data.root_pos_w, self.ref_asset.data.root_quat_w

    def _setup_visualization(self, cfg: ObservationTermCfg, env: ManagerBasedEnv, object_cfg: SceneEntityCfg):
        self.visualize_enabled = cfg.params.get("visualize", False)
        if not self.visualize_enabled:
            return
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.visualize_env_ids = cfg.params.get("visualize_env_ids", [0])
        ref_cfg: SceneEntityCfg = cfg.params.get("ref_cfg", cfg.params.get("robot_cfg"))
        obj_name = object_cfg.name
        body_suffix = "_" + "_".join(ref_cfg.body_names) if ref_cfg.body_names else ""
        ref_name = ref_cfg.name + body_suffix
        self._debug_label = f"{obj_name}_in_{ref_name}"
        marker_cfg = VisualizationMarkersCfg(
            prim_path=f"/Visuals/pointcloud_{self._debug_label}",
            markers={
                "pt": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=cfg.params.get("marker_color", (1.0, 0.0, 0.0)),
                    ),
                ),
            },
        )
        self._pc_markers = VisualizationMarkers(marker_cfg)

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfg: SceneEntityCfg,
        ref_cfg: SceneEntityCfg | None = None,
        robot_cfg: SceneEntityCfg | None = None,
        num_points: int = 128,
        resample_on_reset: bool = False,
        visualize: bool = False,
        visualize_env_ids: list[int] | None = None,
        marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> torch.Tensor:
        # Re-sample mesh points for envs that have just reset (episode_length_buf == 0).
        # This breaks the "fixed init points" memorization channel for symmetric objects.
        if self.resample_on_reset and hasattr(env, "episode_length_buf"):
            reset_mask = env.episode_length_buf == 0
            if reset_mask.any():
                fresh = utils.sample_object_point_cloud(
                    num_envs=env.num_envs,
                    num_points=self.num_points,
                    prim_path_pattern=self._prim_path_pattern,
                    device=str(env.device),
                )  # [num_envs, num_points, 3]
                self.canonical_points[reset_mask] = fresh[reset_mask]

        obj_pos_w = self.object_asset.data.root_pos_w
        obj_quat_w = self.object_asset.data.root_quat_w

        points_w = math_utils.quat_apply(
            obj_quat_w.unsqueeze(1).expand(-1, self.num_points, -1),
            self.canonical_points,
        ) + obj_pos_w.unsqueeze(1)

        if self.visualize_enabled:
            vis_pts = points_w[self.visualize_env_ids].reshape(-1, 3)
            self._pc_markers.visualize(translations=vis_pts)

        ref_pos_w, ref_quat_w = self._get_ref_pose_w()
        ref_quat_w_inv = math_utils.quat_inv(ref_quat_w)

        points_ref = math_utils.quat_apply(
            ref_quat_w_inv.unsqueeze(1).expand(-1, self.num_points, -1),
            points_w - ref_pos_w.unsqueeze(1),
        )

        return points_ref.reshape(env.num_envs, -1)


def ratio_to_counts(ratios, total: int) -> tuple[int, ...]:
    """Convert per-class ``ratios`` -> integer point counts that sum EXACTLY to
    ``total`` (largest-remainder rounding). Used to enforce a fixed robot/insertive/
    receptive split in the scene point clouds (e.g. (0.5, 0.25, 0.25))."""
    raw = [r * total for r in ratios]
    counts = [int(x) for x in raw]  # floor (ratios are non-negative)
    rem = int(total - sum(counts))
    order = sorted(range(len(ratios)), key=lambda i: raw[i] - counts[i], reverse=True)
    for k in range(max(rem, 0)):
        counts[order[k % len(counts)]] += 1
    return tuple(counts)


class ScenePointCloud(ManagerTermBase):
    """Combined pointcloud from robot + insertive + receptive in a configurable frame.

    At init: samples canonical points from all robot body meshes, insertive object,
    and receptive object. Transforms all to world frame using default poses, concatenates,
    and applies FPS to select ``num_points`` total. Records each selected point's source
    (robot body index or object) and its local-frame coordinates.

    At runtime: transforms each point from its local frame to world frame using current
    body/object poses, then transforms all to the reference frame given by ``ref_cfg``.
    No runtime FPS — the selected points are fixed for the entire training run.

    The output frame is set by the ``ref_cfg`` param (a ``SceneEntityCfg``):
      * Articulation WITH ``body_names`` (e.g. ``wrist_3_link``) -> end-effector frame.
        Preferred: the task is wrist-centric, so EE-frame points are nearly invariant
        to arm pose and learn much better than base-frame points.
      * Articulation WITHOUT ``body_names`` -> robot root (base) frame.
      * Unset -> defaults to ``robot_cfg`` (base frame) for back-compat.

    The reference-frame transform is applied at runtime only; the canonical-points
    cache is frame-agnostic, so switching frames does not invalidate cached points.

    Returns flattened ``[num_envs, num_points * 3]``.
    """

    # Source type constants
    _SRC_ROBOT = 0
    _SRC_INSERTIVE = 1
    _SRC_RECEPTIVE = 2

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from pytorch3d.ops import sample_farthest_points

        self.num_points: int = cfg.params.get("num_points", 512)
        oversample: int = cfg.params.get("oversample", 2)
        # Optional per-point segmentation channel (same scheme as OccludedScenePointCloud): when
        # enabled the cloud is (num_points, 4) = xyz + a per-point class label, so the policy can tell
        # robot / insertive / receptive apart instead of inferring it from geometry. LUT is indexed by
        # source code (_SRC_ROBOT=0 / _SRC_INSERTIVE=1 / _SRC_RECEPTIVE=2). Occluded subclass sets its
        # own copy in its __init__ and appends in its own __call__, so this only drives the clean path.
        self.include_segmentation: bool = cfg.params.get("include_segmentation", False)
        _seg = cfg.params.get("segmentation_labels", None) or {"robot": 0.0, "insertive": -1.0, "receptive": 1.0}
        self._seg_lut = torch.tensor(
            [_seg["robot"], _seg["insertive"], _seg["receptive"]], dtype=torch.float32, device=env.device
        )
        # See MeshPointCloud.resample_on_reset — same idea but for the ins/rec
        # subsets of the scene PC. Robot points stay fixed (robot has no
        # rotational symmetry the network could exploit).
        self.resample_on_reset: bool = cfg.params.get("resample_on_reset", False)
        # When True, ALSO randomize per-env which subset of each robot body's
        # oversampled-mesh pool gets used as the robot points each reset. Without
        # this, robot points are sampled once at init via global RNG state — and
        # because that state depends on num_envs / sim init order, the policy can
        # overfit to a specific point selection that doesn't reproduce at deploy.
        self.resample_on_reset_robot: bool = cfg.params.get("resample_on_reset_robot", False)

        # Enforced per-source split (robot, insertive, receptive). When given, the
        # canonical selection devotes exactly these fractions of num_points to each
        # source (e.g. (0.5, 0.25, 0.25)); None keeps the legacy fixed _BUDGET_SPLIT.
        # NOTE: the teacher/expert ScenePC must stay on the legacy split (the JIT
        # expert was trained on it), so this is opt-in per obs term, not global.
        self.class_ratios = cfg.params.get("class_ratios", None)
        self._budget = (
            ratio_to_counts(self.class_ratios, self.num_points)
            if self.class_ratios is not None
            else self._BUDGET_SPLIT
        )

        robot_cfg: SceneEntityCfg = cfg.params["robot_cfg"]
        ins_cfg: SceneEntityCfg = cfg.params["insertive_cfg"]
        rec_cfg: SceneEntityCfg = cfg.params["receptive_cfg"]

        self.robot: Articulation = env.scene[robot_cfg.name]
        self.insertive: RigidObject = env.scene[ins_cfg.name]
        self.receptive: RigidObject = env.scene[rec_cfg.name]

        # -- Output reference frame --
        # The final PC is expressed in this frame. ``ref_cfg`` is a SceneEntityCfg:
        #   * Articulation WITH body_names (e.g. wrist_3_link) -> end-effector frame.
        #     Strongly preferred: the task is wrist-centric, so EE-frame points are
        #     nearly invariant to arm pose and far more effective for learning.
        #   * Articulation WITHOUT body_names -> robot root (base) frame.
        # Defaults to ``robot_cfg`` (base frame) for back-compat when unset.
        # ``.get`` only substitutes the default for a MISSING key; an explicit
        # ``ref_cfg=None`` (NotEECentric configs) must also fall back to the robot
        # base frame -- hence the explicit ``or robot_cfg``.
        ref_cfg: SceneEntityCfg = cfg.params.get("ref_cfg", robot_cfg) or robot_cfg
        self.ref_asset = env.scene[ref_cfg.name]
        self.ref_body_idx: int | None = None
        if ref_cfg.body_names and isinstance(self.ref_asset, Articulation):
            ref_cfg.resolve(env.scene)
            self.ref_body_idx = ref_cfg.body_ids[0]  # type: ignore

        device = env.device

        # -- Multi-task detection --
        ins_usd_paths = utils.get_usd_paths_from_spawn_cfg(self.insertive.cfg.spawn)
        rec_usd_paths = utils.get_usd_paths_from_spawn_cfg(self.receptive.cfg.spawn)
        self.num_task_types: int = len(ins_usd_paths)
        self.is_multitask: bool = self.num_task_types > 1
        if self.is_multitask:
            self.task_type_ids = torch.arange(env.num_envs, device=device) % self.num_task_types

        # -- Canonical-points cache (hardware-invariant point selection) --
        # pytorch3d's sampling kernels produce different selections across CPU
        # vs GPU (and potentially across NVIDIA archs). To guarantee identical
        # ScenePC observations on every machine, we cache the (T, num_points, ...)
        # canonical-point tensors keyed by the (robot, ins paths, rec paths,
        # num_points, oversample, budget) tuple. Cache lives on HF Hub for shared
        # distribution and falls back to a local ~/.cache/uwlab dir when missing.
        # Caching is skipped when resample_on_reset[/_robot] is enabled because
        # those modes also need the oversampled pools (more state to cache).
        self._cache_key = self._compute_cache_key(ins_usd_paths, rec_usd_paths, oversample)
        cache_usable = not (self.resample_on_reset or self.resample_on_reset_robot)
        generate_mode = os.environ.get("UWLAB_GENERATE_SCENE_PC_CACHE") == "1"

        if cache_usable and not generate_mode:
            # Preferred path: load from HF Hub for deterministic points across machines.
            # If HF is unavailable (rate-limit, 404, network down), fall back to local
            # procedural sampling so training is not blocked by asset availability.
            try:
                self._load_canonical_cache_strict(device)
            except Exception as e:
                print(
                    f"[ScenePointCloud] WARNING: HF cache unavailable for key {self._cache_key} "
                    f"({type(e).__name__}: {e}). Falling back to procedural sampling — "
                    f"points may differ across machines."
                )
                self._sample_and_select_canonical(
                    oversample, ins_usd_paths, rec_usd_paths, env.num_envs, device,
                )
        else:
            # GENERATE path: explicit cache-generation script sets the env var;
            # or resample mode (which can't use a static cache anyway).
            self._sample_and_select_canonical(
                oversample, ins_usd_paths, rec_usd_paths, env.num_envs, device,
            )
            if cache_usable and generate_mode:
                self._save_canonical_cache_local()

        # -- Back-compat aliases for single-task runtime path --
        self.selected_local = self.selected_local_task[0]
        self.selected_source_type = self.selected_source_type_task[0]
        self.selected_body_idx = self.selected_body_idx_task[0]
        self.robot_mask = self.selected_source_type == self._SRC_ROBOT
        self.ins_mask = self.selected_source_type == self._SRC_INSERTIVE
        self.rec_mask = self.selected_source_type == self._SRC_RECEPTIVE

        # Per-env selected_local override for per-reset resampling. When this is
        # populated, ``__call__`` routes through the multi-task style per-env
        # transform path so each env can have its own ins/rec point pattern.
        self.selected_local_per_env: torch.Tensor | None = None
        if self.resample_on_reset or self.resample_on_reset_robot:
            # Initialize per-env table from the per-task canonical (broadcast).
            tt_ids = self.task_type_ids if self.is_multitask else torch.zeros(
                env.num_envs, dtype=torch.long, device=device
            )
            self.selected_local_per_env = self.selected_local_task[tt_ids].clone().contiguous()
            # Cache slot indices for each task type (which positions in the
            # selected_local hold ins / rec points). In single-task all envs
            # share the same source pattern.
            self._ins_slots_task = [
                (self.selected_source_type_task[t] == self._SRC_INSERTIVE).nonzero(as_tuple=True)[0]
                for t in range(self.num_task_types)
            ]
            self._rec_slots_task = [
                (self.selected_source_type_task[t] == self._SRC_RECEPTIVE).nonzero(as_tuple=True)[0]
                for t in range(self.num_task_types)
            ]
            # ``_ins_canonical_pool`` / ``_rec_canonical_pool`` are populated by
            # ``_sample_and_select_canonical`` whenever ``resample_on_reset`` is True.
            self._ins_pool_size = self._ins_canonical_pool.shape[1] if getattr(self, "_ins_canonical_pool", None) is not None else 0
            self._rec_pool_size = self._rec_canonical_pool.shape[1] if getattr(self, "_rec_canonical_pool", None) is not None else 0
            # For robot resampling: per-task, group robot slots by body_idx so we
            # can vectorize the per-reset gather per body.
            self._robot_slots_by_body_task: list[dict[int, torch.Tensor]] = []
            for t in range(self.num_task_types):
                src_t = self.selected_source_type_task[t]
                body_t = self.selected_body_idx_task[t]
                robot_slots = (src_t == self._SRC_ROBOT).nonzero(as_tuple=True)[0]
                slots_by_body: dict[int, torch.Tensor] = {}
                for slot in robot_slots:
                    bi = int(body_t[slot].item())
                    slots_by_body.setdefault(bi, []).append(int(slot.item()))
                slots_by_body_tensor = {
                    bi: torch.tensor(slots, dtype=torch.long, device=device)
                    for bi, slots in slots_by_body.items()
                }
                self._robot_slots_by_body_task.append(slots_by_body_tensor)

        for t in range(self.num_task_types):
            src = self.selected_source_type_task[t]
            n_r = (src == self._SRC_ROBOT).sum().item()
            n_i = (src == self._SRC_INSERTIVE).sum().item()
            n_re = (src == self._SRC_RECEPTIVE).sum().item()
            label = f"task {t}" if self.is_multitask else "single-task"
            # Determinism diagnostic: hash of selected_local tells us if the
            # SPECIFIC points are the same across runs (counts can match while
            # points differ if our selection algorithm is num_envs-dependent).
            points_hash = hex(int(
                (self.selected_local_task[t].double().sum() * 1e9).long() & 0xFFFFFFFF
            ))
            print(
                f"[ScenePointCloud] {label}: {self.num_points} points: "
                f"robot={n_r}, insertive={n_i}, receptive={n_re}, points_hash={points_hash}"
            )

        # -- Visualization --
        self.visualize_enabled = cfg.params.get("visualize", False)
        if self.visualize_enabled:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self.visualize_env_ids = cfg.params.get("visualize_env_ids", [0])
            colors = {
                self._SRC_ROBOT: (0.0, 0.5, 1.0),      # blue for robot
                self._SRC_INSERTIVE: (1.0, 0.0, 0.0),   # red for insertive
                self._SRC_RECEPTIVE: (0.0, 1.0, 0.0),   # green for receptive
            }
            self._pc_markers = {}
            for src_type, color in colors.items():
                label = ["robot", "insertive", "receptive"][src_type]
                marker_cfg = VisualizationMarkersCfg(
                    prim_path=f"/Visuals/scene_pc_{label}",
                    markers={
                        "pt": sim_utils.SphereCfg(
                            radius=0.004,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                        ),
                    },
                )
                self._pc_markers[src_type] = VisualizationMarkers(marker_cfg)

    # ------------------------------------------------------------------------
    # Canonical-points cache
    # ------------------------------------------------------------------------
    _BUDGET_SPLIT = (285, 87, 140)  # (robot, ins, rec) — see selection comment

    def _compute_cache_key(
        self,
        ins_usd_paths: list[str],
        rec_usd_paths: list[str],
        oversample: int,
    ) -> str:
        """SHA256-keyed cache identifier from canonical inputs."""
        robot_id = ",".join(self.robot.body_names)
        parts = "|".join([
            f"robot:{robot_id}",
            "ins:" + ",".join(sorted(ins_usd_paths)),
            "rec:" + ",".join(sorted(rec_usd_paths)),
            f"num_points:{self.num_points}",
            f"oversample:{oversample}",
            f"budget:{self._budget[0]},{self._budget[1]},{self._budget[2]}",
        ])
        return hashlib.sha256(parts.encode()).hexdigest()[:16]

    def _cache_url(self, key: str) -> str:
        return f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset/scene_pc_cache/{key}.pt"

    def _cache_local_fallback(self, key: str) -> str:
        return os.path.join(SCENE_PC_LOCAL_CACHE_DIR, f"{key}.pt")

    def _load_canonical_cache_strict(self, device) -> None:
        """Load canonical points from HF Hub. Raise if missing (no silent drift).

        For generation, set ``UWLAB_GENERATE_SCENE_PC_CACHE=1`` to bypass this
        path entirely and sample fresh — see ``scripts/generate_scene_pc_cache.py``.
        """
        key = self._cache_key
        cache_url = self._cache_url(key)
        try:
            local_path = resolve_cloud_path(cache_url)
        except Exception as e:
            raise RuntimeError(
                f"\n[ScenePointCloud] No canonical points cached on HF for key {key}.\n"
                f"  This usually means a new asset combination (robot+ins+rec or num_points)\n"
                f"  was introduced. To generate the cache locally, run:\n"
                f"      UWLAB_GENERATE_SCENE_PC_CACHE=1 python scripts/generate_scene_pc_cache.py \\\n"
                f"          --task <task_id> [hydra overrides]\n"
                f"  Then publish to HF Hub (every machine will load identical points):\n"
                f"      huggingface-cli upload patrickhaoy/uwlab-assets \\\n"
                f"          {self._cache_local_fallback(key)} \\\n"
                f"          Datasets/OmniReset/scene_pc_cache/{key}.pt \\\n"
                f"          --repo-type=dataset\n"
                f"  Underlying error: {e}"
            ) from e
        try:
            blob = torch.load(local_path, map_location=device)
            self.selected_local_task = blob["selected_local_task"].to(device)
            self.selected_source_type_task = blob["selected_source_type_task"].to(device)
            self.selected_body_idx_task = blob["selected_body_idx_task"].to(device)
        except Exception as e:
            raise RuntimeError(
                f"[ScenePointCloud] cache file at {local_path} is corrupt or missing "
                f"expected keys: {e}"
            ) from e
        print(f"[ScenePointCloud] cache HIT (HF) for key {key} from {local_path}")

    def _save_canonical_cache_local(self) -> None:
        """Save freshly-sampled points to local cache dir (generate mode only).

        After this saves, the user MUST upload to HF Hub for other machines to
        see identical points — the upload command is printed for convenience.
        """
        key = self._cache_key
        local_path = self._cache_local_fallback(key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        torch.save(
            {
                "selected_local_task": self.selected_local_task.cpu(),
                "selected_source_type_task": self.selected_source_type_task.cpu(),
                "selected_body_idx_task": self.selected_body_idx_task.cpu(),
                "num_points": self.num_points,
                "num_task_types": self.num_task_types,
            },
            local_path,
        )
        print(f"[ScenePointCloud] cache SAVED to {local_path}")
        print(
            f"[ScenePointCloud] *** UPLOAD REQUIRED ***  Run:\n"
            f"    huggingface-cli upload patrickhaoy/uwlab-assets \\\n"
            f"        {local_path} \\\n"
            f"        Datasets/OmniReset/scene_pc_cache/{key}.pt \\\n"
            f"        --repo-type=dataset"
        )

    # ------------------------------------------------------------------------
    # Canonical sampling (extracted from __init__ so the cache can skip it)
    # ------------------------------------------------------------------------
    def _sample_and_select_canonical(
        self,
        oversample: int,
        ins_usd_paths: list[str],
        rec_usd_paths: list[str],
        env_num_envs: int,
        device,
    ) -> None:
        """Sample per-body and per-object point pools, run deterministic per-source
        selection, and populate ``selected_local_task`` / ``_source_type_task`` /
        ``_body_idx_task``.

        Note: ``ins_usd_paths`` / ``rec_usd_paths`` are accepted for symmetry with
        the cache key signature even though the sampling itself reads USD prims
        from the live scene rather than the path strings.
        """
        robot_prim_path_env0 = self.robot.cfg.prim_path.replace("env_.*", "env_0")
        body_names = self.robot.body_names

        robot_local_parts: list[torch.Tensor] = []
        robot_body_idx_parts: list[int] = []
        # Keep the pre-FPS oversampled pool per body when resample_on_reset_robot
        # is enabled. The pool is indexed by body_idx_int.
        self._robot_pool_per_body: dict[int, torch.Tensor] = {}
        for body_idx_int, body_name in enumerate(body_names):
            body_prim_path = f"{robot_prim_path_env0}/{body_name}"
            try:
                # Per-body deterministic seed so the selection is reproducible
                # across env-init contexts (training, eval, distributed ranks)
                # regardless of upstream RNG consumption.
                body_points, body_pool = self._sample_body_points(
                    body_prim_path, oversample * 64, 64, device,
                    return_pool=self.resample_on_reset_robot,
                    seed=42 + body_idx_int,
                )
                if body_points is not None and body_points.shape[0] > 0:
                    robot_local_parts.append(body_points)
                    robot_body_idx_parts.extend([body_idx_int] * body_points.shape[0])
                    if body_pool is not None:
                        self._robot_pool_per_body[body_idx_int] = body_pool
            except Exception:
                continue
        robot_local = torch.cat(robot_local_parts, dim=0).to(device)
        robot_body_idx = torch.tensor(robot_body_idx_parts, dtype=torch.long, device=device)

        # -- Sample canonical object points (per-env, heterogeneous-aware) --
        ins_prim_pattern = self.insertive.cfg.prim_path.replace("{ENV_REGEX_NS}", ".*")
        ins_canonical_all = utils.sample_object_point_cloud(
            num_envs=env_num_envs, num_points=oversample * 128,
            prim_path_pattern=ins_prim_pattern, device=str(device),
        )
        rec_prim_pattern = self.receptive.cfg.prim_path.replace("{ENV_REGEX_NS}", ".*")
        rec_canonical_all = utils.sample_object_point_cloud(
            num_envs=env_num_envs, num_points=oversample * 128,
            prim_path_pattern=rec_prim_pattern, device=str(device),
        )
        if self.resample_on_reset:
            self._ins_canonical_pool = ins_canonical_all
            self._rec_canonical_pool = rec_canonical_all

        # -- Build (num_task_types, num_points, ...) canonical selections --
        selected_local_list: list[torch.Tensor] = []
        selected_source_list: list[torch.Tensor] = []
        selected_body_idx_list: list[torch.Tensor] = []
        n_robot_pts = robot_local.shape[0]
        for t in range(self.num_task_types):
            if self.is_multitask:
                rep_env = int((self.task_type_ids == t).nonzero(as_tuple=True)[0][0].item())
            else:
                rep_env = 0

            parts_local = [robot_local]
            parts_source = [torch.full((n_robot_pts,), self._SRC_ROBOT, dtype=torch.long, device=device)]
            parts_body = [robot_body_idx]

            if ins_canonical_all is not None:
                ins_pts = ins_canonical_all[rep_env]
                parts_local.append(ins_pts)
                parts_source.append(torch.full((ins_pts.shape[0],), self._SRC_INSERTIVE, dtype=torch.long, device=device))
                parts_body.append(torch.full((ins_pts.shape[0],), -1, dtype=torch.long, device=device))
            if rec_canonical_all is not None:
                rec_pts = rec_canonical_all[rep_env]
                parts_local.append(rec_pts)
                parts_source.append(torch.full((rec_pts.shape[0],), self._SRC_RECEPTIVE, dtype=torch.long, device=device))
                parts_body.append(torch.full((rec_pts.shape[0],), -1, dtype=torch.long, device=device))

            task_local = torch.cat(parts_local, dim=0)
            task_source = torch.cat(parts_source, dim=0)
            task_body = torch.cat(parts_body, dim=0)

            # Deterministic per-source first-N selection (avoids global-FPS's
            # num_envs / physics-warmup dependence).
            robot_target, ins_target, rec_target = self._budget
            n_robot_avail = int((task_source == self._SRC_ROBOT).sum().item())
            n_ins_avail = int((task_source == self._SRC_INSERTIVE).sum().item())
            n_rec_avail = int((task_source == self._SRC_RECEPTIVE).sum().item())
            n_robot_target = min(robot_target, n_robot_avail)
            n_ins_target = min(ins_target, n_ins_avail)
            n_rec_target = min(self.num_points - n_robot_target - n_ins_target, n_rec_avail)
            shortfall = self.num_points - n_robot_target - n_ins_target - n_rec_target
            if shortfall > 0:
                n_robot_target = min(n_robot_target + shortfall, n_robot_avail)

            robot_idx = (task_source == self._SRC_ROBOT).nonzero(as_tuple=True)[0][:n_robot_target]
            ins_idx = (task_source == self._SRC_INSERTIVE).nonzero(as_tuple=True)[0][:n_ins_target]
            rec_idx = (task_source == self._SRC_RECEPTIVE).nonzero(as_tuple=True)[0][:n_rec_target]
            sel = torch.cat([robot_idx, ins_idx, rec_idx], dim=0)
            if sel.numel() < self.num_points:
                pad = self.num_points - sel.numel()
                sel = torch.cat([sel, torch.zeros(pad, dtype=torch.long, device=device)], dim=0)

            selected_local_list.append(task_local[sel])
            selected_source_list.append(task_source[sel])
            selected_body_idx_list.append(task_body[sel])

        self.selected_local_task = torch.stack(selected_local_list, dim=0)
        self.selected_source_type_task = torch.stack(selected_source_list, dim=0)
        self.selected_body_idx_task = torch.stack(selected_body_idx_list, dim=0)

    def _sample_body_points(
        self, body_prim_path: str, n_oversample: int, n_points: int, device,
        return_pool: bool = False, seed: int = 42,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Sample FPS points from all collision meshes under a robot body prim.

        ``pytorch3d.sample_points_from_meshes`` and ``sample_farthest_points``
        consume from the global ``torch.random`` state, so without seed-locking
        the specific points selected at __init__ depend on RNG state at the call
        site (which varies with num_envs, sim init order, distributed rank, etc).
        That made trained teachers overfit to their training-time selection and
        fail at deploy. We wrap the sampling block in ``temporary_seed(seed)`` so
        the result is reproducible across env-init contexts: training and eval
        get identical points iff they call this function with the same seed AND
        identical mesh geometry.

        Returns ``(selected, pool)``. ``selected`` is the FPS-down-sampled
        ``n_points`` set used for the active selection. ``pool`` is the full
        merged-and-FPS-thinned set used as a per-reset random-subset source when
        ``resample_on_reset_robot=True``; ``None`` when ``return_pool=False``.
        """
        from isaaclab.sim import get_all_matching_child_prims
        from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
        from pytorch3d.structures import Meshes

        collider_prims = get_all_matching_child_prims(
            body_prim_path,
            lambda p: p.HasAPI(UsdPhysics.CollisionAPI) and (
                p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Gprim)
            ),
        )
        if not collider_prims:
            return None, None

        verts_list, faces_list = [], []
        for cp in collider_prims:
            tm = utils.prim_to_trimesh(cp, relative_to_world=False)
            # Apply collider's local transform relative to body
            import omni.usd
            import numpy as np
            body_prim = omni.usd.get_context().get_stage().GetPrimAtPath(body_prim_path)
            body_world_tf = np.array(omni.usd.get_world_transform_matrix(body_prim)).T
            coll_world_tf = np.array(omni.usd.get_world_transform_matrix(cp)).T
            # Relative transform: body_inv @ collider_world
            body_inv = np.linalg.inv(body_world_tf)
            rel_tf = body_inv @ coll_world_tf
            tm.apply_transform(rel_tf)
            v = torch.from_numpy(tm.vertices.astype("float32")).to(device)
            f = torch.from_numpy(tm.faces.astype("int64")).to(device)
            verts_list.append(v)
            faces_list.append(f)

        meshes = Meshes(verts=verts_list, faces=faces_list)
        with utils.temporary_seed(seed):
            samp = sample_points_from_meshes(meshes, n_oversample)  # (n_meshes, n_oversample, 3)
            merged = samp.reshape(1, -1, 3)  # (1, n_meshes*n_oversample, 3)
            # Pool: FPS-thin to a stable, well-spread subset (size = n_oversample)
            # and use as the resample source. This gives more diverse subsets than
            # uniform-random over raw mesh samples while remaining cheap to sample
            # at reset time.
            pool = None
            if return_pool:
                pool_size = min(n_oversample, merged.shape[1])
                pool, _ = sample_farthest_points(merged, K=pool_size)  # (1, pool_size, 3)
                pool = pool[0]  # (pool_size, 3)
            selected, _ = sample_farthest_points(merged, K=min(n_points, merged.shape[1]))
        return selected[0], pool  # (n_points, 3) in body-local frame

    def _transform_to_world_env(
        self, pts_local: torch.Tensor, source_type: torch.Tensor,
        body_idx: torch.Tensor, env_id: int,
    ) -> torch.Tensor:
        """Transform local points to world frame using a specific env's current poses.

        Used at init for FPS selection. Single-task uses env 0; multi-task uses one
        representative env per task type so the selection reflects that task's geometry.
        """
        pts_world = torch.zeros_like(pts_local)

        # Robot body points
        robot_mask = source_type == self._SRC_ROBOT
        if robot_mask.any():
            for bi in body_idx[robot_mask].unique():
                mask_bi = robot_mask & (body_idx == bi)
                pos_w = self.robot.data.body_pos_w[env_id, bi]
                quat_w = self.robot.data.body_quat_w[env_id, bi]
                pts_world[mask_bi] = math_utils.quat_apply(
                    quat_w.unsqueeze(0).expand(mask_bi.sum(), -1),
                    pts_local[mask_bi],
                ) + pos_w

        # Insertive object points
        ins_mask = source_type == self._SRC_INSERTIVE
        if ins_mask.any():
            pos_w = self.insertive.data.root_pos_w[env_id]
            quat_w = self.insertive.data.root_quat_w[env_id]
            pts_world[ins_mask] = math_utils.quat_apply(
                quat_w.unsqueeze(0).expand(ins_mask.sum(), -1),
                pts_local[ins_mask],
            ) + pos_w

        # Receptive object points
        rec_mask = source_type == self._SRC_RECEPTIVE
        if rec_mask.any():
            pos_w = self.receptive.data.root_pos_w[env_id]
            quat_w = self.receptive.data.root_quat_w[env_id]
            pts_world[rec_mask] = math_utils.quat_apply(
                quat_w.unsqueeze(0).expand(rec_mask.sum(), -1),
                pts_local[rec_mask],
            ) + pos_w

        return pts_world

    def _get_ref_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pos_w, quat_w) of the output reference frame, shape [N, 3] and [N, 4]."""
        if self.ref_body_idx is not None:
            return (
                self.ref_asset.data.body_pos_w[:, self.ref_body_idx],
                self.ref_asset.data.body_quat_w[:, self.ref_body_idx],
            )
        return self.ref_asset.data.root_pos_w, self.ref_asset.data.root_quat_w

    def __call__(
        self,
        env: ManagerBasedEnv,
        robot_cfg: SceneEntityCfg,
        insertive_cfg: SceneEntityCfg,
        receptive_cfg: SceneEntityCfg,
        ref_cfg: SceneEntityCfg | None = None,
        num_points: int = 512,
        oversample: int = 2,
        resample_on_reset: bool = False,
        resample_on_reset_robot: bool = False,
        class_ratios: tuple[float, float, float] | None = None,
        visualize: bool = False,
        visualize_env_ids: list[int] | None = None,
        include_segmentation: bool = False,
        segmentation_labels: dict | None = None,
    ) -> torch.Tensor:
        # include_segmentation / segmentation_labels are consumed in __init__ (-> self.include_segmentation,
        # self._seg_lut); they appear here only so the ObservationManager accepts them as term params.
        N = env.num_envs
        P = self.num_points
        device = env.device

        # Per-reset resampling: replace ins/rec slots in selected_local_per_env
        # for envs that just reset (episode_length_buf == 0). Uses the
        # oversampled pools captured at init.
        do_resample = (self.resample_on_reset or self.resample_on_reset_robot)
        if do_resample and self.selected_local_per_env is not None and hasattr(env, "episode_length_buf"):
            reset_mask = env.episode_length_buf == 0
            if reset_mask.any():
                reset_ids = reset_mask.nonzero(as_tuple=True)[0]
                n_reset = reset_ids.shape[0]
                # Per-env slot indices (single-task: same for all envs; multi-task: per task type).
                if self.is_multitask:
                    rid_tt = self.task_type_ids[reset_ids]
                else:
                    rid_tt = torch.zeros(n_reset, dtype=torch.long, device=device)
                # Process each task type's reset envs in a single tensor op.
                for t in range(self.num_task_types):
                    in_t = (rid_tt == t).nonzero(as_tuple=True)[0]
                    if in_t.numel() == 0:
                        continue
                    rids_t = reset_ids[in_t]  # env_ids in task t
                    n_t = rids_t.shape[0]
                    if self.resample_on_reset:
                        ins_slots = self._ins_slots_task[t]  # [n_ins_t]
                        rec_slots = self._rec_slots_task[t]
                        if ins_slots.numel() > 0 and self._ins_pool_size > 0:
                            # Random subset: pick ins_slots.numel() indices from pool per env.
                            ins_perm = torch.argsort(
                                torch.rand(n_t, self._ins_pool_size, device=device), dim=1
                            )[:, : ins_slots.numel()]  # [n_t, n_ins_t]
                            ins_pool_t = self._ins_canonical_pool[rids_t]  # [n_t, pool, 3]
                            gathered = torch.gather(
                                ins_pool_t, 1, ins_perm.unsqueeze(-1).expand(-1, -1, 3)
                            )  # [n_t, n_ins_t, 3]
                            self.selected_local_per_env[rids_t.unsqueeze(1), ins_slots.unsqueeze(0)] = gathered
                        if rec_slots.numel() > 0 and self._rec_pool_size > 0:
                            rec_perm = torch.argsort(
                                torch.rand(n_t, self._rec_pool_size, device=device), dim=1
                            )[:, : rec_slots.numel()]
                            rec_pool_t = self._rec_canonical_pool[rids_t]
                            gathered = torch.gather(
                                rec_pool_t, 1, rec_perm.unsqueeze(-1).expand(-1, -1, 3)
                            )
                            self.selected_local_per_env[rids_t.unsqueeze(1), rec_slots.unsqueeze(0)] = gathered
                    if self.resample_on_reset_robot:
                        # Per-body random subset of robot points. For each body
                        # appearing in this task's robot slots, gather n_slots
                        # random points from that body's pool per reset env.
                        slots_by_body = self._robot_slots_by_body_task[t]
                        for bi, bi_slots in slots_by_body.items():
                            if bi not in self._robot_pool_per_body:
                                continue
                            pool = self._robot_pool_per_body[bi]  # [pool_size, 3]
                            pool_size = pool.shape[0]
                            n_slots = int(bi_slots.numel())
                            # (n_t, n_slots) random indices into the pool.
                            rand_idx = torch.randint(0, pool_size, (n_t, n_slots), device=device)
                            gathered = pool[rand_idx]  # [n_t, n_slots, 3]
                            self.selected_local_per_env[rids_t.unsqueeze(1), bi_slots.unsqueeze(0)] = gathered

        # Route through the multi-task per-env path when per-env data is present
        # (either is_multitask, or resample_on_reset built selected_local_per_env).
        if self.is_multitask or self.selected_local_per_env is not None:
            # Per-env canonical selections.
            # env_local:       (N, P, 3) — uses per-env override if resample_on_reset is on
            # env_source_type: (N, P)
            # env_body_idx:    (N, P)
            if self.selected_local_per_env is not None:
                env_local = self.selected_local_per_env
            else:
                env_local = self.selected_local_task[self.task_type_ids]
            tt_ids = self.task_type_ids if self.is_multitask else torch.zeros(N, dtype=torch.long, device=device)
            env_source_type = self.selected_source_type_task[tt_ids]
            env_body_idx = self.selected_body_idx_task[tt_ids]

            robot_mask_np = env_source_type == self._SRC_ROBOT  # (N, P)
            ins_mask_np = env_source_type == self._SRC_INSERTIVE
            rec_mask_np = env_source_type == self._SRC_RECEPTIVE

            # Per-point robot body transform via gather over body dimension.
            body_pos_w = self.robot.data.body_pos_w  # (N, B, 3)
            body_quat_w = self.robot.data.body_quat_w  # (N, B, 4)
            safe_body_idx = env_body_idx.clamp(min=0)  # non-robot points masked away below
            pos_per_pt = torch.gather(body_pos_w, 1, safe_body_idx.unsqueeze(-1).expand(-1, -1, 3))
            quat_per_pt = torch.gather(body_quat_w, 1, safe_body_idx.unsqueeze(-1).expand(-1, -1, 4))
            robot_world = math_utils.quat_apply(quat_per_pt, env_local) + pos_per_pt

            # Insertive / receptive transforms broadcast per env.
            ins_pos = self.insertive.data.root_pos_w.unsqueeze(1).expand(-1, P, -1)
            ins_quat = self.insertive.data.root_quat_w.unsqueeze(1).expand(-1, P, -1)
            ins_world = math_utils.quat_apply(ins_quat, env_local) + ins_pos

            rec_pos = self.receptive.data.root_pos_w.unsqueeze(1).expand(-1, P, -1)
            rec_quat = self.receptive.data.root_quat_w.unsqueeze(1).expand(-1, P, -1)
            rec_world = math_utils.quat_apply(rec_quat, env_local) + rec_pos

            pts_world = torch.zeros(N, P, 3, device=device)
            pts_world = torch.where(robot_mask_np.unsqueeze(-1), robot_world, pts_world)
            pts_world = torch.where(ins_mask_np.unsqueeze(-1), ins_world, pts_world)
            pts_world = torch.where(rec_mask_np.unsqueeze(-1), rec_world, pts_world)
        else:
            # Single-task fast path: one canonical table shared across all envs.
            pts_world = torch.zeros(N, P, 3, device=device)

            if self.robot_mask.any():
                robot_local = self.selected_local[self.robot_mask]
                robot_body_ids = self.selected_body_idx[self.robot_mask]
                for bi in robot_body_ids.unique():
                    mask_bi = robot_body_ids == bi
                    n_bi = mask_bi.sum()
                    local_pts = robot_local[mask_bi]
                    pos_w = self.robot.data.body_pos_w[:, bi]
                    quat_w = self.robot.data.body_quat_w[:, bi]
                    expanded_local = local_pts.unsqueeze(0).expand(N, -1, -1)
                    expanded_quat = quat_w.unsqueeze(1).expand(-1, n_bi, -1)
                    world_pts = math_utils.quat_apply(expanded_quat, expanded_local) + pos_w.unsqueeze(1)
                    robot_indices = torch.where(self.robot_mask)[0]
                    bi_indices = robot_indices[mask_bi]
                    pts_world[:, bi_indices] = world_pts

            if self.ins_mask.any():
                ins_local = self.selected_local[self.ins_mask]
                n_ins = ins_local.shape[0]
                pos_w = self.insertive.data.root_pos_w
                quat_w = self.insertive.data.root_quat_w
                expanded_local = ins_local.unsqueeze(0).expand(N, -1, -1)
                expanded_quat = quat_w.unsqueeze(1).expand(-1, n_ins, -1)
                world_pts = math_utils.quat_apply(expanded_quat, expanded_local) + pos_w.unsqueeze(1)
                ins_indices = torch.where(self.ins_mask)[0]
                pts_world[:, ins_indices] = world_pts

            if self.rec_mask.any():
                rec_local = self.selected_local[self.rec_mask]
                n_rec = rec_local.shape[0]
                pos_w = self.receptive.data.root_pos_w
                quat_w = self.receptive.data.root_quat_w
                expanded_local = rec_local.unsqueeze(0).expand(N, -1, -1)
                expanded_quat = quat_w.unsqueeze(1).expand(-1, n_rec, -1)
                world_pts = math_utils.quat_apply(expanded_quat, expanded_local) + pos_w.unsqueeze(1)
                rec_indices = torch.where(self.rec_mask)[0]
                pts_world[:, rec_indices] = world_pts

        # -- Visualize in world frame --
        if self.visualize_enabled:
            # Per-env masks because in multi-task, different envs may have different
            # source_type assignments for the same point index.
            for src_type, marker in self._pc_markers.items():
                vis_chunks = []
                for vid in self.visualize_env_ids:
                    if self.is_multitask:
                        env_src = self.selected_source_type_task[self.task_type_ids[vid]]
                    else:
                        env_src = self.selected_source_type
                    mask = env_src == src_type
                    if mask.any():
                        indices = torch.where(mask)[0]
                        vis_chunks.append(pts_world[vid, indices])
                if vis_chunks:
                    marker.visualize(translations=torch.cat(vis_chunks, dim=0))

        # -- Transform to the output reference frame --
        # Robot base frame by default; end-effector (e.g. wrist_3_link) frame when
        # ``ref_cfg`` carries body_names. See ``__init__`` / ``_get_ref_pose_w``.
        ref_pos_w, ref_quat_w = self._get_ref_pose_w()  # (N, 3), (N, 4)
        ref_quat_inv = math_utils.quat_inv(ref_quat_w)

        pts_ref = math_utils.quat_apply(
            ref_quat_inv.unsqueeze(1).expand(-1, P, -1),
            pts_world - ref_pos_w.unsqueeze(1),
        )

        # Optional 4th channel: per-point class label (robot/insertive/receptive). The per-point
        # source code is constant for the run; pick the per-env table (multi-task / resampled path)
        # or the shared single-task table, then map through the label LUT.
        if self.include_segmentation:
            if self.is_multitask or self.selected_local_per_env is not None:
                tt_ids = self.task_type_ids if self.is_multitask else torch.zeros(N, dtype=torch.long, device=device)
                src_per_env = self.selected_source_type_task[tt_ids]              # (N, P)
            else:
                src_per_env = self.selected_source_type.unsqueeze(0).expand(N, P)  # (N, P)
            labels = self._seg_lut[src_per_env]                                   # (N, P) class label
            pts_ref = torch.cat([pts_ref, labels.unsqueeze(-1)], dim=-1)          # (N, P, 4)

        return pts_ref.reshape(N, -1)


class OccludedScenePointCloud(ScenePointCloud):
    """Robot-only point cloud in the front-camera frame with sim2real augmentation.

    Subclass of :class:`ScenePointCloud` for the no-object sim2real scene. It
    samples a dense robot point cloud **area-weighted across all robot bodies**
    (big links get proportionally more points, so coverage is even rather than
    concentrated on the many small gripper meshes), then on every step:

        transform robot cloud -> front-camera optical frame
          -> frustum cull (keep only points inside the camera FOV, in front)
          -> HPR self-occlusion (keep only points not occluded from the camera)
          -> edge bleed (boundary skirt toward the table)
          -> low-frequency surface bias
          -> light dropout
          -> resample to a fixed ``num_points``.

    Visibility (frustum + occlusion) uses pure geometry -- projection and the
    Katz HPR convex hull -- **no ray tracing**, matching what the real D455 +
    FoundationStereo will produce at eval time.

    Output is expressed in the **front-camera optical frame** (z forward, x
    right, y down), flattened to ``[num_envs, num_points * 3]`` -- the same frame
    the real camera point cloud arrives in.

    The augmentation runs on CPU (numpy/scipy) so keep ``num_envs`` modest --
    this term is intended for sanity-checking, not large-scale training.

    Visualization (when ``visualize=True``) draws the final camera-visible
    augmented cloud (red) as markers for the ``visualize_env_ids`` envs.
    """

    # opengl -> optical (ROS) convention: rotate 180 deg about the local x-axis.
    _QUAT_X180 = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z)

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # NOTE: deliberately skip ScenePointCloud.__init__ (it requires
        # insertive/receptive scene entities, which this scene does not have).
        ManagerTermBase.__init__(self, cfg, env)
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp import pc_sim2real
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp import pc_sim2real_torch

        self._aug = pc_sim2real  # numpy reference (AugParams lives here)
        self._augt = pc_sim2real_torch  # batched GPU pipeline (used at runtime)
        device = env.device

        self.num_points: int = cfg.params.get("num_points", 2000)
        oversample: int = cfg.params.get("oversample", 3)

        # Enforced (robot, insertive, receptive) split. When set, BOTH stages honor it:
        #   1. the DENSE pre-occlusion cloud is allocated per-class by this ratio (so
        #      more DISTINCT object points survive the z-buffer cull -> real signal for
        #      the maxpool encoder), and
        #   2. the final resample is STRATIFIED to these exact output counts.
        # None -> legacy area-weighted dense sample + uniform resample.
        self.class_ratios = cfg.params.get("class_ratios", None)

        robot_cfg: SceneEntityCfg = cfg.params["robot_cfg"]
        self.robot: Articulation = env.scene[robot_cfg.name]

        # Flags so any inherited helper that branches on them behaves single-task.
        self.num_task_types = 1
        self.is_multitask = False
        self.selected_local_per_env = None
        self.resample_on_reset = False
        self.resample_on_reset_robot = False

        # -- Optional output reference frame --
        # By default the cloud stays in the front-camera optical frame (the frame the
        # real D455 + FoundationStereo cloud arrives in). When ``ref_cfg`` carries
        # body_names (e.g. wrist_3_link), the final cloud is instead re-expressed in
        # that body's (end-effector) frame -- nearly arm-pose-invariant, matching the
        # clean ScenePointCloud's EE-frame option. The occlusion itself is unchanged
        # (still computed from the camera viewpoint); only the output coordinates move.
        # NOTE: at deploy, apply the SAME camera->EE transform (known from the camera
        # extrinsics + wrist FK) to the real cloud, or the frame won't match training.
        ref_cfg: SceneEntityCfg | None = cfg.params.get("ref_cfg", None)
        self.ref_asset = None
        self.ref_body_idx: int | None = None
        if ref_cfg is not None:
            self.ref_asset = env.scene[ref_cfg.name]
            if ref_cfg.body_names and isinstance(self.ref_asset, Articulation):
                ref_cfg.resolve(env.scene)
                self.ref_body_idx = ref_cfg.body_ids[0]  # type: ignore

        # -- Optional scene objects to ALSO sample (insertive / receptive) --
        # The dense cloud becomes robot + objects, so the SAME camera-frame HPR /
        # frustum occlusion is computed jointly: the robot occludes objects, an
        # object occludes the robot, and objects occlude each other -- no special
        # cross-object code, it just falls out of one shared visibility pass.
        # Each rigid object is a RigidObject moved by its root pose at runtime.
        # Source codes reuse the parent's constants (robot=0, ins=1, rec=2) so the
        # visualization can colour them distinctly.
        self.object_assets: list[RigidObject] = []
        self.object_src_codes: list[int] = []
        for key, src_code in (("insertive_cfg", self._SRC_INSERTIVE), ("receptive_cfg", self._SRC_RECEPTIVE)):
            ocfg = cfg.params.get(key, None)
            if ocfg is not None:
                self.object_assets.append(env.scene[ocfg.name])
                self.object_src_codes.append(src_code)

        # -- Optional robot-body include-list --
        # ``robot_body_names``: list of regex patterns resolved against
        # ``self.robot.body_names``; only matching bodies contribute points (e.g.
        # gripper-only clouds for policies that shouldn't see the arm). None (default)
        # -> all bodies, the legacy behavior. Body indices stay ABSOLUTE articulation
        # indices, so the runtime per-body pose transform is untouched.
        # ``include_wrist_camera_mesh``: when False, the collision-only wrist D415
        # camera mesh is NOT sampled (see _load_body_meshes).
        self._include_wrist_cam: bool = cfg.params.get("include_wrist_camera_mesh", True)
        # ``occlude_excluded_bodies``: excluded bodies still get sampled as OCCLUDER-ONLY
        # points (``occluder_points`` budget, area-weighted, D415 mount kept) -- they feed
        # the z-buffer so the arm still hides the gripper/objects behind it, but can never
        # appear in the output cloud. Off by default (legacy: excluded bodies vanish).
        self._occlude_excluded: bool = cfg.params.get("occlude_excluded_bodies", False)
        self._occluder_points: int = cfg.params.get("occluder_points", 6144)
        # Occluder points are splatted over a (2r+1)^2 z-buffer neighbourhood: one cell
        # per point is far too sparse to block a big surface (arm ~5-12k cells vs a few
        # thousand front-face occluder samples). r=2 -> ~13mm footprint at 0.5m.
        self._occluder_splat: int = int(cfg.params.get("occluder_splat_radius", 2))
        body_patterns: list[str] | None = cfg.params.get("robot_body_names", None)
        self._robot_body_include: set[int] | None = None
        if body_patterns is not None:
            self._robot_body_include = {
                i for i, name in enumerate(self.robot.body_names)
                if any(re.fullmatch(pat, name) for pat in body_patterns)
            }
            # An explicit EMPTY list means "no robot output points at all" (objects-only
            # cloud; pair with occlude_excluded_bodies=True + a zero robot class ratio so
            # the whole robot still blocks the view). Non-empty patterns matching nothing
            # are a user error.
            if not self._robot_body_include and body_patterns:
                raise ValueError(
                    f"[OccludedScenePointCloud] robot_body_names patterns {body_patterns} matched no "
                    f"bodies; available: {list(self.robot.body_names)}"
                )
            included = [self.robot.body_names[i] for i in sorted(self._robot_body_include)]
            excluded = [n for n in self.robot.body_names if n not in included]
            print(f"[OccludedScenePointCloud] robot bodies INCLUDED ({len(included)}): {included}")
            print(f"[OccludedScenePointCloud] robot bodies EXCLUDED ({len(excluded)}): {excluded}")

        # -- Dense scene cloud, area-weighted across ALL robot bodies + objects --
        robot_prim_path_env0 = self.robot.cfg.prim_path.replace("env_.*", "env_0")
        env_prefix = robot_prim_path_env0.rsplit("/", 1)[0]  # ".../env_0"
        (self.dense_local, self.dense_body_idx, self.dense_obj_idx,
         self.dense_source, self.dense_occluder) = self._sample_scene_cloud_area_weighted(
            robot_prim_path_env0, env_prefix, total=self.num_points * oversample, device=device,
        )
        n_obj = int((self.dense_obj_idx >= 0).sum())
        n_occ = int(self.dense_occluder.sum())
        print(f"[OccludedScenePointCloud] dense scene cloud: {self.dense_local.shape[0]} points "
              f"({self.dense_local.shape[0] - n_obj - n_occ} robot + {n_obj} object + {n_occ} occluder-only, "
              f"{'ratio ' + str(self.class_ratios) if self.class_ratios else 'area-weighted'} over "
              f"{len(self.robot.body_names)} bodies + {len(self.object_assets)} objects) "
              f"-> {self.num_points} in camera frame")

        # Per-class output targets (src_code, count) for the stratified final resample.
        # Summed counts == num_points exactly; None -> legacy uniform resample.
        self._out_targets = None
        if self.class_ratios is not None:
            nr, ni, nrec = ratio_to_counts(self.class_ratios, self.num_points)
            self._out_targets = [
                (self._SRC_ROBOT, nr), (self._SRC_INSERTIVE, ni), (self._SRC_RECEPTIVE, nrec),
            ]
            print(f"[OccludedScenePointCloud] enforced output split robot/ins/rec = {nr}/{ni}/{nrec}")

        # -- Zero-pad fully occluded classes --
        # When a class has ZERO visible points in an env (e.g. peg fully hidden behind the
        # gripper), the stratified resample's default fills that class's slots from the env's
        # full visible set (ratio drifts, no explicit signal). With ``zero_pad_missing_class``
        # those slots become (0,0,0) in the OUTPUT frame instead — an explicit "class absent"
        # signal. Seg labels on padded slots keep the slot's OWN class code (slot layout is
        # fixed by _out_targets), so a 4D cloud reads "insertive: all zeros". Requires
        # class_ratios. Real-side deploys must replicate the same convention.
        self.zero_pad_missing: bool = cfg.params.get("zero_pad_missing_class", False)
        self._slot_codes: torch.Tensor | None = None
        if self.zero_pad_missing:
            if self._out_targets is None:
                raise ValueError("[OccludedScenePointCloud] zero_pad_missing_class requires class_ratios.")
            self._slot_codes = torch.cat(
                [torch.full((cnt,), code, dtype=torch.long, device=device) for code, cnt in self._out_targets]
            )  # (num_points,) class code each output slot belongs to
            print("[OccludedScenePointCloud] zero_pad_missing_class=True (fully occluded class -> zeros)")

        # -- Optional PER-PRIM labeling --
        # When enabled, the cloud is the SAME ratio-enforced occluded cloud, but each output
        # point additionally carries the SOURCE PRIM it belongs to (robot body link /
        # insertive / receptive) as an extra trailing channel. A downstream model can then
        # select which prims to consume (drop individual links, all robot links, or an object)
        # and zero-pad to a fixed size AT TRAINING TIME. No per-prim cap / padding is applied
        # here. Off by default -> legacy flat cloud, so existing datasets / experts are
        # unaffected. Requires class_ratios (the ratio-enforced cloud) to be set.
        self.per_prim: bool = cfg.params.get("per_prim", False)
        if self.per_prim:
            self._build_prim_ids()

        # -- Fixed front-camera offset (relative to the /Robot root prim) --
        # Defaults match camera_align_cfg.py front_camera (opengl convention).
        off_pos = cfg.params.get("camera_offset_pos", (1.0770121, -0.1679045, 0.4486344))
        off_quat_opengl = cfg.params.get("camera_offset_quat", (0.70564552, 0.46613815, 0.25072644, 0.47107948))
        self.cam_offset_pos = torch.tensor(off_pos, dtype=torch.float32, device=device)  # (3,)
        # Convert opengl-convention offset to optical (z forward) once at init.
        q_opengl = torch.tensor(off_quat_opengl, dtype=torch.float32, device=device)
        q_x180 = torch.tensor(self._QUAT_X180, dtype=torch.float32, device=device)
        self.cam_offset_quat = math_utils.quat_mul(q_opengl, q_x180)  # (4,) optical
        self.bg_plane_z: float = cfg.params.get("bg_plane_z", -0.02)

        # -- Extrinsics domain randomization (per-env, resampled on reset) --
        # The calibrated offset above is the CENTER; each env samples an offset in
        # a box/cone around it so the policy is robust to real-rig miscalibration.
        # ``camera_offset_pos_range`` is +/- meters per axis (uniform);
        # ``camera_offset_rot_range_deg`` is the max angle (uniform in +/-, random
        # axis). Both default to 0 -> a single fixed (calibrated) extrinsic.
        self._off_pos_range = torch.tensor(
            cfg.params.get("camera_offset_pos_range", (0.0, 0.0, 0.0)), dtype=torch.float32, device=device
        )  # (3,)
        self._off_rot_range = float(cfg.params.get("camera_offset_rot_range_deg", 0.0)) * (torch.pi / 180.0)
        self._dr_extrinsics = bool(self._off_pos_range.abs().sum() > 0 or self._off_rot_range > 0)
        # Per-env offset buffers (start at the calibrated center).
        N = env.num_envs
        self.cam_offset_pos_env = self.cam_offset_pos.unsqueeze(0).repeat(N, 1).contiguous()  # (N, 3)
        self.cam_offset_quat_env = self.cam_offset_quat.unsqueeze(0).repeat(N, 1).contiguous()  # (N, 4) optical

        # -- Augmentation params (overridable via cfg.params["aug_params"]) --
        aug_params = cfg.params.get("aug_params", None)
        self.aug_params = aug_params if aug_params is not None else pc_sim2real.AugParams()
        # -- Focal length (mm): ``focal_length_mm`` overrides the AugParams default;
        # ``focal_length_mm_range`` is +/- mm, resampled per env on reset with the
        # extrinsics (intrinsics DR for the frustum/z-buffer FOV).
        focal_override = cfg.params.get("focal_length_mm", None)
        if focal_override is not None:
            self.aug_params = dataclasses.replace(self.aug_params, focal_length_mm=float(focal_override))
        self._focal_center: float = float(self.aug_params.focal_length_mm)
        self._focal_range: float = float(cfg.params.get("focal_length_mm_range", 0.0))
        self._dr_extrinsics = self._dr_extrinsics or self._focal_range > 0
        self.focal_mm_env = torch.full((N,), self._focal_center, dtype=torch.float32, device=device)  # (N,)
        # Per-env RNG seed base. Fixed per env => stable noise pattern across
        # steps (occlusion still updates as the robot moves), nicer for video.
        self._rng_seed: int = cfg.params.get("rng_seed", 0)
        # Device-resident RNG for the batched torch augmentation (no host syncs).
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(self._rng_seed))
        # Occlusion / z-buffer grid (Hz, Wz). The scatter-min buffer is
        # N*Hz*Wz floats, so this trades occlusion fidelity for memory at scale.
        self._zbuf_hw: tuple[int, int] = tuple(
            cfg.params.get("occlusion_grid", pc_sim2real_torch.DEFAULT_ZBUF_HW)
        )
        # Sanity overlay: also draw the OLD numpy-pipeline visibility (frustum +
        # Katz HPR) in a distinct colour, to eyeball torch (z-buffer) vs numpy.
        self._viz_compare_numpy: bool = cfg.params.get("viz_compare_numpy", False)
        # When True, visualize the FULL augmented obs cloud (incl. edge bleed +
        # fliers) instead of the occluded source-coloured visibility -- lets you
        # see the flier outliers against the real overlay.
        self._visualize_augmented: bool = cfg.params.get("visualize_augmented", False)

        # -- Optional per-point segmentation channel --
        # When enabled, the obs is (num_points, 4): xyz + a class label per point,
        # so the policy can tell robot / insertive / receptive apart instead of
        # inferring it from geometry. Labels are configurable; the default matches
        # the user's scheme (robot=0, insertive=-1, receptive=+1). The label rides
        # through the augmentation via point_source (see augment_pointcloud_batched).
        self.include_segmentation: bool = cfg.params.get("include_segmentation", False)
        seg = cfg.params.get("segmentation_labels", None) or {"robot": 0.0, "insertive": -1.0, "receptive": 1.0}
        # LUT indexed by source code: _SRC_ROBOT=0, _SRC_INSERTIVE=1, _SRC_RECEPTIVE=2.
        self._seg_lut = torch.tensor(
            [seg["robot"], seg["insertive"], seg["receptive"]], dtype=torch.float32, device=device
        )

        # -- Visualization --
        # -- Optional real point-cloud overlay (sim-vs-real comparison) --
        # A .ply captured by the real front camera (FoundationStereo), in the
        # SAME camera optical frame as our output. Rendered as a distinct color
        # so the sim2real domain gap is directly visible. Viz-only; does not
        # affect the returned observation.
        self.overlay_local: torch.Tensor | None = None
        overlay_path = cfg.params.get("overlay_pc_path", None)
        if overlay_path:
            self.overlay_local = self._load_overlay_pc(
                overlay_path, cfg.params.get("overlay_pc_max_points", 6000), device,
            )
            print(f"[OccludedScenePointCloud] overlay real cloud: {self.overlay_local.shape[0]} points "
                  f"from {overlay_path}")

        self.visualize_enabled = cfg.params.get("visualize", False)
        if self.visualize_enabled:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self.visualize_env_ids = cfg.params.get("visualize_env_ids", [0])
            # One marker per source so robot / insertive / receptive are visually
            # distinct in the sanity check: robot=red, insertive=green, receptive=blue.
            src_colors = {
                self._SRC_ROBOT: ("robot", (1.0, 0.1, 0.1)),
                self._SRC_INSERTIVE: ("insertive", (0.1, 0.9, 0.2)),
                self._SRC_RECEPTIVE: ("receptive", (1.0, 0.2, 1.0)),  # magenta (blue clashes with table strips)
            }
            self._src_markers: dict[int, object] = {}
            for code, (label, color) in src_colors.items():
                m_cfg = VisualizationMarkersCfg(
                    prim_path=f"/Visuals/sim2real_pc_{label}",
                    markers={
                        "pt": sim_utils.SphereCfg(
                            radius=0.01,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                        ),
                    },
                )
                self._src_markers[code] = VisualizationMarkers(m_cfg)

            # Optional augmented-cloud marker (red) -- the full obs cloud with
            # edge bleed + fliers, for showing the synthetic outliers.
            self._aug_marker = None
            if self._visualize_augmented:
                aug_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/sim2real_pc_augmented",
                    markers={
                        "pt": sim_utils.SphereCfg(
                            radius=0.01,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
                        ),
                    },
                )
                self._aug_marker = VisualizationMarkers(aug_cfg)

            # Optional numpy-pipeline comparison marker (yellow).
            self._numpy_marker = None
            if self._viz_compare_numpy:
                np_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/sim2real_pc_numpy",
                    markers={
                        "pt": sim_utils.SphereCfg(
                            radius=0.01,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)),
                        ),
                    },
                )
                self._numpy_marker = VisualizationMarkers(np_cfg)

            self._overlay_marker = None
            if self.overlay_local is not None:
                overlay_color = tuple(cfg.params.get("overlay_pc_color", (0.1, 0.4, 1.0)))
                overlay_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/sim2real_pc_real",
                    markers={
                        "pt": sim_utils.SphereCfg(
                            radius=0.01,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=overlay_color),
                        ),
                    },
                )
                self._overlay_marker = VisualizationMarkers(overlay_marker_cfg)

    def _load_overlay_pc(self, path: str, max_points: int, device) -> torch.Tensor:
        """Load a .ply point cloud (front-camera optical frame) -> (P, 3) tensor."""
        import trimesh

        mesh = trimesh.load(path, process=False)
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        if max_points and pts.shape[0] > max_points:
            rng = np.random.default_rng(0)
            pts = pts[rng.choice(pts.shape[0], size=max_points, replace=False)]
        return torch.from_numpy(pts).to(device)

    # ------------------------------------------------------------------------
    # Area-weighted scene mesh sampling (robot bodies + rigid objects)
    # ------------------------------------------------------------------------
    def _sample_scene_cloud_area_weighted(
        self, robot_prim_path_env0: str, env_prefix: str, total: int, device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``total`` points over all robot bodies AND scene objects, with
        the per-source budget allocated proportional to each source's mesh area.

        A "source" is either a robot body (transformed at runtime by that body's
        pose) or a rigid object (transformed by its root pose). pytorch3d's
        ``sample_points_from_meshes`` is already area-uniform within a source; the
        per-source area weighting keeps the overall surface density consistent so
        small parts (gripper meshes, the peg) aren't over- or under-sampled
        relative to the big arm links.

        Returns ``(local (M,3), body_idx (M,), obj_idx (M,), source (M,))`` in each
        source's local frame. For robot points ``obj_idx == -1`` and ``body_idx``
        is the articulation body index; for object points ``body_idx == 0`` and
        ``obj_idx`` indexes ``self.object_assets``. ``source`` is the colour code
        (robot=0, insertive=1, receptive=2).
        """
        from pytorch3d.ops import sample_points_from_meshes

        debug = os.environ.get("UWLAB_PC_DEBUG") == "1"
        if debug:
            self._debug_enumerate_robot_meshes(robot_prim_path_env0)

        # Collect every source as (meshes, area, kind, idx, src_code, seed).
        sources: list[dict] = []
        for body_idx_int, body_name in enumerate(self.robot.body_names):
            # Optional include-list (e.g. gripper-only cloud). Indices stay absolute.
            if self._robot_body_include is not None and body_idx_int not in self._robot_body_include:
                continue
            try:
                meshes, area = self._load_body_meshes(f"{robot_prim_path_env0}/{body_name}", device)
            except Exception as e:
                if debug:
                    print(f"[PC_DEBUG] body '{body_name}': _load_body_meshes raised {type(e).__name__}: {e}")
                meshes, area = None, 0.0
            if debug:
                print(f"[PC_DEBUG] body '{body_name}': meshes={0 if meshes is None else len(meshes)} area={area:.5f}")
            if meshes is not None and area > 0.0:
                sources.append(dict(meshes=meshes, area=area, kind="robot", idx=body_idx_int,
                                    src=self._SRC_ROBOT, seed=42 + body_idx_int))

        for obj_i, asset in enumerate(self.object_assets):
            obj_prim = asset.cfg.prim_path.replace("env_.*", "env_0").replace("{ENV_REGEX_NS}", env_prefix)
            try:
                meshes, area = self._load_body_meshes(obj_prim, device)
            except Exception as e:
                if debug:
                    print(f"[PC_DEBUG] object '{obj_prim}': _load_body_meshes raised {type(e).__name__}: {e}")
                meshes, area = None, 0.0
            if debug:
                print(f"[PC_DEBUG] object '{obj_prim}': meshes={0 if meshes is None else len(meshes)} area={area:.5f}")
            if meshes is not None and area > 0.0:
                sources.append(dict(meshes=meshes, area=area, kind="object", idx=obj_i,
                                    src=self.object_src_codes[obj_i], seed=1000 + obj_i))

        total_area = sum(s["area"] for s in sources)

        # Per-source point budget. Legacy: area-weighted over ALL sources (robot wins
        # by surface area -> ~90%+ robot). With class_ratios: split `total` into per-class
        # budgets by ratio FIRST, then area-weight WITHIN each class. This raises the
        # dense object density so more DISTINCT peg/hole points survive occlusion.
        if self.class_ratios is not None:
            cls_dense = ratio_to_counts(self.class_ratios, total)  # (robot, ins, rec)
            cls_budget = {self._SRC_ROBOT: cls_dense[0], self._SRC_INSERTIVE: cls_dense[1],
                          self._SRC_RECEPTIVE: cls_dense[2]}
            cls_area: dict[int, float] = {}
            for s in sources:
                cls_area[s["src"]] = cls_area.get(s["src"], 0.0) + s["area"]

            def _n_for(s) -> int:
                ca = cls_area.get(s["src"], 0.0)
                budget = cls_budget.get(s["src"], 0)
                if ca <= 0.0 or budget <= 0:
                    return 0
                return max(1, int(round(budget * s["area"] / ca)))
        else:
            def _n_for(s) -> int:
                return max(1, int(round(total * s["area"] / total_area)))

        # -- Occluder-only sources: EXCLUDED robot bodies still physically block the
        # camera's view of the gripper/objects. When enabled, sample them too (with the
        # D415 stand-in kept -- the real mount occludes), flagged so the augmentation
        # keeps them for the z-buffer but never selects them into the output.
        occ_sources: list[dict] = []
        if self._occlude_excluded and self._robot_body_include is not None:
            for body_idx_int, body_name in enumerate(self.robot.body_names):
                if body_idx_int in self._robot_body_include:
                    continue
                try:
                    meshes, area = self._load_body_meshes(
                        f"{robot_prim_path_env0}/{body_name}", device, keep_d415=True
                    )
                except Exception:
                    meshes, area = None, 0.0
                if meshes is not None and area > 0.0:
                    occ_sources.append(dict(meshes=meshes, area=area, kind="robot", idx=body_idx_int,
                                            src=self._SRC_ROBOT, seed=42 + body_idx_int))
            occ_area = sum(s["area"] for s in occ_sources)
            for s in occ_sources:
                s["n_occ"] = max(1, int(round(self._occluder_points * s["area"] / occ_area)))

        local_parts: list[torch.Tensor] = []
        body_idx_parts: list[int] = []
        obj_idx_parts: list[int] = []
        source_parts: list[int] = []
        occluder_parts: list[bool] = []
        for s in sources + occ_sources:
            is_occ = "n_occ" in s
            n_b = s["n_occ"] if is_occ else _n_for(s)
            if n_b <= 0:
                continue
            with utils.temporary_seed(s["seed"]):
                pts = sample_points_from_meshes(s["meshes"], n_b)[0]  # (n_b, 3) local
            local_parts.append(pts)
            n = pts.shape[0]
            if s["kind"] == "robot":
                body_idx_parts.extend([s["idx"]] * n)
                obj_idx_parts.extend([-1] * n)
            else:
                body_idx_parts.extend([0] * n)
                obj_idx_parts.extend([s["idx"]] * n)
            source_parts.extend([s["src"]] * n)
            occluder_parts.extend([is_occ] * n)

        local = torch.cat(local_parts, dim=0).to(device)
        body_idx = torch.tensor(body_idx_parts, dtype=torch.long, device=device)
        obj_idx = torch.tensor(obj_idx_parts, dtype=torch.long, device=device)
        source = torch.tensor(source_parts, dtype=torch.long, device=device)
        occluder = torch.tensor(occluder_parts, dtype=torch.bool, device=device)

        if debug:
            # Count points that land inside the d415 camera mesh's local bbox to
            # confirm the wrist camera is actually getting sampled.
            self._debug_count_in_mesh(
                robot_prim_path_env0, "robotiq_base_link", "d415_and_cable",
                local[obj_idx < 0], body_idx[obj_idx < 0], device,
            )
        return local, body_idx, obj_idx, source, occluder

    def _debug_count_in_mesh(self, robot_prim_path_env0, body_name, mesh_leaf,
                             local, body_idx, device) -> None:
        import numpy as _np
        import omni.usd
        from isaaclab.sim import get_all_matching_child_prims

        bidx = list(self.robot.body_names).index(body_name)
        body_path = f"{robot_prim_path_env0}/{body_name}"
        hits = get_all_matching_child_prims(
            body_path, lambda p: p.IsA(UsdGeom.Mesh) and p.GetName() == mesh_leaf
        )
        if not hits:
            print(f"[PC_DEBUG] mesh '{mesh_leaf}' not found under {body_name}")
            return
        stage = omni.usd.get_context().get_stage()
        body_prim = stage.GetPrimAtPath(body_path)
        body_inv = _np.linalg.inv(_np.array(omni.usd.get_world_transform_matrix(body_prim)).T)
        tm = utils.prim_to_trimesh(hits[0], relative_to_world=False)
        tm.apply_transform(body_inv @ _np.array(omni.usd.get_world_transform_matrix(hits[0])).T)
        lo = torch.tensor(tm.vertices.min(0), dtype=torch.float32, device=device)
        hi = torch.tensor(tm.vertices.max(0), dtype=torch.float32, device=device)
        body_pts = local[body_idx == bidx]
        inside = ((body_pts >= lo) & (body_pts <= hi)).all(dim=1).sum().item()
        print(f"[PC_DEBUG] body '{body_name}' total sampled pts={body_pts.shape[0]}, "
              f"inside '{mesh_leaf}' bbox={inside}")

    def _debug_enumerate_robot_meshes(self, robot_prim_path_env0: str) -> None:
        """Print every UsdGeom.Mesh prim under the robot root, with area + per-body
        classification. Gated by ``UWLAB_PC_DEBUG=1``; used to track down meshes
        (e.g. the wrist camera) that aren't getting sampled."""
        import omni.usd
        from isaaclab.sim import get_all_matching_child_prims

        body_name_set = set(self.robot.body_names)
        all_meshes = get_all_matching_child_prims(robot_prim_path_env0, lambda p: p.IsA(UsdGeom.Mesh))
        print(f"[PC_DEBUG] === {len(all_meshes)} total UsdGeom.Mesh prims under {robot_prim_path_env0} ===")
        for p in all_meshes:
            path = str(p.GetPath())
            # which articulation body (if any) is this mesh under?
            rel = path[len(robot_prim_path_env0) + 1:] if path.startswith(robot_prim_path_env0) else path
            top = rel.split("/")[0] if rel else ""
            under_body = top if top in body_name_set else f"NOT-A-BODY({top})"
            try:
                tm = utils.prim_to_trimesh(p, relative_to_world=False)
                area = float(tm.area)
            except Exception as e:
                area = -1.0
                print(f"[PC_DEBUG]   prim_to_trimesh FAILED: {type(e).__name__}: {e}")
            kind = "collision" if "/collisions/" in path else "visual"
            print(f"[PC_DEBUG]   [{kind:9s}] body={under_body:24s} area={area:.6f}  {path}")

    def _load_body_meshes(self, body_prim_path: str, device, keep_d415: bool | None = None):
        """Load the camera-visible meshes under a robot body as a pytorch3d ``Meshes``.

        What the external camera physically sees = the renderable **visual**
        meshes, PLUS any **collision-only** geometry that stands in for a real
        prop with no visual twin -- notably the wrist-mounted D415 camera
        (``robotiq_base_link/collisions/d415_and_cable``), which is collision-only
        in the USD but is a real object the external camera observes.

        Selection rule (paths are scoped ``.../visuals/...`` vs ``.../collisions/...``):
          - all visual meshes (the appearance), and
          - collision meshes whose name has no same-named visual sibling
            (collision-only props), skipping collision duplicates of visuals so
            the gripper isn't double-weighted.

        Returns ``(Meshes | None, total_area)`` with vertices in the body-local frame.
        """
        import numpy as _np
        import omni.usd
        from isaaclab.sim import get_all_matching_child_prims
        from pytorch3d.structures import Meshes

        all_meshes = get_all_matching_child_prims(body_prim_path, lambda p: p.IsA(UsdGeom.Mesh))
        if not all_meshes:
            return None, 0.0
        visual_prims = [p for p in all_meshes if "/collisions/" not in str(p.GetPath())]
        collision_prims = [p for p in all_meshes if "/collisions/" in str(p.GetPath())]
        visual_names = {p.GetName() for p in visual_prims}
        # collision-only extras (e.g. the d415 camera): a collision mesh with no
        # same-named visual mesh in this body.
        extra_prims = [p for p in collision_prims if p.GetName() not in visual_names]
        # ``keep_d415`` overrides the cfg flag -- occluder-only sampling always keeps the
        # D415 stand-in (the physical camera mount blocks the view even when the OUTPUT
        # cloud excludes it).
        if not (self._include_wrist_cam if keep_d415 is None else keep_d415):
            # Gripper-only clouds may want pure gripper geometry: drop the wrist
            # D415 camera stand-in (collision-only, name contains "d415").
            extra_prims = [p for p in extra_prims if "d415" not in p.GetName().lower()]
        use_prims = visual_prims + extra_prims
        if not use_prims:
            return None, 0.0

        stage = omni.usd.get_context().get_stage()
        body_prim = stage.GetPrimAtPath(body_prim_path)
        body_world_tf = _np.array(omni.usd.get_world_transform_matrix(body_prim)).T
        body_inv = _np.linalg.inv(body_world_tf)

        # Merge all of this body's meshes into a SINGLE mesh (concatenate verts,
        # offset faces). This must be one mesh, not a batch: the caller samples
        # ``sample_points_from_meshes(meshes, n_b)[0]``, which would otherwise
        # take points from only the first mesh and silently drop the rest (e.g.
        # the wrist d415 camera, which is a second mesh under robotiq_base_link).
        verts_list, faces_list, total_area = [], [], 0.0
        vert_offset = 0
        for cp in use_prims:
            tm = utils.prim_to_trimesh(cp, relative_to_world=False)
            coll_world_tf = _np.array(omni.usd.get_world_transform_matrix(cp)).T
            tm.apply_transform(body_inv @ coll_world_tf)
            total_area += float(tm.area)
            verts_list.append(torch.from_numpy(tm.vertices.astype("float32")).to(device))
            faces_list.append(torch.from_numpy(tm.faces.astype("int64")).to(device) + vert_offset)
            vert_offset += tm.vertices.shape[0]
        if not verts_list:
            return None, 0.0
        merged_verts = torch.cat(verts_list, dim=0)
        merged_faces = torch.cat(faces_list, dim=0)
        return Meshes(verts=[merged_verts], faces=[merged_faces]), total_area

    # ------------------------------------------------------------------------
    # Per-prim labeling
    # ------------------------------------------------------------------------
    def _build_prim_ids(self) -> None:
        """Assign every dense point a PRIM id (robot body link / insertive / receptive) and
        build the ordered prim-name table. This is the ONLY thing per-prim mode adds at
        collection: the cloud itself is the normal ratio-enforced occluded cloud; each point
        just additionally carries the prim it belongs to (rode through the resample as a label,
        emitted as an extra channel). NO per-prim cap / padding here -- zero-padding is a
        training-time concern (when a run selects a subset of prims).

        Prim order: robot body links present in the dense cloud (articulation body-index
        order), then scene objects (insertive, then receptive). ``prim_id`` indexes
        ``self.prim_names``; objects/robot points absent from the dense cloud never appear.

        Populates ``self.dense_prim_id (M,)`` (float, for label-gather) and
        ``self.prim_names`` (the public id->name table the collector records)."""
        robot_mask = self.dense_obj_idx < 0
        prim_id = torch.full_like(self.dense_body_idx, -1)
        prim_names: list[str] = []
        next_id = 0
        present_bodies = sorted(int(b) for b in self.dense_body_idx[robot_mask].unique().tolist())
        for bi in present_bodies:
            prim_id[robot_mask & (self.dense_body_idx == bi)] = next_id
            prim_names.append(self.robot.body_names[bi])
            next_id += 1
        obj_label = {self._SRC_INSERTIVE: "insertive", self._SRC_RECEPTIVE: "receptive"}
        for obj_i, src_code in enumerate(self.object_src_codes):
            m = self.dense_obj_idx == obj_i
            if not bool(m.any()):
                continue
            prim_id[m] = next_id
            prim_names.append(obj_label.get(src_code, f"object_{obj_i}"))
            next_id += 1

        self.prim_names = prim_names
        # Float so it rides the resample via the same gather as the seg label.
        self.dense_prim_id = prim_id.to(torch.float32)
        print(f"[OccludedScenePointCloud] PER-PRIM labeling: {len(prim_names)} prims "
              f"(each output point tagged with its prim; no cap) -> {prim_names}")

    # ------------------------------------------------------------------------
    # Runtime transforms
    # ------------------------------------------------------------------------
    def _resample_extrinsics(self, env: ManagerBasedEnv) -> None:
        """Resample per-env camera offsets (around the calibrated center) for envs
        that just reset. No-op unless extrinsics DR is enabled."""
        if not self._dr_extrinsics or not hasattr(env, "episode_length_buf"):
            return
        reset_mask = env.episode_length_buf == 0
        n = int(reset_mask.sum())
        if n == 0:
            return
        device = self.cam_offset_pos.device
        # position: uniform box around the calibrated center
        dp = (torch.rand(n, 3, generator=self._generator, device=device) * 2 - 1) * self._off_pos_range
        self.cam_offset_pos_env[reset_mask] = self.cam_offset_pos.unsqueeze(0) + dp
        # orientation: uniform angle in [-range, range] about a random axis
        if self._off_rot_range > 0:
            axis = torch.randn(n, 3, generator=self._generator, device=device)
            axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-9)
            ang = (torch.rand(n, generator=self._generator, device=device) * 2 - 1) * self._off_rot_range
            dq = math_utils.quat_from_angle_axis(ang, axis)  # (n, 4)
            self.cam_offset_quat_env[reset_mask] = math_utils.quat_mul(
                dq, self.cam_offset_quat.unsqueeze(0).expand(n, -1)
            )
        # focal length: uniform in [-range, range] around the center
        if self._focal_range > 0:
            df = (torch.rand(n, generator=self._generator, device=device) * 2 - 1) * self._focal_range
            self.focal_mm_env[reset_mask] = self._focal_center + df

    def _camera_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Front-camera world pose (optical convention). Returns ``(pos (N,3), quat (N,4))``.

        Uses the per-env offset buffers, so extrinsics DR (when enabled) gives each
        env its own camera pose."""
        root_pos_w = self.robot.data.root_pos_w  # (N, 3)
        root_quat_w = self.robot.data.root_quat_w  # (N, 4)
        cam_pos_w = root_pos_w + math_utils.quat_apply(root_quat_w, self.cam_offset_pos_env)
        cam_quat_w = math_utils.quat_mul(root_quat_w, self.cam_offset_quat_env)
        return cam_pos_w, cam_quat_w

    def _ref_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose of the output reference frame (``ref_cfg``). Returns ``(pos
        (N,3), quat (N,4))`` -- the named body's pose when ``ref_body_idx`` is set
        (end-effector frame), else the asset root pose (base frame)."""
        if self.ref_body_idx is not None:
            return (self.ref_asset.data.body_pos_w[:, self.ref_body_idx],
                    self.ref_asset.data.body_quat_w[:, self.ref_body_idx])
        return self.ref_asset.data.root_pos_w, self.ref_asset.data.root_quat_w

    def _robot_points_cam(self, env: ManagerBasedEnv):
        """Transform the dense scene cloud (robot + objects) into the front-camera
        optical frame.

        Robot points are placed by their articulation body pose; object points by
        their rigid-object root pose. The result is one combined cloud, so the
        downstream HPR / frustum visibility pass naturally handles robot<->object
        and object<->object occlusion.

        Returns ``(pts_cam (N,M,3), cam_pos_w (N,3), cam_quat_w (N,4))``.
        """
        N = env.num_envs
        M = self.dense_local.shape[0]
        device = env.device
        local = self.dense_local.unsqueeze(0).expand(N, -1, -1)  # (N, M, 3)
        pts_world = torch.zeros(N, M, 3, device=device)

        # Robot points: per-point body transform.
        robot_mask = self.dense_obj_idx < 0
        if bool(robot_mask.any()):
            bidx = self.dense_body_idx[robot_mask]  # (m,)
            pos_per_pt = self.robot.data.body_pos_w[:, bidx]  # (N, m, 3)
            quat_per_pt = self.robot.data.body_quat_w[:, bidx]  # (N, m, 4)
            pts_world[:, robot_mask] = (
                math_utils.quat_apply(quat_per_pt, local[:, robot_mask]) + pos_per_pt
            )

        # Object points: each object placed by its root pose.
        for obj_i, asset in enumerate(self.object_assets):
            omask = self.dense_obj_idx == obj_i
            if not bool(omask.any()):
                continue
            m = int(omask.sum())
            opos = asset.data.root_pos_w  # (N, 3)
            oquat = asset.data.root_quat_w  # (N, 4)
            pts_world[:, omask] = (
                math_utils.quat_apply(oquat.unsqueeze(1).expand(-1, m, -1), local[:, omask])
                + opos.unsqueeze(1)
            )

        cam_pos_w, cam_quat_w = self._camera_pose_w()
        cam_quat_inv = math_utils.quat_inv(cam_quat_w)
        pts_cam = math_utils.quat_apply(
            cam_quat_inv.unsqueeze(1).expand(-1, M, -1),
            pts_world - cam_pos_w.unsqueeze(1),
        )  # (N, M, 3) optical frame
        return pts_cam, cam_pos_w, cam_quat_w

    def _visualize_sources(self, pts_cam: torch.Tensor, cam_pos_w: torch.Tensor, cam_quat_w: torch.Tensor) -> None:
        """Draw the frustum+HPR-visible dense cloud, one marker colour per source.

        Operates on the same per-point ``dense_source`` labels as the combined
        cloud, so robot / insertive / receptive points stay distinguishable after
        the joint occlusion pass.
        """
        params = self.aug_params
        # Same visibility (frustum + z-buffer occlusion) as the obs, on-device, so
        # the coloured sanity view matches exactly what the policy cloud is built
        # from. (Edge-bleed / bias / dropout are obs-only; not shown here.)
        valid = torch.ones(pts_cam.shape[:2], dtype=torch.bool, device=pts_cam.device)
        if params.enable_frustum_cull:
            valid = valid & self._augt.frustum_cull_mask(pts_cam, params, focal_mm=self.focal_mm_env)
        if params.enable_hpr:
            valid = valid & self._augt.zbuffer_visible_mask(
                pts_cam, valid, params, self._zbuf_hw, focal_mm=self.focal_mm_env
            )

        # accumulate world-frame survivors per source across viz envs
        per_src: dict[int, list[torch.Tensor]] = {code: [] for code in self._src_markers}
        for vid in self.visualize_env_ids:
            m = valid[vid]  # (M,)
            if not bool(m.any()):
                continue
            p = pts_cam[vid][m]  # (m, 3)
            world = math_utils.quat_apply(
                cam_quat_w[vid].unsqueeze(0).expand(p.shape[0], -1), p
            ) + cam_pos_w[vid]
            src = self.dense_source[m]
            for code in per_src:
                sel = world[src == code]
                if sel.shape[0] > 0:
                    per_src[code].append(sel)
        for code, marker in self._src_markers.items():
            pts = per_src[code]
            if pts:
                marker.visualize(translations=torch.cat(pts, dim=0))

        # Optional: overlay the OLD numpy pipeline's visibility (frustum + Katz
        # HPR) in yellow, on the SAME dense cloud, for a torch-vs-numpy check.
        if self._numpy_marker is not None:
            np_world = []
            for vid in self.visualize_env_ids:
                p = pts_cam[vid].detach().cpu().numpy()
                idx = np.arange(p.shape[0])
                if params.enable_frustum_cull:
                    fx, fy, cx, cy = params.intrinsics()
                    keep = self._aug.frustum_cull(p, fx, fy, cx, cy, params.img_width, params.img_height, params.near)
                    p, idx = p[keep], idx[keep]
                if len(p) and params.enable_hpr:
                    vis = self._aug.occlude_hpr(p, np.zeros(3, dtype=np.float32), params.hpr_radius_scale)
                    p, idx = p[vis], idx[vis]
                if len(p) == 0:
                    continue
                p_t = torch.from_numpy(p.astype(np.float32)).to(pts_cam.device)
                np_world.append(
                    math_utils.quat_apply(cam_quat_w[vid].unsqueeze(0).expand(p_t.shape[0], -1), p_t)
                    + cam_pos_w[vid]
                )
            if np_world:
                self._numpy_marker.visualize(translations=torch.cat(np_world, dim=0))

        if os.environ.get("UWLAB_PC_DEBUG") == "1":
            label = {self._SRC_ROBOT: "robot", self._SRC_INSERTIVE: "insertive", self._SRC_RECEPTIVE: "receptive"}
            counts = {label[c]: int(sum(t.shape[0] for t in per_src[c])) for c in per_src}
            print(f"[PC_DEBUG] visible (frustum+HPR) per source: {counts}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        robot_cfg: SceneEntityCfg,
        insertive_cfg: SceneEntityCfg | None = None,
        receptive_cfg: SceneEntityCfg | None = None,
        num_points: int = 2000,
        oversample: int = 3,
        class_ratios: tuple[float, float, float] | None = None,
        ref_cfg: SceneEntityCfg | None = None,
        camera_offset_pos=None,
        camera_offset_quat=None,
        camera_offset_pos_range=None,
        camera_offset_rot_range_deg: float = 0.0,
        focal_length_mm: float | None = None,
        focal_length_mm_range: float = 0.0,
        bg_plane_z: float = -0.02,
        aug_params=None,
        rng_seed: int = 0,
        occlusion_grid=None,
        viz_compare_numpy: bool = False,
        visualize_augmented: bool = False,
        visualize: bool = False,
        visualize_env_ids: list[int] | None = None,
        overlay_pc_path: str | None = None,
        overlay_pc_color=None,
        overlay_pc_max_points: int = 6000,
        include_segmentation: bool = False,
        segmentation_labels: dict | None = None,
        per_prim: bool = False,
        robot_body_names: list[str] | None = None,
        include_wrist_camera_mesh: bool = True,
        zero_pad_missing_class: bool = False,
        occlude_excluded_bodies: bool = False,
        occluder_points: int = 6144,
        occluder_splat_radius: int = 2,
    ) -> torch.Tensor:
        N = env.num_envs
        device = env.device

        self._resample_extrinsics(env)  # per-env extrinsics DR for just-reset envs (no-op if disabled)
        pts_cam, cam_pos_w, cam_quat_w = self._robot_points_cam(env)  # (N, M, 3) optical

        # Background (table) plane expressed in each env's camera frame, for the
        # edge-bleed skirt. Table is horizontal at world z = base_z + bg_plane_z.
        base_pos_w = self.robot.data.root_pos_w  # (N, 3)
        normal_w = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, -1)
        point_w = base_pos_w.clone()
        point_w[:, 2] = point_w[:, 2] + self.bg_plane_z
        cam_quat_inv = math_utils.quat_inv(cam_quat_w)
        plane_normal_cam = math_utils.quat_apply(cam_quat_inv, normal_w)  # (N, 3)
        plane_point_cam = math_utils.quat_apply(cam_quat_inv, point_w - cam_pos_w)  # (N, 3)

        # Fully batched, on-device augmentation (frustum -> z-buffer occlusion ->
        # edge bleed -> surface bias -> dropout -> resample). No host syncs.
        # With segmentation on, also gather a per-output source code via the same
        # resample indices (out_src: (N, num_points), values in {robot,ins,rec} codes).
        # Source codes are threaded through the pipeline when we need a per-point
        # label out (segmentation) OR when the final resample is stratified by class.
        out_src = None
        out_prim = None  # per-prim: per-output-point prim id, appended as the trailing channel
        pad_mask = None  # zero-pad: (N, num_points) True on slots of fully occluded classes
        occ_mask = self.dense_occluder if bool(self.dense_occluder.any()) else None
        if self.per_prim:
            # Per-prim labeling: the NORMAL ratio-enforced occluded cloud, plus a per-point prim
            # id gathered through the same resample. No cap / padding (training-time concern).
            out_t, out_src, out_prim, *rest = self._augt.augment_pointcloud_batched(
                pts_cam, self.aug_params, plane_point_cam, plane_normal_cam,
                self.num_points, self._zbuf_hw, self._generator,
                point_source=self.dense_source,  # class codes: needed to stratify by class_ratios
                class_targets=self._out_targets,
                point_prim=self.dense_prim_id,
                return_pad_mask=self.zero_pad_missing,
                occluder_mask=occ_mask,
                occluder_splat_radius=self._occluder_splat,
                focal_mm=self.focal_mm_env,
            )
            pad_mask = rest[0] if rest else None
            if not self.include_segmentation:
                out_src = None  # class label requested only to stratify; don't emit a seg channel
        elif self.include_segmentation or self._out_targets is not None:
            out_t, out_src, *rest = self._augt.augment_pointcloud_batched(
                pts_cam, self.aug_params, plane_point_cam, plane_normal_cam,
                self.num_points, self._zbuf_hw, self._generator,
                point_source=self.dense_source, class_targets=self._out_targets,
                return_pad_mask=self.zero_pad_missing,
                occluder_mask=occ_mask,
                occluder_splat_radius=self._occluder_splat,
                focal_mm=self.focal_mm_env,
            )
            pad_mask = rest[0] if rest else None
        else:
            out_t = self._augt.augment_pointcloud_batched(
                pts_cam, self.aug_params, plane_point_cam, plane_normal_cam,
                self.num_points, self._zbuf_hw, self._generator,
                occluder_mask=occ_mask,
                occluder_splat_radius=self._occluder_splat,
                focal_mm=self.focal_mm_env,
            )  # (N, num_points, 3) camera frame

        # -- Visualize the camera-visible cloud, coloured by source --
        # We re-run only the *visibility* stages (frustum + HPR) on the combined
        # cloud so each surviving point keeps its source label (robot / insertive
        # / receptive). This is exactly the occlusion the policy obs is built on;
        # the returned obs additionally applies edge-bleed / bias / dropout.
        if self.visualize_enabled:
            if self._aug_marker is not None:
                # Draw the FULL augmented obs cloud (incl. fliers) so the
                # synthetic outliers are visible against the real overlay.
                aug_world = []
                for vid in self.visualize_env_ids:
                    pc = out_t[vid]  # (P, 3) camera frame, post-augmentation
                    aug_world.append(
                        math_utils.quat_apply(cam_quat_w[vid].unsqueeze(0).expand(pc.shape[0], -1), pc)
                        + cam_pos_w[vid]
                    )
                if aug_world:
                    self._aug_marker.visualize(translations=torch.cat(aug_world, dim=0))
            else:
                self._visualize_sources(pts_cam, cam_pos_w, cam_quat_w)

            # Real-cloud overlay: same camera optical frame -> world via the sim
            # camera pose, for the first visualize env.
            if self._overlay_marker is not None and self.overlay_local is not None and self.visualize_env_ids:
                vid = self.visualize_env_ids[0]
                P = self.overlay_local.shape[0]
                ov_world = math_utils.quat_apply(
                    cam_quat_w[vid].unsqueeze(0).expand(P, -1), self.overlay_local,
                ) + cam_pos_w[vid]
                self._overlay_marker.visualize(translations=ov_world)

        # Re-express the cloud from the camera optical frame into the output
        # reference frame (e.g. end-effector), if configured. Camera-frame -> world
        # -> ref. Done after visualization (which assumes camera frame) and before the
        # seg channel is appended, so labels stay aligned.
        if self.ref_asset is not None:
            P = self.num_points
            xyz_world = (
                math_utils.quat_apply(cam_quat_w.unsqueeze(1).expand(-1, P, -1), out_t) + cam_pos_w.unsqueeze(1)
            )
            ref_pos_w, ref_quat_w = self._ref_pose_w()
            ref_quat_inv = math_utils.quat_inv(ref_quat_w)
            out_t = math_utils.quat_apply(
                ref_quat_inv.unsqueeze(1).expand(-1, P, -1), xyz_world - ref_pos_w.unsqueeze(1)
            )  # (N, num_points, 3) in the reference frame

        # Zero-pad fully occluded classes: coords -> (0,0,0) in the OUTPUT frame (must run
        # AFTER the ref-frame re-expression, or zeros would land at the camera origin), and
        # seg labels -> the slot's own class code (so 4D reads "class X: all zeros").
        if pad_mask is not None and bool(pad_mask.any()):
            out_t = torch.where(pad_mask.unsqueeze(-1), torch.zeros_like(out_t), out_t)
            if out_src is not None:
                out_src = torch.where(pad_mask, self._slot_codes.unsqueeze(0).expand_as(out_src), out_src)

        # Append the per-point class label as a 4th channel -> (N, num_points, 4).
        # Done after visualization (which expects xyz-only) and after fliers, so the
        # label stays aligned with the final points.
        if self.include_segmentation:
            labels = self._seg_lut[out_src]  # (N, num_points) source code -> class label
            out_t = torch.cat([out_t, labels.unsqueeze(-1)], dim=-1)  # (N, num_points, 4)

        # PER-PRIM: append the per-point prim id as the FINAL channel (after xyz and any seg
        # channel). The collector peels this last channel into a separate `scene_pc_prim_id`
        # dataset; training uses it to select prims + zero-pad. Cloud values are unchanged.
        if out_prim is not None:
            out_t = torch.cat([out_t, out_prim.unsqueeze(-1)], dim=-1)  # (N, num_points, base+1)

        return out_t.reshape(N, -1)
