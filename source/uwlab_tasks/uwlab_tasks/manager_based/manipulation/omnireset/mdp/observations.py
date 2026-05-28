# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import os

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


class ScenePointCloud(ManagerTermBase):
    """Combined pointcloud from robot + insertive + receptive in robot base frame.

    At init: samples canonical points from all robot body meshes, insertive object,
    and receptive object. Transforms all to world frame using default poses, concatenates,
    and applies FPS to select ``num_points`` total. Records each selected point's source
    (robot body index or object) and its local-frame coordinates.

    At runtime: transforms each point from its local frame to world frame using current
    body/object poses, then transforms all to robot base frame. No runtime FPS — the
    selected points are fixed for the entire training run.

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

        robot_cfg: SceneEntityCfg = cfg.params["robot_cfg"]
        ins_cfg: SceneEntityCfg = cfg.params["insertive_cfg"]
        rec_cfg: SceneEntityCfg = cfg.params["receptive_cfg"]

        self.robot: Articulation = env.scene[robot_cfg.name]
        self.insertive: RigidObject = env.scene[ins_cfg.name]
        self.receptive: RigidObject = env.scene[rec_cfg.name]

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
            f"budget:{self._BUDGET_SPLIT[0]},{self._BUDGET_SPLIT[1]},{self._BUDGET_SPLIT[2]}",
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
            robot_target, ins_target, rec_target = self._BUDGET_SPLIT
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

    def __call__(
        self,
        env: ManagerBasedEnv,
        robot_cfg: SceneEntityCfg,
        insertive_cfg: SceneEntityCfg,
        receptive_cfg: SceneEntityCfg,
        num_points: int = 512,
        oversample: int = 2,
        resample_on_reset: bool = False,
        resample_on_reset_robot: bool = False,
        visualize: bool = False,
        visualize_env_ids: list[int] | None = None,
    ) -> torch.Tensor:
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

        # -- Transform to robot base frame --
        base_pos_w = self.robot.data.root_pos_w  # (N, 3)
        base_quat_w = self.robot.data.root_quat_w  # (N, 4)
        base_quat_inv = math_utils.quat_inv(base_quat_w)

        pts_base = math_utils.quat_apply(
            base_quat_inv.unsqueeze(1).expand(-1, P, -1),
            pts_world - base_pos_w.unsqueeze(1),
        )

        return pts_base.reshape(N, -1)
