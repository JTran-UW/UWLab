# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

from uwlab_tasks.manager_based.manipulation.omnireset.assembly_keypoints import Offset
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils


def asset_pose_in_gc_goal_frame(
    env: ManagerBasedEnv,
    target: str = "insertive_object",
    ee_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wrist_3_link"),
    event_term_name: str = "reset_from_reset_states",
    rotation_repr: str = "axis_angle",
):
    """Current pose of an asset expressed in the frame of its GCMRM-sampled goal pose.

    ``target`` selects the asset: a rigid object name (its root pose vs the goal root pose), or
    ``"ee"`` (the ``ee_asset_cfg`` body vs the FK-derived goal EE pose). Goal poses are env-relative
    in ``goal_state``; env origins are added before the frame subtraction.
    """
    goal_state = env.event_manager.get_term_cfg(event_term_name).func.goal_state

    if target == "ee":
        goal_pose = goal_state["articulation"]["robot"]["ee_pose"]
        ee_asset: Articulation = env.scene[ee_asset_cfg.name]
        body_idx = 0 if isinstance(ee_asset_cfg.body_ids, slice) else ee_asset_cfg.body_ids
        current_pos = ee_asset.data.body_link_pos_w[:, body_idx].view(-1, 3)
        current_quat = ee_asset.data.body_link_quat_w[:, body_idx].view(-1, 4)
    else:
        goal_pose = goal_state["rigid_object"][target]["root_pose"]
        asset: RigidObject = env.scene[target]
        current_pos = asset.data.root_pos_w
        current_quat = asset.data.root_quat_w

    goal_pos_w = goal_pose[:, :3] + env.scene.env_origins
    pos_b, quat_b = math_utils.subtract_frame_transforms(goal_pos_w, goal_pose[:, 3:7], current_pos, current_quat)

    if rotation_repr == "axis_angle":
        return torch.cat([pos_b, math_utils.axis_angle_from_quat(quat_b)], dim=1)
    elif rotation_repr == "quat":
        return torch.cat([pos_b, quat_b], dim=1)
    else:
        raise ValueError(f"Invalid rotation_repr: {rotation_repr}. Must be one of: 'quat', 'axis_angle'")


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


def target_asset_pose_in_root_asset_frame_with_gap(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset=None,
    root_asset_offset=None,
    rotation_repr: str = "quat",
    pos_noise_std: float | None = None,
    rot_noise_std: float | None = None,
    pos_bias=None,
    rot_bias=None,
) -> torch.Tensor:
    """``target_asset_pose_in_root_asset_frame`` with optional observation-gap corruption.

    Two kinds of corruption, each on the position (m) and rotation (axis-angle, rad) parts:

    * ``pos_noise_std`` / ``rot_noise_std``: zero-mean additive gaussian noise, resampled every
      step. None or <= 0 disables.
    * ``pos_bias`` / ``rot_bias``: a fixed constant additive offset -- a scalar (added to every
      component) or a 3-sequence (per-component). None or all-zero adds nothing.

    Defaults (None here; 0.0 / [0,0,0] in the task cfgs so scalar and list CLI overrides pass
    ``update_class_from_dict``'s type check) are a clean passthrough, so swapping this in for the
    plain function leaves the task unchanged, and the knobs are hydra-overridable, e.g.::

        env.observations.policy.insertive_asset_pose.params.pos_noise_std=0.005
        env.observations.policy.insertive_asset_pose.params.pos_bias=[0.01,0,0]

    Any rotation knob requires ``rotation_repr='axis_angle'`` (additive corruption of a
    quaternion is not well-defined).
    """
    rot_gap_active = (rot_noise_std is not None and rot_noise_std > 0.0) or (
        rot_bias is not None and torch.as_tensor(rot_bias).abs().sum() > 0
    )
    if rot_gap_active and rotation_repr != "axis_angle":
        raise ValueError("Rotation noise/bias requires rotation_repr='axis_angle'.")

    obs = target_asset_pose_in_root_asset_frame(
        env, target_asset_cfg, root_asset_cfg, target_asset_offset, root_asset_offset, rotation_repr
    )
    if pos_noise_std is not None and pos_noise_std > 0.0:
        obs[:, :3] += torch.randn_like(obs[:, :3]) * pos_noise_std
    if rot_noise_std is not None and rot_noise_std > 0.0:
        obs[:, 3:6] += torch.randn_like(obs[:, 3:6]) * rot_noise_std
    if pos_bias is not None:
        obs[:, :3] += torch.as_tensor(pos_bias, dtype=obs.dtype, device=obs.device)
    if rot_bias is not None:
        obs[:, 3:6] += torch.as_tensor(rot_bias, dtype=obs.dtype, device=obs.device)
    return obs


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
    return asset.root_physx_view.get_material_properties().view(env.num_envs, -1)


def get_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_masses().view(env.num_envs, -1)


def get_joint_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction_coeff.view(env.num_envs, -1)


def get_joint_armature(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_armature.view(env.num_envs, -1)


def get_joint_stiffness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness.view(env.num_envs, -1)


def get_joint_damping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping.view(env.num_envs, -1)


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
    grayscale: bool = False,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        process_image: Whether to normalize the image. Defaults to True.
        grayscale: If True, collapse an rgb image to a single luma channel (ITU-R BT.601:
            0.299 R + 0.587 G + 0.114 B), giving (..., 1, H, W). Only valid for rgb data types.

    Returns:
        The images produced at the last time-step
    """
    is_depth = "distance_to" in data_type or "depth" in data_type
    assert data_type == "rgb" or is_depth, f"Unsupported data_type: {data_type}"
    assert not (grayscale and is_depth), "grayscale=True is only valid for rgb, not depth."
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type].clone()

    start_dims = torch.arange(len(images.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    current_size = (images.shape[s + 1], images.shape[s + 2])

    # Convert to float32 and normalize in-place
    images = images.to(dtype=torch.float32)  # Avoid redundant .float() and .type() calls
    if is_depth:
        images[images == float("inf")] = 0.0
    else:
        images.div_(255.0).clamp_(0.0, 1.0)  # Normalize and clip in-place
    images = images.permute(start_dims + [s + 3, s + 1, s + 2])

    if grayscale:
        # Collapse rgb(a) to one luma channel before the resize: both the weighted channel sum and
        # bilinear interpolation are linear, so they commute -- doing it here resizes 1 channel
        # instead of 3. Slicing to :3 tolerates an rgba source.
        luma = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype)
        images = (images[..., :3, :, :] * luma.view(-1, 1, 1)).sum(dim=-3, keepdim=True)

    if current_size != output_size:
        # Perform resize operation
        images = F.interpolate(images, size=output_size, mode="bilinear", antialias=True)

    # rgb image normalization
    if not process_image:
        assert not is_depth, "process_image=False (uint8 packing) is only supported for rgb, not depth."
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


def process_multi_camera_image(
    env: ManagerBasedEnv,
    sensor_cfgs: list[SceneEntityCfg],
    data_type: str = "distance_to_camera",
    output_size: tuple = (64, 64),
    grayscale: bool = False,
) -> torch.Tensor:
    """Images from multiple camera sensors, concatenated along the channel dimension.

    Each camera is processed via :func:`process_image` independently (resize, depth
    inf-handling, optional rgb->grayscale, etc.), then stacked as extra input channels: shape
    (N, len(sensor_cfgs) * channels_per_camera, *output_size). With ``grayscale=True`` each camera
    contributes exactly one channel, so the result is (N, len(sensor_cfgs), *output_size).
    """
    images = [
        process_image(
            env,
            sensor_cfg=cfg,
            data_type=data_type,
            process_image=True,
            output_size=output_size,
            grayscale=grayscale,
        )
        for cfg in sensor_cfgs
    ]
    return torch.cat(images, dim=1)


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
