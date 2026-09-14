# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

from uwlab_tasks.manager_based.manipulation.omnireset.assembly_keypoints import Offset
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils


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

    target_pos = target_asset.data.body_link_pos_w.torch[:, target_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w.torch[:, target_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w.torch[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w.torch[:, root_body_idx].view(-1, 4)

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

        target_pos = self.target_asset.data.body_link_pos_w.torch[:, target_body_idx].view(-1, 3)
        target_quat = self.target_asset.data.body_link_quat_w.torch[:, target_body_idx].view(-1, 4)
        root_pos = self.root_asset.data.body_link_pos_w.torch[:, root_body_idx].view(-1, 3)
        root_quat = self.root_asset.data.body_link_quat_w.torch[:, root_body_idx].view(-1, 4)

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

    # `.torch` is required, not cosmetic: Isaac Lab 3.0 returns asset data as a
    # warp-backed ProxyArray. Most torch ops go through its deprecation bridge,
    # but `subtract_frame_transforms` is torch.jit.script'd and rejects anything
    # that is not a real Tensor. Both sides are (x, y, z, w) here -- 3.0 data and
    # 3.0 math utils share the convention -- so no quaternion reordering.
    # base_link body pose (root pose differs from it on the Newton backend; see task_space_actions)
    root_pos_w = root_asset.data.body_link_pos_w.torch[:, 0]
    root_quat_w = root_asset.data.body_link_quat_w.torch[:, 0]

    asset_lin_vel_b, _ = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        target_asset.data.body_lin_vel_w.torch[:, target_body_idx].view(-1, 3),
    )
    asset_ang_vel_b, _ = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        target_asset.data.body_ang_vel_w.torch[:, target_body_idx].view(-1, 3),
    )

    return torch.cat([asset_lin_vel_b, asset_ang_vel_b], dim=1)


def _as_torch(value) -> torch.Tensor:
    """Return ``value`` as a torch tensor.

    Isaac Lab 3.0 hands back warp-backed data from both asset ``.data`` properties
    (as :class:`ProxyArray`) and physics view accessors (as raw ``wp.array``).
    Neither supports torch instance methods such as ``view()`` -- ``wp.array.view()``
    is a dtype reinterpret with a different signature -- so convert explicitly.
    """
    if hasattr(value, "torch"):  # ProxyArray
        return value.torch
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    return value


def _newton_model():
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
    except ImportError:
        return None
    return NewtonManager._model if NewtonManager._solver is not None else None


def _newton_collider_shape_ids(env: ManagerBasedRLEnv, asset, model) -> torch.Tensor:
    """(num_envs, S) indices of the asset's collision shapes in the Newton model (cached on the asset)."""
    cached = getattr(asset, "_uwlab_newton_shape_ids", None)
    if cached is not None:
        return cached
    import re
    prim_path = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    # prim_path is a regex like /World/envs/env_.*/Robot once the scene exists
    pat = re.compile("^" + prim_path.replace("env_.*", "env_(?P<env>[0-9]+)") + "(/|$)")
    body_labels = list(model.body_label)
    shape_body = model.shape_body.numpy(); flags = model.shape_flags.numpy()
    per_env: dict[int, list[int]] = {}
    for i, b in enumerate(shape_body):
        if b < 0 or not (flags[i] & 0x2):
            continue
        m = pat.match(body_labels[b])
        if m:
            per_env.setdefault(int(m.group("env")), []).append(i)
    counts = {len(v) for v in per_env.values()}
    if len(per_env) != env.num_envs or len(counts) != 1:
        raise RuntimeError(f"newton collider shapes for {asset_cfg_name(asset)}: envs={len(per_env)} counts={counts}")
    ids = torch.tensor([per_env[e] for e in range(env.num_envs)], device=env.device, dtype=torch.long)
    asset._uwlab_newton_shape_ids = ids
    return ids


def asset_cfg_name(asset) -> str:
    return getattr(asset.cfg, "prim_path", "?")


def get_material_properties(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    num_shapes: int | None = None,
):
    """Per-shape (static friction, dynamic friction, restitution), flattened.

    Newton: read from the model's shape materials (mu used for both frictions). ``num_shapes`` lets
    the Newton layout match PhysX's shape count (averaged over the asset's colliders and tiled) so
    the privileged critic keeps the dimension the PhysX-trained checkpoints expect.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    model = _newton_model()
    if model is None:
        # `root_view` replaces the now-deprecated `root_physx_view` in Isaac Lab 3.0.
        return _as_torch(asset.root_view.get_material_properties()).view(env.num_envs, -1)
    ids = _newton_collider_shape_ids(env, asset, model)
    mu = _as_torch(model.shape_material_mu)[ids]
    rest = _as_torch(model.shape_material_restitution)[ids]
    props = torch.stack([mu, mu, rest], dim=-1)  # (N, S, 3)
    if num_shapes is not None and num_shapes != props.shape[1]:
        props = props.mean(dim=1, keepdim=True).expand(-1, num_shapes, -1)
    return props.reshape(env.num_envs, -1)


def get_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if _newton_model() is not None:
        return _as_torch(asset.data.default_mass).view(env.num_envs, -1)
    return _as_torch(asset.root_view.get_masses()).view(env.num_envs, -1)


def get_joint_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction_coeff.torch.view(env.num_envs, -1)


def get_joint_armature(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_armature.torch.view(env.num_envs, -1)


def get_joint_stiffness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness.torch.view(env.num_envs, -1)


def get_joint_damping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping.torch.view(env.num_envs, -1)


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

    Returns:
        The images produced at the last time-step
    """
    assert data_type == "rgb", "Only RGB images are supported for now."
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type].clone()

    start_dims = torch.arange(len(images.shape) - 3).tolist()
    s = start_dims[-1] if len(start_dims) > 0 else -1
    current_size = (images.shape[s + 1], images.shape[s + 2])

    # Convert to float32 and normalize in-place
    images = images.to(dtype=torch.float32)  # Avoid redundant .float() and .type() calls
    images.div_(255.0).clamp_(0.0, 1.0)  # Normalize and clip in-place
    images = images.permute(start_dims + [s + 3, s + 1, s + 2])

    if current_size != output_size:
        # Perform resize operation
        images = F.interpolate(images, size=output_size, mode="bilinear", antialias=True)

    # rgb/depth image normalization
    if not process_image:
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


def joint_pos_signed(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), negate_joints: tuple[str, ...] = ()
) -> torch.Tensor:
    """Joint positions with selected joints negated (asset joint-sign conventions -> policy convention).

    Used on Newton with the swapped Robotiq USD, whose two inner_finger_knuckle joints carry the
    opposite sign from the PhysX asset the policies were trained on.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = _as_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    if not negate_joints:
        return q
    key = "_joint_pos_signed_" + asset_cfg.name
    sign = getattr(env, key, None)
    if sign is None or sign.shape[0] != q.shape[1]:
        names = list(asset_cfg.joint_names) if asset_cfg.joint_names is not None else list(asset.joint_names)
        sign = torch.ones(q.shape[1], device=q.device)
        for n in negate_joints:
            if n in names:
                sign[names.index(n)] = -1.0
        setattr(env, key, sign)
    return q * sign
