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

        self.object_asset: RigidObject = env.scene[object_cfg.name]
        self.ref_asset = env.scene[ref_cfg.name]

        self.ref_body_idx: int | None = None
        if ref_cfg.body_names and isinstance(self.ref_asset, Articulation):
            ref_cfg.resolve(env.scene)
            self.ref_body_idx = ref_cfg.body_ids[0]  # type: ignore

        prim_path_pattern = self.object_asset.cfg.prim_path.replace("{ENV_REGEX_NS}", ".*")
        self.canonical_points = utils.sample_object_point_cloud(
            num_envs=env.num_envs,
            num_points=self.num_points,
            prim_path_pattern=prim_path_pattern,
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
        visualize: bool = False,
        visualize_env_ids: list[int] | None = None,
        marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> torch.Tensor:
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
