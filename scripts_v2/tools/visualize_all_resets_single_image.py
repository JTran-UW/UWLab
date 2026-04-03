# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize saved states from HDF5 dataset."""

from __future__ import annotations

import argparse
import math
import os
import torch
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Visualize saved reset states from a dataset directory. Use --headless (app launcher group) to run without a display."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./reset_state_datasets",
    help="Directory containing reset-state datasets saved as <hash>.pt (ignored if --combine_four_resets).",
)
parser.add_argument(
    "--combine_four_resets",
    action="store_true",
    help="Load from four dataset folders (25%% each): ObjectAnywhereEEAnywhere, ObjectRestingEEGrasped, ObjectAnywhereEEGrasped, ObjectPartiallyAssembledEEGrasped under --dataset_parent.",
)
parser.add_argument(
    "--dataset_parent",
    type=str,
    default="./reset_state_datasets",
    help="Parent directory for --combine_four_resets subfolders.",
)
parser.add_argument("--debug_clone_pose", action="store_true", help="Print captured vs clone gripper pose to debug offset.")
parser.add_argument("--record", type=str, default=None, help="Output path for video (e.g. resets.mp4). If set, records the viewer after each reset.")
parser.add_argument("--video_fps", type=int, default=30, help="Frames per second for the recorded video.")
parser.add_argument("--record_hold_seconds_start", type=float, default=0.1, help="Seconds of video to record per reset at the start (viewer hold time); linearly interpolated to record_hold_seconds_end.")
parser.add_argument("--record_hold_seconds_end", type=float, default=0.05, help="Seconds of video to record per reset at the end (linear interpolate from start).")
parser.add_argument(
    "--camera_pos",
    type=str,
    default="ObjectAnywhereEEAnywhere",
    choices=["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"],
    help="Orbit camera preset. With --combine_four_resets, orbit always uses ObjectAnywhereEEAnywhere (wider); this flag only affects single-dataset runs.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# When recording, cameras are required for env.render(); enable so headless + --record works
if args_cli.record:
    args_cli.enable_cameras = True

# launch omniverse app (use --headless to run without a display)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import contextlib
import gymnasium as gym
from tqdm import tqdm

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm

from uwlab_tasks.manager_based.manipulation.reset_states.mdp import events as task_mdp
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
import omni.usd
import isaacsim.core.utils.prims as prim_utils
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, UsdLux

LIGHT_TYPES = {
    "DomeLight": UsdLux.DomeLight,
    "RectLight": UsdLux.RectLight,
    "SphereLight": UsdLux.SphereLight,
    "DiskLight": UsdLux.DiskLight,
}

# Orbit camera presets per --camera_pos. Each tuple: (axis_x, axis_y, speed_rad_per_sec, radius, camera_height, target_z).
CAMERA_ORBIT_PRESETS: dict[str, tuple[float, float, float, float, float, float]] = {
    "ObjectAnywhereEEAnywhere": (0.4, 0.0, 0.05, 1.6, 0.6, 0.1),
    "ObjectRestingEEGrasped": (0.6, 0.0, 0.05, 0.6, 0.3, 0.05), # (0.6, 0.0, 0.05, 0.6, 0.3, 0.1),
    "ObjectAnywhereEEGrasped": (0.5, 0.0, 0.05, 0.8, 0.6, 0.3), # (0.6, 0.0, 0.05, 0.6, 0.4, 0.2),
    "ObjectPartiallyAssembledEEGrasped": (0.46, 0.06, 0.05, 0.8, 0.25, 0.05),
}

# Subfolder names under --dataset_parent for --combine_four_resets (must match MultiResetManager .pt layout per assembly hash).
FOUR_RESET_DATASET_SUBDIRS: tuple[str, ...] = (
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
)


def update_orbit_camera(
    env,
    current_angle: list[float],
    video_fps: float,
    orbit_params: tuple[float, float, float, float, float, float],
) -> None:
    """Set camera to current orbit position; advance angle by one frame of VIDEO time so spin rate in the video is constant."""
    axis_x, axis_y, speed, radius, cam_height, target_z = orbit_params
    angle = current_angle[0]
    eye = (
        axis_x + radius * math.cos(angle),
        axis_y + radius * math.sin(angle),
        cam_height,
    )
    target = (axis_x, axis_y, target_z)
    env.unwrapped.sim.set_camera_view(eye=eye, target=target)
    dt_video = 1.0 / max(1, video_fps)
    current_angle[0] += dt_video * speed


def set_clone_material_translucent(
    prim_path: str,
    shadow_subpaths: list[str],
    stage: Usd.Stage | None = None,
    opacity: float = 0.01,
) -> None:
    """Set the PreviewSurface shader under prim_path to given opacity and IOR 1 (for translucent clone).
    Disable shadow casting on the visual mesh prims specified by shadow_subpaths."""
    if stage is None:
        stage = omni.usd.get_context().get_stage()
    shader_path = prim_path + "/visuals/Looks/PreviewSurface/Shader"
    shader_prim = stage.GetPrimAtPath(shader_path)
    if not shader_prim.IsValid():
        return
    for attr_name, value in [("inputs:opacity", opacity), ("inputs:ior", 1.0)]:
        attr = stage.GetAttributeAtPath(Sdf.Path(shader_path).AppendProperty(attr_name))
        if attr:
            try:
                attr.Set(float(value))
            except Exception:
                pass
    for subpath in shadow_subpaths:
        visuals_prim = stage.GetPrimAtPath(prim_path + "/" + subpath)
        if visuals_prim.IsValid():
            attr = visuals_prim.GetAttribute("primvars:doNotCastShadows")
            if not attr:
                attr = visuals_prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool)
            if attr:
                attr.Set(True)


def disable_collisions_under_prim(prim_path: str, opacity: float = 0.8) -> None:
    """Disable collision on prim and all descendants so clones are visual-only."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root.GetPath()):
            continue

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)


def make_articulation_clone_visual_only(prim_path: str) -> None:
    """Strip articulation and physics from a robot clone so PhysX does not try to create joints/bodies.
    Call after SetInstanceable(False). Then call disable_collisions_under_prim."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    root_path = root.GetPath()
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root_path):
            continue
        # Remove articulation root from every prim (reference may put it on a child)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        # Disable joints so PhysX does not create them
        if prim.IsA(UsdPhysics.Joint):
            attr = prim.GetAttribute("physics:jointEnabled")
            if attr:
                attr.Set(False)
    disable_collisions_under_prim(prim_path)


def set_prim_world_pose(prim_path: str, pos: torch.Tensor, quat: torch.Tensor) -> None:
    """Set a prim so its world pose equals (pos, quat). Computes local xform from parent's world transform."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(0.0, 0.0, 0.0)
    )
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    )
    if not prim.IsValid():
        return
    pos = pos.detach().cpu().float().numpy().flatten()
    quat = quat.detach().cpu().float().numpy().flatten()
    pos_gf = Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
    quat_gf = Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3])))
    world_mat = Gf.Matrix4d()
    world_mat.SetRotateOnly(Gf.Rotation(quat_gf))
    world_mat.SetTranslateOnly(pos_gf)
    parent_path = prim.GetPath().GetParentPath()
    if parent_path and str(parent_path):
        parent_prim = stage.GetPrimAtPath(parent_path)
        parent_world = omni.usd.get_world_transform_matrix(parent_prim) if parent_prim.IsValid() else Gf.Matrix4d(1.0)
    else:
        parent_world = Gf.Matrix4d(1.0)
    local_mat = parent_world.GetInverse() * world_mat
    local_trans = local_mat.ExtractTranslation()
    local_rot = local_mat.ExtractRotation()
    local_quat = local_rot.GetQuat()
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if not ops:
        xform.AddTranslateOp().Set(local_trans)
        xform.AddOrientOp().Set(local_quat)
    else:
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(local_trans)
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(local_quat)
                break


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic for diverse pose discovery
    env_cfg.seed = None

    if args_cli.combine_four_resets:
        base_paths = [os.path.join(args_cli.dataset_parent, name) for name in FOUR_RESET_DATASET_SUBDIRS]
        reset_probs = [1.0, 1.0, 1.0, 1.0]
        # Receptive / insertive poses change across reset families; refresh clones every reset (not single frozen receptive clone).
        fix_objects = False
    else:
        base_paths = [args_cli.dataset_dir]
        reset_probs = [1.0]
        fix_objects = args_cli.camera_pos in [
            "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped",
            "ObjectPartiallyAssembledEEGrasped",
        ]

    # Set up the MultiResetManager to load states from the computed dataset(s)
    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": base_paths,
            "probs": reset_probs,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

    if fix_objects:
        print("Fixing objects")
    else:
        print("Not fixing objects")

    # Add the reset manager to the environment configuration
    env_cfg.events.reset_from_reset_states = reset_from_reset_states

    # create environment
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    env.reset()
    env.render_mode = "rgb_array"

    # Enable translucency so clone materials with opacity < 1 render correctly
    try:
        import carb
        import isaacsim.core.utils.carb as carb_utils
        settings = carb.settings.get_settings()
        carb_utils.set_carb_setting(settings, "/rtx/translucency/enabled", True)
    except Exception:
        pass

    orbit_camera_key = (
        "ObjectAnywhereEEAnywhere" if args_cli.combine_four_resets else args_cli.camera_pos
    )
    orbit_params = CAMERA_ORBIT_PRESETS[orbit_camera_key]
    current_orbit_angle: list[float] = [0.0]

    # Initialize variables
    print(f"Starting visualization of saved states from: {base_paths}")
    print("Press Ctrl+C to stop")

    # import pdb; pdb.set_trace()
    to_hide = [
        # "/World/envs/env_0/Robot/robotiq_base_link",
        "/World/envs/env_0/UR5MetalSupport",
        "/World/envs/env_0/Robot/base_link",
        "/World/envs/env_0/Robot/shoulder_link",
        "/World/envs/env_0/Robot/upper_arm_link",
        "/World/envs/env_0/Robot/forearm_link",
        "/World/envs/env_0/Robot/wrist_1_link",
        "/World/envs/env_0/Robot/wrist_2_link",
        "/World/envs/env_0/Robot/wrist_3_link",
        "/World/envs/env_0/Robot/shoulder_pan_joint",
        "/World/envs/env_0/Robot/shoulder_lift_joint",
        "/World/envs/env_0/Robot/elbow_joint",
        "/World/envs/env_0/Robot/wrist_1_joint",
        "/World/envs/env_0/Robot/wrist_2_joint",
        "/World/envs/env_0/Robot/wrist_3_joint",
    ]
    import isaaclab.sim as sim_utils
    import cv2
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG

    for asset_path in to_hide:
        asset = sim_utils.find_matching_prims(asset_path)[0]
        sim_utils.set_prim_visibility(asset, False)

    # Ensure Clones container exists
    stage = omni.usd.get_context().get_stage()
    clones_path = "/World/envs/env_0/Clones"
    if not stage.GetPrimAtPath(clones_path).IsValid():
        UsdGeom.Xform.Define(stage, clones_path)

    # Turn off skylight
    # skylight_path = "/World/skyLight"
    # stage.GetPrimAtPath(skylight_path).SetActive(False)

    # Pose marker for cloned EE poses (replaces visible robotiq_base_link per clone)
    robot = env.unwrapped.scene["robot"]
    ee_body_name = "robotiq_base_link"
    ee_body_idx = next(i for i, n in enumerate(robot.body_names) if n == ee_body_name)
    # frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    # frame_marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    # pose_marker = VisualizationMarkers(
    #     frame_marker_cfg.replace(prim_path=clones_path + "/EEPoseMarkers")
    # )
    all_ee_pos = []
    all_ee_quat = []

    video_writer = None
    if args_cli.record:
        print(
            f"Recording video to {args_cli.record} at {args_cli.video_fps} fps, "
            f"hold seconds: linear {args_cli.record_hold_seconds_start}s -> {args_cli.record_hold_seconds_end}s per reset"
        )
        frames_to_record = max(1, int(args_cli.record_hold_seconds_start * args_cli.video_fps))
        for _ in range(frames_to_record):
            update_orbit_camera(env, current_orbit_angle, args_cli.video_fps, orbit_params)
            img = env.render()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if video_writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    args_cli.record, fourcc, args_cli.video_fps, (w, h)
                )
            video_writer.write(img)

    num_resets = 1000
    with contextlib.suppress(KeyboardInterrupt):
        for i in tqdm(range(num_resets), desc="Resets", unit="reset"):
            asset = env.unwrapped.scene["robot"]
            # specific for robotiq
            gripper_joint_positions = asset.data.joint_pos[:, asset.find_joints(["right_inner_finger_joint"])[0][0]]
            gripper_closed_fraction = (
                torch.abs(gripper_joint_positions) / env_cfg.actions.gripper.close_command_expr["finger_joint"]
            )
            gripper_mask = gripper_closed_fraction > 0.1
            # Step the simulation
            # for _ in range(5):
            #     action = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
            #     action[gripper_mask, -1] = -1.0
            #     action[~gripper_mask, -1] = 1.0
            #     env.step(action)
            for _ in range(3):
                env.unwrapped.sim.step()
            success = env.unwrapped.reward_manager.get_term_cfg("progress_context").func.success

            # Read original poses and full link state before we reset
            insertive = env.unwrapped.scene["insertive_object"]
            receptive = env.unwrapped.scene["receptive_object"]
            robot = env.unwrapped.scene["robot"]
            original_pos_w = insertive.data.root_pos_w[0].clone()
            original_quat_w = insertive.data.root_quat_w[0].clone()
            receptive_pos_w = receptive.data.root_pos_w[0].clone()
            receptive_quat_w = receptive.data.root_quat_w[0].clone()
            robot_root_pos_w = robot.data.root_pos_w[0].clone()
            robot_root_quat_w = robot.data.root_quat_w[0].clone()
            robot_body_link_state = robot.data.body_state_w[0, :, :7].clone()
            # print(robot_body_link_state)

            # Insertive object clone
            clone_path = "/World/envs/env_0/Clones/MyClone_" + str(i)
            prim_utils.create_prim(clone_path, "Xform", usd_path=insertive.cfg.spawn.usd_path)
            clone_prim = stage.GetPrimAtPath(clone_path)
            if clone_prim.IsValid():
                clone_prim.SetInstanceable(False)
            disable_collisions_under_prim(clone_path)
            # disable_lights(clone_path, stage=stage)
            set_clone_material_translucent(clone_path, ["visuals/bolt", "visuals/leg"], stage=stage)

            if not fix_objects or (fix_objects and i == 0):
                # Receptive object clone
                receptive_clone_path = "/World/envs/env_0/Clones/ReceptiveClone_" + str(i)
                prim_utils.create_prim(receptive_clone_path, "Xform", usd_path=receptive.cfg.spawn.usd_path)
                receptive_clone_prim = stage.GetPrimAtPath(receptive_clone_path)
                if receptive_clone_prim.IsValid():
                    receptive_clone_prim.SetInstanceable(False)
                disable_collisions_under_prim(receptive_clone_path)
                # disable_lights(receptive_clone_path, stage=stage)
                set_clone_material_translucent(
                    receptive_clone_path,
                    [
                        "visuals/table_top",
                        "visuals/wall",
                        "visuals/wall1",
                        "visuals/wall2",
                        "visuals/wall3",
                        "visuals/hole",
                        "visuals/hole1",
                        "visuals/hole2",
                        "visuals/hole3",
                    ],
                    stage=stage,
                )

            # Robot clone (same pattern: clone -> reset -> set pose -> disable collisions)
            robot_clone_path = "/World/envs/env_0/Clones/RobotClone_" + str(i)
            prim_utils.create_prim(robot_clone_path, "Xform", usd_path=robot.cfg.spawn.usd_path)
            # Strip articulation/physics before any sim step so PhysX never tries to CreateJoint
            robot_clone_prim = stage.GetPrimAtPath(robot_clone_path)
            if robot_clone_prim.IsValid():
                robot_clone_prim.SetInstanceable(False)
            # disable_lights(robot_clone_path, stage=stage)
            make_articulation_clone_visual_only(robot_clone_path)

            # Reset the environment to load a new state
            env.reset()
            if fix_objects and i > 0:
                for obj in (receptive, ):
                    obj_prims = sim_utils.find_matching_prims(obj.cfg.prim_path)
                    for p in obj_prims:
                        sim_utils.set_prim_visibility(p, False)
            set_prim_world_pose(clone_path, original_pos_w, original_quat_w)
            if not fix_objects or (fix_objects and i == 0):
                set_prim_world_pose(clone_path, original_pos_w, original_quat_w)
                set_prim_world_pose(receptive_clone_path, receptive_pos_w, receptive_quat_w)
                set_prim_world_pose(robot_clone_path, robot_root_pos_w, robot_root_quat_w)
            root = stage.GetPrimAtPath(robot_clone_path)
            xform = UsdGeom.Xformable(root)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(0.0, 0.0, 0.0)
            )
            xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
            )
            # Set each link's world pose so the clone matches the captured arm pose
            links_set_count = 0
            for link_idx, link_name in enumerate(robot.body_names):
                link_prim_path = robot_clone_path + "/" + link_name
                if stage.GetPrimAtPath(link_prim_path).IsValid():
                    # print("Setting link pose: ", link_name)
                    pos = robot_body_link_state[link_idx, :3]
                    quat = robot_body_link_state[link_idx, 3:7]
                    set_prim_world_pose(link_prim_path, pos, quat)
                    links_set_count += 1
            # Replace cloned EE with pose marker at same pose
            pos_ee = robot_body_link_state[ee_body_idx, :3].clone()
            quat_ee = robot_body_link_state[ee_body_idx, 3:7].clone()
            all_ee_pos.append(pos_ee)
            all_ee_quat.append(quat_ee)
            poses_pos = torch.stack(all_ee_pos, dim=0).to(device=env.device)
            poses_quat = torch.stack(all_ee_quat, dim=0).to(device=env.device)
            # pose_marker.visualize(poses_pos, poses_quat)

            # Delete the robot clone entirely now that the pose marker replaces it
            stage.RemovePrim(robot_clone_path)

            # Record current view before we capture/reset (so video shows this state, then the reset)
            if args_cli.record:
                # Linear interpolate hold time from start to end over the run (gradual speedup)
                t = i / max(1, num_resets - 1)
                hold_seconds = (1.0 - t) * args_cli.record_hold_seconds_start + t * args_cli.record_hold_seconds_end
                frames_to_record = max(1, int(hold_seconds * args_cli.video_fps))
                for _ in range(frames_to_record):
                    update_orbit_camera(env, current_orbit_angle, args_cli.video_fps, orbit_params)
                    img = env.render()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if video_writer is None:
                        h, w = img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            args_cli.record, fourcc, args_cli.video_fps, (w, h)
                        )
                    video_writer.write(img)
            # Keep final frame as still image (full scene with new clone)
            update_orbit_camera(env, current_orbit_angle, args_cli.video_fps, orbit_params)
            img = env.render()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("final_viz.png", rgb)

    if video_writer is not None:
        video_writer.release()
        print(f"Saved video to {args_cli.record}")
    env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
