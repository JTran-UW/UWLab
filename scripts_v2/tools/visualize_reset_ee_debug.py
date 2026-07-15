"""Visualize reset states + approach debug geometry with viser.

Shows: peg, hole, arm (cylinders via FK), gripper mesh, approach ray, depth slider.

Usage:
    conda run -n env_uwlab python scripts_v2/tools/visualize_reset_ee_debug.py \
        --dataset ./Datasets/OmniReset/Resets/Peg__PegHole/resets_GravityTrickPartialAssembly.pt \
        --port 8090
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import trimesh
import viser


# ============================================================================
# Geometry helpers
# ============================================================================

def quat_to_rotmat(q):
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def load_usd_mesh(usd_path, visual_only=True, scale=None):
    """Load visual mesh from USD file. scale overrides auto-detection."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(usd_path)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if scale is not None:
        meters_per_unit = scale
    all_verts, all_faces = [], []
    offset = 0
    for mp in stage.Traverse():
        if not mp.IsA(UsdGeom.Mesh):
            continue
        path_str = str(mp.GetPath()).lower()
        if visual_only and "collision" in path_str:
            continue
        # Skip adapter/D415 meshes that have wrong scale or complex faces
        leaf_name = path_str.split("/")[-1]
        if "adapter" in leaf_name or "d415" in leaf_name:
            continue
        mesh = UsdGeom.Mesh(mp)
        pts = np.array(mesh.GetPointsAttr().Get())
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        if len(pts) == 0:
            continue
        pts = pts * meters_per_unit  # convert to meters
        # Handle triangles and quads
        faces = []
        idx = 0
        for c in counts:
            if c == 3:
                faces.append(indices[idx:idx+3] + offset)
            elif c == 4:
                # Split quad into two triangles
                faces.append(indices[[idx, idx+1, idx+2]] + offset)
                faces.append(indices[[idx, idx+2, idx+3]] + offset)
            idx += c
        if faces:
            all_verts.append(pts)
            all_faces.extend(faces)
            offset += len(pts)
    if not all_verts:
        return trimesh.primitives.Box(extents=[0.01, 0.01, 0.01])
    return trimesh.Trimesh(np.concatenate(all_verts), np.array(all_faces))


# ============================================================================
# UR5e FK — returns per-link transforms
# ============================================================================

# Standard DH parameters for UR5e
DH_A = np.array([0, -0.42500, -0.39225, 0, 0, 0])
DH_D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
DH_ALPHA = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
# 180deg Z rotation (base_link_inertia -> base_link)
R180Z = np.diag([-1.0, -1.0, 1.0, 1.0])


def ur5e_fk_all_links(joints):
    """UR5e FK returning transforms for all 7 frames (base + 6 joints).

    Returns list of 4x4 transforms in base_link frame.
    """
    transforms = [R180Z.copy()]  # base
    T = R180Z.copy()
    for i in range(6):
        ct, st = np.cos(joints[i]), np.sin(joints[i])
        ca, sa = np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
        Ti = np.array([
            [ct, -st*ca, st*sa, DH_A[i]*ct],
            [st, ct*ca, -ct*sa, DH_A[i]*st],
            [0, sa, ca, DH_D[i]],
            [0, 0, 0, 1],
        ])
        T = T @ Ti
        transforms.append(T.copy())
    return transforms


def make_arm_cylinders(transforms, radius=0.02):
    """Create cylinder meshes between consecutive link origins."""
    meshes = []
    for i in range(len(transforms) - 1):
        p0 = transforms[i][:3, 3]
        p1 = transforms[i+1][:3, 3]
        length = np.linalg.norm(p1 - p0)
        if length < 0.001:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length)
        # Align cylinder (default along Z) to p0->p1
        direction = (p1 - p0) / length
        midpoint = (p0 + p1) / 2
        # Build transform
        z_axis = direction
        x_axis = np.cross([0, 0, 1], z_axis) if abs(z_axis[2]) < 0.99 else np.cross([1, 0, 0], z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        R = np.eye(4)
        R[:3, 0] = x_axis
        R[:3, 1] = y_axis
        R[:3, 2] = z_axis
        R[:3, 3] = midpoint
        cyl.apply_transform(R)
        meshes.append(cyl)
    if meshes:
        return trimesh.util.concatenate(meshes)
    return trimesh.primitives.Box(extents=[0.01, 0.01, 0.01])


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--peg_usd", type=str, default="/tmp/mesh_check/hf_download/Peg/peg.usd")
    parser.add_argument("--peghole_usd", type=str, default="/tmp/mesh_check/hf_download/PegHole/peg_hole.usd")
    parser.add_argument("--robot_usd", type=str, default=None, help="Robot USD for gripper mesh")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    # Auto-find robot USD from HF cache
    if args.robot_usd is None:
        try:
            from huggingface_hub import hf_hub_download
            args.robot_usd = hf_hub_download(
                "yandabao/uwlab-assets",
                "Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd",
                repo_type="dataset",
            )
        except Exception:
            args.robot_usd = None

    # Load dataset
    data = torch.load(args.dataset, map_location="cpu")
    states = data["initial_state"]
    n_states = len(states["articulation"]["robot"]["joint_position"])
    print(f"Loaded {n_states} states")

    # Load meshes (peg/hole USDs have vertices in mm despite metersPerUnit=1.0)
    peg_mesh = load_usd_mesh(args.peg_usd, scale=0.001)   # peg verts in mm
    hole_mesh = load_usd_mesh(args.peghole_usd)            # hole verts already in meters
    # Sample surface points properly (on faces, not just vertices)
    peg_surface_pts = peg_mesh.sample(64)  # [64, 3] in mesh local frame (meters)
    gripper_mesh = load_usd_mesh(args.robot_usd) if args.robot_usd else None  # robot USD already in meters
    if gripper_mesh is not None:
        # Gripper mesh has approach along +Y, but body frame has approach along +X
        # Rotate mesh: Y->X, X->Y (-90° around Z)
        rot_mesh_to_body = np.array([
            [ 0, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 1],
        ], dtype=float)
        gripper_mesh.vertices = (rot_mesh_to_body @ gripper_mesh.vertices.T).T

    # Constants
    GRIPPER_APPROACH_DIR = np.array([1.0, 0.0, 0.0])
    GRIPPER_OFFSET = np.array([0.1345, 0.0, 0.0])

    # Start viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    current_idx = [0]
    current_depth = [0.5]

    def show_state(idx, depth_frac=None):
        idx = idx % n_states
        current_idx[0] = idx
        if depth_frac is not None:
            current_depth[0] = depth_frac

        # State data
        jp = states["articulation"]["robot"]["joint_position"][idx][:6].numpy()
        gripper_jp = states["articulation"]["robot"]["joint_position"][idx][6:].numpy()
        robot_root = states["articulation"]["robot"]["root_pose"][idx][:3].numpy()
        ins_pose = states["rigid_object"]["insertive_object"]["root_pose"][idx].numpy()
        rec_pose = states["rigid_object"]["receptive_object"]["root_pose"][idx].numpy()
        ins_pos, ins_quat = ins_pose[:3], ins_pose[3:]
        rec_pos, rec_quat = rec_pose[:3], rec_pose[3:]

        # FK
        link_transforms = ur5e_fk_all_links(jp)
        # Offset all transforms by robot root position
        for T in link_transforms:
            T[:3, 3] += robot_root
        ee_T = link_transforms[-1]  # wrist_3_link
        ee_pos = ee_T[:3, 3]
        ee_R = ee_T[:3, :3]

        # Approach direction in world
        approach_world = ee_R @ GRIPPER_APPROACH_DIR
        fingertip = ee_pos + ee_R @ GRIPPER_OFFSET

        # --- Peg ---
        T_ins = np.eye(4)
        T_ins[:3, :3] = quat_to_rotmat(ins_quat)
        T_ins[:3, 3] = ins_pos
        peg_t = peg_mesh.copy()
        peg_t.apply_transform(T_ins)
        peg_t.visual.face_colors = [100, 200, 100, 220]
        server.scene.add_mesh_trimesh("/peg", peg_t)

        # --- Hole ---
        T_rec = np.eye(4)
        T_rec[:3, :3] = quat_to_rotmat(rec_quat)
        T_rec[:3, 3] = rec_pos
        hole_t = hole_mesh.copy()
        hole_t.apply_transform(T_rec)
        hole_t.visual.face_colors = [100, 100, 220, 220]
        server.scene.add_mesh_trimesh("/hole", hole_t)

        # --- Arm (cylinders) ---
        arm_mesh = make_arm_cylinders(link_transforms, radius=0.025)
        arm_mesh.visual.face_colors = [180, 180, 180, 200]
        server.scene.add_mesh_trimesh("/arm", arm_mesh)

        # --- Recorded EE point (red, semi-transparent) ---
        server.scene.add_point_cloud(
            "/ee_pt", points=ee_pos.reshape(1, 3).astype(np.float32),
            colors=np.array([[255, 50, 50]], dtype=np.uint8), point_size=0.012,
        )

        # --- Object center (green) ---
        server.scene.add_point_cloud(
            "/obj_center", points=ins_pos.reshape(1, 3).astype(np.float32),
            colors=np.array([[0, 255, 0]], dtype=np.uint8), point_size=0.008,
        )

        # --- Depth-controlled gripper position ---
        # Match actual reset_ee_toward_object: random surface point, approach from EE
        peg_local_pts = peg_surface_pts
        peg_world_pts = (quat_to_rotmat(ins_quat) @ peg_local_pts.T).T + ins_pos

        # Use a fixed random surface point per state (seeded by state index)
        rng = np.random.RandomState(idx)
        entry_idx = rng.randint(0, len(peg_world_pts))
        entry_pos = peg_world_pts[entry_idx]

        # Approach direction: EE -> entry point (same as actual code)
        approach_dir = entry_pos - ee_pos
        approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-8)

        # Max depth: project all mesh points along approach from entry
        offsets_from_entry = peg_world_pts - entry_pos
        proj_from_entry = offsets_from_entry @ approach_dir
        max_depth_from_entry = max(proj_from_entry.max(), 0.001)

        # Fingertip target at current depth
        fingertip_target = entry_pos + approach_dir * current_depth[0] * max_depth_from_entry
        # EE position = fingertip - offset along approach
        gripper_offset_dist = np.linalg.norm(GRIPPER_OFFSET)
        sim_ee_pos = fingertip_target - approach_dir * gripper_offset_dist

        # Build simulated EE transform: x-axis (approach) = approach_dir
        x_axis = approach_dir / (np.linalg.norm(approach_dir) + 1e-8)
        up = np.array([0, 0, -1.0])
        if abs(np.dot(x_axis, up)) > 0.99:
            up = np.array([0, 1, 0])
        z_axis = np.cross(x_axis, up)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        sim_ee_T = np.eye(4)
        sim_ee_T[:3, 0] = x_axis
        sim_ee_T[:3, 1] = y_axis
        sim_ee_T[:3, 2] = z_axis
        sim_ee_T[:3, 3] = sim_ee_pos

        # Gripper mesh at simulated position
        if gripper_mesh is not None:
            verts = gripper_mesh.vertices.copy()
            faces = gripper_mesh.faces.copy()
            verts = (sim_ee_T[:3, :3] @ verts.T).T + sim_ee_T[:3, 3]
            g = trimesh.Trimesh(vertices=verts, faces=faces)
            g.visual.face_colors = [200, 200, 200, 220]
            server.scene.add_mesh_trimesh("/gripper", g)

        # Simulated fingertip: offset along approach direction (not ee_R)
        sim_fingertip = sim_ee_pos + approach_dir * np.linalg.norm(GRIPPER_OFFSET)
        server.scene.add_point_cloud(
            "/fingertip_pt", points=sim_fingertip.reshape(1, 3).astype(np.float32),
            colors=np.array([[255, 255, 0]], dtype=np.uint8), point_size=0.01,
        )

        # Cyan dot: the randomly sampled entry point on mesh surface
        server.scene.add_point_cloud(
            "/entry_pos", points=entry_pos.reshape(1, 3).astype(np.float32),
            colors=np.array([[0, 255, 255]], dtype=np.uint8), point_size=0.008,
        )

        # Show all canonical surface points as small white dots
        server.scene.add_point_cloud(
            "/all_surface_pts", points=peg_world_pts.astype(np.float32),
            colors=np.full((len(peg_world_pts), 3), 200, dtype=np.uint8), point_size=0.003,
        )

        # Approach line (from EE through object)
        ray_start = ee_pos - approach_dir * 0.05
        ray_end = entry_pos + approach_dir * max_depth_from_entry
        server.scene.add_spline_catmull_rom(
            "/approach_ray",
            positions=np.array([ray_start, ray_end], dtype=np.float32),
            color=(255, 0, 255),
        )

        # Line from simulated fingertip to object center
        server.scene.add_spline_catmull_rom(
            "/line_to_obj",
            positions=np.array([sim_fingertip, ins_pos], dtype=np.float32),
            color=(255, 255, 255),
        )

        # Info
        dist_fingertip_obj = np.linalg.norm(sim_fingertip - ins_pos)
        dist_fingertip_entry = np.linalg.norm(sim_fingertip - entry_pos)
        info_label.value = (
            f"State **{idx}/{n_states-1}** | "
            f"fingertip→obj: **{dist_fingertip_obj*1000:.1f}mm** | "
            f"fingertip→entry: **{dist_fingertip_entry*1000:.1f}mm** | "
            f"approach·down: **{approach_dir[2]:.2f}** | "
            f"depth: **{current_depth[0]:.2f}**"
        )

    # GUI
    server.gui.add_markdown("### Reset EE Debug Viewer")
    info_label = server.gui.add_markdown("...")
    slider_state = server.gui.add_slider("State", min=0, max=max(n_states-1, 1), step=1, initial_value=0)
    slider_depth = server.gui.add_slider("Depth", min=0.0, max=1.0, step=0.01, initial_value=0.5)
    btn_prev = server.gui.add_button("Prev")
    btn_next = server.gui.add_button("Next")

    @slider_state.on_update
    def _on_state(event):
        show_state(slider_state.value, current_depth[0])

    @slider_depth.on_update
    def _on_depth(event):
        show_state(current_idx[0], slider_depth.value)

    @btn_prev.on_click
    def _on_prev(event):
        slider_state.value = (current_idx[0] - 1) % n_states

    @btn_next.on_click
    def _on_next(event):
        slider_state.value = (current_idx[0] + 1) % n_states

    show_state(0)
    print(f"Open http://localhost:{args.port}")

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
