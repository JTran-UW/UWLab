"""Visualize convex-hull approach reset strategy with viser.

Shows: object mesh, convex hull (wireframe), sampled surface point + normal,
EE IK target at varying distance d along the normal.

Usage:
    conda run -n env_uwlab python scripts_v2/tools/visualize_convex_hull_approach.py \
        --usd /tmp/mesh_check/hf_download/Peg/peg.usd --scale 0.001 --port 8090

    conda run -n env_uwlab python scripts_v2/tools/visualize_convex_hull_approach.py \
        --usd /tmp/mesh_check/hf_download/PegHole/peg_hole.usd --port 8090
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import trimesh
import viser


def load_usd_mesh(usd_path, scale=None):
    """Load visual mesh from USD file."""
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
        if "collision" in path_str:
            continue
        leaf_name = path_str.split("/")[-1]
        if "adapter" in leaf_name or "d415" in leaf_name:
            continue
        mesh = UsdGeom.Mesh(mp)
        pts = np.array(mesh.GetPointsAttr().Get())
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        if len(pts) == 0:
            continue
        pts = pts * meters_per_unit
        faces = []
        idx = 0
        for c in counts:
            if c == 3:
                faces.append(indices[idx : idx + 3] + offset)
            elif c == 4:
                faces.append(indices[[idx, idx + 1, idx + 2]] + offset)
                faces.append(indices[[idx, idx + 2, idx + 3]] + offset)
            idx += c
        if faces:
            all_verts.append(pts)
            all_faces.extend(faces)
            offset += len(pts)
    if not all_verts:
        return trimesh.primitives.Box(extents=[0.01, 0.01, 0.01])
    return trimesh.Trimesh(np.concatenate(all_verts), np.array(all_faces))


def main():
    parser = argparse.ArgumentParser(description="Visualize convex hull approach strategy")
    parser.add_argument("--usd", type=str, required=True, help="Path to insertive object USD file")
    parser.add_argument("--receptive_usd", type=str, default=None, help="Path to receptive object USD file (for occlusion filter test)")
    parser.add_argument("--receptive_scale", type=float, default=None, help="Override meters-per-unit for receptive object")
    parser.add_argument("--scale", type=float, default=None, help="Override meters-per-unit (e.g. 0.001 for mm)")
    parser.add_argument("--gripper_offset", type=float, default=0.1425, help="Distance from EE frame to fingerpad center")
    parser.add_argument("--d_range", type=float, default=0.0175, help="Half-range of d (fingerpad half-length)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .pt dataset to load object poses from")
    parser.add_argument("--dataset_idx", type=int, default=0, help="Which entry in the dataset to use")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    # Load insertive mesh and compute convex hull
    mesh = load_usd_mesh(args.usd, scale=args.scale)
    hull = mesh.convex_hull
    bbox_extents = mesh.bounding_box.extents
    bbox_radius = np.linalg.norm(bbox_extents) / 2
    print(f"Insertive mesh: {len(mesh.vertices)} verts, bbox extents: {bbox_extents}")
    print(f"Convex hull: {len(hull.vertices)} verts, {len(hull.faces)} faces")
    print(f"Bbox radius: {bbox_radius:.4f}m")

    # Load receptive mesh for occlusion filtering (optional)
    rec_mesh = None
    rec_hull = None
    if args.receptive_usd:
        rec_mesh = load_usd_mesh(args.receptive_usd, scale=args.receptive_scale)
        rec_hull = rec_mesh.convex_hull
        print(f"Receptive mesh: {len(rec_mesh.vertices)} verts")
        print(f"Receptive convex hull: {len(rec_hull.vertices)} verts, {len(rec_hull.faces)} faces")

    # Precompute convex hull face normals and areas for sampling
    hull_face_normals = hull.face_normals  # [n_faces, 3], outward-pointing
    hull_face_areas = hull.area_faces  # [n_faces]

    # Default positions
    obj_pos = np.array([0.4, 0.2, 0.05])
    obj_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    rec_pos = np.array([0.4, 0.1, 0.0])
    rec_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    # Override from dataset if provided
    if args.dataset:
        import torch
        ds = torch.load(args.dataset, map_location="cpu")
        idx = args.dataset_idx
        state_data = ds["initial_state"]
        ins_pose = state_data["rigid_object"]["insertive_object"]["root_pose"][idx].numpy()
        obj_pos = ins_pose[:3].copy()
        obj_quat_wxyz = ins_pose[3:7].copy()
        if "receptive_object" in state_data["rigid_object"]:
            rec_pose = state_data["rigid_object"]["receptive_object"]["root_pose"][idx].numpy()
            rec_pos = rec_pose[:3].copy()
            rec_quat_wxyz = rec_pose[3:7].copy()
        n_entries = len(state_data["rigid_object"]["insertive_object"]["root_pose"])
        print(f"Loaded dataset entry {idx}/{n_entries}: obj_pos={obj_pos}, rec_pos={rec_pos}")

    def quat_to_rotmat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    rng = np.random.RandomState(42)

    MAX_SAMPLES = 50

    # State
    state = {
        "points_local": [],   # list of surface points in local frame
        "normals_local": [],  # list of outward normals in local frame
        "n_visible": 5,       # how many to show
        "d": 0.05,            # distance along normal from surface
    }

    def sample_points(n):
        """Sample n random points on the convex hull surface + outward normals."""
        probs = hull_face_areas / hull_face_areas.sum()
        state["points_local"] = []
        state["normals_local"] = []
        for _ in range(n):
            face_idx = rng.choice(len(hull.faces), p=probs)
            face = hull.faces[face_idx]
            v0, v1, v2 = hull.vertices[face]
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
            state["points_local"].append(point)
            state["normals_local"].append(hull_face_normals[face_idx].copy())

    sample_points(MAX_SAMPLES)

    # Start viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    def clear_sample_visuals():
        """Remove all per-sample scene nodes."""
        for i in range(MAX_SAMPLES):
            for name in [f"/sample_{i}/surface", f"/sample_{i}/fingertip", f"/sample_{i}/ee",
                         f"/sample_{i}/ray", f"/sample_{i}/axis_x", f"/sample_{i}/axis_y", f"/sample_{i}/axis_z"]:
                try:
                    server.scene.remove(name)
                except Exception:
                    pass

    def occlusion_check(points_world):
        """Check which world-frame points are inside the receptive object's convex hull.

        Returns boolean array: True = inside (occluded), False = valid.
        """
        if rec_hull is None:
            return np.zeros(len(points_world), dtype=bool)
        R_rec = quat_to_rotmat(rec_quat_wxyz)
        # Transform to receptive object local frame: p_local = R^T @ (p_world - t)
        # Row-vector convention: (p - t) @ R gives R^T @ (p - t) in column convention
        pts_local = (points_world - rec_pos) @ R_rec
        return rec_hull.contains(pts_local)

    def update_scene():
        R = quat_to_rotmat(obj_quat_wxyz)
        d = state["d"]
        n = state["n_visible"]

        # Transform insertive mesh to world
        T_obj = np.eye(4)
        T_obj[:3, :3] = R
        T_obj[:3, 3] = obj_pos

        # Insertive object mesh (green, semi-transparent)
        obj_world = mesh.copy()
        obj_world.apply_transform(T_obj)
        obj_world.visual.face_colors = [100, 200, 100, 180]
        server.scene.add_mesh_trimesh("/object", obj_world)

        # Convex hull (blue wireframe — show as transparent mesh)
        hull_world = hull.copy()
        hull_world.apply_transform(T_obj)
        hull_world.visual.face_colors = [80, 80, 255, 60]
        server.scene.add_mesh_trimesh("/convex_hull", hull_world)

        # Receptive object mesh (orange, semi-transparent)
        if rec_mesh is not None:
            R_rec = quat_to_rotmat(rec_quat_wxyz)
            T_rec = np.eye(4)
            T_rec[:3, :3] = R_rec
            T_rec[:3, 3] = rec_pos
            rec_world = rec_mesh.copy()
            rec_world.apply_transform(T_rec)
            rec_world.visual.face_colors = [220, 150, 50, 120]
            server.scene.add_mesh_trimesh("/receptive_object", rec_world)
            # Receptive convex hull
            rec_hull_world = rec_hull.copy()
            rec_hull_world.apply_transform(T_rec)
            rec_hull_world.visual.face_colors = [220, 100, 0, 40]
            server.scene.add_mesh_trimesh("/receptive_hull", rec_hull_world)

        # Object center (small green dot)
        server.scene.add_point_cloud(
            "/obj_center",
            points=obj_pos.reshape(1, 3).astype(np.float32),
            colors=np.array([[0, 200, 0]], dtype=np.uint8),
            point_size=0.006,
        )

        clear_sample_visuals()

        # Compute all surface points in world frame and run occlusion check in batch
        all_sp_world = np.array([R @ state["points_local"][i] + obj_pos for i in range(n)])
        occluded = occlusion_check(all_sp_world)
        n_occluded = int(occluded.sum())
        n_valid = n - n_occluded

        # Draw each visible sample
        for i in range(n):
            sp_world = all_sp_world[i]
            is_occluded = occluded[i]
            normal_world = R @ state["normals_local"][i]
            normal_world = normal_world / (np.linalg.norm(normal_world) + 1e-8)

            fingertip_pos = sp_world + normal_world * d
            ee_target_pos = fingertip_pos + normal_world * args.gripper_offset

            approach_dir = -normal_world
            x_axis = approach_dir / (np.linalg.norm(approach_dir) + 1e-8)
            up = np.array([0.0, 0.0, -1.0])
            if abs(np.dot(x_axis, up)) > 0.99:
                up = np.array([0.0, 1.0, 0.0])
            z_axis = np.cross(x_axis, up)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
            y_axis = np.cross(z_axis, x_axis)

            # Color: green=valid, red=occluded
            if is_occluded:
                pt_color = np.array([[255, 50, 50]], dtype=np.uint8)
                ray_color = (255, 80, 80)
            else:
                pt_color = np.array([[50, 255, 50]], dtype=np.uint8)
                ray_color = (50, 200, 50)

            # Surface point
            server.scene.add_point_cloud(
                f"/sample_{i}/surface",
                points=sp_world.reshape(1, 3).astype(np.float32),
                colors=pt_color,
                point_size=0.008,
            )

            # Fingertip target (dimmer version of same color)
            server.scene.add_point_cloud(
                f"/sample_{i}/fingertip",
                points=fingertip_pos.reshape(1, 3).astype(np.float32),
                colors=pt_color,
                point_size=0.005,
            )

            # EE target
            server.scene.add_point_cloud(
                f"/sample_{i}/ee",
                points=ee_target_pos.reshape(1, 3).astype(np.float32),
                colors=pt_color,
                point_size=0.006,
            )

            # Ray from EE through surface point into object
            into_object = sp_world - normal_world * 0.01
            server.scene.add_spline_catmull_rom(
                f"/sample_{i}/ray",
                positions=np.array([ee_target_pos, into_object], dtype=np.float32),
                color=ray_color,
            )

            # Small axes at EE target (only for valid points to reduce clutter)
            if not is_occluded:
                axis_len = 0.02
                server.scene.add_spline_catmull_rom(
                    f"/sample_{i}/axis_x",
                    positions=np.array([ee_target_pos, ee_target_pos + x_axis * axis_len], dtype=np.float32),
                    color=(255, 0, 0),
                )
                server.scene.add_spline_catmull_rom(
                    f"/sample_{i}/axis_y",
                    positions=np.array([ee_target_pos, ee_target_pos + y_axis * axis_len], dtype=np.float32),
                    color=(0, 255, 0),
                )
                server.scene.add_spline_catmull_rom(
                    f"/sample_{i}/axis_z",
                    positions=np.array([ee_target_pos, ee_target_pos + z_axis * axis_len], dtype=np.float32),
                    color=(0, 0, 255),
                )

        # Info
        occ_info = f" | **{n_valid}** valid, **{n_occluded}** occluded" if rec_hull is not None else ""
        info_label.value = (
            f"Showing **{n}** samples | "
            f"d: **{d * 1000:.1f}mm** | "
            f"ee offset: **{args.gripper_offset * 1000:.1f}mm** | "
            f"pad contact: **{(args.gripper_offset + d) * 1000:.1f}mm** from EE | "
            f"bbox radius: **{bbox_radius * 1000:.1f}mm**"
            f"{occ_info}"
        )

    # GUI
    server.gui.add_markdown("### Convex Hull Approach Visualizer")
    if rec_hull is not None:
        server.gui.add_markdown(
            "**Green** = valid surface point | "
            "**Red** = occluded (inside receptive hull) | "
            "**Orange mesh** = receptive object"
        )
    else:
        server.gui.add_markdown(
            "**Green** = surface point on convex hull | "
            "Tip: pass `--receptive_usd` to test occlusion filtering"
        )
    info_label = server.gui.add_markdown("...")
    slider_n = server.gui.add_slider("Num samples", min=1, max=MAX_SAMPLES, step=1, initial_value=20)
    slider_d = server.gui.add_slider(
        "d (fingerpad sweep)",
        min=-args.d_range,
        max=args.d_range,
        step=0.001,
        initial_value=0.0,
    )
    btn_resample = server.gui.add_button("Resample All Points")
    slider_yaw = server.gui.add_slider("Object Yaw (deg)", min=-180, max=180, step=5, initial_value=0)

    # Insertive object position sliders
    server.gui.add_markdown("#### Insertive Object Position")
    slider_obj_x = server.gui.add_slider("Obj X", min=0.0, max=0.8, step=0.01, initial_value=obj_pos[0])
    slider_obj_y = server.gui.add_slider("Obj Y", min=-0.3, max=0.5, step=0.01, initial_value=obj_pos[1])
    slider_obj_z = server.gui.add_slider("Obj Z", min=-0.1, max=0.4, step=0.01, initial_value=obj_pos[2])

    if rec_hull is not None:
        server.gui.add_markdown("#### Receptive Object Position")
        slider_rec_x = server.gui.add_slider("Rec X", min=0.0, max=0.8, step=0.01, initial_value=rec_pos[0])
        slider_rec_y = server.gui.add_slider("Rec Y", min=-0.3, max=0.5, step=0.01, initial_value=rec_pos[1])
        slider_rec_z = server.gui.add_slider("Rec Z", min=-0.1, max=0.4, step=0.01, initial_value=rec_pos[2])

    @slider_n.on_update
    def _on_n(event):
        state["n_visible"] = int(slider_n.value)
        update_scene()

    @slider_d.on_update
    def _on_d(event):
        state["d"] = slider_d.value
        update_scene()

    @btn_resample.on_click
    def _on_resample(event):
        sample_points(MAX_SAMPLES)
        update_scene()

    @slider_yaw.on_update
    def _on_yaw(event):
        nonlocal obj_quat_wxyz
        yaw = np.radians(slider_yaw.value)
        obj_quat_wxyz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        update_scene()

    @slider_obj_x.on_update
    def _on_obj_x(event):
        obj_pos[0] = slider_obj_x.value
        update_scene()

    @slider_obj_y.on_update
    def _on_obj_y(event):
        obj_pos[1] = slider_obj_y.value
        update_scene()

    @slider_obj_z.on_update
    def _on_obj_z(event):
        obj_pos[2] = slider_obj_z.value
        update_scene()

    if rec_hull is not None:
        @slider_rec_x.on_update
        def _on_rec_x(event):
            rec_pos[0] = slider_rec_x.value
            update_scene()

        @slider_rec_y.on_update
        def _on_rec_y(event):
            rec_pos[1] = slider_rec_y.value
            update_scene()

        @slider_rec_z.on_update
        def _on_rec_z(event):
            rec_pos[2] = slider_rec_z.value
            update_scene()

    update_scene()
    print(f"Open http://localhost:{args.port}")

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
