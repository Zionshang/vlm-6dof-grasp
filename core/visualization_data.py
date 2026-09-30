"""Shared preparation of point clouds and backend-specific O3D geometries."""
import cv2
import numpy as np


DEPTH_MAX_M = 3.0


def graspgenx_projection_images(image, grasps, intrinsic, gripper, top_k=20):
    """Project the exact O3D/GraspGenX gripper wireframe into RGB images."""
    local_points = np.asarray(
        gripper.control_points_visualization[0], dtype=np.float64,
    )[:, :3]
    if local_points.ndim != 2 or local_points.shape[1] != 3:
        raise ValueError(
            "GraspGenX gripper visualization points must have shape (N, 3+)"
        )
    images = []
    for i, grasp in enumerate(grasps[:top_k]):
        camera = (local_points @ grasp.rotation_matrix.T
                  + grasp.translation)
        valid = camera[:, 2] > 1e-4
        uv = camera @ np.asarray(intrinsic, dtype=np.float64).T
        pts = np.zeros((len(camera), 2), dtype=int)
        pts[valid] = np.rint(uv[valid, :2] / uv[valid, 2:3]).astype(int)
        vis = image.copy()
        for start in range(len(pts) - 1):
            if valid[start] and valid[start + 1]:
                cv2.line(
                    vis, tuple(pts[start]), tuple(pts[start + 1]),
                    (0, 255, 255), 3,
                )
        origin = intrinsic @ grasp.translation
        label_at = (tuple(np.rint(origin[:2] / origin[2]).astype(int))
                    if origin[2] > 1e-4 else (10, 25))
        cv2.putText(vis, f"ID:{i} score:{grasp.score:.3f}", label_at,
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)
        images.append(vis)
    return images


def point_cloud_arrays(color, depth, intrinsic, max_points=None):
    """Return camera-frame XYZ/RGB arrays used by desktop and web viewers."""
    depth = np.asarray(depth)
    color = np.asarray(color)
    height, width = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid = (depth > 0) & (depth < DEPTH_MAX_M) & np.isfinite(depth)
    z = depth[valid]
    points = np.column_stack(((u[valid] - cx) * z / fx,
                              (v[valid] - cy) * z / fy, z))
    colors = color[valid].astype(np.float64) / 255.0
    if max_points and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points, colors = points[indices], colors[indices]
    return points.astype(np.float64), colors


def grasp_geometries(grasps, color=(0, 0, 0)):
    """Build backend-provided meshes or a neutral 6-DOF pose frame."""
    geometries = []
    if grasps is not None:
        for grasp in grasps:
            if hasattr(grasp, "to_open3d_geometry"):
                geometry = grasp.to_open3d_geometry(color=color)
            else:
                import open3d as o3d
                size = max(0.03, grasp.width, grasp.depth)
                geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=size
                )
                geometry.transform(grasp.pose)
            geometries.extend(geometry if isinstance(geometry, list) else [geometry])
    return geometries


def graspgenx_geometries(grasps, gripper, show_top_mesh=True,
                         top_mesh_color=(0.1, 0.5, 1.0)):
    """Build GraspGenX gripper wireframes and the best-candidate mesh."""
    if grasps is None or len(grasps) == 0:
        return []

    import open3d as o3d

    local_points = np.asarray(
        gripper.control_points_visualization[0], dtype=np.float64,
    )[:, :3]
    scores = np.clip(np.asarray(grasps.scores), 0.0, 1.0)
    points, lines, colors = [], [], []
    for pose, score in zip(grasps.poses, scores):
        transformed = local_points @ pose[:3, :3].T + pose[:3, 3]
        start = len(points)
        points.extend(transformed)
        lines.extend(
            [start + index, start + index + 1]
            for index in range(len(transformed) - 1)
        )
        colors.extend([[1.0 - score, score, 0.0]] * (len(transformed) - 1))

    wire = o3d.geometry.LineSet()
    wire.points = o3d.utility.Vector3dVector(np.asarray(points))
    wire.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    wire.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    geometries = [wire]

    if show_top_mesh and gripper.visual_mesh is not None:
        best = grasps.poses[int(scores.argmax())]
        vertices = np.asarray(gripper.visual_mesh.vertices)
        vertices = vertices @ best[:3, :3].T + best[:3, 3]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(gripper.visual_mesh.faces, dtype=np.int32)
        )
        mesh.paint_uniform_color(np.asarray(top_mesh_color, dtype=float))
        mesh.compute_vertex_normals()
        geometries.append(mesh)
    return geometries
