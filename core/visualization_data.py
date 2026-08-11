"""Shared RGB-D point cloud and GraspNet gripper geometry preparation."""
import numpy as np


DEPTH_MAX_M = 3.0


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
    """Build the exact GraspNetAPI meshes used by the Open3D viewer."""
    geometries = []
    if grasps is not None:
        for grasp in grasps:
            geometry = grasp.to_open3d_geometry(color=color)
            geometries.extend(geometry if isinstance(geometry, list) else [geometry])
    return geometries
