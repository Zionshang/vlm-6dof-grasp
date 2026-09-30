"""Depth-image geometry without coordinate-frame transformations."""
import numpy as np


def mask_depth_target(depth, mask, intrinsic, max_depth_m=.85,
                      min_points=100):
    """Return the masked 3D median's nearest measured camera-frame point."""
    depth, mask = np.asarray(depth), np.asarray(mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError(f"depth/mask未对齐: {depth.shape}/{mask.shape}")
    valid = (mask & np.isfinite(depth) & (depth > 0)
             & (depth < float(max_depth_m)))
    v, u = np.nonzero(valid)
    if len(u) < int(min_points):
        return None
    z = depth[v, u].astype(float)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    median = np.median(points, axis=0)
    return points[np.argmin(np.linalg.norm(points - median, axis=1))]
