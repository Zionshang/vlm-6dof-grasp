"""Shared geometry operations for grasp pipelines and application workflows."""
import numpy as np
from scipy.spatial.transform import Rotation


def expand_boxes(boxes, image_shape, scale=1.5):
    """Scale xyxy boxes about their centres and clip them to the image."""
    height, width = image_shape[:2]
    expanded = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box_width, box_height = (x2 - x1) * scale, (y2 - y1) * scale
        expanded.append([
            max(0, int(cx - box_width / 2)),
            max(0, int(cy - box_height / 2)),
            min(width, int(cx + box_width / 2)),
            min(height, int(cy + box_height / 2)),
        ])
    return expanded


def box_center_to_base(depth, box, intrinsic, ee_pose, hand_eye_r,
                       hand_eye_t, factor_depth=1.0, patch_size=5):
    """Project robust box-centre depth through camera→EE→base transforms."""
    x1, y1, x2, y2 = box
    u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
    height, width = depth.shape
    u, v = np.clip(u, 0, width - 1), np.clip(v, 0, height - 1)
    half = patch_size // 2
    patch = depth[max(0, v-half):min(height, v+half+1),
                  max(0, u-half):min(width, u+half+1)]
    valid = patch[patch > 0]
    if not valid.size:
        return None

    z = float(np.median(valid)) / factor_depth
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    point = np.array([(u-cx)*z/fx, (v-cy)*z/fy, z, 1.0])
    camera_to_ee = np.eye(4)
    camera_to_ee[:3, :3], camera_to_ee[:3, 3] = hand_eye_r, hand_eye_t
    ee_to_base = np.eye(4)
    ee_to_base[:3, :3] = Rotation.from_euler("xyz", ee_pose[3:]).as_matrix()
    ee_to_base[:3, 3] = ee_pose[:3]
    return (ee_to_base @ camera_to_ee @ point)[:3]


def filter_grasps_by_orientation(grasps, keep_topk, max_x_deg=50, max_y_deg=100):
    """Filter a score-ordered GraspGroup by approach and closing axes."""
    keep = []
    for index, grasp in enumerate(grasps):
        if len(keep) >= keep_topk:
            break
        rotation = grasp.rotation_matrix
        angle_x = np.arccos(np.clip(rotation[:, 0] @ [0, 0, 1], -1, 1))
        angle_y = np.arccos(np.clip(rotation[:, 1] @ [1, 0, 0], -1, 1))
        if angle_x < np.deg2rad(max_x_deg) and angle_y < np.deg2rad(max_y_deg):
            keep.append(index)
    return grasps[keep] if keep else grasps[:keep_topk]
