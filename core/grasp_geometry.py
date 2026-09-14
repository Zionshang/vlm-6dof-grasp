"""Shared geometry operations for grasp pipelines and application workflows."""
import numpy as np


def expand_boxes(boxes, image_shape, scale=1.25):
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


def filter_grasps_by_orientation(grasps, keep_topk, max_x_deg=50, max_y_deg=100):
    """Filter a score-ordered GraspGroup by approach and closing axes."""
    total, keep = len(grasps), []
    for index, grasp in enumerate(grasps):
        if len(keep) >= keep_topk:
            break
        rotation = grasp.rotation_matrix
        angle_x = np.arccos(np.clip(rotation[:, 0] @ [0, 0, 1], -1, 1))
        angle_y = np.arccos(np.clip(rotation[:, 1] @ [1, 0, 0], -1, 1))
        if angle_x < np.deg2rad(max_x_deg) and angle_y < np.deg2rad(max_y_deg):
            keep.append(index)
    result = grasps[keep] if keep else grasps[:keep_topk]
    print(f"[3D筛选] {max_x_deg}°/{max_y_deg}°: "
          f"输入={total}, 通过={len(keep)}, 输出={len(result)}"
          + ("（兜底）" if not keep else ""))
    return result
