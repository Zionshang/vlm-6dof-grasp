"""Shared rigid transforms for camera, TCP, targets, OBBs, and grasps."""
import numpy as np
from scipy.spatial.transform import Rotation


def rigid_transform(rotation, translation):
    """Build a homogeneous transform from a 3x3 rotation and XYZ translation."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def pose_transform(pose):
    """Convert base-frame [x,y,z,rx,ry,rz] using the project xyz convention."""
    pose = np.asarray(pose, dtype=float)
    return rigid_transform(
        Rotation.from_euler("xyz", pose[3:]).as_matrix(), pose[:3],
    )


def camera_to_base_transform(ee_pose, handeye_rot, handeye_trans):
    """Return camera→base from TCP→base and calibrated camera→TCP."""
    return pose_transform(ee_pose) @ rigid_transform(
        handeye_rot, handeye_trans,
    )


def camera_frame_to_base(position, rotation, ee_pose,
                         handeye_rot, handeye_trans):
    """Transform a camera-frame position and orientation into the base frame."""
    camera_to_base = camera_to_base_transform(
        ee_pose, handeye_rot, handeye_trans,
    )
    return (
        camera_to_base[:3, :3] @ np.asarray(position)+camera_to_base[:3, 3],
        camera_to_base[:3, :3] @ np.asarray(rotation),
    )


def camera_pose_to_ee(camera_position, camera_rotation,
                      handeye_rot, handeye_trans):
    """Recover base-frame TCP position/rotation for a desired camera pose."""
    camera_rotation = np.asarray(camera_rotation, dtype=float)
    ee_rotation = camera_rotation @ np.asarray(handeye_rot, dtype=float).T
    ee_position = (np.asarray(camera_position, dtype=float)
                   - ee_rotation @ np.asarray(handeye_trans, dtype=float))
    return ee_position, ee_rotation


def box_center_to_base(depth, box, intrinsic, ee_pose, handeye_rot,
                       handeye_trans, factor_depth=1.0, patch_size=5):
    """Project robust detection-box centre depth into the robot base frame."""
    x1, y1, x2, y2 = box
    u, v = int((x1+x2)/2), int((y1+y2)/2)
    height, width = depth.shape
    u, v = np.clip(u, 0, width-1), np.clip(v, 0, height-1)
    half = patch_size//2
    patch = depth[max(0, v-half):min(height, v+half+1),
                  max(0, u-half):min(width, u+half+1)]
    valid = patch[patch > 0]
    if not valid.size:
        return None

    z = float(np.median(valid))/factor_depth
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    point = np.array([(u-cx)*z/fx, (v-cy)*z/fy, z, 1.])
    return (camera_to_base_transform(
        ee_pose, handeye_rot, handeye_trans,
    ) @ point)[:3]


def convert_new(grasp_translation, grasp_rotation_mat, current_ee_pose,
                handeye_rot, handeye_trans, grasp_depth):
    """Convert an EconomicGrasp camera-frame grasp into a base-frame TCP pose."""
    grasp_depth = float(grasp_depth)
    if not np.isfinite(grasp_depth) or grasp_depth < 0:
        raise ValueError(f"Invalid grasp depth: {grasp_depth}")

    grasp_to_camera = rigid_transform(
        grasp_rotation_mat, grasp_translation,
    )
    alignment = rigid_transform(
        np.diag([1., -1., -1.]), [grasp_depth, 0., 0.],
    )
    gripper_to_base = (
        camera_to_base_transform(
            current_ee_pose, handeye_rot, handeye_trans,
        )
        @ grasp_to_camera @ alignment
    )
    return np.r_[
        gripper_to_base[:3, 3],
        Rotation.from_matrix(gripper_to_base[:3, :3]).as_euler("xyz"),
    ].tolist()
