"""Shared rigid transforms for camera, TCP, targets, and grasps."""
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


def offset_pose(pose, offset=(0.0, 0.0, 0.0),
                local_offset=(0.0, 0.0, 0.0), rpy=None):
    """Apply world/local translations using the final commanded orientation."""
    result = np.asarray(pose, dtype=float).copy()
    if rpy is not None:
        result[3:] = rpy
    result[:3] += np.asarray(offset, dtype=float)
    result[:3] += Rotation.from_euler("xyz", result[3:]).apply(local_offset)
    return result


def target_orbit_poses(target, offset, rpy, angles_deg=(-30.0, 0.0, 30.0)):
    """Rotate one target-facing observation pose around base Z."""
    target, offset = np.asarray(target, float), np.asarray(offset, float)
    base_rotation = Rotation.from_euler("xyz", rpy)
    poses = []
    for angle in angles_deg:
        yaw = Rotation.from_euler("z", np.deg2rad(angle))
        rotation = yaw * base_rotation
        poses.append(np.r_[target + yaw.apply(offset),
                           rotation.as_euler("xyz")])
    return poses


PIPER_FLANGE_T_TCP = pose_transform(
    [0.0, 0.0, 0.13, 0.0, -np.pi / 2, 0.0]
)


def camera_to_base_transform(ee_pose, handeye_rot, handeye_trans):
    """Return camera→base from TCP→base and calibrated camera→TCP."""
    return pose_transform(ee_pose) @ rigid_transform(
        handeye_rot, handeye_trans,
    )


def camera_point_to_base(point, ee_pose, handeye_rot, handeye_trans):
    """Transform one XYZ point from the camera frame to the robot base."""
    point = np.r_[np.asarray(point, dtype=float), 1.0]
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
    tcp_to_grasp = rigid_transform(
        np.diag([1., -1., -1.]), [grasp_depth, 0., 0.],
    )
    tcp_to_base_goal = (
        camera_to_base_transform(
            current_ee_pose, handeye_rot, handeye_trans,
        )
        @ grasp_to_camera @ tcp_to_grasp
    )
    return np.r_[
        tcp_to_base_goal[:3, 3],
        Rotation.from_matrix(tcp_to_base_goal[:3, :3]).as_euler("xyz"),
    ].tolist()


def graspgenx_grasp_to_base(grasp_translation, grasp_rotation_mat,
                            capture_ee_pose, handeye_rot, handeye_trans):
    """Convert a GraspGenX grasp to the Piper controller TCP pose.

    GraspGenX uses +Z for approach and +X for closing.  Piper's URDF gripper
    uses +Z/+Y, while its configured controller TCP uses +X/+Y.
    """
    camera_T_graspgenx = rigid_transform(
        grasp_rotation_mat, grasp_translation,
    )
    graspgenx_T_tcp = rigid_transform(
        Rotation.from_euler("z", np.pi / 2).as_matrix(), [0.0] * 3,
    ) @ PIPER_FLANGE_T_TCP
    base_T_tcp = (
        camera_to_base_transform(
            capture_ee_pose, handeye_rot, handeye_trans,
        )
        @ camera_T_graspgenx
        @ graspgenx_T_tcp
    )
    return np.r_[
        base_T_tcp[:3, 3],
        Rotation.from_matrix(base_T_tcp[:3, :3]).as_euler("xyz"),
    ].tolist()
