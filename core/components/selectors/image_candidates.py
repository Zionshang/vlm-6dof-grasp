"""Shared 2D candidate preparation for image-based selectors."""
import cv2
import numpy as np

from saver import save_reject, try_save


FINGER_BACK_M = -0.024


def project_grasp(translation, rotation, width, depth, intrinsic):
    half_width = width / 2
    local = np.array([
        [depth, -half_width, 0], [FINGER_BACK_M, -half_width, 0],
        [FINGER_BACK_M, half_width, 0], [depth, half_width, 0], [0, 0, 0],
    ]).T
    camera = rotation @ local + np.asarray(translation).reshape(3, 1)
    z = np.where(np.abs(camera[2]) < 1e-6, 1e-6, camera[2])
    u = intrinsic[0, 0] * camera[0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[1] / z + intrinsic[1, 2]
    return np.rint(np.column_stack((u, v))).astype(int)


def _draw(image, points, candidate_id, score):
    cv2.line(image, tuple(points[0]), tuple(points[1]), (255, 0, 0), 3)
    cv2.line(image, tuple(points[1]), tuple(points[2]), (0, 255, 0), 3)
    cv2.line(image, tuple(points[2]), tuple(points[3]), (255, 0, 0), 3)
    cv2.circle(image, tuple(points[4]), 5, (0, 255, 255), -1)
    anchor = tuple(np.mean(points[1:3], axis=0).astype(int))
    cv2.putText(image, f"ID:{candidate_id} score:{score:.3f}", anchor,
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)


def prepare_image_candidates(color, grasps, intrinsic, output_dir, top_k):
    """Project and geometrically validate EconomicGrasp candidates."""
    images, candidates = [], []
    for source_index, grasp in enumerate(grasps[:top_k]):
        translation = np.asarray(grasp.translation)
        rotation = np.asarray(grasp.rotation_matrix)
        if translation.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("抓取位姿结构错误")
        if not (np.isfinite(translation).all() and np.isfinite(rotation).all()):
            raise ValueError("抓取位姿包含NaN/Inf")
        if translation[2] <= 0:
            continue
        points = project_grasp(
            translation, rotation, grasp.width, grasp.depth, intrinsic,
        )
        width_px = np.linalg.norm(points[0] - points[3])
        base = points[2] - points[1]
        root = np.mean(points[1:3], axis=0)
        tip = np.mean(points[[0, 3]], axis=0)
        approach = tip - root
        denominator = np.linalg.norm(base) * np.linalg.norm(approach)
        angle = 90.0 if denominator == 0 else np.degrees(np.arccos(
            np.clip(base @ approach / denominator, -1.0, 1.0)
        ))
        angle = min(angle, 180.0 - angle)
        rejected = (
            width_px < 55
            or (points[2, 0] < points[1, 0]
                and points[2, 1] > points[1, 1])
            or angle < 45
            or tip[1] > root[1]
        )
        image = color.copy()
        _draw(image, points, len(candidates), float(grasp.score))
        if rejected:
            try_save("筛除图", save_reject, output_dir, image, source_index)
            continue

        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, translation
        images.append(image)
        candidates.append({
            "id": len(candidates), "source_index": source_index,
            "pose_matrix": pose.tolist(), "score": float(grasp.score),
            "width": float(grasp.width), "depth": float(grasp.depth),
            "translation": translation.tolist(), "rotation": rotation.tolist(),
        })
    return images, candidates
