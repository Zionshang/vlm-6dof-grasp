"""统一的输出保存工具。

集中处理三件事,避免散落各处的重复:
  1) 父目录自动创建(mkdir);
  2) RGB <-> BGR 转换(cv2 用 BGR);
  3) 各类输出(检测框 / 分割 mask / RGBD capture / 2D grasp / reject)的路径与命名。
     OBB 投影叠加图也集中在这里保存。

所有保存都走这里,调用方只需一行。
"""
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

from obb import OBB_EDGES
from transform import camera_to_base_transform


def _ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_image(path, img, is_rgb=True):
    """保存图片:自动创建父目录;is_rgb=True 时把 RGB 转 BGR。"""
    p = _ensure_parent(path)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if is_rgb else img
    if not cv2.imwrite(str(p), out):
        raise OSError(f"无法写入 {p}")
    return p


def try_save(name, save, *args, **kwargs):
    """Best-effort output; saving must never stop robot work."""
    try:
        return save(*args, **kwargs)
    except Exception as exc:
        print(f"[输出] {name}保存失败: {str(exc).splitlines()[0]}")
        return None


def save_mask(path, mask):
    """保存 0/1(或 bool)mask 为 png(乘 255)。"""
    return save_image(path, (np.asarray(mask) > 0).astype(np.uint8) * 255, is_rgb=False)


def save_vlm_boxes(output_dir, color, boxes, run_id=None, tag="vlm"):
    """画检测框并保存到 output_dir/vlm/{run_id}_{tag}.png。"""
    img = cv2.cvtColor(np.asarray(color), cv2.COLOR_RGB2BGR)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    name = f"{run_id}_{tag}.png" if run_id else f"{tag}.png"
    return save_image(Path(output_dir) / "vlm" / name, img, is_rgb=False)


def _project_obb(obb, ee_pose, hand_eye_r, hand_eye_t, intrinsic):
    camera_to_base = camera_to_base_transform(
        ee_pose, hand_eye_r, hand_eye_t,
    )
    points_camera = ((np.asarray(obb.vertices, dtype=float)
                      - camera_to_base[:3, 3]) @ camera_to_base[:3, :3])
    if np.any(points_camera[:, 2] <= 1e-6):
        raise ValueError("OBB顶点不在相机前方")

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    pixels = np.column_stack((
        points_camera[:, 0]*fx/points_camera[:, 2]+cx,
        points_camera[:, 1]*fy/points_camera[:, 2]+cy,
    ))
    return pixels


def _draw_obb(img, pixels, color, thickness=2):
    for start, end in OBB_EDGES:
        cv2.line(img, tuple(pixels[start].astype(int)),
                 tuple(pixels[end].astype(int)), color, thickness)


def save_obb_overlay(output_dir, color, obb, ee_pose, hand_eye_r,
                     hand_eye_t, intrinsic, run_id=None):
    """Project a base-frame fused OBB onto its observation RGB and save it."""
    pixels = _project_obb(obb, ee_pose, hand_eye_r, hand_eye_t, intrinsic)
    img = cv2.cvtColor(np.asarray(color), cv2.COLOR_RGB2BGR)
    _draw_obb(img, pixels, (0, 140, 255))
    name = f"{run_id}_obb.png" if run_id else "obb.png"
    return save_image(Path(output_dir) / "obb" / name, img, is_rgb=False)


def save_obb_samples(output_dir, color, obbs, ee_pose, hand_eye_r,
                     hand_eye_t, intrinsic, tag):
    """Project all accepted base-frame OBB samples onto one observation RGB."""
    img = cv2.cvtColor(np.asarray(color), cv2.COLOR_RGB2BGR)
    for index, obb in enumerate(obbs):
        color_bgr = cv2.cvtColor(np.uint8([[
            [180*index//max(1, len(obbs)-1), 220, 255],
        ]]), cv2.COLOR_HSV2BGR)[0, 0]
        pixels = _project_obb(obb, ee_pose, hand_eye_r, hand_eye_t, intrinsic)
        _draw_obb(img, pixels, tuple(int(value) for value in color_bgr))
        cv2.putText(img, str(index+1), tuple(pixels[0].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, tuple(int(v) for v in color_bgr), 1)
    path = save_image(Path(output_dir) / "obb" / "raw" / f"{tag}.png",
                      img, is_rgb=False)
    print(f"[输出] OBB采样图已保存: {path}")
    return path


def save_seg_mask(output_dir, mask, run_id=None):
    """保存分割 mask 到 output_dir/sam/{run_id}_sam.png。"""
    name = f"{run_id}_sam.png" if run_id else "seg_result.png"
    return save_mask(Path(output_dir) / "sam" / name, mask)


def save_capture(output_dir, color, depth, timestamp):
    """保存 RGBD capture 到 output_dir/captures/{ts}_color.png + _depth.png。"""
    base = Path(output_dir) / "captures"
    save_image(base / f"{timestamp}_color.png", color, is_rgb=True)
    depth = np.asarray(depth)
    # FFS outputs metres as float; store lossless millimetres in uint16 PNG.
    if np.issubdtype(depth.dtype, np.floating):
        depth = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    save_image(base / f"{timestamp}_depth.png", depth, is_rgb=False)


def save_grasp_result(output_dir, result, run_id):
    """Save the selected camera-frame grasp without coupling to its backend."""
    path = _ensure_parent(Path(output_dir) / "grasp" / f"{run_id}_first.json")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"[输出] first抓取结果已保存: {path}")
    return path


def save_2d_grasp(output_dir, imgs, subdir=None):
    """保存 2D grasp 图到 output_dir/2D_grasp[/subdir]/{i}.jpg。"""
    d = Path(output_dir) / "2D_grasp"
    if subdir:
        d /= subdir
    if d.exists():
        shutil.rmtree(d)
    return [str(save_image(d / f"{i}.jpg", img, is_rgb=True)) for i, img in enumerate(imgs)]


def save_reject(output_dir, img, idx):
    """保存被拒抓取的调试图到 output_dir/vlm/origin_{idx}.jpg。"""
    return save_image(Path(output_dir) / "vlm" / f"origin_{idx}.jpg", img, is_rgb=True)
