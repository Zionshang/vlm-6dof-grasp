"""统一的输出保存工具。

集中处理三件事,避免散落各处的重复:
  1) 父目录自动创建(mkdir);
  2) RGB <-> BGR 转换(cv2 用 BGR);
  3) 各类输出(检测框 / 分割 mask / RGBD capture / 2D grasp / reject)的路径与命名。

所有保存都走这里,调用方只需一行。
"""
import shutil
import cv2
import numpy as np
from pathlib import Path


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
