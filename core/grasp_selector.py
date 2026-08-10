"""O3D 同源 2D 抓取投影与可插拔选择器。"""
import numpy as np
import cv2
from typing import Protocol

from saver import save_reject, save_2d_grasp


O3D_FINGER_BACK = -0.024  # -(depth_base 0.020 + finger_width 0.004)

def _compress_image(image, max_dim=480):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim: return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def project_grasp_to_2d(trans, rot, width, depth, intrinsic):
    """按 O3D 夹爪局部几何投影开口内边中线。"""
    hw = width / 2
    points_g = np.array([
        [depth,          -hw, 0],
        [O3D_FINGER_BACK, -hw, 0],
        [O3D_FINGER_BACK,  hw, 0],
        [depth,           hw, 0],
        [0,                0, 0],  # EconomicGrasp translation
    ]).T

    points_c = rot @ points_g + np.asarray(trans).reshape(3, 1)

    # 投影到像素坐标
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    Z = np.where(np.abs(points_c[2, :]) < 1e-6, 1e-6, points_c[2, :])
    X = points_c[0, :]
    Y = points_c[1, :]

    U = (fx * X / Z) + cx
    V = (fy * Y / Z) + cy

    return np.rint(np.stack([U, V], axis=1)).astype(int)

def _draw_id_label(vis_img, pts, valid_idx):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    label = f"ID: {valid_idx}"
    font_scale = 1.0
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # 优先放在包围盒上方
    text_x = int((min_x + max_x) // 2 - tw // 2)
    text_y = int(min_y - 40)

    # 如果上方超出图片上边缘，则放到下方
    if text_y - th < 0:
            text_y = int(max_y + th + 40)

    # 左右边界保护
    text_x = max(2, min(text_x, vis_img.shape[1] - tw - 2))

    # 绘制白色实心背景框
    box_tl = (text_x - 4, text_y - th - 4)
    box_br = (text_x + tw + 4, text_y + 4)

    cv2.rectangle(vis_img, box_tl, box_br, (255, 255, 255), -1)
    cv2.putText(vis_img, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


def vlm_grasp_visualize_batch(image, trans, rot, width, depths, intrinsic,
                              top_k=8, output_dir="output"):
    """
    生成【多张】独立的图像，每张图只包含一个抓取候选。
    彻底解决重叠问题。

    Returns:
        vis_images: List[np.ndarray] 图片列表
        candidates: List[dict] 抓取信息列表
    """

    # 统一格式处理
    trans = np.array(trans)
    rot = np.array(rot)
    width = np.array(width)
    depths = np.array(depths)

    if trans.ndim == 1: trans = trans[np.newaxis, :]
    if rot.ndim == 2:   rot = rot[np.newaxis, ...]
    if width.ndim == 0: width = width[np.newaxis]
    if depths.ndim == 0: depths = depths[np.newaxis]

    lengths = {len(trans), len(rot), len(width), len(depths)}
    if len(lengths) != 1:
        raise ValueError("Grasp translation/rotation/width/depth counts differ")
    num_grasps = min(len(trans), top_k)

    vis_images = []
    candidates = []
    valid_idx = 0

    best_candidate_backup = None

    for i in range(num_grasps):
        # 每次都复制一张干净的背景图
        vis_img = image.copy()

        t = trans[i]
        r = rot[i]
        w = width[i]
        d = depths[i]

        pts = project_grasp_to_2d(t, r, w, d, intrinsic)

        if t[2] <= 0: continue

        # 1. 夹爪2D宽度
        width_px = np.linalg.norm(pts[0] - pts[3])
        # 2. 指根相对位置
        dx = pts[2][0] - pts[1][0]
        dy = pts[2][1] - pts[1][1]
        # 3. 抓取夹角
        center_base = np.mean(pts[1:3], axis=0).astype(int)
        center_tip = np.mean([pts[0], pts[3]], axis=0).astype(int)
        vec_base = pts[2] - pts[1]
        vec_arrow = center_tip - center_base

        angle_deg = 0.0
        norm_base = np.linalg.norm(vec_base)
        norm_arrow = np.linalg.norm(vec_arrow)
        if norm_base > 0 and norm_arrow > 0:
            cos_theta = np.dot(vec_base, vec_arrow) / (norm_base * norm_arrow)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_theta))
            if angle_deg > 90: angle_deg = 180 - angle_deg

        print(f"[Debug G{i}] Width={width_px:.1f}px | Depth={d:.3f}m | "
              f"dPos=({dx}, {dy}) | Angle={angle_deg:.1f} deg")

        # --- 无论好坏，先画出几何结构 (Review用) ---
        color_finger = (255, 0, 0) # Blue (BGR)
        color_base = (0, 255, 0)   # Green (BGR)
        thick = 3
        cv2.line(vis_img, tuple(pts[0]), tuple(pts[1]), color_finger, thick)
        cv2.line(vis_img, tuple(pts[1]), tuple(pts[2]), color_base, thick)
        cv2.line(vis_img, tuple(pts[2]), tuple(pts[3]), color_finger, thick)
        cv2.circle(vis_img, tuple(pts[2]), 8, (0, 0, 255), -1)
        cv2.arrowedLine(vis_img, tuple(center_base), tuple(center_tip), (0, 0, 255), 3, tipLength=0.35)
        cv2.circle(vis_img, tuple(pts[4]), 5, (0, 255, 255), -1)

        if best_candidate_backup is None:
            best_candidate_backup = (vis_img.copy(), pts, t, r, w, d, i)

        # --- 统一筛选 ---
        cond_width = width_px < 55
        cond_pose = pts[2][0] < pts[1][0] and pts[2][1] > pts[1][1]
        cond_angle = angle_deg < 45
        cond_down = center_tip[1] > center_base[1]

        if cond_width or cond_pose or cond_angle or cond_down:
            reasons = []
            if cond_width: reasons.append("TooNarrow")
            if cond_pose: reasons.append("BadPose")
            if cond_angle: reasons.append("BadAngle")
            if cond_down: reasons.append("PointingDown")
            fail_str = ', '.join(reasons)
            print(f" -> SKIPPED G{i}: {fail_str}")

            # [Added] 保存被筛选掉的图片用于Debug
            cv2.putText(vis_img, f"REJECT: {fail_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            save_reject(output_dir, vis_img, i)
            continue

        # --- 如果通过筛选，绘制ID并添加到列表 ---
        _draw_id_label(vis_img, pts, valid_idx)

        pose_mat = np.eye(4)
        pose_mat[:3, :3] = r
        pose_mat[:3, 3] = t

        vis_images.append(_compress_image(vis_img))
        candidates.append({
            "id": valid_idx,
            "source_index": i,
            "pose_matrix": pose_mat.tolist(),
            "width": float(w),
            "depth": float(d),
            "translation": t.tolist(),
            "rotation": r.tolist()
        })
        valid_idx += 1

    # [Added] 兜底逻辑：如果没有任何候选通过筛选，强制保留第一个
    if not candidates and best_candidate_backup is not None:
        print("[Warn] All candidates rejected. Force keeping the ID:0 candidate.")
        vis_img, pts, t, r, w, d, source_index = best_candidate_backup

        # 重新绘制 Label (Hardcoded ID: 0)
        _draw_id_label(vis_img, pts, 0)

        pose_mat = np.eye(4)
        pose_mat[:3, :3] = r
        pose_mat[:3, 3] = t

        vis_images.append(_compress_image(vis_img))
        candidates.append({
            "id": 0,
            "source_index": source_index,
            "pose_matrix": pose_mat.tolist(),
            "width": float(w),
            "depth": float(d),
            "translation": t.tolist(),
            "rotation": r.tolist()
        })

    return vis_images, candidates


# =============================================================================
# 抓取选择(可插拔):GraspSelector 协议 + VLM / 首个 两种实现
# =============================================================================

class GraspSelector(Protocol):
    """抓取选择接口:从多个抓取候选中选一个。可插拔。

    实现例:VLMSelector(VLM 视觉二次选优)、FirstGraspSelector(跳过 VLM,取筛选后首个)。
    """
    def select(self, color, trans_list, rot_list, width_list, depth_list, intrinsic,
               top_k=8, output_dir="output"):
        """返回 (best_idx, candidates)，候选保留 pose/width/depth。"""
        ...


class VLMSelector:
    """适配 vlm.src.apps.grasp_selection.GraspSelectionApp。"""

    def __init__(self, model_name: str, prompts_dir: str = "prompts"):
        from vlm.src.apps.grasp_selection import GraspSelectionApp
        self.app = GraspSelectionApp(model_name=model_name, prompts_dir=prompts_dir)

    def select(self, color, trans_list, rot_list, width_list, depth_list, intrinsic,
               top_k=8, output_dir="output"):
        """
        3D 抓取 -> 2D 渲染 -> VLM 选择。
        返回 (best_idx, candidates)。candidates 含 translation/rotation/width/pose_matrix。
        渲染的 reject 图与 2D grasp 候选图都落在 output_dir 下(vlm/、2D_grasp/)。
        """
        imgs, candidates = vlm_grasp_visualize_batch(
            color, trans_list, rot_list, width_list, depth_list, intrinsic,
            top_k=top_k, output_dir=output_dir,
        )
        img_paths = save_2d_grasp(output_dir, imgs)

        vlm_res = self.app.run(img_paths)
        print(f"[VLM] Full Response: {vlm_res}")
        best_id = int(vlm_res.get("selected_id", 0)) if isinstance(vlm_res, dict) else 0
        idx = best_id if 0 <= best_id < len(candidates) else 0
        print(f"[VLM] Final Decision -> ID: {idx}")
        return idx, candidates


class FirstGraspSelector:
    """不做 VLM 二次选择:复用几何筛选生成候选,直接取首个。实现 GraspSelector。"""

    def select(self, color, trans_list, rot_list, width_list, depth_list, intrinsic,
               top_k=8, output_dir="output"):
        imgs, candidates = vlm_grasp_visualize_batch(
            color, trans_list, rot_list, width_list, depth_list, intrinsic,
            top_k=top_k, output_dir=output_dir,
        )
        save_2d_grasp(output_dir, imgs)
        paths = save_2d_grasp(output_dir, imgs[:1], subdir="first_select")
        if paths:
            print(f"[Select] Saved first filtered grasp: {paths[0]}")
        print("[Select] No VLM; using first candidate (ID: 0).")
        return 0, candidates
