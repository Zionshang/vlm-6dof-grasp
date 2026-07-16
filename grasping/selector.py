"""抓取选择模块。

包含两部分:
  1) 3D 抓取 -> 2D 投影渲染(原 economic_grasp/utils/vlm_utils.py,逻辑原样迁入);
  2) VLMSelector:把"渲染 -> VLM 选择 -> 最佳 id"封装成一步
     (逻辑原样来自 run_grasp_lcm / run_realtime 的选择段)。
"""
import numpy as np
import cv2
from typing import Protocol

from saver import save_reject, save_2d_grasp


# =============================================================================
# 3D 抓取 -> 2D 投影渲染(原 economic_grasp/utils/vlm_utils.py,原样保留)
# =============================================================================

def _compress_image(image, max_dim=480):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim: return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def project_grasp_to_2d(trans, rot, width, intrinsic, depth=0.04):
    """
    将 3D 抓取投影到 2D 图像平面。
    这里进行了【坐标对齐修正】，使得可视化位置与机器人实际执行位置一致。
    机器人执行时会沿着抓取方向前移 4cm (T_align)。
    因此可视化也需要前移，才能正确显示抓取落点。
    """

    # 1. 构建原始抓取矩阵 T_grasp2cam
    T_grasp2cam = np.eye(4)
    T_grasp2cam[:3, :3] = rot
    T_grasp2cam[:3, 3] = trans

    # 2. 定义前移修正矩阵 T_align
    T_align = np.eye(4)
    T_align[:3, 3] = [0.04, 0, 0]

    # 3. 计算修正后的抓取位姿
    T_final = T_grasp2cam @ T_align

    # 4. 提取新的旋转和平移
    rot_final = T_final[:3, :3]
    trans_final = T_final[:3, 3]

    hw = width / 2
    d = depth

    # 定义关键点 (在抓取局部坐标系下)
    # Origin (X=0) 现在是【修正后】的指尖位置
    # Tip 在 0, Base 在 -d
    points_g = np.array([
        [0,   -hw, 0],  # 0: 左指尖 (Tip)
        [-d,  -hw, 0],  # 1: 左指根 (Base)
        [-d,   hw, 0],  # 2: 右指根 (Base) (在此处绘制蓝点)
        [0,    hw, 0],  # 3: 右指尖 (Tip)
    ]).T # (3, 4)

    # 变换到相机坐标系 (使用修正后的 rot_final 和 trans_final)
    points_c = rot_final @ points_g + trans_final.reshape(3, 1) # (3, 4)

    # 投影到像素坐标
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    Z = points_c[2, :]
    X = points_c[0, :]
    Y = points_c[1, :]

    # 避免除以零
    Z[Z==0] = 0.001

    U = (fx * X / Z) + cx
    V = (fy * Y / Z) + cy

    return np.stack([U, V], axis=1).astype(int)

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


def vlm_grasp_visualize_batch(image, trans, rot, width, intrinsic, top_k=8, output_dir="output"):
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

    if trans.ndim == 1: trans = trans[np.newaxis, :]
    if rot.ndim == 2:   rot = rot[np.newaxis, ...]
    if width.ndim == 0: width = width[np.newaxis]

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

        pts = project_grasp_to_2d(t, r, w, intrinsic)

        if i == 0:
            best_candidate_backup = (vis_img.copy(), pts, t, r, w)

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

        print(f"[Debug G{i}] Width={width_px:.1f}px | dPos=({dx}, {dy}) | Angle={angle_deg:.1f} deg")

        # --- 无论好坏，先画出几何结构 (Review用) ---
        color_finger = (255, 0, 0) # Red
        color_base = (0, 255, 0)   # Green
        thick = 3
        cv2.line(vis_img, tuple(pts[0]), tuple(pts[1]), color_finger, thick)
        cv2.line(vis_img, tuple(pts[1]), tuple(pts[2]), color_base, thick)
        cv2.line(vis_img, tuple(pts[2]), tuple(pts[3]), color_finger, thick)
        cv2.circle(vis_img, tuple(pts[2]), 8, (0, 0, 255), -1)
        cv2.arrowedLine(vis_img, tuple(center_base), tuple(center_tip), (0, 0, 255), 3, tipLength=0.35)

        # --- 统一筛选 ---
        cond_width = width_px < 75
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
            "pose_matrix": pose_mat.tolist(),
            "width": float(w),
            "translation": t.tolist(),
            "rotation": r.tolist()
        })
        valid_idx += 1

    # [Added] 兜底逻辑：如果没有任何候选通过筛选，强制保留第一个
    if not candidates and best_candidate_backup is not None:
        print("[Warn] All candidates rejected. Force keeping the ID:0 candidate.")
        vis_img, pts, t, r, w = best_candidate_backup

        # 重新绘制 Label (Hardcoded ID: 0)
        _draw_id_label(vis_img, pts, 0)

        pose_mat = np.eye(4)
        pose_mat[:3, :3] = r
        pose_mat[:3, 3] = t

        vis_images.append(_compress_image(vis_img))
        candidates.append({
            "id": 0,
            "pose_matrix": pose_mat.tolist(),
            "width": float(w),
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
    def select(self, color, trans_list, rot_list, width_list, intrinsic,
               top_k=8, output_dir="output"):
        """返回 (best_idx, candidates)。candidates[i] 含 translation/rotation/width。"""
        ...


class VLMSelector:
    """适配 vlm.src.apps.grasp_selection.GraspSelectionApp。"""

    def __init__(self, model_name: str, prompts_dir: str = "prompts"):
        from vlm.src.apps.grasp_selection import GraspSelectionApp
        self.app = GraspSelectionApp(model_name=model_name, prompts_dir=prompts_dir)

    def select(self, color, trans_list, rot_list, width_list, intrinsic,
               top_k=8, output_dir="output"):
        """
        3D 抓取 -> 2D 渲染 -> VLM 选择。
        返回 (best_idx, candidates)。candidates 含 translation/rotation/width/pose_matrix。
        渲染的 reject 图与 2D grasp 候选图都落在 output_dir 下(vlm/、2D_grasp/)。
        """
        imgs, candidates = vlm_grasp_visualize_batch(
            color, trans_list, rot_list, width_list, intrinsic, top_k=top_k, output_dir=output_dir
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

    def select(self, color, trans_list, rot_list, width_list, intrinsic,
               top_k=8, output_dir="output"):
        _, candidates = vlm_grasp_visualize_batch(
            color, trans_list, rot_list, width_list, intrinsic, top_k=top_k, output_dir=output_dir
        )
        print("[Select] No VLM; using first candidate (ID: 0).")
        return 0, candidates
