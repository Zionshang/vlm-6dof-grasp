
import numpy as np
import cv2

def project_grasp_to_2d(trans, rot, width, intrinsic, depth=0.04):
    """
    将 3D 抓取投影到 2D 图像平面。
    GraspNet 坐标系定义 (参考 graspnetAPI):
    - 局部坐标系原点 (Model Origin): 位于两指根部连线中心。
    - X轴: 接近方向 Approach (从底座指向指尖)。指尖位于 X=depth 处。
    - Y轴: 闭合方向 Width (两指张开方向)。
    - Z轴: 垂直方向 Orthogonal。
    
    Args:
        trans: (3,) 平移 [x, y, z] (Model Origin 在相机坐标系下的位置)
        rot: (3, 3) 旋转矩阵
        width: float 抓取宽度
        intrinsic: (3, 3) 相机内参
        depth: float 手指长度 (默认 0.04m, 需与 3D 可视化一致)
    
    Returns:
        pts_2d: (4, 2) 投影后的像素坐标 [u, v]
    """
    hw = width / 2
    d = depth
    
    # 定义关键点 (在抓取局部坐标系下)
    # 修正：Origin 在 Base，Tip 在 +X 方向
    points_g = np.array([
        [d,  -hw, 0],  # 0: 左指尖
        [0,  -hw, 0],  # 1: 左指根
        [0,   hw, 0],  # 2: 右指根
        [d,   hw, 0],  # 3: 右指尖
    ]).T # (3, 4)

    # 变换到相机坐标系
    points_c = rot @ points_g + trans.reshape(3, 1) # (3, 4)
    
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

def vlm_grasp_visualize(image, trans, rot, width, intrinsic, top_k=5):
    # 此函数保持不变，用于单图概览（Legacy）
    """
    将 EconomicGrasp 抓取结果可视化在图像上，用于传给 VLM。
    """
    vis_img = image.copy()
    
    # 统一格式处理 (兼顾单条与批量)
    trans = np.array(trans)
    rot = np.array(rot)
    width = np.array(width)
    
    if trans.ndim == 1: trans = trans[np.newaxis, :]
    if rot.ndim == 2:   rot = rot[np.newaxis, ...]
    if width.ndim == 0: width = width[np.newaxis]
    
    num_grasps = min(len(trans), top_k)
    candidates = []
    
    for i in range(num_grasps):
        t = trans[i]
        r = rot[i]
        w = width[i]
        
        if t[2] <= 0: continue # 过滤相机后方的点
        
        # 投影关键点
        pts = project_grasp_to_2d(t, r, w, intrinsic)
        
        center = np.mean(pts[1:3], axis=0).astype(int)
        
        # 绘制 U型 夹爪
        color_finger = (0, 0, 255) # 红色
        color_base = (0, 255, 0)   # 绿色
        thick = 2
        
        cv2.line(vis_img, tuple(pts[0]), tuple(pts[1]), color_finger, thick)
        cv2.line(vis_img, tuple(pts[1]), tuple(pts[2]), color_base, thick)
        cv2.line(vis_img, tuple(pts[2]), tuple(pts[3]), color_finger, thick)
        
        # 画主体黑色字
        cv2.putText(vis_img, str(i), tuple(center), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        
        # 构建 4x4 变换矩阵供机器人使用
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = r
        pose_mat[:3, 3] = t
        
        candidates.append({
            "id": i,
            "pose_matrix": pose_mat.tolist(),
            "width": float(w),
            "translation": t.tolist(),
            "rotation": r.tolist()
        })
        
    return vis_img, candidates

def vlm_grasp_visualize_batch(image, trans, rot, width, intrinsic, top_k=5):
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
    
    for i in range(num_grasps):
        # 每次都复制一张干净的背景图
        vis_img = image.copy()
        
        t = trans[i]
        r = rot[i]
        w = width[i]
        
        if t[2] <= 0: continue 
        
        pts = project_grasp_to_2d(t, r, w, intrinsic)
        
        # 绘制 U型 夹爪
        # image is RGB, so use RGB colors
        color_finger = (255, 0, 0) # Red
        color_base = (0, 255, 0)   # Green
        thick = 3 # 加粗一点，反正不重叠
        
        cv2.line(vis_img, tuple(pts[0]), tuple(pts[1]), color_finger, thick)
        cv2.line(vis_img, tuple(pts[1]), tuple(pts[2]), color_base, thick)
        cv2.line(vis_img, tuple(pts[2]), tuple(pts[3]), color_finger, thick)
        
        # 绘制抓取方向箭头 (蓝色箭头: 底座 -> 指尖)
        # 计算中心点
        center_base = np.mean(pts[1:3], axis=0).astype(int)
        center_tip = np.mean([pts[0], pts[3]], axis=0).astype(int)
        # 绘制箭头
        cv2.arrowedLine(vis_img, tuple(center_base), tuple(center_tip), (0, 0, 255), 4, tipLength=0.3)
        
        # 绘制详细 Label
        # 策略：计算抓取投影的 2D 包围盒，将文字放在上方或下方，彻底杜绝遮挡
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        
        label = f"ID: {i}"
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

        pose_mat = np.eye(4)
        pose_mat[:3, :3] = r
        pose_mat[:3, 3] = t
        
        vis_images.append(vis_img)
        candidates.append({
            "id": i,
            "pose_matrix": pose_mat.tolist(),
            "width": float(w),
            "translation": t.tolist(),
            "rotation": r.tolist()
        })
        
    return vis_images, candidates
