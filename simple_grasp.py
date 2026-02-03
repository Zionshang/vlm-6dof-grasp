# import sys
# import time
# import cv2
# import numpy as np
# import math
# from pathlib import Path
# from scipy.spatial.transform import Rotation as R

# # 1. 环境与路径设置
# ROOT = Path(__file__).resolve().parent
# sys.path.extend([str(ROOT)])

# from communication.lcm.lcm_client import Arx5LcmClient
# from realsense_driver import RealSenseD405
# from vlm.src.apps.static_detection import StaticDetectionApp
# from vlm.src.apps.orientation_prediction import OrientationPredictionApp

# # 2. 标定参数与相机内参
# # 手眼标定 (Camera -> End Effector)
# HAND_EYE_R = np.array([
#     [-0.00609262, -0.30277251,  0.95304338],
#     [-0.9999547,  -0.00512515, -0.00802072],
#     [ 0.00731294, -0.95304907, -0.30272757]
# ])
# HAND_EYE_T = np.array([-0.19322195, 0.01031036, 0.10957433])

# # 相机内参
# CAMERA_MATRIX = np.array([
#     [435.75600787, 0.0, 423.51396062],
#     [0.0, 435.67414187, 243.52287949],
#     [0.0, 0.0, 1.0]
# ])

# def get_base_coordinates(u, v, depth_mm, current_ee_pose):
#     """
#     像素坐标(u,v)+深度 -> 机械臂基座坐标(x,y,z)
#     """
#     # 1. 反投影 (Pixel -> Camera)
#     z_cam = depth_mm / 1000.0  # mm 转 m
#     if z_cam <= 0.01: return None # 无效深度
    
#     x_cam = (u - CAMERA_MATRIX[0, 2]) * z_cam / CAMERA_MATRIX[0, 0]
#     y_cam = (v - CAMERA_MATRIX[1, 2]) * z_cam / CAMERA_MATRIX[1, 1]
#     p_cam = np.array([x_cam, y_cam, z_cam])

#     # 2. 变换 (Camera -> End Effector)
#     p_ee = HAND_EYE_R @ p_cam + HAND_EYE_T

#     # 3. 变换 (End Effector -> Base)
#     # current_ee_pose: [x, y, z, rx, ry, rz]
#     t_base_ee = current_ee_pose[:3]
#     # 假设使用 XYZ 欧拉角 (与 convert.py 保持一致)
#     r_base_ee = R.from_euler('xyz', current_ee_pose[3:]).as_matrix()
    
#     p_base = r_base_ee @ p_ee + t_base_ee
#     return p_base

# def main():
#     # 初始化
#     print("Initializing Robot & Camera...")
#     # 请根据实际情况调整 address/port
#     client = Arx5LcmClient(url="", address="239.255.76.67", port=7667, ttl=1)
#     cam = RealSenseD405()
    
#     # 加载 VLM 应用
#     print("Loading VLM Apps...")
#     prompts_path = str(ROOT / "vlm/prompts")
#     detect_app = StaticDetectionApp(model_name="qwen2.5-vl", prompts_dir=prompts_path)
#     orient_app = OrientationPredictionApp(model_name="qwen2.5-vl", prompts_dir=prompts_path)

#     # 移动到初始观测位置
#     print("Moving to Ready Pose...")
#     client.reset_to_home()
#     # 这一步是为了让相机能看到工作台。根据你的场景调整。
#     ready_pose = np.array([0.25, 0.0, 0.17, 0.0, 1.0, 0.0]) 
#     client.set_ee_pose(ready_pose, gripper_pos=0.08, preview_time=2.0)
#     time.sleep(3.0)

#     try:
#         while True:
#             prompt_text = input("\n请输入抓取目标 (Prompt), 输入 'q' 退出: ").strip()
#             if prompt_text.lower() == 'q': break
            
#             # 1. 获取图像
#             color, depth = cam.get_frames()
#             if color is None: 
#                 print("Camera error."); continue
                
#             # 保存临时图供 VLM 使用
#             img_path = ROOT / "output" / "temp_detect.jpg"
#             img_path.parent.mkdir(parents=True, exist_ok=True)
#             cv2.imwrite(str(img_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

#             # 2. 调用 VLM 检测 (Detection)
#             print(f"Detecting '{prompt_text}'...")
#             res_det = detect_app.run(str(img_path), prompt_text)
            
#             if not res_det['success'] or not res_det['pixel_boxes']:
#                 print(f"未检测到物体: {prompt_text}")
#                 continue
            
#             # 取第一个检测框
#             box = res_det['pixel_boxes'][0] # [x1, y1, x2, y2]
#             cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
            
#             # 显示一下
#             vis_img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
#             cv2.rectangle(vis_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
#             cv2.circle(vis_img, (cx, cy), 5, (0,0,255), -1)
#             cv2.imshow("Detection", vis_img)
#             cv2.waitKey(100)

#             # 3. 计算基座坐标 (Position)
#             # 获取深度 (取中心附近中值以防噪点)
#             d_crop = depth[max(0,cy-2):cy+3, max(0,cx-2):cx+3]
#             if d_crop.size == 0: d_val = 0
#             else: d_val = np.median(d_crop)
            
#             if d_val <= 0:
#                 print("深度无效，跳过。")
#                 continue

#             state = client.get_state()
#             p_base = get_base_coordinates(cx, cy, d_val, state['ee_pose'])
#             if p_base is None: continue
            
#             obj_x, obj_y, obj_z = p_base
#             print(f"物体位置 (Base): X={obj_x:.3f}, Y={obj_y:.3f}, Z={obj_z:.3f}")

#             # 4. 调用 VLM 获取角度 (Orientation)
#             print("Predicting Orientation...")
#             res_ori = orient_app.run(str(img_path), prompt_text)
#             angle_deg = res_ori.get('rotation_angle', 0.0)
#             print(f"预测角度 q: {angle_deg}°")

#             # 5. 计算目标位姿
#             # 你的公式: x=x-0.2*cos(-q) y = y-0.2sin(-q) z= z+0.2
#             # 你的公式: rx=0 ry=0.85 rz=-q
#             q_rad = math.radians(angle_deg)
#             neg_q = -q_rad 

#             target_x = obj_x - 0.2 * math.cos(neg_q)
#             target_y = obj_y - 0.2 * math.sin(neg_q)
#             target_z = obj_z + 0.2
            
#             target_rx = 0.0
#             target_ry = 0.85
#             target_rz = neg_q

#             target_pose = np.array([target_x, target_y, target_z, target_rx, target_ry, target_rz])
#             print(f"目标机械臂位姿: {target_pose}")

#             # 6. 执行
#             if input("是否执行? (y/n): ").lower() == 'y':
#                 print("Executing...")
#                 client.set_ee_pose(target_pose, gripper_pos=0.08, preview_time=2.0)
#                 time.sleep(2.5)
#                 print("Execution Done.")
                
#     finally:
#         cam.release()
#         cv2.destroyAllWindows()
#         client.reset_to_home() # 退出时回零

# if __name__ == "__main__":
#     main()
