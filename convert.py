import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_new(
        grasp_translation,  # GraspNet 输出的平移 (相机坐标系下)
        grasp_rotation_mat,  # GraspNet 输出的旋转矩阵 (相机坐标系下, 3x3)
        current_ee_pose,  # 机械臂当前末端在基座坐标系下的位姿 [x, y, z, rx, ry, rz]
        handeye_rot,  # 手眼标定旋转矩阵 (相机→末端)
        handeye_trans,  # 手眼标定平移向量 (相机→末端)
        gripper_length=0.05
):
    """
    根据 GraspNet 输出 (相机系下的抓取位姿)，计算在机械臂基座系下的抓取位姿。

    -----------------------------------------------------------------------------------
        * GraspNet 默认抓取朝向是局部 x 轴；本项目中机械臂末端坐标系约定 x 前、y 左、z 上，
            因此将 "GraspNet.x" 直接对齐到 "Robot.x"（无需轴变换）。
        * 同时把夹爪长度的补偿 (沿末端主轴 -X 方向后退 gripper_length) 也放进该对齐矩阵里，一并完成。
    * current_ee_pose 给出的末端姿态是在基座坐标系下 (x,y,z,rx,ry,rz)，
      若你的机械臂控制器/SDK 使用的是其它顺序，需要在此函数内对应修改 R.from_euler() 的顺序。
    -----------------------------------------------------------------------------------

    返回 [base_x, base_y, base_z, base_rx, base_ry, base_rz]，这里的 base_rx, base_ry, base_rz
    是按某种欧拉角顺序 (示例中是 'XYZ' 或 'ZYX') 输出，你可根据机械臂需求进行调整。
    """

    # =============== 1) 构造：GraspNet输出【抓取坐标系 → 相机坐标系】的变换矩阵 ================
    T_grasp2cam = np.eye(4, dtype=float)
    T_grasp2cam[:3, :3] = grasp_rotation_mat
    T_grasp2cam[:3, 3] = grasp_translation

    # =============== 2) 在 GraspNet 的输出上做「轴对齐 + 夹爪补偿」 ================

    # 轴对齐：保持 x 轴不变，y 轴取反，z 轴取反（等价于绕 x 轴旋转 180°）
    R_align = np.array([[1.0,  0.0,  0.0],
                        [0.0, -1.0,  0.0],
                        [0.0,  0.0, -1.0]], dtype=float)

    T_align = np.eye(4, dtype=float)
    T_align[:3, :3] = R_align

    # gripper_length 补偿：新坐标系 X 为抓取主轴，让末端后退 (沿 -X 方向)
    # 若想“探出去”，可以改成 [+gripper_length, 0, 0]
    T_align[:3, 3] = [gripper_length, 0, 0]

    # 得到【修正后的】抓取姿态 (相机坐标系下)
    T_gripper2cam = T_grasp2cam @ T_align
    # T_gripper2cam = T_align @ T_grasp2cam
    # =============== 3) 手眼标定：构造【相机坐标系 → 末端坐标系】 ================
    #   如果实际标定结果是 (末端→相机)，就需要再取逆；此处假设 handeye_rot, handeye_trans
    #   的确代表 “相机→末端”。
    #
    T_cam2ee = np.eye(4, dtype=float)
    T_cam2ee[:3, :3] = handeye_rot
    T_cam2ee[:3, 3] = handeye_trans
    # print(f"相机坐标系 → 末端坐标系(平移):\n{handeye_trans}\n")
    # print(f"相机坐标系 → 末端坐标系(旋转):\n{handeye_rot}\n")

    # =============== 4) 当前末端姿态：构造【末端坐标系 → 基座坐标系】的变换 ================
    #   如果你的机械臂 API 返回的 (x,y,z,rx,ry,rz) 本身就表示“末端在基座系的位姿”，
    #   那么做法是：T_ee2base * [0,0,0,1] = [x,y,z,1]，并把欧拉角对应旋转填进去。

    x_ee, y_ee, z_ee, rx_ee, ry_ee, rz_ee = current_ee_pose

    # 例：机械臂有些驱动器/SDK喜欢 'ZYX' 顺序，也有喜欢 'XYZ'。下面仅做示例：
    # 如果你确定 rx_ee, ry_ee, rz_ee 是以 "XYZ" 顺序，则要用 R.from_euler('XYZ',[...])。
    # 如果你确定是 "ZYX" ，则要 R.from_euler('ZYX',[rz_ee, ry_ee, rx_ee])。
    #
    # 以下演示用 'XYZ'，根据你实际情况来：
    R_ee2base = R.from_euler('xyz', [rx_ee, ry_ee, rz_ee], degrees=False).as_matrix()

    T_ee2base = np.eye(4, dtype=float)
    T_ee2base[:3, :3] = R_ee2base
    T_ee2base[:3, 3] = [x_ee, y_ee, z_ee]
    # print(f"末端坐标系 → 基座坐标系(平移):\n{[x_ee, y_ee, z_ee]}\n")
    # print(f"末端坐标系 → 基座坐标系(旋转):\n{R_ee2base}\n")
    # =============== 5) 计算最终【抓取坐标系(对齐后) → 基座坐标系】 ================
    #
    #   T_gripper2base = T_ee2base * (T_cam2ee * T_gripper2cam)
    #
    #   这样就把"修正后"的抓取位姿从相机系一路转换到基座系。
    #
    T_gripper2base = T_ee2base @ (T_cam2ee @ T_gripper2cam)

    # 分离出旋转 + 平移
    final_rot_mat = T_gripper2base[:3, :3]
    final_trans = T_gripper2base[:3, 3]
    # print(f"抓取坐标系 → 基座坐标系(平移):\n{final_trans}\n")
    # print(f"抓取坐标系 → 基座坐标系(旋转):\n{final_rot_mat}\n")

    # =============== 6) 将最终旋转矩阵变为欧拉角 (如果你需要欧拉角作为机械臂指令) ================
    #   再次强调，具体要什么顺序，需要和你的机械臂驱动匹配。
    #   演示这里输出 "XYZ" 顺序的 [rx, ry, rz]。
    #
    final_euler = R.from_matrix(final_rot_mat).as_euler('xyz', degrees=False)
    base_rx, base_ry, base_rz = final_euler

    # 拼装输出 [x, y, z, rx, ry, rz]
    result = [
        final_trans[0],
        final_trans[1],
        final_trans[2],
        base_rx,
        base_ry,
        base_rz
    ]

    return result
