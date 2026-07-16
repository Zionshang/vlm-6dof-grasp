import time
import numpy as np
from dataclasses import dataclass


@dataclass
class GraspStep:
    """抓取执行序列中的一个状态步骤(纯数据,描述"这一步怎么动")。

    借鉴 ros_base 的状态分离理念:把"怎么动"收敛成有序步骤 + 统一调度与错误出口。
    本项目为事件驱动的一次性任务,不做高频 tick。
    """
    name: str                         # approach / reach / grasp / lift / home / reopen(状态名,用于日志与错误定位)
    gripper: str = "max"              # "max"(张开到 grip_max) | "target"(合到目标宽度)
    preview: float = 0.5             # set_ee_pose 的 preview_time
    wait: float = 0.0                # 运动后 sleep 时长(correct=True 的步骤内部已 sleep,wait 不再生效)
    correct: bool = False            # 是否走闭环纠偏 _safe_move_with_correction
    offset: tuple = (0.0, 0.0, 0.0)  # 相对 arm_cmd 的 xyz 偏移(use_home_pose=True 时忽略)
    fix_rpy: tuple = (None, None, None)  # 强制 (rx, ry, rz);元素 None = 不覆盖原值
    use_home_pose: bool = False      # True 时位姿取 hw.home_pose(用于 home 步)


class GraspExecutor:
    """抓取运动执行器(共享框架):遍历 GraspStep 序列,统一移动 / 纠偏 / 失败处理。

    注入 client / hw / grip_max,可插拔不硬编码。在各 app 的 __init__ 里建一次复用。
    """

    def __init__(self, client, hw, grip_max):
        self.client = client
        self.hw = hw
        self.grip_max = grip_max

    def _resolve_gripper(self, mode, target_width):
        return self.grip_max if mode == "max" else target_width

    def _safe_move_with_correction(self, target_pose, gripper_pos, preview_time=0.5, retries=1):
        """
        Executes a move with closed-loop error correction based on end-effector state feedback.
        """
        # 1. Initial Command
        self.client.set_ee_pose(target_pose, gripper_pos=gripper_pos, preview_time=preview_time)
        time.sleep(preview_time + 0.3)

        current_target = target_pose.copy()  # We correct *relative to* the initial command

        for i in range(retries):
            # 2. Get Actual State vs Original Target
            curr_state = self.client.get_state()
            if not curr_state:
                break

            curr_pos = np.array(curr_state['ee_pose'][:3])
            desired_pos = target_pose[:3]

            # 3. Calculate Global Error (Where I am vs Where I wanted to go)
            pos_error = desired_pos - curr_pos
            error_norm = np.linalg.norm(pos_error)

            # 4. Threshold (5mm)
            if error_norm < 0.005:
                break

            print(f"[Control] Correction Loop {i+1}: Global Error={error_norm*1000:.1f}mm")

            # 5. Compute Correction (Gain = 0.8)
            correction = pos_error * 0.8
            current_target[:3] += correction

            # 6. Safety Clamp
            x, y, z = current_target[:3]
            # Ensure we don't correct into unsafe zones
            if not self.hw.in_workspace(x, y, z):
                print(f"[Control] Correction unsafe {current_target[:3]}. Aborting.")
                break

            # 7. Apply Correction
            self.client.set_ee_pose(current_target, gripper_pos=gripper_pos, preview_time=0.3)
            time.sleep(0.4)

    def run_sequence(self, arm_cmd, target_width, steps):
        """按 steps 有序执行抓取序列。

        返回 (success, reason):全部完成 -> (True, "success");任一步异常 -> (False, step.name)。
        """
        for step in steps:
            # 1. Compute Target Pose
            if step.use_home_pose:
                pose = np.array(self.hw.home_pose, dtype=float).copy()
            else:
                pose = np.array(arm_cmd, dtype=float).copy()
                pose[0] += step.offset[0]
                pose[1] += step.offset[1]
                pose[2] += step.offset[2]

            # 2. Force RPY (if any)
            rx, ry, rz = step.fix_rpy
            if rx is not None:
                pose[3] = rx
            if ry is not None:
                pose[4] = ry
            if rz is not None:
                pose[5] = rz

            # 3. Resolve Gripper
            grip = self._resolve_gripper(step.gripper, target_width)
            print(f"[Grasp] step={step.name} pose={np.array2string(pose, precision=3)} grip={grip:.3f}")

            # 4. Execute (correct or plain move)
            try:
                if step.correct:
                    self._safe_move_with_correction(pose, grip, preview_time=step.preview, retries=1)
                else:
                    self.client.set_ee_pose(pose, gripper_pos=grip, preview_time=step.preview)
                    if step.wait:
                        time.sleep(step.wait)
            except Exception as e:
                print(f"[Grasp] FAILED at '{step.name}': {e}")
                return False, step.name

        return True, "success"
