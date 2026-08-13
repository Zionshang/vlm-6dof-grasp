import time
import numpy as np
from dataclasses import dataclass


@dataclass
class GraspStep:
    """One configured robot/gripper step in a grasp sequence."""
    name: str
    gripper: str = "max"
    preview: float = 0.5
    wait: float = 0.0
    offset: tuple = (0.0, 0.0, 0.0)
    rpy: tuple | None = None
    use_home_pose: bool = False


class GraspExecutor:
    """抓取运动执行器:按步骤直接发送目标位姿。

    注入 client / hw / grip_max,可插拔不硬编码。在各 app 的 __init__ 里建一次复用。
    """

    def __init__(self, client, hw, grip_max, steps=None):
        self.client = client
        self.hw = hw
        self.grip_max = grip_max
        self.steps = list(steps or [])

    def _resolve_gripper(self, mode, target_width):
        return self.grip_max if mode == "max" else target_width

    def run_sequence(self, arm_cmd, target_width, steps=None,
                     arrival_timeout=None):
        """按 steps 有序执行抓取序列。

        返回 (success, reason):全部完成返回 success；异常包含步骤名和原始原因。
        """
        steps = self.steps if steps is None else steps
        if not steps:
            raise ValueError("Grasp sequence has no configured steps")
        self.last_state = None
        for step in steps:
            if step.use_home_pose:
                pose = np.array(self.hw.home_pose, dtype=float).copy()
            else:
                pose = np.array(arm_cmd, dtype=float).copy()
                pose[:3] += step.offset
                if step.rpy is not None:
                    pose[3:] = step.rpy

            grip = self._resolve_gripper(step.gripper, target_width)
            try:
                if arrival_timeout is None:
                    self.client.set_ee_pose(
                        pose, gripper_pos=grip, preview_time=step.preview,
                    )
                else:
                    from robot_safety import move_to_pose_and_wait
                    self.last_state = move_to_pose_and_wait(
                        self.client, self.hw, pose, grip, arrival_timeout,
                        f"Grasp {step.name}", step.gripper == "max",
                    )
                if step.wait:
                    time.sleep(step.wait)
            except Exception as exc:
                return False, str(exc)

        return True, "success"
