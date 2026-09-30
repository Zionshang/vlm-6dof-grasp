import time
import numpy as np
from dataclasses import dataclass

from transform import offset_pose


@dataclass
class GraspStep:
    """One configured robot/gripper step in a grasp sequence."""
    name: str
    gripper: str = "max"
    preview: float = 0.5
    wait: float = 0.0
    offset: tuple = (0.0, 0.0, 0.0)
    local_offset: tuple = (0.0, 0.0, 0.0)
    rpy: tuple | None = None
    use_home_pose: bool = False
    speed_percent: int | None = None
    gripper_duration: float = 0.0


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

    def _resolve_pose(self, arm_cmd, step):
        if step.use_home_pose:
            return np.array(self.hw.home_pose, dtype=float).copy()
        return offset_pose(
            arm_cmd, step.offset, step.local_offset, step.rpy,
        )

    def pose_for_step(self, arm_cmd, name):
        """Resolve one configured step pose without executing it."""
        matches = [step for step in self.steps if step.name == name]
        if len(matches) != 1:
            raise ValueError(
                f"Grasp sequence requires exactly one '{name}' step, "
                f"found {len(matches)}"
            )
        return self._resolve_pose(arm_cmd, matches[0])

    def run_sequence(self, arm_cmd, target_width, steps=None,
                     arrival_timeout=None):
        """按 steps 有序执行抓取序列。

        返回 (success, reason):全部完成返回 success；异常包含步骤名和原始原因。
        """
        steps = self.steps if steps is None else steps
        if not steps:
            raise ValueError("Grasp sequence has no configured steps")
        self.last_state, speed = None, 100
        success, reason = True, "success"
        try:
            for step in steps:
                if step.speed_percent is not None and step.speed_percent != speed:
                    self.client.set_speed_percent(step.speed_percent)
                    speed = step.speed_percent
                pose = self._resolve_pose(arm_cmd, step)
                grip = self._resolve_gripper(step.gripper, target_width)
                if arrival_timeout is None:
                    self.client.set_ee_pose(
                        pose, gripper_pos=grip, preview_time=step.preview,
                        gripper_duration=step.gripper_duration,
                    )
                else:
                    from robot_safety import move_to_pose_and_wait
                    self.last_state = move_to_pose_and_wait(
                        self.client, self.hw, pose, grip, arrival_timeout,
                        f"Grasp {step.name}", step.gripper == "max",
                        step.gripper_duration,
                    )
                if step.wait:
                    time.sleep(step.wait)
        except Exception as exc:
            success, reason = False, str(exc)
        finally:
            if speed != 100:
                try:
                    self.client.set_speed_percent(100)
                except Exception as exc:
                    success, reason = False, f"{reason}; 恢复速度失败: {exc}"
        return success, reason
