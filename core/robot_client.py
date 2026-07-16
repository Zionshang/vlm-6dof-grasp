"""机械臂驱动层:可插拔。

定义 RobotClient 协议与工厂 make_robot_client。换机械臂(其他型号 / 其他通信)只需:
1. 新增一个实现 RobotClient 接口的 client 类
2. 在 make_robot_client 里按 hw.robot_kind 增加一个分支
入口 / pipeline / executor 无需改动(client 已通过构造注入)。
"""
from typing import Protocol, Any
import numpy as np


class RobotConfig(Protocol):
    """机械臂配置(由 client 提供,如夹爪宽度)。"""
    gripper_width: float


class RobotClient(Protocol):
    """机械臂接口:末端位姿控制 + 状态读取 + 复位。"""
    def set_ee_pose(self, pose, gripper_pos, preview_time) -> None: ...
    def get_state(self) -> Any: ...
    def reset_to_home(self) -> None: ...
    def get_robot_config(self) -> RobotConfig: ...


def make_robot_client(hw):
    """根据硬件 profile 的 robot.kind 选择机械臂 client。换臂时在此加分支。"""
    kind = getattr(hw, "robot_kind", "arx5_lcm")
    if kind == "arx5_lcm":
        from communication.lcm.lcm_client import Arx5LcmClient
        return Arx5LcmClient(
            url="",
            address=hw.lcm_arm_address,
            port=hw.lcm_arm_port,
            ttl=hw.lcm_arm_ttl,
        )
    raise ValueError(f"Unknown robot kind: {kind}")
