"""硬件配置加载:从 profile yaml 读取硬件/机械臂/通信参数。

设计为"可插拔":换硬件(机械臂 / 相机 / 通信)时,新建一个 profile yaml
并把 DEFAULT_PROFILE 指向它(或构造时传 profile 路径),业务代码无需改动。
"""
from pathlib import Path
import numpy as np
import yaml

# 当前使用的硬件 profile(换硬件时改这里,或构造 HardwareConfig 时传别的 profile)
DEFAULT_PROFILE = "config/hardware/x5_umi_d405.yaml"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_profile(profile):
    p = Path(profile) if profile else Path(DEFAULT_PROFILE)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


class HardwareConfig:
    """硬件参数集合。所有字段均从 profile yaml 原样读取。"""

    def __init__(self, profile=None):
        with open(_resolve_profile(profile)) as f:
            c = yaml.safe_load(f)

        self.name = c.get("name")

        # ---- robot(驱动类型,由 core/robot_client.make_robot_client 分发)----
        self.robot_kind = c.get("robot", {}).get("kind", "arx5_lcm")

        # ---- camera ----
        cam = c["camera"]
        self.camera_matrix = np.array(cam["intrinsic"], dtype=float)
        self.dist_coeffs = np.array(cam["dist_coeffs"], dtype=float)
        self.factor_depth = cam["factor_depth"]

        # ---- hand eye(相机 -> 末端)----
        he = c["hand_eye"]
        self.hand_eye_r = np.array(he["rotation"], dtype=float)
        self.hand_eye_t = np.array(he["translation"], dtype=float)

        # ---- gripper ----
        self.gripper_max_width = c["gripper"]["max_width"]
        self.gripper_approach_width = c["gripper"]["approach_width"]

        # ---- workspace bounds ----
        wb = c["workspace_bounds"]
        self.ws_x = tuple(wb["x"])
        self.ws_y = tuple(wb["y"])
        self.ws_z_max = wb["z_max"]

        # ---- poses ----
        ps = c["poses"]
        self.home_pose = np.array(ps["home"], dtype=float)
        self.drop_pose = np.array(ps["drop"], dtype=float)
        self.ready_views = [np.array(p, dtype=float) for p in ps["ready_views"]]
        self.realtime_ready_action = np.array(ps["realtime_ready_action"], dtype=float)
        self.realtime_ready_main = np.array(ps["realtime_ready_main"], dtype=float)

        # ---- lcm ----
        self.lcm_task_url = c["lcm"]["task_url"]
        self.lcm_cmd_channel = c["lcm"]["cmd_channel"]
        self.lcm_callback_channel = c["lcm"]["callback_channel"]
        self.lcm_arm_address = c["lcm"]["arm"]["address"]
        self.lcm_arm_port = c["lcm"]["arm"]["port"]
        self.lcm_arm_ttl = c["lcm"]["arm"]["ttl"]

    def in_workspace(self, x, y, z):
        """位姿是否在工作空间安全边界内(统一边界,供各入口共用)。"""
        return (self.ws_x[0] <= x <= self.ws_x[1]
                and self.ws_y[0] <= y <= self.ws_y[1]
                and z <= self.ws_z_max)
