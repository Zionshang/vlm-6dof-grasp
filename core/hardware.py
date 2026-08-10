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

        # ---- robot backend selected by the component registry ----
        self.robot_kind = c.get("robot", {}).get("kind", "arx5_lcm")
        self.robot_driver_root = c.get("robot", {}).get("driver_root")

        # ---- camera ----
        cam = c["camera"]
        self.camera_kind = cam.get("kind", "d405")
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
        self.ws_z_max = wb.get("z_max")

        # ---- poses ----
        ps = c["poses"]
        self.home_pose = self._optional_array(ps.get("home"))
        self.drop_pose = self._optional_array(ps.get("drop"))
        ready_views = ps.get("ready_views")
        self.ready_views = None if ready_views is None else [np.array(p, dtype=float) for p in ready_views]
        self.realtime_ready_action = self._optional_array(ps.get("realtime_ready_action"))
        self.realtime_ready_main = self._optional_array(ps.get("realtime_ready_main"))

        # ---- target-relative close observation pose ----
        approach = c.get("target_approach") or {}
        self.target_approach_offset = self._optional_array(approach.get("offset"))
        self.target_approach_rpy = self._optional_array(approach.get("rpy"))

        # ---- application motion policy (robot-specific) ----
        self.grasp_policy = c.get("grasp_policy")

        # ---- lcm ----
        self.lcm_task_url = c["lcm"]["task_url"]
        self.lcm_cmd_channel = c["lcm"]["cmd_channel"]
        self.lcm_callback_channel = c["lcm"]["callback_channel"]
        arm_lcm = c["lcm"]["arm"]
        self.lcm_arm_url = arm_lcm.get("url")
        self.lcm_arm_address = arm_lcm.get("address")
        self.lcm_arm_port = arm_lcm.get("port")
        self.lcm_arm_ttl = arm_lcm.get("ttl")

    @staticmethod
    def _optional_array(value):
        return None if value is None else np.array(value, dtype=float)

    def missing_for_grasp_lcm(self):
        """Return unset parameters required by the live LCM grasp application."""
        required = {
            "poses.home": self.home_pose,
            "poses.drop": self.drop_pose,
            "poses.ready_views": self.ready_views,
            "grasp_policy": self.grasp_policy,
            "lcm.task_url": self.lcm_task_url,
            "lcm.cmd_channel": self.lcm_cmd_channel,
            "lcm.callback_channel": self.lcm_callback_channel,
        }
        return [name for name, value in required.items() if value is None]

    def in_workspace(self, x, y, z):
        """Check configured workspace axes; an omitted Z bound disables its check."""
        return (self.ws_x[0] <= x <= self.ws_x[1]
                and self.ws_y[0] <= y <= self.ws_y[1]
                and (self.ws_z_max is None or z <= self.ws_z_max))
