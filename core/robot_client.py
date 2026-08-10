"""Robot client contracts and adapters used by robot plugins."""
from pathlib import Path
from typing import Protocol, Any
import sys


class RobotConfig(Protocol):
    """机械臂配置(由 client 提供,如夹爪宽度)。"""
    gripper_width: float


class RobotClient(Protocol):
    """机械臂接口:末端位姿控制 + 状态读取 + 复位。"""
    def set_ee_pose(self, pose, gripper_pos, preview_time) -> None: ...
    def get_state(self) -> Any: ...
    def reset_to_home(self) -> None: ...
    def get_robot_config(self) -> RobotConfig: ...


class SafeRobotClient:
    """Transparent driver wrapper with an explicit safety lifecycle hook."""

    def __init__(self, client, safe_stop_enabled=True):
        self._managed_client = client
        self._safe_stop_enabled = safe_stop_enabled

    def __getattr__(self, name):
        return getattr(self._managed_client, name)

    def enable_safe_stop(self):
        self._safe_stop_enabled = True

    def disable_safe_stop(self):
        self._safe_stop_enabled = False

    @property
    def safe_stop_enabled(self):
        return self._safe_stop_enabled

    def safe_stop(self):
        if self._safe_stop_enabled:
            self._managed_client.reset_to_home()
            self._safe_stop_enabled = False


class PiperLcmRobotClient:
    """Adapter from agx_control ArmLcmClient to the local RobotClient protocol.

    Both agx_control and ``scipy.Rotation.from_euler('xyz', ...)`` represent
    ``[roll, pitch, yaw]`` as fixed-axis XYZ rotations, with the resulting
    matrix ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. No Euler remapping is needed.
    """

    def __init__(self, client):
        self._client = client

    @classmethod
    def from_hardware(cls, hw):
        driver_root = getattr(hw, "robot_driver_root", None)
        if driver_root:
            root = str(Path(driver_root).expanduser().resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
        try:
            from communication.lcm.arm_lcm_client import ArmLcmClient
        except ImportError as exc:
            raise ImportError(
                "Cannot import Piper ArmLcmClient; install agx_control or set "
                "robot.driver_root in the hardware profile"
            ) from exc

        url = getattr(hw, "lcm_arm_url", None)
        if not url:
            raise ValueError("Piper profile requires lcm.arm.url")
        return cls(ArmLcmClient(url=url))

    def set_ee_pose(self, pose, gripper_pos, *args, **kwargs):
        """Send Piper pose and gripper only; Piper has no duration control."""
        self._client.set_cartesian_cmd(
            tcp_pose=list(pose), gripper=float(gripper_pos),
        )

    def get_state(self):
        state = self._client.get_state()
        if state is None:
            return None
        result = dict(state)
        result["ee_pose"] = list(state["tcp_pose"])
        return result

    def reset_to_home(self):
        self._client.set_to_home()

    def set_to_passive(self):
        self._client.set_to_passive()
