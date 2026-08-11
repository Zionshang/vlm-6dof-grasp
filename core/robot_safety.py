"""Piper command feedback and recovery checks."""
import time

import numpy as np
from scipy.spatial.transform import Rotation

GRIPPER_TOLERANCE_M = 0.003
POSITION_TOLERANCE_M = 0.01
ORIENTATION_TOLERANCE_RAD = 0.05
ARM_ERRORS = {
    1: "急停", 2: "无逆解", 3: "奇异点", 4: "目标位置超限",
    5: "关节通信异常", 6: "关节抱闸未释放", 7: "发生碰撞",
    8: "示教超速", 9: "关节状态异常", 10: "控制器异常",
    14: "主控过温", 15: "电阻过温",
}


def _values(state):
    if not state:
        return None
    values = state.get("ee_pose")
    return state.get("tcp_pose") if values is None else values


def _pose(state):
    values = _values(state)
    return "" if values is None else "[" + ", ".join(f"{x:.4f}" for x in values) + "]"


def _pose_error(state, target):
    actual = _values(state)
    if actual is None or target is None:
        return float("inf"), float("inf")
    position = np.linalg.norm(np.subtract(actual[:3], target[:3]))
    orientation = (Rotation.from_euler("xyz", actual[3:]).inv()
                   * Rotation.from_euler("xyz", target[3:])).magnitude()
    return float(position), float(orientation)


def _status_error(state, motion=False):
    status = None if not state else state.get("arm_status")
    if status not in (None, 0):
        return f"{ARM_ERRORS.get(status, '未知机械臂异常')} (arm_status={status})"
    status = None if not state else state.get("motion_status")
    return f"运动失败 (motion_status={status})" if motion and status not in (None, 0) else None


def wait_for_robot_state(robot, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = robot.get_state()
        if state:
            print("[就绪] ARM_STATE")
            return state
    raise RuntimeError(
        f"No ARM_STATE received within {timeout:.1f}s. Check arm_lcm_server, "
        "LCM multicast route, firewall, and lcm.arm.url."
    )


def require_not_emergency_stopped(state):
    if state.get("arm_status") == 1:
        raise RuntimeError("机械臂处于急停，需人工解除后重新运行")


def monitor_robot_state(robot, duration, label="Robot"):
    deadline, state = time.monotonic() + duration, None
    while time.monotonic() < deadline:
        state = robot.get_state()
        if not state:
            raise RuntimeError(f"{label} 期间 ARM_STATE 通信中断")
        error = _status_error(state, motion=True)
        if error:
            raise RuntimeError(f"{label} 失败: {error}, 当前pose={_pose(state)}")
    return state


def wait_for_target_reached(robot, previous_target_utime, timeout, label="Robot",
                            expected_gripper=None, expected_pose=None):
    deadline = time.monotonic() + timeout
    previous, accepted, state = int(previous_target_utime or 0), None, None
    pose_error = (float("inf"), float("inf"))

    while time.monotonic() < deadline:
        feedback = robot.get_state()
        if not feedback:
            continue
        state = feedback
        target = int(state.get("target_utime", 0))
        if not state.get("has_target") or target in (0, previous):
            continue
        accepted = target if accepted is None else accepted
        if target != accepted:
            raise RuntimeError(
                f"Robot target changed while waiting for {label}: "
                f"expected {accepted}, received {target}"
            )
        error = _status_error(state)
        if error:
            raise RuntimeError(f"{label} 失败: {error}, 当前pose={_pose(state)}")

        gripper = state.get("gripper_pos")
        grip_ok = (expected_gripper is None or gripper is not None and
                   abs(float(gripper) - float(expected_gripper))
                   <= GRIPPER_TOLERANCE_M)
        pose_error = _pose_error(state, expected_pose)
        pose_ok = (expected_pose is None or
                   pose_error[0] <= POSITION_TOLERANCE_M and
                   pose_error[1] <= ORIENTATION_TOLERANCE_RAD)
        if state.get("target_reached") and grip_ok and pose_ok:
            print(f"[到达] {label}: pose={_pose(state)}")
            return state

    error = _status_error(state, motion=True)
    if accepted is None:
        reason = "控制器未接收命令"
    elif error:
        reason = error
    elif (expected_pose is not None and
          (pose_error[0] > POSITION_TOLERANCE_M or
           pose_error[1] > ORIENTATION_TOLERANCE_RAD)):
        reason = (f"位姿未到（位置误差 {pose_error[0]:.3f} m，"
                  f"姿态误差 {pose_error[1]:.3f} rad）")
    elif expected_gripper is not None:
        reason = f"夹爪未到 {expected_gripper:.3f} m"
    else:
        reason = "控制器未返回 target_reached"
    actual = _pose(state)
    raise RuntimeError(
        f"{label} 未到达（{timeout:.0f}秒）: {reason}"
        + (f", 当前pose={actual}" if actual else "")
    )


def _command_and_wait(robot, command, timeout, label, gripper=None,
                      expected_pose=None, send_without_state=False):
    state = robot.get_state()
    if not state and not send_without_state:
        raise RuntimeError(f"ARM_STATE unavailable before {label} command")
    previous = state.get("target_utime", 0) if state else 0
    current_gripper = state.get("gripper_pos") if state else None
    expected_gripper = (
        gripper if gripper is not None and current_gripper is not None and
        abs(float(current_gripper) - gripper) > GRIPPER_TOLERANCE_M else None
    )
    command()
    if not state:
        raise RuntimeError(
            f"{label} command sent, but ARM_STATE was unavailable for verification"
        )
    return wait_for_target_reached(
        robot, previous, timeout, label, expected_gripper, expected_pose,
    )


def move_to_pose_and_wait(robot, hw, pose, gripper, timeout, label="Robot",
                          verify_gripper=True):
    if not hw.in_workspace(*pose[:3]):
        raise RuntimeError(f"{label} 目标超出工作空间: pose={_pose({'ee_pose': pose})}")
    print(f"[发送] {label}: pose={_pose({'ee_pose': pose})}, grip={gripper:.3f}")
    return _command_and_wait(
        robot, lambda: robot.set_ee_pose(pose, gripper), timeout, label,
        gripper if verify_gripper else None, pose,
    )


def reset_to_home_and_wait(robot, timeout, home_pose):
    print("[发送] HOME")
    return _command_and_wait(
        robot, robot.reset_to_home, timeout, "HOME", expected_pose=home_pose,
    )


def safe_stop_and_wait(robot, timeout, home_pose):
    state = robot.get_state()
    if state and state.get("arm_status") == 1:
        robot.disable_safe_stop()
        raise RuntimeError("机械臂处于急停，未再次发送 HOME")
    print("[发送] HOME（安全恢复）")
    return _command_and_wait(
        robot, robot.safe_stop, timeout, "HOME", expected_pose=home_pose,
        send_without_state=True,
    )
