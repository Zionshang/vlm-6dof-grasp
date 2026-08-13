"""Piper command feedback and recovery checks."""
import time

import numpy as np
from scipy.spatial.transform import Rotation

GRIPPER_TOLERANCE_M = 0.01
POSITION_TOLERANCE_M = 0.015
ORIENTATION_TOLERANCE_RAD = np.deg2rad(5.5)
STABLE_TIME_S = 0.3
HOLD_POSITION_M = 0.03
HOLD_ORIENTATION_RAD = np.deg2rad(10)
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


def _read(robot, label):
    state = robot.get_state()
    if not state:
        raise RuntimeError(f"{label} 期间 ARM_STATE 通信中断")
    status = state.get("arm_status")
    if status not in (None, 0):
        error = ARM_ERRORS.get(status, "未知机械臂异常")
        raise RuntimeError(f"{label} 失败: {error} (arm_status={status}), "
                           f"当前pose={_pose(state)}")
    return state


def wait_for_robot_state(robot, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = robot.get_state()
        if state:
            print("[就绪] ARM_STATE")
            return state
    raise RuntimeError(f"{timeout:.1f}秒内未收到ARM_STATE")


def require_not_emergency_stopped(state):
    if state.get("arm_status") == 1:
        raise RuntimeError("机械臂处于急停，需人工解除后重新运行")


def monitor_robot_state(robot, duration, expected_pose, label="Robot"):
    deadline, state = time.monotonic() + duration, None
    while time.monotonic() < deadline:
        state = _read(robot, label)
        position, orientation = _pose_error(state, expected_pose)
        if position > HOLD_POSITION_M or orientation > HOLD_ORIENTATION_RAD:
            raise RuntimeError(
                f"{label} 位姿漂移: {position:.3f} m, "
                f"{np.rad2deg(orientation):.1f}°, 当前pose={_pose(state)}"
            )
    return state


def _wait_for_arrival(robot, previous_target_utime, timeout, label,
                      expected_gripper, expected_pose):
    deadline = time.monotonic() + timeout
    previous, accepted, stable, state = int(previous_target_utime or 0), None, None, None
    pose_error, grip_ok = (float("inf"), float("inf")), False

    while time.monotonic() < deadline:
        state = _read(robot, label)
        target = int(state.get("target_utime", 0))
        if target in (0, previous):
            continue
        accepted = target if accepted is None else accepted
        if target != accepted:
            raise RuntimeError(f"等待{label}时目标被新命令覆盖")

        gripper = state.get("gripper_pos")
        grip_ok = (expected_gripper is None or gripper is not None and
                   abs(float(gripper) - float(expected_gripper))
                   < GRIPPER_TOLERANCE_M)
        pose_error = _pose_error(state, expected_pose)
        pose_ok = (expected_pose is None or
                   pose_error[0] <= POSITION_TOLERANCE_M and
                   pose_error[1] < ORIENTATION_TOLERANCE_RAD)
        stable = stable or time.monotonic() if pose_ok and grip_ok else None
        if stable and time.monotonic() - stable >= STABLE_TIME_S:
            print(f"[到达] {label}: pose={_pose(state)}")
            return state

    if accepted is None:
        reason = "控制器未接收命令"
    elif (expected_pose is not None and
          (pose_error[0] > POSITION_TOLERANCE_M or
           pose_error[1] >= ORIENTATION_TOLERANCE_RAD)):
        reason = (f"位姿未到（位置误差 {pose_error[0]:.3f} m，"
                  f"姿态误差 {np.rad2deg(pose_error[1]):.1f}°）")
    elif expected_gripper is not None and not grip_ok:
        gripper = None if not state else state.get("gripper_pos")
        actual = "无反馈" if gripper is None else f"{float(gripper):.3f} m"
        reason = f"夹爪未到 {expected_gripper:.3f} m（当前 {actual}）"
    else:
        reason = f"位姿未连续稳定 {STABLE_TIME_S:.1f} 秒"
    actual = _pose(state)
    raise RuntimeError(
        f"{label} 未到达（{timeout:.0f}秒）: {reason}"
        + (f", 当前pose={actual}" if actual else "")
    )


def _command_and_wait(robot, command, timeout, label, gripper=None,
                      expected_pose=None, send_without_state=False):
    state = robot.get_state()
    if not state and not send_without_state:
        raise RuntimeError(f"{label}发送前无ARM_STATE")
    previous = state.get("target_utime", 0) if state else 0
    current_gripper = state.get("gripper_pos") if state else None
    expected_gripper = (
        gripper if gripper is not None and current_gripper is not None and
        abs(float(current_gripper) - gripper) >= GRIPPER_TOLERANCE_M else None
    )
    command()
    return _wait_for_arrival(
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
