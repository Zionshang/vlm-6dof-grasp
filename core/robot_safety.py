"""Robot communication readiness helpers."""
import time

GRIPPER_TOLERANCE_M = 0.003
ARM_ERRORS = {
    1: "急停", 2: "无逆解", 3: "奇异点", 4: "目标位置超限",
    5: "关节通信异常", 6: "关节抱闸未释放", 7: "发生碰撞",
    8: "示教超速", 9: "关节状态异常", 10: "控制器异常",
    14: "主控过温", 15: "电阻过温",
}


def _pose(state):
    values = state.get("ee_pose")
    if values is None:
        values = state.get("tcp_pose")
    return "" if values is None else "[" + ", ".join(f"{x:.4f}" for x in values) + "]"


def wait_for_robot_state(robot, timeout):
    """Require feedback communication before enabling automatic safe-stop."""
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


def wait_for_target_reached(robot, previous_target_utime, timeout, label="Robot",
                            expected_gripper=None):
    """Wait until feedback confirms completion of a newly accepted command.

    ``previous_target_utime`` prevents a stale reached state from the preceding
    HOME or Cartesian command from satisfying this wait.
    """
    deadline = time.monotonic() + timeout
    accepted_target = None
    last_state = None
    while time.monotonic() < deadline:
        state = robot.get_state()
        if not state:
            continue
        last_state = state
        target_utime = int(state.get("target_utime", 0))
        is_new_target = (
            bool(state.get("has_target"))
            and target_utime != 0
            and target_utime != int(previous_target_utime or 0)
        )
        if not is_new_target:
            continue
        if accepted_target is None:
            accepted_target = target_utime
        elif target_utime != accepted_target:
            raise RuntimeError(
                f"Robot target changed while waiting for {label}: "
                f"expected {accepted_target}, received {target_utime}"
            )
        gripper = state.get("gripper_pos")
        gripper_reached = (
            expected_gripper is None
            or (gripper is not None and abs(
                float(gripper) - float(expected_gripper)
            ) <= GRIPPER_TOLERANCE_M)
        )
        if bool(state.get("target_reached")) and gripper_reached:
            print(f"[到达] {label}: pose={_pose(state)}")
            return state

    if accepted_target is None:
        reason = "控制器未接收命令"
    elif last_state.get("arm_status") not in (None, 0):
        status = last_state["arm_status"]
        reason = f"{ARM_ERRORS.get(status, '未知机械臂异常')} (arm_status={status})"
    elif last_state.get("motion_status") not in (None, 0):
        reason = f"运动失败 (motion_status={last_state['motion_status']})"
    elif expected_gripper is not None:
        reason = f"夹爪未到 {expected_gripper:.3f} m"
    else:
        reason = "控制器未返回 target_reached"
    actual = _pose(last_state)
    if actual:
        reason += f", 当前pose={actual}"
    raise RuntimeError(f"{label} 未到达（{timeout:.0f}秒）: {reason}")


def move_to_pose_and_wait(robot, hw, pose, gripper, timeout, label="Robot",
                          verify_gripper=True):
    """Send an exact pose/gripper command and wait for its correlated feedback."""
    if not hw.in_workspace(*pose[:3]):
        raise RuntimeError(f"{label} 目标超出工作空间: pose={_pose({'ee_pose': pose})}")
    print(f"[发送] {label}: pose={_pose({'ee_pose': pose})}, grip={gripper:.3f}")
    return _command_and_wait(
        robot, lambda: robot.set_ee_pose(pose, gripper), timeout, label,
        gripper if verify_gripper else None,
    )


def _command_and_wait(robot, command, timeout, label,
                      gripper=None, send_without_state=False):
    state = robot.get_state()
    if not state and not send_without_state:
        raise RuntimeError(f"ARM_STATE unavailable before {label} command")
    previous_target = state.get("target_utime", 0) if state else 0
    current_gripper = state.get("gripper_pos") if state else None
    expected_gripper = (
        gripper if gripper is not None and current_gripper is not None
        and abs(float(current_gripper) - gripper) > GRIPPER_TOLERANCE_M
        else None
    )
    command()
    if not state:
        raise RuntimeError(
            f"{label} command sent, but ARM_STATE was unavailable for verification"
        )
    return wait_for_target_reached(
        robot, previous_target, timeout, label, expected_gripper,
    )


def reset_to_home_and_wait(robot, timeout):
    """Send HOME and wait for the matching controller completion feedback."""
    print("[发送] HOME")
    return _command_and_wait(robot, robot.reset_to_home, timeout, "HOME")


def safe_stop_and_wait(robot, timeout):
    """Issue the one-shot safe HOME command and verify controller arrival."""
    print("[发送] HOME（安全恢复）")
    return _command_and_wait(
        robot, robot.safe_stop, timeout, "HOME", send_without_state=True,
    )
