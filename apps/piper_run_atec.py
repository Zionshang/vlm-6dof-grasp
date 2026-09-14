"""Safe Piper perception, selection and feedback-verified grasp test."""
import argparse
import logging
import sys
import time
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

from grasp_perception import GraspPerception, retry
from hardware import HardwareConfig
from manager import GraspManager
from robot_safety import (
    move_to_pose_and_wait, require_not_emergency_stopped,
    reset_to_home_and_wait, safe_stop_and_wait, wait_for_robot_state,
)
from saver import save_capture, save_vlm_boxes, try_save
from transform import box_center_to_base, convert_new


ROOT = paths.PROJECT_ROOT
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TASK_NAME = "全流程抓取"


def _brief(exc):
    lines = str(exc).splitlines()
    return lines[0] if lines else type(exc).__name__


def _step(name, call):
    try:
        return call()
    except Exception as exc:
        raise RuntimeError(f"{name}: {_brief(exc)}") from exc


def _web(dashboard, action, call):
    if dashboard:
        try:
            return call()
        except Exception as exc:
            print(f"[网页] {action}不可用: {_brief(exc)}")


def _web_context(dashboard, detail=False):
    if not dashboard:
        return nullcontext()
    try:
        return dashboard.details() if detail else dashboard.capture_output()
    except Exception as exc:
        print(f"[网页] 日志不可用: {_brief(exc)}")
        return nullcontext()


def locate_target(perception, hw, target, ee_pose, flush_frames):
    attempt = 0

    def locate_once():
        nonlocal attempt
        attempt += 1
        color, depth = perception.capture(flush_frames)
        detection = perception.detector.detect(color, target)
        if not detection or not detection.boxes:
            raise RuntimeError("未检测到目标")
        try_save("远距离检测图", save_vlm_boxes, perception.output_dir,
                 color, detection.boxes,
                 time.strftime("%Y%m%d-%H%M%S") + f"_far{attempt}")
        return box_center_to_base(
            depth, detection.boxes[0], perception.grasp_engine.intrinsic,
            ee_pose, hw.hand_eye_r, hw.hand_eye_t,
        )
    return retry("目标定位", locate_once, empty="目标框中心无有效深度")


def visualize(manager, color, depth, grasps, seconds):
    try:
        visualizer = manager.require("visualizer")
        visualizer.update_cloud(color, depth)
        visualizer.update_grasps(grasps)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and visualizer.poll():
            visualizer.render()
            time.sleep(0.02)
    except Exception as exc:
        raise RuntimeError(f"O3D可视化失败: {_brief(exc)}") from exc


def adjust_ry(command):
    if 0.0 <= command[4] <= 0.9:
        command[4] = 0.9 + command[4] / 18.0
    return command


def execute_selected(manager, robot, hw, selected, timeout, steps, width):
    state = robot.get_state()
    if not state:
        raise RuntimeError("执行前无ARM_STATE")
    command = _step("抓取位姿转换失败", lambda: adjust_ry(convert_new(
        np.asarray(selected["translation"]), np.asarray(selected["rotation"]),
        state["ee_pose"], hw.hand_eye_r, hw.hand_eye_t, selected["depth"],
    )))
    command[3:] = selected.get("fallback_rpy", command[3:])
    executor = _step("执行器初始化失败", lambda: manager.require("executor"))
    success, reason = executor.run_sequence(
        command, width, steps=steps, arrival_timeout=timeout)
    if not success:
        raise RuntimeError(reason)
    return executor.last_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", default="all",
        help="YOLO类别: watermelon/can/lunch_box/red_bag，逗号分隔，all为全部",
    )
    parser.add_argument("--hardware-profile", default="config/hardware/piper_d405.yaml")
    parser.add_argument("--app-config", default="config/apps/piper_run_test.yaml")
    parser.add_argument("--output-dir", default="output/piper_run_test")
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument(
        "--arrival-timeout", type=float, default=10.0,
        help="maximum wait for each Piper command (default: 10 seconds)",
    )
    parser.add_argument(
        "--flush-frames", type=int, default=None,
        help="override camera component discard_frames (default: component value)",
    )
    parser.add_argument("--visualize-seconds", type=float, default=30.0)
    return parser.parse_args()


def validate_hardware(hw):
    if (hw.home_pose is None or not hw.ready_views
            or hw.target_approach_offset is None
            or hw.target_approach_rpy is None):
        raise RuntimeError("Piper配置缺少HOME、远观测位姿或接近参数")


@contextmanager
def initialize_system(args):
    """Build configured components and establish a safe robot baseline."""
    hw = _step("硬件配置加载失败", lambda: HardwareConfig(args.hardware_profile))
    validate_hardware(hw)
    manager = _step("应用配置加载失败", lambda: GraspManager.from_yaml(
        ROOT / args.app_config, hw=hw, eager=False,
    ))
    try:
        dashboard = manager.require("dashboard")
        dashboard.set_output_dir(ROOT / args.output_dir)
        dashboard.set_task(TASK_NAME, args.target)
    except Exception as exc:
        dashboard = None
        print(f"[网页] 初始化不可用: {_brief(exc)}")
    robot = None
    safe = False
    with _web_context(dashboard):
        print(f"[任务] {TASK_NAME} | 目标: {args.target}")
        try:
            robot = _step("机械臂驱动初始化失败", lambda: manager.require("robot"))
            state = wait_for_robot_state(robot, args.state_timeout)
            require_not_emergency_stopped(state)
            robot.enable_safe_stop()
            safe = True
            reset_to_home_and_wait(robot, args.arrival_timeout, hw.home_pose)
            print("[流程] 加载感知组件")
            with _web_context(dashboard, True):
                roles = ["detector", "camera", "depth", "segmenter",
                         "grasp_engine", "selector", "executor"]
                try:
                    manager.initialize(roles)
                    perception = GraspPerception.from_manager(
                        manager, ROOT / args.output_dir,
                    )
                except Exception as exc:
                    raise RuntimeError(f"感知初始化失败: {_brief(exc)}") from exc
                if not manager.handshake():
                    detail = manager.handshake_error or "首帧超时"
                    raise RuntimeError(f"相机握手失败: {_brief(detail)}")
                validate_target = getattr(
                    manager.get("detector"), "validate_target", None,
                )
                if callable(validate_target):
                    validate_target(args.target)
            print("[就绪] 感知组件")
            yield hw, manager, robot, perception, dashboard
        except BaseException as exc:
            print(f"[失败] {_brief(exc)}")
            setattr(exc, "_reported", True)
            raise
        finally:
            if safe and robot.safe_stop_enabled:
                try:
                    safe_stop_and_wait(
                        robot, args.arrival_timeout, hw.home_pose,
                    )
                except Exception as home_error:
                    print(f"[失败] HOME 恢复: {home_error}")
            with _web_context(dashboard, True):
                manager.release_resources()


def prepare_grasp(args, hw, manager, robot, perception, dashboard):
    # ----- Far observation and target localization -----
    print("[流程] 远距离定位")
    far_state = move_to_pose_and_wait(
        robot, hw, hw.ready_views[0], hw.gripper_approach_width,
        args.arrival_timeout, "Far observation",
    )
    with _web_context(dashboard, True):
        target = locate_target(
            perception, hw, args.target, far_state["ee_pose"], args.flush_frames,
        )

    # ----- Close observation -----
    print("[流程] 近距离观测")
    offset, rpy = hw.approach_for(args.target)
    approach_pose = np.r_[target + offset, rpy]
    move_to_pose_and_wait(
        robot, hw, approach_pose, hw.gripper_approach_width,
        args.arrival_timeout, "Close observation",
    )

    # ----- Perception, visualization, selection and execution -----
    print("[流程] 检测 → 分割 → 抓取生成")
    color, depth = perception.capture(args.flush_frames)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    try_save("RGB-D", save_capture, perception.output_dir, color, depth, run_id)
    grasps = perception.generate(color, depth, args.target, run_id)
    if grasps is None or not len(grasps):
        raise RuntimeError("抓取生成无结果")
    _web(dashboard, "3D更新", lambda: dashboard.update_scene(
        color, depth, grasps, perception.grasp_engine.intrinsic))
    if manager.specs.get("visualizer", {}).get("enabled", True):
        print("[流程] O3D 可视化")
        with _web_context(dashboard, True):
            visualize(manager, color, depth, grasps, args.visualize_seconds)
    print("[流程] first 筛选")
    selected = perception.select(color, grasps)
    if selected is None:
        raise RuntimeError("二维筛选无候选")
    return selected


def run_grasp(args, hw, manager, robot, perception, dashboard):
    """Execute the configured approach → reach → grasp → lift → home sequence."""
    selected = prepare_grasp(args, hw, manager, robot, perception, dashboard)
    executor = manager.require("executor")
    width = np.clip(selected["width"] - 0.04, 0.0, hw.gripper_max_width)
    print("[流程] 全流程抓取")
    execute_selected(manager, robot, hw, selected, args.arrival_timeout,
                     executor.steps, float(width))
    robot.disable_safe_stop()
    print("[成功] 全流程抓取完成")


def main():
    args = parse_args()
    try:
        with initialize_system(args) as system:
            run_grasp(args, *system)
    except Exception as exc:
        if not getattr(exc, "_reported", False):
            print(f"[失败] {_brief(exc)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
