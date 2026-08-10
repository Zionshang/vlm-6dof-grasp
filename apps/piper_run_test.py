"""Safe Piper perception, selection and feedback-verified grasp test."""
import argparse
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

from grasp_geometry import box_center_to_base
from grasp_perception import GraspPerception
from hardware import HardwareConfig
from manager import GraspManager
from robot_safety import (
    move_to_pose_and_wait, reset_to_home_and_wait, safe_stop_and_wait,
    wait_for_robot_state,
)
from saver import save_capture
from transform import convert_new


ROOT = paths.PROJECT_ROOT
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@contextmanager
def quiet_components():
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        yield


def capture(perception, flush_frames, label):
    color, depth = perception.capture(flush_frames)
    if color is None or depth is None:
        raise RuntimeError(f"D405/FFS frame unavailable at {label}")
    return color, depth


def locate_target(perception, hw, prompt, ee_pose, flush_frames):
    color, depth = capture(perception, flush_frames, "far observation pose")
    detection = perception.detect(color, prompt)
    if not detection or not detection.boxes:
        raise RuntimeError("Target not detected from far observation pose")
    target = box_center_to_base(
        depth, detection.boxes[0], perception.grasp_engine.intrinsic, ee_pose,
        hw.hand_eye_r, hw.hand_eye_t,
    )
    if target is None:
        raise RuntimeError("No valid target depth at detected box center")
    return target


def visualize(manager, color, depth, grasps, seconds):
    visualizer = manager.require("visualizer")
    visualizer.update_cloud(color, depth)
    visualizer.update_grasps(grasps)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and visualizer.poll():
        visualizer.render()
        time.sleep(0.02)


def adjust_ry(command):
    if 0.0 <= command[4] <= 0.8:
        command[4] = 0.8 + command[4] / 16.0
    return command


def execute_grasp(manager, robot, hw, selected, timeout):
    state = robot.get_state()
    if not state:
        raise RuntimeError("ARM_STATE unavailable before grasp execution")
    command = adjust_ry(convert_new(
        np.asarray(selected["translation"]), np.asarray(selected["rotation"]),
        state["ee_pose"], hw.hand_eye_r, hw.hand_eye_t,
    ))
    width = float(np.clip(selected["width"] - 0.03, 0.0, hw.gripper_max_width))
    success, reason = manager.require("executor").run_sequence(
        command, width, arrival_timeout=timeout,
    )
    if not success:
        raise RuntimeError(reason)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="orange")
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
        raise SystemExit(
            "[Config] Piper home, far observation and target approach are required"
        )


@contextmanager
def initialize_system(args):
    """Build configured components and establish a safe robot baseline."""
    hw = HardwareConfig(args.hardware_profile)
    validate_hardware(hw)
    manager = robot = None
    safe = False
    try:
        with open(ROOT / args.app_config) as stream:
            manager = GraspManager(yaml.safe_load(stream), hw=hw, eager=False)
        robot = manager.require("robot")
        wait_for_robot_state(robot, args.state_timeout)
        robot.enable_safe_stop()
        safe = True
        reset_to_home_and_wait(robot, args.arrival_timeout)
        print("[流程] 加载感知组件")
        with quiet_components():
            manager.initialize()
            perception = GraspPerception(manager, ROOT / args.output_dir)
            if not manager.handshake():
                raise RuntimeError("Camera handshake failed")
        print("[就绪] 感知组件")
        yield hw, manager, robot, perception
    finally:
        if safe and robot.safe_stop_enabled:
            try:
                safe_stop_and_wait(robot, args.arrival_timeout)
            except Exception as home_error:
                print(f"[失败] HOME 恢复: {home_error}")
        if manager is not None:
            with quiet_components():
                manager.release_resources()


def run_test(args, hw, manager, robot, perception):
    # ----- Far observation and target localization -----
    print("[流程] 远距离定位")
    far_state = move_to_pose_and_wait(
        robot, hw, hw.ready_views[0], hw.gripper_approach_width,
        args.arrival_timeout, "Far observation",
    )
    with quiet_components():
        target = locate_target(
            perception, hw, args.prompt, far_state["ee_pose"], args.flush_frames,
        )

    # ----- Close observation -----
    print("[流程] 近距离观测")
    approach_pose = np.r_[target + hw.target_approach_offset,
                          hw.target_approach_rpy]
    move_to_pose_and_wait(
        robot, hw, approach_pose, hw.gripper_approach_width,
        args.arrival_timeout, "Close observation",
    )

    # ----- Perception, visualization, selection and execution -----
    print("[流程] 检测 → 分割 → 抓取生成")
    color, depth = capture(perception, args.flush_frames, "close observation pose")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    save_capture(perception.output_dir, color, depth, run_id)
    grasps = perception.generate(color, depth, args.prompt, run_id)
    if grasps is None or not len(grasps):
        raise RuntimeError("VLM/FastSAM/EconomicGrasp produced no grasp")
    print("[流程] O3D 可视化")
    with quiet_components():
        visualize(manager, color, depth, grasps, args.visualize_seconds)
    print("[流程] first 筛选")
    selected = perception.select(color, grasps)
    if selected is None:
        raise RuntimeError("First selector produced no grasp")
    print("[流程] 执行抓取")
    execute_grasp(manager, robot, hw, selected, args.arrival_timeout)
    robot.disable_safe_stop()  # The configured final HOME was already verified.
    print("[成功] 抓取流程完成")


def main():
    args = parse_args()
    try:
        with initialize_system(args) as system:
            run_test(args, *system)
    except Exception as exc:
        print(f"[失败] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
