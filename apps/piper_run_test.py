"""Safe Piper perception, selection and feedback-verified grasp test."""
import argparse
import logging
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

from grasp_geometry import box_center_to_base, expand_boxes
from grasp_perception import GraspPerception
from hardware import HardwareConfig
from manager import GraspManager
from robot_safety import (
    monitor_robot_state, move_to_pose_and_wait, require_not_emergency_stopped,
    reset_to_home_and_wait, safe_stop_and_wait, wait_for_robot_state,
)
from saver import save_capture, save_seg_mask, save_vlm_boxes
from transform import convert_new
from vlm.src.utils.image_utils import make_bbox_mask


ROOT = paths.PROJECT_ROOT
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def capture(perception, flush_frames, label):
    color, depth = perception.capture(flush_frames)
    if color is None or depth is None:
        raise RuntimeError(f"D405/FFS frame unavailable at {label}")
    return color, depth


def capture_components(manager, flush_frames):
    ctx, camera = manager.ctx, manager.require("camera")
    ctx.color = ctx.depth = ctx.ir = None
    camera.capture(ctx, discard_frames=flush_frames)
    manager.require("depth").step(ctx)
    return ctx.color, ctx.depth


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


def execute_selected(manager, robot, hw, selected, timeout, steps, width):
    state = robot.get_state()
    if not state:
        raise RuntimeError("ARM_STATE unavailable before grasp")
    command = adjust_ry(convert_new(
        np.asarray(selected["translation"]), np.asarray(selected["rotation"]),
        state["ee_pose"], hw.hand_eye_r, hw.hand_eye_t,
        selected["depth"],
    ))
    success, reason = manager.require("executor").run_sequence(
        command, width, steps=steps, arrival_timeout=timeout)
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
    parser.add_argument("--mode", choices=("grasp", "reach", "sam"),
                        default="reach")
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
    with open(ROOT / args.app_config) as stream:
        manager = GraspManager(yaml.safe_load(stream), hw=hw, eager=False)
    dashboard = manager.require("dashboard")
    dashboard.set_output_dir(ROOT / args.output_dir)
    robot = None
    safe = False
    with dashboard.capture_output():
        try:
            robot = manager.require("robot")
            state = wait_for_robot_state(robot, args.state_timeout)
            require_not_emergency_stopped(state)
            robot.enable_safe_stop()
            safe = True
            reset_to_home_and_wait(robot, args.arrival_timeout, hw.home_pose)
            print("[流程] 加载感知组件")
            with dashboard.details():
                roles = (("detector", "camera", "depth", "segmenter")
                         if args.mode == "sam" else None)
                manager.initialize(roles)
                perception = (None if args.mode == "sam" else
                              GraspPerception(manager, ROOT / args.output_dir))
                if not manager.handshake():
                    detail = manager.handshake_error or "first frame timeout"
                    raise RuntimeError(f"D405 stereo unavailable: {detail}")
            print("[就绪] 感知组件")
            yield hw, manager, robot, perception, dashboard
        except BaseException as exc:
            dashboard.log(f"[失败] {exc}", "flow")
            raise
        finally:
            if safe and robot.safe_stop_enabled:
                try:
                    safe_stop_and_wait(
                        robot, args.arrival_timeout, hw.home_pose,
                    )
                except Exception as home_error:
                    print(f"[失败] HOME 恢复: {home_error}")
            with dashboard.details():
                manager.release_resources()


def prepare_grasp(args, hw, manager, robot, perception, dashboard):
    # ----- Far observation and target localization -----
    print("[流程] 远距离定位")
    far_state = move_to_pose_and_wait(
        robot, hw, hw.ready_views[0], hw.gripper_approach_width,
        args.arrival_timeout, "Far observation",
    )
    with dashboard.details():
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
        raise RuntimeError("Perception pipeline produced no grasp")
    dashboard.update_scene(color, depth, grasps,
                           perception.grasp_engine.intrinsic)
    print("[流程] O3D 可视化")
    with dashboard.details():
        visualize(manager, color, depth, grasps, args.visualize_seconds)
    print("[流程] first 筛选")
    selected = perception.select(color, grasps)
    if selected is None:
        raise RuntimeError("First selector produced no grasp")
    return selected


def run_test(args, hw, manager, robot, perception, dashboard):
    """Approach/reach accuracy test retained for later calibration."""
    selected = prepare_grasp(args, hw, manager, robot, perception, dashboard)
    print("[流程] Reach 精度测试")
    steps = manager.require("executor").steps[:2]
    execute_selected(manager, robot, hw, selected, args.arrival_timeout,
                     steps, hw.gripper_max_width)
    print("[测试] Reach 保持 30 秒")
    monitor_robot_state(robot, 30, "Reach 保持")
    reset_to_home_and_wait(robot, args.arrival_timeout, hw.home_pose)
    robot.disable_safe_stop()
    print("[成功] Reach 测试完成")


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


def sam_test(args, hw, manager, robot, perception, dashboard):
    """Retry the configured box-prompt segmenter for at most 30 seconds."""
    camera, detector = manager.require("camera"), manager.require("detector")
    segmenter = manager.require("segmenter")
    intrinsic = np.array([[camera.color_fx, 0, camera.color_cx],
                          [0, camera.color_fy, camera.color_cy], [0, 0, 1.]])

    far = move_to_pose_and_wait(robot, hw, hw.ready_views[0],
                                hw.gripper_approach_width,
                                args.arrival_timeout, "Far observation")
    color, depth = capture_components(manager, args.flush_frames)
    detection = detector.detect(color, args.prompt)
    if not detection or not detection.boxes:
        raise RuntimeError("SAM test target not detected")
    target = box_center_to_base(depth, detection.boxes[0], intrinsic,
                                far["ee_pose"], hw.hand_eye_r, hw.hand_eye_t)
    if target is None:
        raise RuntimeError("SAM test target depth unavailable")
    pose = np.r_[target + hw.target_approach_offset, hw.target_approach_rpy]
    move_to_pose_and_wait(robot, hw, pose, hw.gripper_approach_width,
                          args.arrival_timeout, "Close observation")

    deadline, attempt = time.monotonic() + 30, 0
    while time.monotonic() < deadline:
        attempt += 1
        color, _ = capture_components(manager, args.flush_frames)
        detection = detector.detect(color, args.prompt)
        if not detection or not detection.boxes:
            continue
        boxes = expand_boxes(detection.boxes, color.shape)
        mask = segmenter.segment(color, boxes)
        if mask is None:
            continue
        tag = time.strftime("%Y%m%d-%H%M%S") + f"_sam_test_{attempt}"
        save_vlm_boxes(ROOT / args.output_dir, color, boxes, tag)
        save_seg_mask(ROOT / args.output_dir, mask, tag)
        box_mask = make_bbox_mask(boxes, *mask.shape)
        inside = np.count_nonzero(mask & box_mask)
        containment = inside / max(1, np.count_nonzero(mask))
        coverage = inside / max(1, np.count_nonzero(box_mask))
        print(f"[SAM] attempt={attempt}, containment={containment:.2f}, "
              f"coverage={coverage:.2f}")
        if containment >= .7 and coverage >= .1:
            reset_to_home_and_wait(robot, args.arrival_timeout, hw.home_pose)
            robot.disable_safe_stop()
            print("[成功] SAM 分割通过")
            return
    raise RuntimeError("SAM test 30秒内未得到合格目标 mask")


def main():
    args = parse_args()
    try:
        with initialize_system(args) as system:
            {"grasp": run_grasp, "reach": run_test, "sam": sam_test}[
                args.mode](args, *system)
    except Exception as exc:
        print(f"[失败] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
