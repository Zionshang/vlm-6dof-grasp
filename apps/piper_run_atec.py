"""Three-view perception, reachability selection, and Piper grasp execution."""
import argparse
import io
import logging
import sys
import time
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

from grasp_perception import GraspPerception
from hardware import HardwareConfig
from manager import GraspManager
from perception_errors import PerceptionEmptyError
from robot_safety import (
    RobotStatusError, move_to_pose_and_wait, require_not_emergency_stopped,
    reset_to_home_and_wait, safe_stop_and_wait, wait_for_robot_state,
)
from saver import (
    save_capture, save_multiview_projections, save_multiview_record,
    try_save,
)
from transform import (
    camera_point_to_base, graspgenx_grasp_to_base, target_orbit_poses,
)
from visualization_data import graspgenx_projection_images


ROOT = paths.PROJECT_ROOT
TASK_NAME = "多视角全流程抓取"
SIDE_ANGLES_DEG = (35.0, 27.0, 17.0)
VIEW_TOPK = 10
TARGET_ATTEMPTS = 3
MAX_SCANS = 3


@dataclass
class System:
    hw: object
    robot: object
    perception: GraspPerception
    reachability: object
    executor: object
    dashboard: object = None


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


@contextmanager
def _quiet_component_initialization(dashboard=None):
    """Send verbose model banners to web details, not the terminal."""
    if dashboard:
        try:
            capture = dashboard.silent_details()
        except Exception as exc:
            print(f"[网页] 详细日志不可用: {_brief(exc)}")
        else:
            with capture:
                yield
            return

    previous_logging_disable = logging.root.manager.disable
    sink = io.StringIO()
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(sink), redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous_logging_disable)


def locate_targets(perception, hw, target, ee_pose, flush_frames):
    tag = time.strftime("%Y%m%d-%H%M%S_far")
    try:
        observations = perception.observe_all(
            target, flush_frames, tag, "far",
        )
    except PerceptionEmptyError:
        _save_far_capture(perception, tag)
        raise
    _save_far_capture(perception, tag, observations[0])
    targets = []
    for observation in observations:
        point = perception.target_point(observation)
        if point is None:
            print(f"[目标] 跳过{observation.label}: mask内有效深度点不足")
            continue
        center = camera_point_to_base(
            point, ee_pose, hw.hand_eye_r, hw.hand_eye_t,
        )
        targets.append({
            "label": observation.label, "score": observation.score,
            "center": center,
            "distance": float(np.linalg.norm(center[:3] - ee_pose[:3])),
        })
    if not targets:
        raise PerceptionEmptyError("全部目标mask内均无有效深度点")
    targets.sort(key=lambda item: item["distance"])
    for index, item in enumerate(targets, 1):
        item["id"] = f"{item['label']}_{index:02d}"
        xyz = ", ".join(f"{value:.3f}" for value in item["center"][:3])
        print(f"[目标] {index}/{len(targets)} {item['label']}: "
              f"score={item['score']:.3f}, 距离={item['distance']:.3f}m, "
              f"base=[{xyz}]")
    return targets


def _save_far_capture(perception, tag, observation=None):
    """Save one far-view RGB-D pair, computing depth once only if needed."""
    color = observation.color if observation else perception.ctx.color
    depth = observation.depth if observation else perception.ctx.depth
    if depth is None and color is not None:
        try:
            perception.depth.step(perception.ctx)
            depth = perception.ctx.depth
        except Exception as exc:
            print(f"[输出] 远观测深度生成失败: {_brief(exc)}")
    if color is None or depth is None:
        missing = "RGB" if color is None else "深度"
        print(f"[输出] 远观测RGB-D未保存: 缺少{missing}")
        return
    try_save("远观测RGB-D", save_capture,
             perception.output_dir, color, depth, tag)


def observe_target(perception, hw, target, ee_pose, flush_frames, tag, label):
    """Select the same physical target by nearest base-frame mask depth."""
    observations = perception.observe_all(
        target["label"], flush_frames, tag, label,
    )
    matches = []
    for observation in observations:
        point = perception.target_point(observation)
        if point is None:
            continue
        center = camera_point_to_base(
            point, ee_pose, hw.hand_eye_r, hw.hand_eye_t,
        )
        matches.append((
            float(np.linalg.norm(center[:3] - target["center"][:3])),
            observation,
        ))
    if not matches:
        raise PerceptionEmptyError(f"{label}目标mask内有效深度点不足")
    distance, observation = min(matches, key=lambda item: item[0])
    print(f"[匹配] {label}: {target['id']}, 三维偏差={distance:.3f}m")
    return observation


def project_grasps(color, grasps, intrinsic, gripper):
    """Best-effort visualization; never gate reachability or execution."""
    try:
        return graspgenx_projection_images(
            color, grasps, intrinsic, gripper, len(grasps),
        )
    except Exception as exc:
        print(f"[可视化] 2D投影跳过: {_brief(exc)}")
        return []


def finalize_run(system, run_id, target_name, target_center, views,
                 projections, reference, decision):
    """Persist optional outputs, publish the dashboard, and return the winner."""
    selected, evaluations, risk_min, risk_limit = decision
    for view, images in projections.items():
        if not images:
            continue
        reachable = [item for item in evaluations
                     if item["view"] == view and item["reachable"]]
        rejected = [item for item in evaluations
                    if item["view"] == view and not item["reachable"]]
        try_save(
            f"{view} IK可达抓取图", save_multiview_projections,
            system.perception.output_dir, run_id, view,
            [images[item["rank"]] for item in reachable],
            [item["rank"] for item in reachable], "selected_all",
        )
        try_save(
            f"{view} IK淘汰抓取图", save_multiview_projections,
            system.perception.output_dir, run_id, view,
            [images[item["rank"]] for item in rejected],
            [item["rank"] for item in rejected], "rejected",
        )
    if selected and projections[selected["view"]]:
        images = projections[selected["view"]]
        try_save(
            "多视角最终抓取图", save_multiview_projections,
            system.perception.output_dir, run_id, "final",
            [images[selected["rank"]]], [selected["rank"]], "selected_best",
        )

    record = {
        "run_id": run_id, "target": target_name,
        "target_center": np.asarray(target_center).tolist(), "views": views,
        "reference_pose": np.asarray(reference["ee_pose"]).tolist(),
        "reference_joints": np.asarray(reference["joint_pos"]).tolist(),
        "candidates": evaluations,
        "decision": {
            "risk_min": risk_min, "risk_limit": risk_limit,
            "selected_id": selected["id"] if selected else None,
        },
    }
    path = try_save(
        "多视角记录", save_multiview_record,
        system.perception.output_dir, run_id, record,
    )
    _web(system.dashboard, "多视角结果更新",
         lambda: system.dashboard.update_multiview(record))
    if path:
        print(f"[输出] 多视角记录: {path}")
    if not selected:
        reason = ("三视角候选均未通过approach/reach可达性筛选"
                  if risk_min is None else
                  f"最低关节风险{risk_min:.3f}超过上限{risk_limit:.3f}")
        print(f"[决策] 无可执行候选: {reason}")
        return None
    print(f"[决策] R_min={risk_min:.3f}, 风险池上限={risk_limit:.3f}")
    print(
        f"[选择] {selected['id']}: score={selected['score']:.4f}, "
        f"R={selected['risk']:.3f}, worst=J{selected['worst_joint']}"
    )
    return selected["tcp_pose"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", default="all",
        help="YOLO类别: watermelon/can/lunch_box/red_bag，逗号分隔，all为全部",
    )
    parser.add_argument("--hardware-profile", default="config/hardware/piper_d405.yaml")
    parser.add_argument("--app-config", default="config/apps/piper_run_atec.yaml")
    parser.add_argument("--output-dir", default="output/piper_run_atec")
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument(
        "--arrival-timeout", type=float, default=10.0,
        help="maximum wait for each Piper command (default: 10 seconds)",
    )
    parser.add_argument(
        "--flush-frames", type=int, default=None,
        help="override camera component discard_frames (default: component value)",
    )
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
            print("[流程] 加载任务组件")
            with _web_context(dashboard, True):
                roles = ["camera", "depth", "instance_model", "grasp_engine",
                         "reachability", "executor"]
                try:
                    with _quiet_component_initialization(dashboard):
                        manager.initialize(roles)
                    perception = GraspPerception.from_manager(
                        manager, ROOT / args.output_dir,
                    )
                    reachability = manager.require("reachability")
                    executor = manager.require("executor")
                except Exception as exc:
                    raise RuntimeError(f"任务组件初始化失败: {_brief(exc)}") from exc
                if not manager.handshake():
                    detail = manager.handshake_error or "首帧超时"
                    raise RuntimeError(f"相机握手失败: {_brief(detail)}")
                validate_target = getattr(
                    manager.get("instance_model"), "validate_target", None,
                )
                if callable(validate_target):
                    validate_target(args.target)
            print("[就绪] 任务组件")
            yield System(
                hw, robot, perception, reachability, executor, dashboard,
            )
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


def prepare_target_grasp(args, system, target, attempt):
    """Collect and select one three-view grasp set for one physical target."""
    hw, robot = system.hw, system.robot
    perception, dashboard = system.perception, system.dashboard
    offset, rpy = hw.approach_for(target["label"])
    front_pose = target.get("front_pose")
    if front_pose is None:
        front_pose = target_orbit_poses(
            target["center"], offset, rpy, (0.0,),
        )[0]
    run_id = (time.strftime("%Y%m%d-%H%M%S")
              + f"_{target['id']}_try{attempt}")
    views, candidates, projections = [], [], {}

    def collect(name, command_pose, state, actual_view, angle=0.0,
                precheck=None, fallback_reason=None):
        tag = f"{run_id}_{name}"
        label = name if actual_view == name else f"{name}(front补偿)"
        observation = observe_target(
            perception, hw, target, state["ee_pose"], args.flush_frames,
            tag, label,
        )
        try_save("RGB-D", save_capture, perception.output_dir,
                 observation.color, observation.depth, tag)
        result = perception.generate(observation, label).grasps[:VIEW_TOPK]
        projections[name] = project_grasps(
            observation.color, result, perception.grasp_engine.intrinsic,
            perception.grasp_engine.gripper_info,
        )

        view = {
            "name": name, "command_pose": np.asarray(command_pose).tolist(),
            "actual_pose": np.asarray(state["ee_pose"]).tolist(),
            "joint_pos": np.asarray(state["joint_pos"]).tolist(),
            "candidate_count": len(result), "actual_view": actual_view,
            "angle_deg": float(angle), "precheck": precheck or [],
            "fallback": actual_view == "front" and name != "front",
            "fallback_reason": fallback_reason,
        }
        views.append(view)
        for rank, grasp in enumerate(result):
            tcp_pose = graspgenx_grasp_to_base(
                grasp.translation, grasp.rotation_matrix, state["ee_pose"],
                hw.hand_eye_r, hw.hand_eye_t,
            )
            approach_pose = system.executor.pose_for_step(tcp_pose, "approach")
            candidates.append({
                "id": f"{name}_{rank:02d}", "view": name, "rank": rank,
                "source_view": actual_view, "view_angle_deg": float(angle),
                "score": float(grasp.score), "tcp_pose": tcp_pose,
                "approach_pose": approach_pose,
                "camera_grasp_pose": np.asarray(grasp.pose).tolist(),
            })
    print("[流程] 三视角检测 → 分割 → 抓取生成")
    front_state = move_to_pose_and_wait(
        robot, hw, front_pose, hw.gripper_approach_width,
        args.arrival_timeout, "front observation",
    )
    collect("front", front_pose, front_state, "front")

    for name, sign in (("left", 1.0), ("right", -1.0)):
        options = [
            {"angle_deg": sign * angle,
             "pose": target_orbit_poses(
                 target["center"], offset, rpy, (sign * angle,),
             )[0]}
            for angle in SIDE_ANGLES_DEG
        ]
        chosen, checks = system.reachability.select_observation_pose(
            options, front_state["joint_pos"],
        )
        for check in checks:
            state = "通过" if check["reachable"] else check["reason"]
            seed = check.get("ik_seed")
            seed_text = (", 备用初值" if seed == "front_nominal" else "")
            margin = check.get("min_margin_deg")
            margin_text = ("" if margin is None else
                           f", 最小关节限位余量={margin:.1f}°")
            print(f"[视角预检] {name} {check['angle_deg']:+.0f}°: "
                  f"{state}{seed_text}{margin_text}")

        side_state, fallback_reason = None, None
        if chosen:
            try:
                side_state = move_to_pose_and_wait(
                    robot, hw, chosen["pose"], hw.gripper_approach_width,
                    args.arrival_timeout,
                    f"{name} observation ({chosen['angle_deg']:+.0f}deg)",
                )
            except RobotStatusError as exc:
                detail = _brief(exc)
                if exc.status not in (2, 3, 4):
                    raise
                fallback_reason = f"controller_rejected: {detail}"
                print(f"[补偿] {name}控制器拒绝，改为前视补采: {detail}")
        else:
            fallback_reason = "all_prechecks_failed"
            print(f"[补偿] {name}三个角度均未通过预检，改为前视补采")

        if side_state is not None:
            try:
                collect(name, chosen["pose"], side_state, name,
                        chosen["angle_deg"], checks)
            except PerceptionEmptyError as exc:
                fallback_reason = f"perception_failed: {exc}"
                print(f"[补偿] {name}感知失败，改为前视补采: {exc}")
            else:
                front_state = move_to_pose_and_wait(
                    robot, hw, front_pose, hw.gripper_approach_width,
                    args.arrival_timeout, "front transit",
                )
                continue

        front_state = move_to_pose_and_wait(
            robot, hw, front_pose, hw.gripper_approach_width,
            args.arrival_timeout, f"{name} fallback front",
        )
        collect(name, front_pose, front_state, "front", 0.0,
                checks, fallback_reason)

    reference = front_state
    decision = system.reachability.select_multiview(
        candidates, reference["joint_pos"],
    )
    return finalize_run(
        system, run_id, target["label"], target["center"], views,
        projections, reference, decision,
    )


def locate_scene_targets(args, system):
    """Locate and order every requested instance from the far view."""
    hw, robot = system.hw, system.robot
    print("[流程] 远距离多目标定位")
    far_state = move_to_pose_and_wait(
        robot, hw, hw.ready_views[0], hw.gripper_approach_width,
        args.arrival_timeout, "Far observation",
    )
    with _web_context(system.dashboard, True):
        targets = locate_targets(
            system.perception, hw, args.target, far_state["ee_pose"],
            args.flush_frames,
        )
    return targets, far_state


def iter_observable_targets(system, targets, reference_joints):
    """Lazily yield distance-ordered targets with reachable front views."""
    for target in targets:
        offset, rpy = system.hw.approach_for(target["label"])
        pose = target_orbit_poses(
            target["center"], offset, rpy, (0.0,),
        )[0]
        chosen, checks = system.reachability.select_observation_pose(
            [{"angle_deg": 0.0, "pose": pose}], reference_joints,
        )
        check = checks[0]
        if chosen is None:
            print(f"[跳过] {target['id']} 正面观察位姿不可达: "
                  f"{check['reason']}")
            continue
        if check.get("ik_seed") == "front_nominal":
            print(f"[视角预检] {target['id']} 正面备用初值通过")
        target["front_pose"] = pose
        yield target


def prepare_multiview_grasp(args, system, target):
    """Give one physical target three complete multiview attempts."""
    print(f"[目标] 开始评估 {target['id']}")
    for attempt in range(1, TARGET_ATTEMPTS + 1):
        print(f"[尝试] {target['id']} 三视角 {attempt}/{TARGET_ATTEMPTS}")
        try:
            command = prepare_target_grasp(args, system, target, attempt)
        except PerceptionEmptyError as exc:
            print(f"[重试] {target['id']} 感知无结果: {_brief(exc)}")
            continue
        if command is not None:
            print(f"[目标] 最终选择 {target['id']}")
            return command
    print(f"[跳过] {target['id']} 三次均无可执行抓取")
    return None


def run_multiview_grasp(args, system):
    completed, scan = [], 0
    while True:
        try:
            targets, far_state = locate_scene_targets(args, system)
        except PerceptionEmptyError as exc:
            if completed:
                print(f"[结束] 远观测无有效目标: {_brief(exc)}")
                break
            raise
        scan += 1
        for target in targets:
            target["id"] = f"s{scan:02d}_{target['id']}"
        executed = selected = False
        for target in iter_observable_targets(
                system, targets, far_state["joint_pos"]):
            command = prepare_multiview_grasp(args, system, target)
            if command is None:
                continue
            selected = True
            if input(
                f"[确认] 是否执行 {target['id']} 抓取？输入 yes: "
            ).strip().lower() != "yes":
                print(f"[跳过] 用户取消 {target['id']} 抓取")
                continue
            success, reason = system.executor.run_sequence(
                command, 0.0, arrival_timeout=args.arrival_timeout,
            )
            if not success:
                raise RuntimeError(reason)
            completed.append(target["id"])
            executed = True
            print(f"[完成] {target['id']} 已抓取、回HOME并松爪")
            break
        if executed:
            if scan >= MAX_SCANS:
                print(f"[结束] 已达到最大扫描次数 {MAX_SCANS}")
                break
            continue
        if not completed and not selected:
            raise RuntimeError("当前全部目标均无可执行抓取")
        break

    reset_to_home_and_wait(
        system.robot, args.arrival_timeout, system.hw.home_pose,
    )
    system.robot.disable_safe_stop()
    result = ", ".join(completed) if completed else "无（均由用户取消）"
    print(f"[成功] 多目标流程完成: {result}")


def main():
    args = parse_args()
    try:
        with initialize_system(args) as system:
            run_multiview_grasp(args, system)
    except Exception as exc:
        if not getattr(exc, "_reported", False):
            print(f"[失败] {_brief(exc)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
