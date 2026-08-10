"""Config-driven LCM grasp service.

Components come from ``config/apps/grasp_lcm.yaml``; this file contains only
the application-specific multi-view workflow and task-LCM protocol.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

from grasp_executor import GraspStep
from grasp_geometry import box_center_to_base
from grasp_perception import GraspPerception
from hardware import HardwareConfig
from manager import GraspManager
from robot_safety import wait_for_robot_state
from saver import save_capture
from transform import convert_new


ROOT = paths.PROJECT_ROOT


class GraspWorkflow:
    """Multi-view locate → grasp inference → robot execution."""

    def __init__(self, manager, output_dir):
        self.manager, self.hw = manager, manager.hw
        self.output_dir = output_dir
        self.perception = GraspPerception(manager, output_dir)
        self.engine = self.perception.grasp_engine
        self.robot = manager.require("robot")
        self.executor = manager.require("executor")
        self.policy = self.hw.grasp_policy
        self.steps = ([GraspStep(**step) for step in self.policy["steps"]]
                      if self.policy else [])

    def grasp(self, prompt):
        status = self._approach_target(prompt)
        if status:
            return False, status

        color, depth = self.capture(self.policy["grasp_flush_frames"])
        if color is None or depth is None:
            return False, "camera_error"
        run_id = time.strftime("%Y%m%d-%H%M%S")
        save_capture(self.output_dir, color, depth, run_id)

        selected = self._select_candidate(color, depth, prompt, run_id)
        if selected is None:
            return False, "detection/grasp_generation_failed"

        state = self.robot.get_state()
        if not state:
            return False, "robot_state_unavailable"
        command = convert_new(
            np.asarray(selected["translation"]), np.asarray(selected["rotation"]),
            state["ee_pose"], self.hw.hand_eye_r, self.hw.hand_eye_t,
            selected["depth"],
        )
        if not self.hw.in_workspace(*command[:3]):
            print(f"[Error] Safety violation: {command}, out of bounds!")
            return False, "safety_violation"

        self._adjust_ry(command)
        width = np.clip(
            selected["width"] + self.policy["target_width_offset"],
            0.0, self.hw.gripper_max_width,
        )
        print(f"[Info] Converted Arm Command: {command}")
        return self.executor.run_sequence(command, float(width), self.steps)

    def release(self):
        cfg, pose = self.policy["release"], self.hw.drop_pose
        state = self.robot.get_state()
        width = float(state.get("gripper_pos", 0.0))
        print(f"[Robot] Executing Release at {pose[:3]}...")
        self.robot.set_ee_pose(pose, width, cfg["move_preview"])
        time.sleep(cfg["move_wait"])
        self.robot.set_ee_pose(pose, self.hw.gripper_max_width, cfg["open_preview"])
        time.sleep(cfg["open_wait"])
        self.robot.reset_to_home()
        return True

    def _approach_target(self, prompt):
        target = None
        for index, pose in enumerate(self.hw.ready_views):
            print(f"[Robot] Moving to Ready Pose {index + 1}...")
            self.robot.set_ee_pose(
                pose, self.hw.gripper_approach_width,
                self.policy["ready_preview_time"],
            )
            time.sleep(self.policy["ready_wait"])
            color, depth = self.capture(self.policy["ready_flush_frames"])
            if color is None or depth is None:
                continue

            run_id = time.strftime(f"%Y%m%d-%H%M%S_ready{index}")
            save_capture(self.output_dir, color, depth, run_id)
            state = self.robot.get_state()
            if state:
                detection = self.perception.detect(color, prompt)
                if detection and detection.boxes:
                    target = box_center_to_base(
                        depth, detection.boxes[0], self.engine.intrinsic,
                        state["ee_pose"], self.hw.hand_eye_r, self.hw.hand_eye_t,
                    )
            if target is not None:
                print(f"[Approach] Target found at Pose {index + 1}")
                break

        if target is None:
            print("[Approach] No target found in any view.")
            return "detect_none"

        cfg = self.policy["coarse_approach"]
        pose = np.r_[np.asarray(target) + cfg["offset"], cfg["rpy"]]
        if not self.hw.in_workspace(*pose[:3]):
            print(f"[Approach] Approach pose unsafe: {pose}. Staying.")
            return "approach_unsafe"
        print(f"[Approach] Moving to closer view: {pose}")
        self.robot.set_ee_pose(pose, self.hw.gripper_approach_width, cfg["preview_time"])
        time.sleep(cfg["wait"])
        return None

    def capture(self, flush_count=None):
        return self.perception.capture(flush_count)

    def generate_grasps(self, color, depth, prompt, run_id):
        return self.perception.generate(color, depth, prompt, run_id)

    def _select_candidate(self, color, depth, prompt, run_id):
        grasps = self.generate_grasps(color, depth, prompt, run_id)
        if grasps is None:
            return None
        return self.perception.select(color, grasps)

    def _adjust_ry(self, command):
        cfg, ry = self.policy.get("ry_alignment"), command[4]
        if not cfg or not cfg["input"][0] <= ry <= cfg["input"][1]:
            return
        in_low, in_high = cfg["input"]
        out_low, out_high = cfg["output"]
        command[4] = out_low + (ry - in_low) / (in_high - in_low) * (out_high - out_low)


class GraspTaskLcmNode:
    """Task JSON transport; grasp logic stays in ``GraspWorkflow``."""

    def __init__(self, workflow):
        import lcm

        self.workflow, self.hw = workflow, workflow.hw
        self.lc = lcm.LCM(self.hw.lcm_task_url)
        self.lc.subscribe(self.hw.lcm_cmd_channel, self._on_task)

    def run(self):
        print(f"[LCM] Listening on {self.hw.lcm_task_url} [{self.hw.lcm_cmd_channel}]")
        try:
            while True:
                self.lc.handle()
        except KeyboardInterrupt:
            self.workflow.manager.release_resources()

    def _on_task(self, channel, data):
        success, reason, task_id = False, "unknown_error", None
        try:
            message = json.loads(data.decode("utf-8"))
            task_id, kind = message.get("id"), int(message.get("kind", -1))
            if kind == 1:
                success, reason = self.workflow.grasp(message.get("obj"))
            elif kind == 2:
                success = self.workflow.release()
                reason = "success" if success else "release_failed"
            else:
                reason = "invalid_command"
        except Exception as exc:
            print(f"[Error] LCM Process Failed: {exc}")
            reason = str(exc)
        if not success:
            self.workflow.robot.reset_to_home()
        if task_id:
            response = {"id": task_id, "kind": int(success),
                        "obj": "success" if success else reason}
            self.lc.publish(self.hw.lcm_callback_channel, json.dumps(response).encode())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-profile", default="config/hardware/piper_d405.yaml")
    parser.add_argument("--app-config", default="config/apps/grasp_lcm.yaml")
    parser.add_argument("--output-dir", default="output")
    return parser.parse_args()


def main():
    args = parse_args()
    hw = HardwareConfig(args.hardware_profile)
    missing = hw.missing_for_grasp_lcm()
    if missing:
        raise SystemExit(f"[Config] '{hw.name}' missing: {', '.join(missing)}")
    with open(ROOT / args.app_config) as stream:
        manager = GraspManager(yaml.safe_load(stream), hw=hw)
    if not manager.handshake():
        manager.release_resources()
        raise SystemExit("[Error] Component handshake failed")

    try:
        workflow = GraspWorkflow(manager, ROOT / args.output_dir)
        if hw.robot_kind == "piper_lcm":
            wait_for_robot_state(workflow.robot, timeout=10.0)
            workflow.robot.enable_safe_stop()
        workflow.robot.reset_to_home()
        GraspTaskLcmNode(workflow).run()
    except BaseException:
        manager.release_resources()
        raise


if __name__ == "__main__":
    main()
