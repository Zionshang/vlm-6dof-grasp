import sys
import time
import argparse
import numpy as np
import cv2
import lcm
import threading
import json
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
ROOT = paths.PROJECT_ROOT

from robot_client import make_robot_client
from pipeline import GraspPipeline
from camera import RealSenseD405
from transform import convert_new
from grasping.selector import VLMSelector
from saver import save_capture
from hardware import HardwareConfig
from grasp_executor import GraspExecutor, GraspStep


# 抓取执行序列(参数为 X5_umi 实测调校值,改动需谨慎)
GRASP_STEPS = [
    GraspStep("approach", gripper="max",    offset=(-0.06, 0.0, 0.05), preview=1.3, correct=True),
    GraspStep("reach",    gripper="max",    preview=0.5, correct=True),
    GraspStep("grasp",    gripper="target", preview=0.5, wait=0.8),
    GraspStep("lift",     gripper="target", offset=(0.05, 0.0, 0.08), fix_rpy=(0.0, None, 0.0), preview=1.0, wait=1.2),
    GraspStep("home",     gripper="target", use_home_pose=True, preview=2.0, wait=2.0),
]


class GraspLcmNode:
    def __init__(self, robot_client, pipeline, cam):
        self.client = robot_client
        self.pipeline = pipeline
        self.cam = cam
        hw = pipeline.hw
        self.grip_max = hw.gripper_max_width
        self.executor = GraspExecutor(self.client, hw, self.grip_max)

        # Initialize Task LCM
        self.lc = lcm.LCM(hw.lcm_task_url)
        self.lc.subscribe(hw.lcm_cmd_channel, self.on_grasp_cmd)
        print(f"[LCM] Listening on {hw.lcm_task_url} [Topic: {hw.lcm_cmd_channel}]")

        # VLM Selector
        model_name = self.pipeline.cfg.get("grasp_selection_model", "qwen3-vl:8b-instruct-q4_K_M")
        self.selector = VLMSelector(model_name=model_name, prompts_dir=str(ROOT / "vlm/prompts"))
        
        # Async Warmup
        def _warmup_thread():
            self.pipeline.detector.warmup()
        
        self.warmup_thread = threading.Thread(target=_warmup_thread, daemon=True)
        self.warmup_thread.start()

    def start(self):
        print("[System] Ready. Waiting for commands...")
        try:
            while True:
                self.lc.handle()
        except KeyboardInterrupt:
            self.shutdown()

    def on_grasp_cmd(self, channel, data):
        success = False
        msg_text = "unknown_error"
        task_id = None
        try:
            msg = json.loads(data.decode('utf-8'))
            task_id = msg.get("id")
            kind = int(msg.get("kind", -1))
            
            print(f"\n[LCM] Received Task ID: {task_id}, Kind: {kind}, Obj: {msg.get('obj')}")
            
            if kind == 1: # Grasp
                success, msg_text = self.execute_grasp(msg.get("obj"))
            elif kind == 2: # Release
                success = self.execute_release()
                msg_text = "success" if success else "release_failed"
            else:
                msg_text = "invalid_command"

        except Exception as e:
            print(f"[Error] LCM Process Failed: {e}")
            msg_text = str(e)

        if not success: self.client.reset_to_home()

        if task_id:
            # kind: 1=Success, 0=Failure. obj: Failure Reason (or 'success')
            res = {
                "id": task_id, 
                "kind": 1 if success else 0, 
                "obj": "success" if success else msg_text
            }
            self.lc.publish(self.pipeline.hw.lcm_callback_channel, json.dumps(res).encode('utf-8'))
            print(f"[LCM] Sent Result: {res}")

    def execute_release(self):
        drop_pose = self.pipeline.hw.drop_pose
        print(f"[Robot] Executing Release at {drop_pose[:3]}...")
        
        # 1. Move to Drop Pose (Maintain Hold)
        curr_width = float(self.client.get_state().get("gripper_pos", 0.0))
        self.client.set_ee_pose(drop_pose, gripper_pos=curr_width, preview_time=1.7)
        time.sleep(2.0)

        # 2. Open Gripper
        self.client.set_ee_pose(drop_pose, gripper_pos=self.grip_max, preview_time=0.5)
        time.sleep(1.0)
        
        # 3. Home
        self.client.reset_to_home()
        return True

    def _approach_target(self, prompt="object"):
        """
        Move to Multi-View Ready Pose -> Detect Target -> Move Closer (if valid)
        """
        # 1. Multi-View Ready Poses
        ready_poses = self.pipeline.hw.ready_views
        
        target_pos = None
        for i, pose in enumerate(ready_poses):
            print(f"[Robot] Moving to Ready Pose {i+1}...")
            self.client.set_ee_pose(pose, self.pipeline.hw.gripper_approach_width, preview_time=1.5)
            time.sleep(1.8)
            
            # Capture & Locate
            for _ in range(10): c, d = self.cam.get_frames()
            if c is None: continue
            
            ts = time.strftime(f"%Y%m%d-%H%M%S_ready{i}")
            save_capture(self.pipeline.output_dir, c, d, ts)
            
            st = self.client.get_state()
            if not st: continue
            tf = (np.array(st['ee_pose']), self.pipeline.hw.hand_eye_r, self.pipeline.hw.hand_eye_t)
            
            # Try to find target
            res = self.pipeline.get_target_position(d, c, prompt, run_id=ts, transform_info=tf)
            if res is not None:
                target_pos = res
                print(f"[Approach] Target found at Pose {i+1}")
                break
        
        if target_pos is None:
            print("[Approach] No target found in any view.")
            return "detect_none"

        # 3. Check Bounds
        x, y, z = target_pos
        # 4. Compute Approach Pose Logic: x-0.17, y, z+0.17 | Rot: 0.0, 0.9, 0.0
        approach_pose = np.array([x - 0.13, y, z + 0.10, 0.0, 0.8, 0.0])
        
        # Check Approach Pose Safety
        ax, ay, az = approach_pose[:3]
        if not self.pipeline.hw.in_workspace(ax, ay, az):
             print(f"[Approach] Approach pose unsafe: {approach_pose}. Staying.")
             return "approach_unsafe"

        # 5. Move Closer
        print(f"[Approach] Moving to closer view: {approach_pose}")
        self.client.set_ee_pose(approach_pose, self.pipeline.hw.gripper_approach_width, preview_time=1.5)
        time.sleep(1.5)

    def execute_grasp(self, prompt):
        # 0. Coarse Approach (Replaces simple Ready Pose)
        approach_res = self._approach_target(prompt)
        if approach_res == "detect_none":
            return False, "detect_none"
        elif approach_res == "approach_unsafe":
            return False, "approach_unsafe"

        # 1. Capture & Detection (Flush buffer first)
        for _ in range(20): 
            color, depth = self.cam.get_frames()
        if color is None: 
            print("[Error] Camera frame not available.")
            return False, "camera_error"
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_capture(self.pipeline.output_dir, color, depth, timestamp)

        # Reuse pipeline with updated prompt
        trans_list, rot_list, width_list = self.pipeline.run(color, depth, prompt=prompt, run_id=timestamp)

        if trans_list is None:
            print("[Error] Detection failed or No valid grasps found.")
            return False, "detection/grasp_generation_failed"

        # 2. Grasp Selection(可插拔 selector:默认 VLM 二次选优,可换 FirstGraspSelector 跳过)
        idx, candidates = self.selector.select(
            color, trans_list, rot_list, width_list,
            self.pipeline.grasp_engine.intrinsic, top_k=8, output_dir=self.pipeline.output_dir
        )
        sel = candidates[idx]
        
        # 3. Coordinate Conversion & Safety Check
        curr_pose = self.client.get_state()['ee_pose']
        arm_cmd = convert_new(np.array(sel['translation']), np.array(sel['rotation']), 
                              curr_pose, self.pipeline.hw.hand_eye_r, self.pipeline.hw.hand_eye_t)
        
        x, y, z = arm_cmd[:3]
        if not self.pipeline.hw.in_workspace(x, y, z):
            print(f"[Error] Safety violation: {arm_cmd}, out of bounds!")
            return False, "safety_violation"
        print(f"[Info] Converted Arm Command: {arm_cmd}")

        # Ry alignment for better pose
        ry = arm_cmd[4]
        if 0 <= ry <= 0.7:
            print(f"[Adjust] Original Ry {ry:.3f}")
            # Map [0, 0.7] -> [0.7, 0.8] linearly
            arm_cmd[4] = 0.7 + (ry / 0.7) * 0.1
            print(f"[Adjust] New Ry -> {arm_cmd[4]:.3f}")

        # 4. Execution Sequence (APPROACH -> REACH -> GRASP -> LIFT -> HOME)
        width = sel['width']
        target_width = max(0.0, width - 0.05)
        return self.executor.run_sequence(arm_cmd, target_width, GRASP_STEPS)
    
    def shutdown(self):
        print("Shutting down...")
        self.cam.release()
        self.client.reset_to_home()

def main():
    parser = GraspPipeline.get_parser()
    parser.set_defaults(prompt="mug")
    args = parser.parse_args()

    # 1. Hardware Init
    print("Initializing Robot & Camera...")
    hw = HardwareConfig()
    robot_client = make_robot_client(hw)
    cam = RealSenseD405()

    for _ in range(5):
        if all(f is not None for f in cam.get_frames()): 
            print("Camera is ready.")
            break
        time.sleep(0.5)
    else:
        sys.exit("[Error] Camera failed")
    
    # 2. Pipeline Init
    pipeline = GraspPipeline(args)
    
    # 3. Robot Ready Pose (Moved to execute_grasp)
    robot_client.reset_to_home()
    
    # 4. Start LCM Node
    node = GraspLcmNode(robot_client, pipeline, cam)
    node.start()

if __name__ == "__main__":
    main()
