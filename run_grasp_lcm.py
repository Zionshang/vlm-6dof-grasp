import sys
import time
import argparse
import numpy as np
import cv2
import lcm
import shutil
import threading
import json
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT)])

from communication.lcm.lcm_client import Arx5LcmClient
from inference_pipeline import GraspPipeline
from realsense_driver import RealSenseD405
from convert import convert_new
from economic_grasp.utils.vlm_utils import vlm_grasp_visualize_batch
from vlm.src.apps.grasp_selection import GraspSelectionApp

# Configuration
TASK_LCM_URL = "udpm://239.255.76.67:50000?ttl=1"
MANI_CMD_CHANNEL = "mani/cmd"
MANI_CALLBACK_CHANNEL = "mani/callback"

HAND_EYE_R = np.array([
    [-0.006092615385294875, -0.3027725149342249, 0.9530433800400533],
    [-0.999954699443327, -0.005125146873484365, -0.008020718841149077],
    [0.007312940514623167, -0.9530490737994176, -0.30272757362205927]
])
HAND_EYE_T = np.array([-0.1932219485813188, 0.010310356659821916, 0.1095743344596426])

class GraspLcmNode:
    def __init__(self, robot_client, pipeline, cam, lcm_url):
        self.client = robot_client
        self.pipeline = pipeline
        self.cam = cam
        self.grip_max = 0.085

        # Initialize Task LCM
        self.lc = lcm.LCM(lcm_url)
        self.lc.subscribe(MANI_CMD_CHANNEL, self.on_grasp_cmd)
        print(f"[LCM] Listening on {lcm_url} [Topic: {MANI_CMD_CHANNEL}]")

        # VLM Selector
        model_name = self.pipeline.cfg.get("grasp_selection_model", "qwen3-vl:8b-instruct-q4_K_M")
        self.vlm_selector = GraspSelectionApp(model_name=model_name, prompts_dir=str(ROOT / "vlm/prompts"))
        
        # Async Warmup
        def _warmup_thread():
            if self.pipeline.vlm:
                self.pipeline.vlm.llm_client.warmup()
            # if self.vlm_selector:
            #     self.vlm_selector.llm_client.warmup()
        
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
            self.lc.publish(MANI_CALLBACK_CHANNEL, json.dumps(res).encode('utf-8'))
            print(f"[LCM] Sent Result: {res}")

    def execute_release(self):
        drop_pose = np.array([0.426, 0.001, 0.235, -0.008, 0.801, 0.002])
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

    def _safe_move_with_correction(self, target_pose, gripper_pos, preview_time=0.5, retries=1):
        """
        Executes a move with closed-loop error correction based on end-effector state feedback.
        """
        # 1. Initial Command
        self.client.set_ee_pose(target_pose, gripper_pos=gripper_pos, preview_time=preview_time)
        time.sleep(preview_time + 0.3) 

        current_target = target_pose.copy() # We correct *relative to* the initial command
        
        for i in range(retries):
            # 2. Get Actual State vs Original Target
            curr_state = self.client.get_state()
            if not curr_state: break
            
            curr_pos = np.array(curr_state['ee_pose'][:3])
            desired_pos = target_pose[:3] 
            
            # 3. Calculate Global Error (Where I am vs Where I wanted to go)
            pos_error = desired_pos - curr_pos
            error_norm = np.linalg.norm(pos_error)
            
            # 4. Threshold (5mm)
            if error_norm < 0.005: 
                break
            
            print(f"[Control] Correction Loop {i+1}: Global Error={error_norm*1000:.1f}mm")
            
            # 5. Compute Correction (Gain = 0.8)
            correction = pos_error * 0.8 
            current_target[:3] += correction
            
            # 6. Safety Clamp 
            x, y, z = current_target[:3]
            # Ensure we don't correct into unsafe zones
            if not ((0 <= x <= 0.75) and (-0.65 <= y <= 0.65) and (z <= 0.7)):
                print(f"[Control] Correction unsafe {current_target[:3]}. Aborting.")
                break

            # 7. Apply Correction
            self.client.set_ee_pose(current_target, gripper_pos=gripper_pos, preview_time=0.3)
            time.sleep(0.4)

    def _approach_target(self, prompt="object"):
        """
        Move to Multi-View Ready Pose -> Detect Target -> Move Closer (if valid)
        """
        # 1. Multi-View Ready Poses
        ready_poses = [
            np.array([0.23, 0.0, 0.28, 0.0, 0.65, 0.0]),
            np.array([0.256, 0.14, 0.26, 0.07, 0.65, 0.57]),
            np.array([0.256, -0.14, 0.26, -0.01, 0.65, -0.57])
        ]
        
        target_pos = None
        for i, pose in enumerate(ready_poses):
            print(f"[Robot] Moving to Ready Pose {i+1}...")
            self.client.set_ee_pose(pose, 0.086, preview_time=1.5)
            time.sleep(1.8)
            
            # Capture & Locate
            for _ in range(10): c, d = self.cam.get_frames()
            if c is None: continue
            
            ts = time.strftime(f"%Y%m%d-%H%M%S_ready{i}")
            self._save_capture(c, d, ts)
            
            st = self.client.get_state()
            if not st: continue
            tf = (np.array(st['ee_pose']), HAND_EYE_R, HAND_EYE_T)
            
            # Try to find target
            res = self.pipeline.get_target_position(d, prompt, run_id=ts, transform_info=tf)
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
        if not ((0 <= ax <= 0.75) and (-0.65 <= ay <= 0.65) and (az <= 0.7)):
             print(f"[Approach] Approach pose unsafe: {approach_pose}. Staying.")
             return "approach_unsafe"

        # 5. Move Closer
        print(f"[Approach] Moving to closer view: {approach_pose}")
        self.client.set_ee_pose(approach_pose, 0.086, preview_time=1.5)
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
        self._save_capture(color, depth, timestamp)

        # Reuse pipeline with updated prompt
        trans_list, rot_list, width_list = self.pipeline.run(color, depth, prompt=prompt, run_id=timestamp)

        if trans_list is None:
            print("[Error] Detection failed or No valid grasps found.")
            return False, "detection/grasp_generation_failed"

        # 2. VLM 2D Selection
        imgs, candidates = vlm_grasp_visualize_batch(
            color, trans_list, rot_list, width_list, 
            self.pipeline.grasp_engine.intrinsic, top_k=8
        )
        
        # Save temp images for VLM
        savedir = ROOT / "output/2D_grasp"
        if savedir.exists(): shutil.rmtree(savedir)
        savedir.mkdir(parents=True)
        img_paths = []
        for i, img in enumerate(imgs):
            p = str(savedir / f"{i}.jpg")
            cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            img_paths.append(p)

        vlm_res = self.vlm_selector.run(img_paths)

        print(f"[VLM] Full Response: {vlm_res}")
        best_id = int(vlm_res.get("selected_id", 0)) if isinstance(vlm_res, dict) else 0
        idx = best_id if 0 <= best_id < len(candidates) else 0
        print(f"[VLM] Final Decision -> ID: {idx}")
        
        sel = candidates[idx]

        # # Debug Confirmation
        # try:
        #      prompt = input(f"\n[DEBUG] Execute Grasp ID {idx}? (y/n) > ").lower()
        #      if prompt != 'y':
        #         print("[DEBUG] User cancelled.")
        #         return False, "cancelled_by_user"
        # except Exception: 
        #     pass # Non-blocking in case of headless
        
        # 3. Coordinate Conversion & Safety Check
        curr_pose = self.client.get_state()['ee_pose']
        arm_cmd = convert_new(np.array(sel['translation']), np.array(sel['rotation']), 
                              curr_pose, HAND_EYE_R, HAND_EYE_T)
        
        x, y, z = arm_cmd[:3]
        if not ((0 <= x <= 0.75) and (-0.65 <= y <= 0.65) and (z <= 0.7)):
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

        # 4. Execution Sequence
        print(f"Executing Grasp at {arm_cmd}...")
        width = sel['width']
        target_width = max(0.0, width - 0.05)
        
        # Approach -> Reach -> Grasp -> Lift -> Return
        # 1. Approach
        pre_pose = arm_cmd.copy()
        pre_pose[0] -= 0.06  # backward -6cm
        pre_pose[2] += 0.05
        self._safe_move_with_correction(pre_pose, gripper_pos=self.grip_max, preview_time=1.3, retries=1)
        
        # 2. Reach
        self._safe_move_with_correction(arm_cmd, gripper_pos=self.grip_max, preview_time=0.5, retries=1)
        # 3. Grasp
        self.client.set_ee_pose(arm_cmd, gripper_pos=target_width, preview_time=0.5)
        time.sleep(0.8)
        
        #4. Lift 
        lift_pose = arm_cmd.copy()
        lift_pose[2] += 0.08  # lift +10cm
        lift_pose[0] += 0.05  # forward +5cm
        lift_pose[3] = 0.0   # rx = 0
        lift_pose[5] = 0.0   # rz = 0
        self.client.set_ee_pose(lift_pose, gripper_pos=target_width, preview_time=1)
        time.sleep(1.2)
        # 5. Return Home
        home_pose = np.array([0.3202, 0.001, 0.1565, -0., 0., 0.])
        self.client.set_ee_pose(home_pose, gripper_pos=target_width, preview_time=2)
        time.sleep(2)
        
        return True, "success"
    
    def _save_capture(self, color, depth, timestamp):
        capture_dir = ROOT / "output" / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(capture_dir / f"{timestamp}_color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(capture_dir / f"{timestamp}_depth.png"), depth)
        print(f"Saved capture -> {capture_dir}")

    def shutdown(self):
        print("Shutting down...")
        self.cam.release()
        self.client.reset_to_home()

def main():
    parser = GraspPipeline.get_parser()
    parser.set_defaults(prompt="mug", no_vis=True) 
    args = parser.parse_args()

    # 1. Hardware Init
    print("Initializing Robot & Camera...")
    robot_client = Arx5LcmClient(url="", address="239.255.76.67", port=7667, ttl=1)
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
    node = GraspLcmNode(robot_client, pipeline, cam, TASK_LCM_URL)
    node.start()

if __name__ == "__main__":
    main()
