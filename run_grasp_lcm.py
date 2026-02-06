import sys
import time
import argparse
import numpy as np
import cv2
import lcm
import shutil
import threading
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
TASK_LCM_URL = "udpm://239.255.76.68:7668?ttl=1"
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
        self.lc.subscribe("GRASP_CMD", self.on_grasp_cmd)
        print(f"[LCM] Listening on {lcm_url} [Topic: GRASP_CMD]")

        # VLM Selector
        model_name = self.pipeline.cfg.get("grasp_selection_model", "qwen3-vl:8b-instruct-q4_K_M")
        self.vlm_selector = GraspSelectionApp(model_name=model_name, prompts_dir=str(ROOT / "vlm/prompts"))
        
        # Async Warmup
        def _warmup_thread():
            if self.pipeline.vlm:
                self.pipeline.vlm.llm_client.warmup()
            if self.vlm_selector:
                self.vlm_selector.llm_client.warmup()
        
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
        # Decode prompt (Simple string assumption, replace with your msg type)
        try:
            prompt = data.decode('utf-8').strip()
        except Exception:
            prompt = "object"   
        print(f"\n[LCM] Received Task: '{prompt}'")

        try:
            success = self.execute_grasp(prompt)
        except Exception as e:
            print(f"[Error] {e}")
            success = False

        if not success: self.client.reset_to_home()
        
        # Send Result (1=Success, 0=Failure)
        result_payload = b'\x01' if success else b'\x00'
        self.lc.publish("GRASP_RESULT", result_payload)
        print(f"[LCM] Result Sent: {'SUCCESS' if success else 'FAILURE'}")

    def execute_grasp(self, prompt):
        # 0. Move to Ready Pose
        ready_pose = np.array([0.25, 0.0, 0.17, 0.0, 1.0, 0.0])
        print(f"[Robot] Moving to Ready Pose...")
        self.client.set_ee_pose(ready_pose, 0.086, preview_time=1.3)
        time.sleep(2)

        # 1. Capture & Detection (Flush buffer first)
        for _ in range(20): 
            color, depth = self.cam.get_frames()
        if color is None: 
            print("[Error] Camera frame not available.")
            return False
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._save_capture(color, depth, timestamp)

        # Reuse pipeline with updated prompt
        trans_list, rot_list, width_list = self.pipeline.run(color, depth, prompt=prompt, run_id=timestamp)

        if trans_list is None:
            print("[Error] Detection failed or No valid grasps found.")
            return False

        # 2. VLM 2D Selection
        imgs, candidates = vlm_grasp_visualize_batch(
            color, trans_list, rot_list, width_list, 
            self.pipeline.grasp_engine.intrinsic, top_k=5
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
        
        # 3. Coordinate Conversion & Safety Check
        curr_pose = self.client.get_state()['ee_pose']
        arm_cmd = convert_new(np.array(sel['translation']), np.array(sel['rotation']), 
                              curr_pose, HAND_EYE_R, HAND_EYE_T)
        
        x, y, z = arm_cmd[:3]
        if not ((0 <= x <= 0.7) and (-0.6 <= y <= 0.6) and (-0.02 <= z <= 0.7)):
            print(f"[Error] Safety violation: {arm_cmd[:3]}")
            return False

        # 4. Execution Sequence
        print(f"Executing Grasp at {arm_cmd}...")
        width = sel['width']
        target_width = max(0.0, width - 0.05)
        
        # Approach -> Reach -> Grasp -> Lift -> Return
        pre_pose = arm_cmd.copy()
        pre_pose[2] += 0.05
        
        self.client.set_ee_pose(pre_pose, gripper_pos=self.grip_max, preview_time=1)
        time.sleep(1.2)
        self.client.set_ee_pose(arm_cmd, gripper_pos=self.grip_max, preview_time=0.5)
        time.sleep(0.7)
        self.client.set_ee_pose(arm_cmd, gripper_pos=target_width, preview_time=0.5)
        time.sleep(0.8)
        
        lift_pose = arm_cmd.copy()
        lift_pose[2] += 0.1
        self.client.set_ee_pose(lift_pose, gripper_pos=target_width, preview_time=0.5)
        time.sleep(0.8)

        home_pose = np.array([0.3202, 0.001, 0.1565, -0., 0., 0.])
        self.client.set_ee_pose(home_pose, gripper_pos=target_width, preview_time=1.0)
        time.sleep(1.2)
        
        return True
    
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
