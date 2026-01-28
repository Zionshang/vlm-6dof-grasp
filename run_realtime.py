import sys
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from pynput import keyboard

# Path setup & Imports
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT)])
from communication.lcm.lcm_client import Arx5LcmClient
from inference_pipeline import GraspPipeline
from realsense_driver import RealSenseD405
from convert import convert_new
from economic_grasp.utils.vlm_utils import vlm_grasp_visualize_batch
from vlm.src.apps.grasp_selection import GraspSelectionApp
import shutil

# Hand-Eye Calibration (Camera -> End Effector)
HAND_EYE_R = np.array([
    [-0.006092615385294875, -0.3027725149342249, 0.9530433800400533],
    [-0.999954699443327, -0.005125146873484365, -0.008020718841149077],
    [0.007312940514623167, -0.9530490737994176, -0.30272757362205927]
])
HAND_EYE_T = np.array([-0.1932219485813188, 0.010310356659821916, 0.1095743344596426])


def get_gripper_max_width(client):
    try:
        return client.get_robot_config().gripper_width
    except AttributeError:
        return 0.085 


class RealtimeGraspController:
    def __init__(self, client, pipeline, cam):
        self.client = client
        self.pipeline = pipeline
        self.cam = cam
        self.current_prompt = "mug"
        self.running = True
        self.grip_max = get_gripper_max_width(client)
        model_name = self.pipeline.cfg.get("default_model", "qwen2.5-vl") if hasattr(self.pipeline, "cfg") else "qwen2.5-vl"
        self.vlm_selector = GraspSelectionApp(model_name=model_name, prompts_dir=str(ROOT / "vlm/prompts"))
        
        # Key Config: Map keys to specific handler functions
        self.key_actions = {
            keyboard.KeyCode.from_char("h"): self.action_grasp,
            keyboard.KeyCode.from_char("q"): self.action_quit,
            keyboard.KeyCode.from_char("n"): self.action_new_prompt,
            keyboard.Key.space: self.action_home
        }
        # State tracking for keys
        self.key_pressed = {key: False for key in self.key_actions}

    def start(self):
        """Starts the keyboard listener and main loop."""
        print("\n=== Realtime Grasp Controller ===")
        print(" [h] Grasp Target")
        print(" [n] Change Prompt")
        print(" [space] Reset Home")
        print(" [q] Quit")
        print("=================================\n")

        # Non-blocking listener
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        try:
            while self.running:
                self.loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def loop(self):
        """Main update loop: Get frames -> Handle Input -> Sleep."""
        # 1. Camera Stream
        color, depth = self.cam.get_frames()
        if color is not None:
            cv2.imshow("Realtime Grasp", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        
        # 2. Handle Key Actions
        # Iterate over keys we care about
        for key, action_func in self.key_actions.items():
            if self.key_pressed.get(key, False):
                # Execute action
                action_func(color, depth)
                
                # Debounce: Wait for key release (unless it's Quit)
                if key != keyboard.KeyCode.from_char("q"): 
                     self._wait_for_release(key)
        
        time.sleep(0.01)

    def on_press(self, key):
        if key in self.key_pressed: self.key_pressed[key] = True

    def on_release(self, key):
        if key in self.key_pressed: self.key_pressed[key] = False

    def _wait_for_release(self, key):
        """Blocks logic (but keeps camera alive) until key is released."""
        while self.key_pressed.get(key, False):
            time.sleep(0.05)
            # Flush camera buffer so stream doesn't lag
            _ = self.cam.get_frames()

    # --- Actions ---
    def action_home(self, color, depth):
        print("Resetting robot to home...")
        self.client.reset_to_home()
    
    def action_quit(self, color, depth):
        self.running = False

    def action_new_prompt(self, color, depth):
        print(f"\nCurrent Prompt: {self.current_prompt}")
        ready_pose = np.array([0.25, 0.0, 0.17, 0.0, 1.0, 0.0])
        self.client.set_ee_pose(ready_pose, get_gripper_max_width(self.client), preview_time=1.5)
        time.sleep(2)
        # Use try-except to handle potential terminal input issues gracefully
        try:
            # Note: might need to click terminal window to type
            new_prompt = input("Enter New Prompt > ").strip()
            if new_prompt:
                self.current_prompt = new_prompt
                print(f"Updated: {self.current_prompt}")
        except EOFError:
            pass

    def action_grasp(self, color, depth):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._save_capture(color, depth, timestamp) 
        
        # Get Top 5
        trans_list, rot_list, width_list = self.pipeline.run(color, depth, prompt=self.current_prompt, run_id=timestamp)
        if trans_list is None: return print("Grasp detection failed.")

        # VLM Selection
        imgs, candidates = vlm_grasp_visualize_batch(
            color, trans_list, rot_list, width_list, 
            self.pipeline.grasp_engine.intrinsic, top_k=5
        )
        savedir = ROOT / "output/2D_grasp"
        if savedir.exists(): shutil.rmtree(savedir)
        savedir.mkdir(parents=True)
        
        paths = []
        for i, img in enumerate(imgs):
            p = savedir / f"{i}.jpg"
            # cv2.imwrite expects BGR, but img is RGB
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            paths.append(str(p))

        print("[VLM] Selecting best grasp...")
        vlm_res = self.vlm_selector.run(paths)
        print(f"[VLM] Full Response: {vlm_res}")
        
        best_id = vlm_res.get("selected_id") if isinstance(vlm_res, dict) else None
        idx = 0
        if best_id is not None:
            best_id = int(best_id)
            if 0 <= best_id < len(candidates):
                idx = best_id
        
        print(f"[VLM] Final Decision -> ID: {idx}")
        if input("Execute grasp? (y/n) > ").lower() != 'y': return

        sel = candidates[idx]
        # Convert
        state = self.client.get_state()
        curr_pose = state['ee_pose']
        arm_cmd = convert_new(np.array(sel['translation']), np.array(sel['rotation']), 
                              curr_pose, HAND_EYE_R, HAND_EYE_T)
        width = sel['width']
        
        print(f"Target Pose (Base): {arm_cmd}")

        # 3. Safety Check & Execution
        x, y, z = arm_cmd[:3]
        if (0 <= x <= 0.7) and (-0.6 <= y <= 0.6) and (-0.02 <= z <= 0.7):
            print("Pose valid. Executing grasp sequence...")
            
            grip_max = self.grip_max
            target_close_width = max(0.0, width - 0.05)
            
            print(f"Gripper Control: Max={grip_max:.3f}m, Target={target_close_width:.3f}m (Obj={width:.3f}m)")

            # 1. Move to Target (Keep Open)
            ready_pose = arm_cmd.copy()
            ready_pose[2] += 0.05
            self.client.set_ee_pose(ready_pose, gripper_pos=grip_max, preview_time=2.0)
            time.sleep(2.5)
            self.client.set_ee_pose(arm_cmd, gripper_pos=grip_max, preview_time=0.5)
            time.sleep(1)
            
            # 2. Grasp (Close to target_close_width)
            # Call set_ee_pose again with same arm pose but new gripper width
            self.client.set_ee_pose(arm_cmd, gripper_pos=target_close_width, preview_time=0.5)
            time.sleep(1.0)
            
            # 3. Lift up 10cm
            lift_pose = arm_cmd.copy()
            lift_pose[2] += 0.1
            self.client.set_ee_pose(lift_pose, gripper_pos=target_close_width, preview_time=1.0)
            time.sleep(1.5)
            
            # Home
            self.action_home(None, None)
            state = self.client.get_state()
            curr_pose = state['ee_pose']
            self.client.set_ee_pose(curr_pose, gripper_pos=grip_max, preview_time=0.5)
            time.sleep(1.0)
            
        else:
            print(f"Safety violation: Pose {arm_cmd[:3]} out of bounds!")

    def _save_capture(self, color, depth, timestamp):
        capture_dir = ROOT / "output" / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(capture_dir / f"{timestamp}_color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(capture_dir / f"{timestamp}_depth.png"), depth)
        print(f"Saved capture -> {capture_dir}")
    
    def shutdown(self):
        print("Shutting down...")
        self.cam.release()
        cv2.destroyAllWindows()
        self.client.reset_to_home()


def main():
    # 1. Parse Args (Inherit pipeline args)
    parser = GraspPipeline.get_parser()
    parser.set_defaults(prompt="mug")
    args = parser.parse_args()

    # 2. Initialize System
    print("Initializing...")
    pipeline = GraspPipeline(args)
    client = Arx5LcmClient(url="", address="239.255.76.67", port=7667, ttl=1)
    cam = RealSenseD405()

    # 3. Check Camera
    print("Checking camera stream...")
    for _ in range(5):
        c, d = cam.get_frames()
        if c is not None and d is not None:
             print("Camera Check Passed.")
             break
        time.sleep(0.5)
    else:
        print("Error: Camera not streaming.")
        sys.exit(1)

    # 4. Reset Robot to Ready Pose
    print("Moving to ready pose...")
    client.reset_to_home()
    ready_pose = np.array([0.25, 0.0, 0.17, 0.0, 1.0, 0.0])
    client.set_ee_pose(ready_pose, get_gripper_max_width(client), preview_time=1.5)
    time.sleep(2)
    # prep_poses = [
    #     np.array([ 0.2, 0.0, 0.17, -0., 0.85, 0. ], dtype=float),
    #     np.array([ 0.2, -0.13, 0.17, 0.0032, 0.85, -0.76], dtype=float),
    #     np.array([ 0.2, 0.13, 0.17, 0.0032, 0.85, 0.76], dtype=float),
    # ]

    # 5. Start Controller
    controller = RealtimeGraspController(client, pipeline, cam)
    controller.current_prompt = args.prompt
    controller.start()


if __name__ == "__main__":
    main()