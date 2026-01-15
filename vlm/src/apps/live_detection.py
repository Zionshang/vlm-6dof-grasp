import cv2
import time
from pathlib import Path
from io import BytesIO
import numpy as np
import sys
from typing import Union

from src.core.llm_client import OllamaClient
from src.core.prompt_manager import PromptManager
from src.parsers.bbox_parser import BoundingBoxParser
from src.core.detector import ObjectDetector
from src.utils.image_utils import draw_on_image
from src.utils.camera import RealSenseCamera

class LiveDetectionApp:
    def __init__(self, 
                 model_name: str, 
                 template_name: str,
                 prompts_dir: str = "prompts",
                 camera_config: dict = None,
                 visualization_config: dict = None,
                 max_image_width: Union[int, str] = 640,
                 show_latency: bool = False):
        
        self.model_name = model_name
        self.template_name = template_name
        self.prompts_dir = prompts_dir
        self.camera_config = camera_config or {}
        self.visualization_config = visualization_config or {}
        self.max_image_width = max_image_width
        self.show_latency = show_latency
        
        self._init_components()

    def _init_components(self):
        print(f"Initializing Live Mode...")
        print(f"Model: {self.model_name}")
        
        self.llm_client = OllamaClient(model_name=self.model_name)
        self.prompt_manager = PromptManager(prompt_dir=self.prompts_dir)
        self.bbox_parser = BoundingBoxParser()
        self.detector = ObjectDetector(self.llm_client, self.bbox_parser)

    def run_detection_in_memory(self, image_np: np.ndarray, target_object: str):
        """
        Run detection on a NumPy Image array without saving to disk.
        Returns the annotated image (NumPy array).
        """
        # Prepare Prompt
        try:
            prompt = self.prompt_manager.format_prompt(self.template_name, target=target_object)
            print(f"Constructed Prompt: {prompt}")
        except ValueError as e:
            print(f"Error constructing prompt: {e}")
            return None

        # Convert Image to Bytes for Ollama
        # Encode as JPEG
        success, encoded_img = cv2.imencode('.jpg', image_np)
        if not success:
            print("Failed to encode image to JPEG")
            return None

        img_bytes = encoded_img.tobytes()

        # Execute Pipeline with Bytes
        print("Running detector...")
        result = self.detector.run(img_bytes, prompt)
        
        print("-" * 30)
        print("Raw Response:")
        print(result['raw_response'])
        
        if self.show_latency:
            print(f"Inference Latency: {result.get('latency_seconds', 0):.4f}s")
        print("-" * 30)
        
        coords = result.get('coordinates', [])
        
        # Start with a copy to draw annotations
        annotated_image = image_np.copy()
        
        if coords:
            print(f"Found {len(coords)} objects.")
            labels = [f"{target_object} {i+1}" for i in range(len(coords))]
            
            # Use visualization settings
            box_color = self.visualization_config.get("box_color", "red")
            box_width = self.visualization_config.get("box_width", 3)
            
            # Draw on the copy
            annotated_image = draw_on_image(annotated_image, coords, labels, color=box_color, width=box_width)
        else:
            print("No objects detected or parsing failed.")
            
        return annotated_image

    def run(self):
        print("Starting Live Mode... (Press SPACE to capture, 'q' to quit)")
        print("Note: Images are processed in memory and not saved to disk.")
        
        # Initialize camera with config
        camera = RealSenseCamera(
            width=self.camera_config.get("width", 1280),
            height=self.camera_config.get("height", 720),
            fps=self.camera_config.get("fps", 30),
            warmup=self.camera_config.get("warmup_seconds", 1.0)
        )

        if not camera.pipeline:
             print("Failed to start camera. Exiting.")
             return

        cv2.namedWindow("RealSense Feed", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                # 1. Get continuous frame
                frame_rgb = camera.get_frame() # Still returns RGB
                if frame_rgb is None:
                    continue
                
                # Convert for OpenCV display/processing (RGB -> BGR)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                cv2.putText(frame_bgr, "Live Preview - Press SPACE to Detect", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("RealSense Feed", frame_bgr)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 32: # SPACE
                    self._handle_capture(camera, frame_bgr)

        except KeyboardInterrupt:
            pass
        finally:
            camera.release()
            cv2.destroyAllWindows()
            print("Exiting Live Mode.")

    def _handle_capture(self, camera, frame_bgr):
        print("\n[Paused] Waiting for input...")
        
        # Show paused state
        frame_paused = frame_bgr.copy()
        cv2.putText(frame_paused, "PAUSED - Check Terminal", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("RealSense Feed", frame_paused)
        cv2.waitKey(1)
        
        try:
            user_input = input("Enter target object: ").strip()
        except EOFError:
            return

        if not user_input:
            print("Cancelled.")
            return
            
        # Prepare Image for Inference (Resize if needed)
        # It's already BGR and numpy array
        img_small = frame_bgr.copy()
        h, w = img_small.shape[:2]
        
        target_width = None
        should_resize = False
        
        if isinstance(self.max_image_width, str) and self.max_image_width.lower() == "default":
             should_resize = False
             target_width = None
        else:
            try:
                target_width = int(self.max_image_width)
                should_resize = True
            except (ValueError, TypeError):
                should_resize = False
                target_width = None
        
        if should_resize and target_width and w > target_width:
            ratio = target_width / float(w)
            new_height = int(float(h) * float(ratio))
            # Resize using OpenCV
            img_small = cv2.resize(img_small, (target_width, new_height), interpolation=cv2.INTER_AREA)
        
        print(f"Running detection (Image size: {img_small.shape[1]}x{img_small.shape[0]})...")
        
        # Run detection on memory image
        annotated_img = self.run_detection_in_memory(img_small, user_input)

        
        if annotated_img is not None:
            # Show result
            cv2.imshow("RealSense Feed", annotated_img)
            print("Result displayed. Press any key to resume...")
            cv2.waitKey(0)

