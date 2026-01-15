import sys
import cv2  # Need cv2 explicitly for reading dimensions
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.core.llm_client import OllamaClient
from src.core.prompt_manager import PromptManager
from src.parsers.bbox_parser import BoundingBoxParser
from src.core.detector import ObjectDetector
from src.utils.image_utils import convert_normalized_yxyx_to_pixel_xyxy # Import converter

class StaticDetectionApp:
    def __init__(self, 
                 model_name: str, 
                 template_name: str = "standard_detection.v2",
                 prompts_dir: str = "prompts",
                 show_latency: bool = False):

        self.model_name = model_name
        self.template_name = template_name
        self.prompts_dir = prompts_dir
        self.show_latency = show_latency
        
        self._init_components()

    def _init_components(self):
        # Initialize components silently or with debug level logging
        self.llm_client = OllamaClient(model_name=self.model_name)
        self.prompt_manager = PromptManager(prompt_dir=self.prompts_dir)
        self.bbox_parser = BoundingBoxParser()
        self.detector = ObjectDetector(self.llm_client, self.bbox_parser, show_latency=self.show_latency)

    def run(self, image_path: str, target_object: str) -> Dict[str, Any]:
        """
        Detects objects in the image.
        Returns a dictionary containing:
        - success: bool
        - pixel_boxes: List[List[int]] [x1, y1, x2, y2]
        - normalized_boxes: List[List[int]] [y1, x1, y2, x2]
        - raw_response: str
        """
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Error: Image '{image_path}' not found.")
            return {"success": False, "error": "Image not found"}

        # Prepare Prompt
        try:
            prompt = self.prompt_manager.format_prompt(self.template_name, target=target_object)
        except Exception as e:
            # Fallback
            prompt = f"Detect {target_object}"

        # Run Detector (pass path directly)
        detect_result = self.detector.run(str(image_path), prompt)
        normalized_boxes = detect_result.get('coordinates', [])
        
        if not normalized_boxes:
            return {
                "success": True, # Call successful, just no objects
                "pixel_boxes": [],
                "normalized_boxes": [],
                "raw_response": detect_result.get('raw_response', ''),
                "latency_seconds": detect_result.get('latency_seconds', 0)
            }
            
        # Convert to pixel coordinates
        # Need to read image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
             return {"success": False, "error": "Failed to read image for dimensions"}
             
        height, width = img.shape[:2]
        pixel_boxes = convert_normalized_yxyx_to_pixel_xyxy(normalized_boxes, width, height)
        
        return {
            "success": True,
            "pixel_boxes": pixel_boxes,
            "normalized_boxes": normalized_boxes,
            "raw_response": detect_result.get('raw_response', ''),
            "latency_seconds": detect_result.get('latency_seconds', 0)
        }



