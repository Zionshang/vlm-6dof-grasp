import json
import re
from pathlib import Path
from typing import Dict, Any
from src.core.llm_client import OllamaClient
from src.core.prompt_manager import PromptManager

class OrientationPredictionApp:
    def __init__(self, model_name: str, template_name: str = "orientation.v1", prompts_dir: str = "prompts"):
        self.model_name = model_name
        self.template_name = template_name
        self.prompts_dir = prompts_dir
        self.llm_client = OllamaClient(model_name=self.model_name)
        self.prompt_manager = PromptManager(prompt_dir=self.prompts_dir)

    def run(self, image_path: str, target_object: str) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
             return {"success": False, "error": f"Image {image_path} not found"}

        try:
            # Ensure prompts are reloaded/loaded correctly
            if not self.prompt_manager.templates:
                 self.prompt_manager._load_prompts()
            
            prompt = self.prompt_manager.format_prompt(self.template_name, target=target_object)
        except Exception as e:
            return {"success": False, "error": f"Prompt formatting error: {e}"}

        print(f"Sending request to {self.model_name}...")
        try:
            response_text = self.llm_client.generate(prompt, images=[str(image_path)])
        except Exception as e:
            return {"success": False, "error": f"LLM generation error: {e}"}
        
        # Parse JSON
        try:
            # Find JSON string in response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group()
                data = json.loads(json_str)
                angle = float(data.get("rotation_angle", 0.0))
                
                # Normalize angle to [-90, 90] if LLM messes up, although we asked implicitly.
                # But user requirement: "Identify ... calculate ... range of -90 to 90"
                # If LLM returns 270, we should probably correct it? 
                # Let's trust the LLM's explicit instruction for now, but simple clamps are safe.
                # Actually, 270 degrees is -90. 
                while angle > 90: angle -= 180
                while angle < -90: angle += 180
                
                return {"success": True, "rotation_angle": angle, "raw_response": response_text}
            else:
                 # Try to find a standalone number if JSON fails?
                 # Regex for number
                 nums = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
                 if len(nums) == 1:
                     angle = float(nums[0])
                     return {"success": True, "rotation_angle": angle, "raw_response": response_text, "note": "Parsed from text, no JSON"}
                     
                 return {"success": False, "error": "No JSON found", "raw_response": response_text}
        except Exception as e:
            return {"success": False, "error": f"Parsing Error: {e}", "raw_response": response_text}
