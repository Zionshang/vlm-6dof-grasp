import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

from src.core.llm_client import OllamaClient
from src.core.prompt_manager import PromptManager

class GraspSelectionApp:
    def __init__(self, 
                 model_name: str, 
                 template_name: str = "grasp_selection.v1",
                 prompts_dir: str = "prompts"):

        self.model_name = model_name
        self.template_name = template_name
        self.prompts_dir = prompts_dir
        
        self.llm_client = OllamaClient(model_name=self.model_name)
        self.prompt_manager = PromptManager(prompt_dir=self.prompts_dir)

    def run(self, image_paths: list[str], instruction: str = "") -> Dict[str, Any]:
        """
        Analyses a list of images to select the best grasp ID.
        """
        valid_paths = [p for p in image_paths if Path(p).exists()]
        if not valid_paths:
            return {"success": False, "error": "No valid images found"}

        try:
            # Pass image count to prompt
            prompt = self.prompt_manager.format_prompt(self.template_name, num_images=len(valid_paths))
            if instruction:
                prompt += f"\nAdditional Instruction: {instruction}"
        except Exception:
            prompt = f"Analyze these {len(valid_paths)} images. Each image shows a grasp candidate with an ID. Select the best ID."

        # Run Inference with MULTIPLE images
        raw_response = self.llm_client.generate(prompt, images=valid_paths)
        
        # Parse Response
        result = self._parse_response(raw_response)
        
        return {
            "success": result is not None,
            "selected_id": result.get("selected_id") if result else None,
            "reason": result.get("reason", "") if result else "",
            "raw_response": raw_response
        }

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extracts JSON from the response string.
        """
        try:
            # Try to find JSON block
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                # Fallback: look for integer if JSON search fails
                match_id = re.search(r'\b\d+\b', response)
                if match_id:
                     return {"selected_id": int(match_id.group(0)), "reason": "Parsed from raw text"}
        except Exception as e:
            print(f"Error parsing grasp selection response: {e}")
            
        return None
