import re
from typing import List, Tuple, Any
from abc import ABC, abstractmethod

class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> Any:
        pass

class BoundingBoxParser(BaseParser):
    def parse(self, content: str) -> List[List[int]]:
        """
        Parse bounding boxes from LLM response.
        Returns: List of [y1, x1, y2, x2] (normalized 0-1000)
        """
        boxes = []
        
        # Strategy 1: JSON format (e.g., "bbox_2d": [y, x, y, x])
        try:
            # Match "bbox_2d" or "box_2d"
            json_bbox_pattern = r'"b?box_2d":\s*\[([\d,\s]+)\]'
            json_matches = re.findall(json_bbox_pattern, content)
            
            if json_matches:
                for match in json_matches:
                    nums = [int(n) for n in re.findall(r'\d+', match)]
                    if len(nums) == 4:
                        # JSON output usually follows [xmin, ymin, xmax, ymax]
                        # But our pipeline expects [ymin, xmin, ymax, xmax] (Qwen native)
                        x1, y1, x2, y2 = nums
                        boxes.append([y1, x1, y2, x2])
                if boxes:
                    return boxes
        except Exception:
            pass

        # Strategy 2: <box>...</box> pattern
        box_pattern = r'<box>(.*?)</box>'
        box_matches = re.findall(box_pattern, content)
        
        if box_matches:
            for box_content in box_matches:
                # Often formatted as (y1, x1), (y2, x2) or similar
                nums = [int(n) for n in re.findall(r'\d+', box_content)]
                if len(nums) == 4:
                    boxes.append(nums)
        
        # Strategy 3: Plain text list [x, y, x, y] or similar fallback if needed
        # (Implementation can be added here)
        
        return boxes
