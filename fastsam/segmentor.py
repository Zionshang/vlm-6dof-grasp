from typing import List, Union, Any, Optional
import numpy as np
import torch
from pathlib import Path
from ultralytics import FastSAM # type: ignore

class ImageSegmentor:
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.model_path = str(model_path)
        self.device = device
        print(f"Loading FastSAM model from {self.model_path} on {self.device}...")
        self.model = FastSAM(self.model_path)

    def segment(self, image_source: Union[str, np.ndarray], bboxes: List[List[int]], conf: float = 0.5) -> Any:
        """
        Segment objects in the image based on bounding boxes.
        image_source: Path to image or numpy array.
        bboxes: List of [x1, y1, x2, y2] in pixels.
        Returns: The Result object from ultralytics containing masks.
        """
        if not bboxes:
            print("No bounding boxes provided for segmentation.")
            return None

        # Run inference
        try:
            results = self.model.predict(
                image_source, 
                device=self.device, 
                bboxes=bboxes, 
                conf=conf, 
                retina_masks=False, 
                verbose=False,
                save=False,      
                save_txt=False,
                save_conf=False,
                exist_ok=True
            )
            
            if results:
                return results[0] # Return the result for the first (and only) image
            return None
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return None
