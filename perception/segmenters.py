"""分割器实现。默认 FastSAM,use_sam=False 时退化到 bbox mask。"""
from typing import Optional
import numpy as np



class FastSAMSegmenter:
    """适配 fastsam.segmentor.ImageSegmentor,含 bbox 退化(use_sam=False 时)。"""

    def __init__(self, weights: str, use_sam: bool = True):
        self.use_sam = use_sam
        self.sam = None
        if use_sam:
            from fastsam.segmentor import ImageSegmentor
            self.sam = ImageSegmentor(weights)

    def segment(self, color: np.ndarray, boxes) -> Optional[np.ndarray]:
        if not self.use_sam or self.sam is None:
            # 退化:直接用检测框生成 mask(原 use_sam=False 分支)
            from vlm.src.utils.image_utils import make_bbox_mask
            h, w = color.shape[:2]
            return make_bbox_mask(boxes, h, w)
        res = self.sam.segment(color, boxes)
        if getattr(res, "masks", None) is not None:
            import torch
            mask_data = res.masks.data
            if torch.is_tensor(mask_data):
                mask_data = mask_data.cpu().numpy()
            mask = np.any(mask_data > 0, axis=0)
            if mask.shape != color.shape[:2]:
                raise ValueError(
                    "FastSAM mask is not in the RGB image grid: "
                    f"mask={mask.shape}, rgb={color.shape[:2]}"
                )
            return mask
        return None
