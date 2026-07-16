"""分割器实现。默认 FastSAM,use_sam=False 时退化到 bbox mask。"""
from typing import Optional
import numpy as np

from .base import register_segmenter


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
            return np.any(mask_data > 0, axis=0)
        return None


# =============================================================================
# Registry 工厂:每个 backend 一个 @register_segmenter
# 加新分割器 = 新增一个工厂,pipeline 无需改动。
# =============================================================================

@register_segmenter("fastsam")
def build_fastsam_segmenter(cfg, root, args=None):
    """默认分割器:FastSAM(use_sam=False 时退化到 bbox mask)。"""
    return FastSAMSegmenter(str(root / args.fastsam), use_sam=args.use_sam)
