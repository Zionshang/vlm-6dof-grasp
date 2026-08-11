"""FastSAM adapter and component registration."""
import numpy as np

from registry import register


class FastSAMSegmenter:
    """Use FastSAM, or a detection-box mask when segmentation is disabled."""

    def __init__(self, weights, use_sam=True):
        self.use_sam = use_sam
        self.sam = None
        if use_sam:
            from fastsam.segmentor import ImageSegmentor
            self.sam = ImageSegmentor(weights)

    def segment(self, color, boxes):
        if not self.use_sam or self.sam is None:
            from vlm.src.utils.image_utils import make_bbox_mask
            h, w = color.shape[:2]
            return make_bbox_mask(boxes, h, w)

        result = self.sam.segment(color, boxes)
        if getattr(result, "masks", None) is None:
            return None
        import torch
        masks = result.masks.data
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        mask = np.any(masks > 0, axis=0)
        if mask.shape != color.shape[:2]:
            raise ValueError(
                "FastSAM mask is not in the RGB image grid: "
                f"mask={mask.shape}, rgb={color.shape[:2]}"
            )
        return mask


@register("segmenter", "fastsam")
def build_fastsam_segmenter(cfg=None, hw=None, ctx=None, dependencies=None):
    import paths
    ROOT = paths.PROJECT_ROOT
    scfg = cfg or {}
    return FastSAMSegmenter(
        str(ROOT / scfg.get("weights", "third_party/fastsam/weight/FastSAM-s.pt")),
        use_sam=scfg.get("use_sam", True),
    )
