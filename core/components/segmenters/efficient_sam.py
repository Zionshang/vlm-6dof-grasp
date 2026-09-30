"""EfficientSAM box-prompt adapter and component registration."""
import numpy as np

from perception_errors import PerceptionEmptyError
from registry import register


class EfficientSAMSegmenter:
    def __init__(self, weights, variant="vitt", device=None,
                 min_containment=.9, min_coverage=.1):
        import torch
        from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam

        dim, heads = {"vitt": (192, 3), "vits": (384, 6)}[variant]
        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.min_containment = float(min_containment)
        self.min_coverage = float(min_coverage)
        print(f"[EfficientSAM] loading {weights} on {self.device}")
        self.model = build_efficient_sam(dim, heads, weights).to(self.device).eval()

    def _select(self, candidates, scores, box):
        """Keep ratio-valid masks and choose by EfficientSAM's IoU score."""
        height, width = candidates.shape[-2:]
        x1, y1, x2, y2 = np.asarray(box, int)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        box_mask = np.zeros((height, width), dtype=bool)
        box_mask[y1:y2, x1:x2] = True
        valid = []
        for index, mask in enumerate(candidates):
            inside = np.count_nonzero(mask & box_mask)
            containment = inside / max(1, np.count_nonzero(mask))
            coverage = inside / max(1, np.count_nonzero(box_mask))
            if (containment >= self.min_containment
                    and coverage >= self.min_coverage):
                valid.append(index)
        if not valid:
            raise PerceptionEmptyError("EfficientSAM候选未通过包含率/覆盖率检查")
        index = max(valid, key=lambda i: float(scores[i]))
        return candidates[index] & box_mask

    def segment_instances(self, color, boxes):
        """Return one validated mask per box from a single model forward."""
        if not boxes:
            return []
        torch = self.torch
        image = (torch.from_numpy(np.ascontiguousarray(color)).permute(2, 0, 1)
                 .to(self.device, dtype=torch.float32).div_(255).unsqueeze(0))
        corners = torch.as_tensor(
            boxes, device=self.device, dtype=torch.float32
        ).reshape(-1, 2, 2)
        points = corners.unsqueeze(0)
        labels = torch.tensor(
            [2, 3], device=self.device,
        ).expand(1, len(boxes), 2)
        with torch.inference_mode():
            logits, iou = self.model(image, points, labels)
            candidates = (logits[0] >= 0).cpu().numpy()
            scores = iou[0].cpu().numpy()
            masks = [
                self._select(group, score, box)
                for group, score, box in zip(candidates, scores, boxes)
            ]
        return masks

    def segment(self, color, boxes):
        masks = self.segment_instances(color, boxes)
        return np.any(masks, axis=0) if masks else None


@register("segmenter", "efficient_sam")
def build_efficient_sam_segmenter(cfg=None, hw=None, ctx=None,
                                  dependencies=None):
    import paths
    cfg = cfg or {}
    return EfficientSAMSegmenter(
        str(paths.PROJECT_ROOT / cfg.get(
            "weights", "third_party/EfficientSAM/weights/efficient_sam_vitt.pt"
        )),
        variant=cfg.get("variant", "vitt"), device=cfg.get("device"),
        min_containment=cfg.get("min_containment", .9),
        min_coverage=cfg.get("min_coverage", .1),
    )
