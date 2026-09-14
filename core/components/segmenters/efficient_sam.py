"""EfficientSAM box-prompt adapter and component registration."""
import numpy as np

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

    @staticmethod
    def _select(candidates, scores, box):
        """Choose a confident, compact mask containing the prompt centre."""
        height, width = candidates.shape[-2:]
        x1, y1, x2, y2 = np.asarray(box, int)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        cx, cy = np.clip([(x1+x2)//2, (y1+y2)//2],
                         [0, 0], [width-1, height-1])

        def rank(item):
            score, mask = item
            if not mask[cy, cx] or x2 <= x1 or y2 <= y1:
                return -np.inf
            crop = mask[y1:y2, x1:x2]
            return float(score)-float(crop.mean())

        index = max(range(len(scores)),
                    key=lambda i: rank((scores[i], candidates[i])))
        if not np.isfinite(rank((scores[index], candidates[index]))):
            raise ValueError("EfficientSAM候选未包含检测框中心")
        return candidates[index]

    def segment(self, color, boxes):
        if not boxes:
            return None
        torch = self.torch
        image = (torch.from_numpy(np.ascontiguousarray(color)).permute(2, 0, 1)
                 .to(self.device, dtype=torch.float32).div_(255).unsqueeze(0))
        corners = torch.as_tensor(
            boxes, device=self.device, dtype=torch.float32
        ).reshape(-1, 2, 2)
        points = torch.cat((corners, corners.mean(dim=1, keepdim=True)), dim=1)
        points = points.unsqueeze(0)
        labels = torch.tensor(
            [2, 3, 1], device=self.device,
        ).expand(1, len(boxes), 3)
        with torch.inference_mode():
            logits, iou = self.model(image, points, labels)
            candidates = (logits[0] >= 0).cpu().numpy()
            scores = iou[0].cpu().numpy()
            mask = np.any([
                self._select(group, score, box)
                for group, score, box in zip(candidates, scores, boxes)
            ], axis=0)
        if mask.shape != color.shape[:2]:
            raise ValueError(
                "EfficientSAM mask is not in the RGB image grid: "
                f"mask={mask.shape}, rgb={color.shape[:2]}"
            )
        if not mask.any():
            raise ValueError("EfficientSAM分割结果为空")
        from vlm.src.utils.image_utils import make_bbox_mask
        box_mask = make_bbox_mask(boxes, *mask.shape)
        inside = np.count_nonzero(mask & box_mask)
        containment = inside / max(1, np.count_nonzero(mask))
        coverage = inside / max(1, np.count_nonzero(box_mask))
        if (containment < self.min_containment
                or coverage < self.min_coverage):
            raise ValueError(
                f"EfficientSAM分割结果不合格: "
                f"containment={containment:.2f}, coverage={coverage:.2f}"
            )
        return mask & box_mask


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
