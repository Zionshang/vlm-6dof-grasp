"""EfficientSAM box-prompt adapter and component registration."""
import numpy as np

from registry import register


class EfficientSAMSegmenter:
    def __init__(self, weights, variant="vitt", device=None):
        import torch
        from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam

        dim, heads = {"vitt": (192, 3), "vits": (384, 6)}[variant]
        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[EfficientSAM] loading {weights} on {self.device}")
        self.model = build_efficient_sam(dim, heads, weights).to(self.device).eval()

    def segment(self, color, boxes):
        if not boxes:
            return None
        torch = self.torch
        image = (torch.from_numpy(np.ascontiguousarray(color)).permute(2, 0, 1)
                 .to(self.device, dtype=torch.float32).div_(255).unsqueeze(0))
        points = torch.as_tensor(
            boxes, device=self.device, dtype=torch.float32
        ).reshape(1, -1, 2, 2)
        labels = torch.tensor([2, 3], device=self.device).expand(1, len(boxes), 2)
        with torch.inference_mode():
            logits, iou = self.model(image, points, labels)
            order = iou.argsort(dim=2, descending=True)
            logits = torch.take_along_dim(logits, order[..., None, None], dim=2)
            mask = (logits[0, :, 0] >= 0).any(dim=0).cpu().numpy()
        if mask.shape != color.shape[:2]:
            raise ValueError(
                "EfficientSAM mask is not in the RGB image grid: "
                f"mask={mask.shape}, rgb={color.shape[:2]}"
            )
        return mask


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
    )
