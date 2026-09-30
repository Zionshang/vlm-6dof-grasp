"""Single-pass Ultralytics YOLO instance-segmentation component."""
import cv2
import numpy as np

from components.yolo_common import bgr_image, class_ids, model_names
from instances import Instance
from registry import register


class YOLOSeg:
    def __init__(self, weights, confidence=.7, iou=.7, imgsz=640,
                 device=None, max_det=20, classes=None):
        from ultralytics import YOLO

        print(f"[YOLO-seg] loading {weights}")
        self.model = YOLO(weights, task="segment")
        self.confidence, self.iou = float(confidence), float(iou)
        self.imgsz, self.device, self.max_det = int(imgsz), device, int(max_det)
        self.configured_classes = classes
        class_ids(self.names, classes)

    @property
    def names(self):
        return model_names(self.model)

    def validate_target(self, target):
        class_ids(self.names, target)

    def predict(self, color, target=None):
        kwargs = {
            "source": bgr_image(color), "conf": self.confidence,
            "iou": self.iou, "imgsz": self.imgsz,
            "max_det": self.max_det, "retina_masks": True,
            "verbose": False,
        }
        selected = class_ids(
            self.names,
            self.configured_classes if target is None else target,
        )
        if selected is not None:
            kwargs["classes"] = selected
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        if not results or results[0].boxes is None or results[0].masks is None:
            return []

        result = results[0]
        boxes = result.boxes.xyxy.detach().round().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        masks = result.masks.data.detach().cpu().numpy()
        if len(boxes) != len(masks):
            raise RuntimeError(
                f"YOLO-seg box/mask数量不一致: {len(boxes)}/{len(masks)}"
            )
        height, width = color.shape[:2]
        names, instances = self.names, []
        for box, score, class_id, mask in zip(boxes, scores, classes, masks):
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height),
                                  interpolation=cv2.INTER_NEAREST)
            mask = np.asarray(mask > .5, dtype=bool)
            if not mask.any():
                continue
            x1, y1, x2, y2 = box
            instances.append(Instance(
                [max(0, min(width, int(x1))), max(0, min(height, int(y1))),
                 max(0, min(width, int(x2))), max(0, min(height, int(y2)))],
                float(score), names.get(int(class_id), str(class_id)),
                mask,
            ))
        return instances


@register("instance_model", "yolo_seg")
def build_yolo_seg(cfg=None, hw=None, ctx=None, dependencies=None):
    import paths

    cfg = cfg or {}
    weights = paths.PROJECT_ROOT / cfg.get(
        "weights", "models/yolo26seg/best.pt",
    )
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO-seg checkpoint not found: {weights}")
    return YOLOSeg(
        str(weights), cfg.get("confidence", .7), cfg.get("iou", .7),
        cfg.get("imgsz", 640), cfg.get("device"), cfg.get("max_det", 20),
        cfg.get("classes"),
    )
