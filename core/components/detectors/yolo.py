"""Ultralytics YOLO detection adapter and component registration."""
from dataclasses import dataclass, field

from components.yolo_common import bgr_image, class_ids, model_names
from registry import register


@dataclass
class Detection:
    boxes: list[list[int]]
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


class YOLODetector:
    """Expose an Ultralytics detection checkpoint through ``detect``."""

    def __init__(self, weights, confidence=0.25, iou=0.7, imgsz=640,
                 device=None, max_det=100, classes=None):
        from ultralytics import YOLO

        print(f"[YOLO] loading {weights}")
        self.model = YOLO(weights, task="detect")
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.max_det = int(max_det)
        self.configured_classes = classes
        self._class_ids(classes)

    @property
    def names(self):
        return model_names(self.model)

    def _class_ids(self, target):
        selection = self.configured_classes if target is None else target
        return class_ids(self.names, selection)

    def detect(self, color, target=None):
        kwargs = {
            "source": bgr_image(color),
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        class_ids = self._class_ids(target)
        if class_ids is not None:
            kwargs["classes"] = class_ids
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        # Keep tensor conversions explicit and backend independent.
        xyxy = boxes.xyxy.detach().round().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        height, width = color.shape[:2]
        converted = []
        for x1, y1, x2, y2 in xyxy:
            converted.append([
                max(0, min(width, int(x1))),
                max(0, min(height, int(y1))),
                max(0, min(width, int(x2))),
                max(0, min(height, int(y2))),
            ])
        names = self.names
        return Detection(
            boxes=converted,
            scores=[float(score) for score in scores],
            labels=[names.get(int(index), str(index)) for index in classes],
        )

    def validate_target(self, target):
        """Fail before entering a live loop when a class selector is invalid."""
        self._class_ids(target)


@register("detector", "yolo")
def build_yolo_detector(cfg=None, hw=None, ctx=None, dependencies=None):
    import paths

    ycfg = cfg or {}
    weights = paths.PROJECT_ROOT / ycfg.get("weights", "models/yolo26/best.pt")
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {weights}")
    return YOLODetector(
        str(weights),
        confidence=ycfg.get("confidence", 0.25),
        iou=ycfg.get("iou", 0.7),
        imgsz=ycfg.get("imgsz", 640),
        device=ycfg.get("device"),
        max_det=ycfg.get("max_det", 100),
        classes=ycfg.get("classes"),
    )
