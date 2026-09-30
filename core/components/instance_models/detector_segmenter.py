"""Compatibility adapter for legacy detector plus segmenter pipelines."""
import numpy as np

from grasp_geometry import expand_boxes
from instances import Instance
from registry import register


class DetectorSegmenter:
    def __init__(self, detector, segmenter, box_scale=1.25):
        self.detector, self.segmenter = detector, segmenter
        self.box_scale = float(box_scale)

    def validate_target(self, target):
        validate = getattr(self.detector, "validate_target", None)
        if callable(validate):
            validate(target)

    def predict(self, color, target=None):
        detection = self.detector.detect(color, target)
        if not detection or not detection.boxes:
            return []
        boxes = expand_boxes(detection.boxes, color.shape, self.box_scale)
        mask = self.segmenter.segment(color, boxes)
        if mask is None:
            return []
        index = int(np.argmax(detection.scores)) if detection.scores else 0
        score = detection.scores[index] if detection.scores else 1.0
        label = detection.labels[index] if detection.labels else str(target)
        return [Instance(detection.boxes[index], float(score), label,
                         np.asarray(mask, dtype=bool))]


@register("instance_model", "detector_segmenter",
          requires=("detector", "segmenter"))
def build_detector_segmenter(cfg=None, hw=None, ctx=None, dependencies=None):
    return DetectorSegmenter(
        dependencies["detector"], dependencies["segmenter"],
        (cfg or {}).get("box_scale", 1.25),
    )
