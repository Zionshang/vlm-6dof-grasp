"""Shared detection, segmentation and grasp-candidate orchestration."""
from pathlib import Path

from grasp_geometry import expand_boxes, filter_grasps_by_orientation
from saver import save_seg_mask, save_vlm_boxes


class GraspPerception:
    """Compose pluggable perception components without owning robot motion."""

    def __init__(self, manager, output_dir, config=None):
        self.ctx = manager.ctx
        self.camera = manager.require("camera")
        self.depth = manager.require("depth")
        self.detector = manager.require("detector")
        self.segmenter = manager.require("segmenter")
        self.grasp_engine = manager.require("grasp_engine")
        self.selector = manager.require("selector")
        self.output_dir = Path(output_dir)
        cfg = config or manager.app_config.get("pipeline") or {}
        self.predict_topk = int(cfg.get("predict_topk", 100))
        self.selector_topk = int(cfg.get("selector_topk", 8))
        self.filter_orientation = bool(cfg.get("filter_orientation", False))
        self.save_debug = bool(cfg.get("save_debug", False))

    def capture(self, flush_count=None):
        self.ctx.color = self.ctx.depth = self.ctx.ir = None
        capture = getattr(self.camera, "capture", None)
        if callable(capture):
            capture(self.ctx, discard_frames=flush_count)
        else:
            # Compatibility for non-RealSense camera plugins.
            count = 0 if flush_count is None else int(flush_count)
            for _ in range(max(0, count) + 1):
                self.camera.step(self.ctx)
        self.depth.step(self.ctx)
        return self.ctx.color, self.ctx.depth

    def detect(self, color, prompt):
        return self.detector.detect(color, prompt)

    def generate(self, color, depth, prompt, run_id=None):
        detection = self.detect(color, prompt)
        if not detection or not detection.boxes:
            return None
        if self.save_debug:
            save_vlm_boxes(
                self.output_dir, color, detection.boxes, run_id, "origin_vlm",
            )

        boxes = expand_boxes(detection.boxes, color.shape)
        if self.save_debug:
            save_vlm_boxes(self.output_dir, color, boxes, run_id)
        mask = self.segmenter.segment(color, boxes)
        if mask is None:
            return None
        rgb_shape = color.shape[:2]
        if depth.shape != rgb_shape or mask.shape != rgb_shape:
            raise ValueError(
                "RGB, aligned depth, and mask must share the color image grid: "
                f"rgb={rgb_shape}, depth={depth.shape}, mask={mask.shape}"
            )
        if self.save_debug:
            save_seg_mask(self.output_dir, mask, run_id)

        grasps, _ = self.grasp_engine.predict(
            color, depth, mask=mask, topk=self.predict_topk,
        )
        if grasps is None or len(grasps) == 0:
            return None
        if self.filter_orientation:
            grasps = filter_grasps_by_orientation(grasps, self.selector_topk)
        return grasps

    def select(self, color, grasps):
        if grasps is None or len(grasps) == 0:
            return None
        index, candidates = self.selector.select(
            color, grasps.translations, grasps.rotation_matrices, grasps.widths,
            self.grasp_engine.intrinsic, top_k=self.selector_topk,
            output_dir=self.output_dir,
        )
        if not candidates or not 0 <= index < len(candidates):
            return None
        return candidates[index]

    def generate_and_select(self, color, depth, prompt, run_id=None):
        return self.select(color, self.generate(color, depth, prompt, run_id))
