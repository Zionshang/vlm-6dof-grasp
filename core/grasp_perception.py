"""Shared detection, segmentation and grasp-candidate orchestration."""
from pathlib import Path

from grasp_geometry import expand_boxes, filter_grasps_by_orientation
from saver import save_seg_mask, save_vlm_boxes, try_save


def _run(name, call):
    try:
        return call()
    except Exception as exc:
        detail = str(exc).splitlines()
        raise RuntimeError(f"{name}: {detail[0] if detail else type(exc).__name__}") from exc


def retry(name, call, valid=lambda value: value is not None, empty="无有效结果"):
    """Try one fallible perception step three times."""
    for attempt in range(1, 4):
        try:
            result, reason = call(), empty
            if valid(result):
                return result
        except Exception as exc:
            reason = str(exc).splitlines()[0] or type(exc).__name__
        if attempt < 3:
            print(f"[重试] {name} {attempt}/3: {reason}")
    raise RuntimeError(f"{name}: {reason}")


def capture_rgbd(ctx, camera, depth, flush_count=None):
    """Capture one fresh RGB-D pair through any camera/depth plugins."""
    def capture_once():
        ctx.color = ctx.depth = ctx.ir = None
        capture = getattr(camera, "capture", None)
        if callable(capture):
            capture(ctx, discard_frames=flush_count)
        else:
            for _ in range(max(0, int(flush_count or 0)) + 1):
                camera.step(ctx)
        depth.step(ctx)
        return ctx.color, ctx.depth
    return retry("相机取帧", capture_once,
                 lambda frames: all(frame is not None for frame in frames),
                 "无有效RGB-D帧")


def detect_target(detector, color, prompt):
    return retry("目标检测", lambda: detector.detect(color, prompt),
                 lambda result: bool(result and result.boxes), "未检测到目标")


class GraspPerception:
    """Compose pluggable perception components without owning robot motion."""

    def __init__(self, ctx, camera, depth, detector, segmenter, grasp_engine,
                 selector, output_dir, config=None):
        self.ctx, self.camera, self.depth = ctx, camera, depth
        self.detector, self.segmenter = detector, segmenter
        self.grasp_engine, self.selector = grasp_engine, selector
        self.output_dir = Path(output_dir)
        self.mask = None
        cfg = config or {}
        self.predict_topk = int(cfg.get("predict_topk", 100))
        self.selector_topk = int(cfg.get("selector_topk", 8))
        self.filter_orientation = bool(cfg.get("filter_orientation", False))
        self.save_debug = bool(cfg.get("save_debug", False))

    @classmethod
    def from_manager(cls, manager, output_dir):
        roles = ("camera", "depth", "detector", "segmenter",
                 "grasp_engine", "selector")
        return cls(manager.ctx, *(manager.require(role) for role in roles),
                   output_dir, manager.app_config.get("pipeline"))

    def capture(self, flush_count=None):
        return capture_rgbd(self.ctx, self.camera, self.depth, flush_count)

    def detect(self, color, prompt):
        return detect_target(self.detector, color, prompt)

    def generate(self, color, depth, prompt, run_id=None):
        detection = self.detect(color, prompt)
        if self.save_debug:
            try_save("检测图", save_vlm_boxes, self.output_dir, color,
                     detection.boxes, run_id, "origin_vlm")

        boxes = expand_boxes(detection.boxes, color.shape)
        if self.save_debug:
            try_save("检测图", save_vlm_boxes, self.output_dir, color,
                     boxes, run_id)
        mask = retry("目标分割", lambda: self.segmenter.segment(color, boxes),
                     empty="无结果")
        self.mask = mask
        rgb_shape = color.shape[:2]
        if depth.shape != rgb_shape or mask.shape != rgb_shape:
            raise RuntimeError(
                "RGB/深度/mask未对齐: "
                f"rgb={rgb_shape}, depth={depth.shape}, mask={mask.shape}"
            )
        if self.save_debug:
            try_save("分割图", save_seg_mask, self.output_dir, mask, run_id)

        grasps, _ = retry(
            "抓取生成", lambda: self.grasp_engine.predict(
                color, depth, mask=mask, topk=self.predict_topk,
            ),
            lambda result: bool(result and result[0] is not None
                                and len(result[0]) > 0), "无结果",
        )
        if self.filter_orientation:
            grasps = filter_grasps_by_orientation(grasps, self.selector_topk)
        return grasps

    def select(self, color, grasps):
        if grasps is None or len(grasps) == 0:
            return None
        index, candidates = _run(
            "二维筛选失败", lambda: self.selector.select(
                color, grasps.translations, grasps.rotation_matrices,
                grasps.widths, grasps.depths, self.grasp_engine.intrinsic,
                top_k=self.selector_topk, output_dir=self.output_dir,
                mask=self.mask,
            ),
        )
        if not candidates or not 0 <= index < len(candidates):
            raise RuntimeError("二维筛选无候选")
        return candidates[index]
