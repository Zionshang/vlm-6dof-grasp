"""Backend-neutral continuous OBB observation service."""
import time

import numpy as np

from grasp_geometry import expand_boxes
from grasp_perception import capture_rgbd, detect_target, retry
from transform import box_center_to_base


def _progress(name, state, done, total):
    width = 20
    filled = 0 if total <= 0 else int(width * min(done / total, 1.0))
    bar = "█" * filled + "░" * (width - filled)
    value = f"{done:.1f}" if isinstance(done, float) else f"{done:g}"
    print(f"\r\033[2K[{name}] |{bar}| {state} {value}/{total:g}",
          end="", flush=True)


class OBBObservation:
    def __init__(self, ctx, camera, depth, detector, segmenter, estimator,
                 fusion, config=None):
        self.ctx, self.camera, self.depth = ctx, camera, depth
        self.detector, self.segmenter = detector, segmenter
        self.estimator, self.fusion = estimator, fusion
        config = config or {}
        self.box_scale = float(config.get("box_scale", 1.25))
        self.warmup = float(config.get("obb_warmup_seconds", 3))
        self.timeout = float(config.get("obb_collection_timeout", 30))
        self.last_rgb = None
        self.last_depth = None
        self.last_detection_box = None
        self.last_expanded_box = None
        self.last_mask = None

    @classmethod
    def from_manager(cls, manager):
        roles = ("camera", "depth", "detector", "segmenter",
                 "obb_estimator", "obb_fusion")
        return cls(manager.ctx, *(manager.require(role) for role in roles),
                   manager.app_config.get("pipeline"))

    @property
    def intrinsic(self):
        return np.array([
            [self.camera.color_fx, 0, self.camera.color_cx],
            [0, self.camera.color_fy, self.camera.color_cy],
            [0, 0, 1],
        ])

    def debug_frame(self):
        """Return copies of the last frame that produced an OBB estimate."""
        if self.last_rgb is None:
            raise RuntimeError("没有可保存的OBB观测帧")
        return (self.last_rgb.copy(), self.last_depth.copy(),
                self.last_detection_box.copy(), self.last_expanded_box.copy(),
                self.last_mask.copy())

    def collect(self, target, ee_pose, name="OBB", flush_frames=None):
        """Warm up, reject jumps, and return one stable base-frame group."""
        self.estimator.reset()
        self.fusion.reset()
        (self.last_rgb, self.last_depth, self.last_detection_box,
         self.last_expanded_box, self.last_mask) = 5*(None,)
        flush_frames = 0 if flush_frames is None else flush_frames
        started = time.monotonic()
        last_reason = ""
        while (not self.fusion.complete
               and time.monotonic()-started < self.timeout):
            try:
                color, depth = capture_rgbd(
                    self.ctx, self.camera, self.depth, flush_frames,
                )
                detection = detect_target(self.detector, color, target)
                indices = [i for i, label in enumerate(detection.labels)
                           if label == target]
                if not indices:
                    raise RuntimeError(f"未检测到{target}")
                scores = getattr(detection, "scores", ())
                index = max(
                    indices,
                    key=lambda i: scores[i] if i < len(scores) else 0,
                )
                box = expand_boxes(
                    [detection.boxes[index]], color.shape, self.box_scale,
                )
                mask = self.segmenter.segment(color, box)
                obb = self.estimator.estimate(depth, mask, label=target)
                self.last_rgb = color
                self.last_depth = depth
                self.last_detection_box = np.asarray(
                    detection.boxes[index]).copy()
                self.last_expanded_box = np.asarray(box[0]).copy()
                self.last_mask = np.asarray(mask).copy()
                elapsed = time.monotonic()-started
                if elapsed < self.warmup:
                    _progress(name, "预热", elapsed, self.warmup)
                    continue
                accepted = self.fusion.add(obb, ee_pose)
                last_reason = getattr(self.fusion, "last_reason", "")
                state = "稳定" if accepted else f"重建窗口 {last_reason}"
                _progress(name, state, len(self.fusion.samples),
                          self.fusion.target_samples)
            except Exception as exc:
                last_reason = str(exc).splitlines()[0] or type(exc).__name__
                self.fusion.reset()
                _progress(name, f"跳过帧 {last_reason}", len(self.fusion.samples),
                          self.fusion.target_samples)
        if not self.fusion.complete:
            print()
            raise RuntimeError(
                f"{name}稳定OBB不足: "
                f"{len(self.fusion.samples)}/{self.fusion.target_samples}; "
                f"最近状态: {last_reason or '采样时间不足'}"
            )
        print()
        return self.fusion.samples

    def locate_box_center(self, target, ee_pose, hand_eye_r, hand_eye_t,
                          flush_frames=None):
        """Locate a target from its detection-box centre depth."""
        intrinsic = self.intrinsic

        def locate_once():
            color, depth = capture_rgbd(
                self.ctx, self.camera, self.depth, flush_frames,
            )
            detection = self.detector.detect(color, target)
            if not detection or not detection.boxes:
                raise RuntimeError("未检测到目标")
            return box_center_to_base(
                depth, detection.boxes[0], intrinsic, ee_pose,
                hand_eye_r, hand_eye_t,
            )

        return retry("目标定位", locate_once, empty="目标框中心无有效深度")
