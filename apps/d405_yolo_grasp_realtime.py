"""D405 live YOLO preview; press 1 for one FFS/SAM/grasp update."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

import components.detectors.yolo  # noqa: F401 - register YOLO for this app
from context import FrameContext
from grasp_geometry import expand_boxes, filter_grasps_by_orientation
from manager import GraspManager
from worker import AsyncWorker


ROOT = paths.PROJECT_ROOT
LIVE_WINDOW = "D405 YOLO Live  [1] grasp  [Q/ESC] quit"
RESULT_WINDOW = "Triggered Detection + EfficientSAM"


def draw_detection(color, detection, mask=None):
    """Return a BGR preview with optional mask and detection boxes."""
    image = color.copy()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        image[mask] = (
            image[mask].astype(np.float32) * 0.35
            + np.array([0, 255, 0], dtype=np.float32) * 0.65
        ).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if detection is None:
        return image
    for index, box in enumerate(detection.boxes):
        x1, y1, x2, y2 = map(int, box)
        label = (detection.labels[index]
                 if index < len(detection.labels) else "object")
        if index < len(detection.scores):
            label += f" {detection.scores[index]:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return image


class KeyGraspHandler:
    """Serialize live YOLO and triggered grasp inference in one CUDA worker."""

    def __init__(self, manager, target):
        self.ctx = manager.ctx
        self.camera = manager.require("camera")
        self.depth = manager.require("depth")
        self.detector = manager.require("detector")
        self.segmenter = manager.require("segmenter")
        self.grasp_engine = manager.require("grasp_engine")
        self.visualizer = manager.require("visualizer")
        self.target = target
        self.detector.validate_target(target)

        cfg = manager.app_config.get("pipeline") or {}
        self.box_scale = float(cfg.get("box_scale", 1.25))
        self.predict_topk = int(cfg.get("predict_topk", 100))
        self.visualize_topk = int(cfg.get("visualize_topk", 20))
        self.min_depth_points = int(cfg.get("min_depth_points", 100))
        self.filter_orientation = bool(cfg.get("filter_orientation", True))

        self.worker = AsyncWorker(self._run_job, name="yolo-grasp-worker")
        self.job_active = False
        self.grasp_pending = False
        self.last_detection = None

    def _run_job(self, job):
        kind, color, ir = job
        started = time.monotonic()
        detection = mask = depth = grasps = None
        try:
            detection = self.detector.detect(color, self.target)
            if kind == "detect" or detection is None:
                status = "未检测到目标" if detection is None else "检测更新"
                return (kind, color, detection, None, None, None, status,
                        time.monotonic() - started)

            local = FrameContext(color=color, ir=ir)
            self.depth.step(local)
            depth = local.depth
            if local.depth is None:
                raise RuntimeError("无有效 FFS 深度")
            boxes = expand_boxes(detection.boxes, color.shape, self.box_scale)
            mask = self.segmenter.segment(color, boxes)
            if mask is None:
                raise RuntimeError("EfficientSAM 无结果")
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != local.depth.shape:
                raise RuntimeError(
                    f"RGB/FFS/mask 未对齐: {color.shape[:2]}/"
                    f"{local.depth.shape}/{mask.shape}"
                )
            valid = mask & (local.depth > 0) & np.isfinite(local.depth)
            if np.count_nonzero(valid) < self.min_depth_points:
                raise RuntimeError("目标区域有效深度点不足")

            grasps, _ = self.grasp_engine.predict(
                color, local.depth, mask=mask, topk=self.predict_topk,
            )
            if grasps is None or len(grasps) == 0:
                raise RuntimeError("无抓取候选")
            grasps = (filter_grasps_by_orientation(grasps, self.visualize_topk)
                      if self.filter_orientation
                      else grasps[:self.visualize_topk])
            labels = ",".join(detection.labels)
            status = f"{labels}: boxes={len(boxes)}, grasps={len(grasps)}"
            return (kind, color, detection, mask, local.depth, grasps,
                    status, time.monotonic() - started)
        except Exception as exc:
            message = str(exc).splitlines()[0] or type(exc).__name__
            return (kind, color, detection, mask, depth, grasps,
                    f"失败: {message}", time.monotonic() - started)

    def _take_result(self):
        result = self.worker.take()
        if result is None:
            return
        self.job_active = False
        kind, color, detection, mask, depth, grasps, status, elapsed = result
        if kind == "detect" or detection is not None:
            self.last_detection = detection
        if kind == "grasp":
            cv2.imshow(RESULT_WINDOW, draw_detection(color, detection, mask))
            self.visualizer.update_grasps(grasps)
            if depth is not None:
                self.ctx.depth = depth
                self.visualizer.update_cloud(color, depth)
            print(f"[抓取结果] {status} | {elapsed:.2f}s")

    def step(self, ctx, components):
        self.camera.step(ctx)
        self._take_result()

        if ctx.color is not None:
            cv2.imshow(LIVE_WINDOW, draw_detection(ctx.color, self.last_detection))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("1"):
            self.grasp_pending = True
            message = ("当前推理结束后执行" if self.job_active
                       else "执行一次抓取检测")
            print(f"[按键] {message}")
        elif key in (ord("q"), 27):
            ctx.state["quit"] = True

        if not self.job_active and ctx.color is not None and ctx.ir is not None:
            kind = "grasp" if self.grasp_pending else "detect"
            if kind == "grasp":
                self.grasp_pending = False
                print("[流程] FFS → YOLO → EfficientSAM → EconomicGrasp")
            frame = (kind, ctx.color.copy(), tuple(x.copy() for x in ctx.ir))
            self.worker.submit(frame)
            self.job_active = True

        if not self.visualizer.poll():
            ctx.state["quit"] = True
        self.visualizer.render()

    def close(self):
        self.worker.stop()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="all",
                        help="YOLO class name/id, comma-separated, or all")
    parser.add_argument("--app-config",
                        default="config/apps/d405_yolo_grasp_realtime.yaml")
    parser.add_argument("--handshake-timeout", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    manager = GraspManager.from_yaml(ROOT / args.app_config, eager=False)
    try:
        roles = ("camera", "depth", "detector", "segmenter",
                 "grasp_engine", "visualizer")
        manager.initialize(roles)
        if not manager.handshake(timeout=args.handshake_timeout):
            raise RuntimeError(
                f"D405 handshake failed: {manager.handshake_error or 'timeout'}"
            )
        print("[就绪] 2D 窗口持续检测；聚焦 2D 窗口按 1 生成抓取，Q/ESC 退出")
        manager.run(KeyGraspHandler(manager, args.target))
    except BaseException:
        manager.release_resources()
        raise


if __name__ == "__main__":
    main()
