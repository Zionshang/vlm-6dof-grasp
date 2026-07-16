"""共用抓取动作链:detect → segment → grasp → select → convert_new → executor.run_sequence。

keyboard_grasp / lcm_grasp 两个 Handler 共用:事件触发 → run(color, depth, prompt)。
逻辑等价于原 run_realtime.action_grasp + run_grasp_lcm.execute_grasp + core/pipeline.run 的感知段。
"""
import numpy as np
import cv2
from transform import convert_new
from grasp_executor import GraspStep


def _expand_boxes(boxes, shape, scale=1.5):
    h, w = shape[:2]
    return [[max(0, int((x1 + x2) / 2 - (x2 - x1) * scale / 2)),
             max(0, int((y1 + y2) / 2 - (y2 - y1) * scale / 2)),
             min(w, int((x1 + x2) / 2 + (x2 - x1) * scale / 2)),
             min(h, int((y1 + y2) / 2 + (y2 - y1) * scale / 2))] for x1, y1, x2, y2 in boxes]


# 默认抓取执行序列(原 run_realtime.REALTIME_STEPS,改动需谨慎)
DEFAULT_STEPS = [
    GraspStep("approach", gripper="max",    offset=(0.0, 0.0, 0.05), preview=1.3, wait=1.5),
    GraspStep("reach",    gripper="max",    preview=0.5, wait=0.7),
    GraspStep("grasp",    gripper="target", preview=0.5, wait=0.8),
    GraspStep("lift",     gripper="target", offset=(0.0, 0.0, 0.06), preview=0.5, wait=0.8),
    GraspStep("home",     gripper="target", use_home_pose=True, preview=1.5, wait=1.5),
    GraspStep("reopen",   gripper="max",    use_home_pose=True, preview=0.5),
]


class GraspAction:
    """完整抓取链。组件由 GraspManager 提供(构造时传入)。"""

    def __init__(self, detector, segmenter, grasp_engine, selector, executor, robot, hw,
                 steps=None, keep_topk=8):
        self.detector = detector
        self.segmenter = segmenter
        self.grasp_engine = grasp_engine
        self.selector = selector
        self.executor = executor
        self.robot = robot
        self.hw = hw
        self.steps = steps or DEFAULT_STEPS
        self.keep_topk = keep_topk

    def run(self, color, depth, prompt):
        """返回 (success: bool, reason: str)。"""
        det = self.detector.detect(color, prompt)
        if not det or not det.boxes:
            return False, f"no detection for '{prompt}'"

        boxes = _expand_boxes(det.boxes, color.shape)
        mask = self.segmenter.segment(color, boxes)
        if mask is None:
            return False, "no mask"
        if mask.shape != depth.shape:
            mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0

        gg, _ = self.grasp_engine.predict(color, depth, mask=mask, topk=100)
        if len(gg) == 0:
            return False, "no grasp"

        idx, candidates = self.selector.select(
            color, gg.translations, gg.rotation_matrices, gg.widths,
            self.grasp_engine.intrinsic, top_k=self.keep_topk)
        sel = candidates[idx]

        curr = self.robot.get_state()["ee_pose"]
        arm_cmd = convert_new(np.array(sel["translation"]), np.array(sel["rotation"]),
                              curr, self.hw.hand_eye_r, self.hw.hand_eye_t)
        x, y, z = arm_cmd[:3]
        if not self.hw.in_workspace(x, y, z):
            return False, f"out of workspace ({x:.2f},{y:.2f},{z:.2f})"

        target_width = max(0.0, sel["width"] - 0.05)
        return self.executor.run_sequence(arm_cmd, target_width, self.steps)
