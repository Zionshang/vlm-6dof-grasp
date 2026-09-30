"""Shared detection, segmentation and grasp-candidate orchestration."""
from dataclasses import dataclass
from pathlib import Path

from depth_geometry import mask_depth_target
from perception_errors import PerceptionEmptyError
from saver import save_seg_mask, save_vlm_boxes, try_save
from target_confirmation import TargetConfirmer


GRASP_ATTEMPTS = 10


@dataclass(frozen=True)
class Observation:
    color: object
    depth: object
    box: list[int]
    score: float
    label: str
    mask: object


@dataclass(frozen=True)
class PerceptionResult:
    observation: Observation
    grasps: object
    stats: dict


def retry(name, call, valid=lambda value: value is not None, empty="无有效结果",
          attempts=3):
    """Retry expected empty results; propagate system and programming errors."""
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            result = call()
        except PerceptionEmptyError as exc:
            reason = str(exc).splitlines()[0] or type(exc).__name__
        else:
            if valid(result):
                return result
            reason = empty
        if attempt < attempts:
            print(f"[重试] {name} {attempt}/{attempts}: {reason}")
    raise PerceptionEmptyError(f"{name}: {reason}")


class GraspPerception:
    """Compose pluggable perception components without owning robot motion."""

    def __init__(self, ctx, camera, depth, instance_model, grasp_engine,
                 output_dir, config=None):
        self.ctx, self.camera, self.depth = ctx, camera, depth
        self.instance_model = instance_model
        self.grasp_engine = grasp_engine
        self.output_dir = Path(output_dir)
        cfg = config or {}
        self.predict_topk = int(cfg.get("predict_topk", 100))
        self.save_debug = bool(cfg.get("save_debug", False))
        self.locate_max_depth = float(cfg.get("locate_max_depth_m", .85))
        self.min_target_depth_points = int(
            cfg.get("min_target_depth_points", 100)
        )
        confirmation = cfg.get("confirmation") or {}
        self.max_frames = int(confirmation.get("max_frames", 40))
        self.confirmation = {
            "high_confidence": confirmation.get("high_confidence", .9),
            "high_frames": confirmation.get("high_frames", 5),
            "normal_confidence": confirmation.get("normal_confidence", .8),
            "normal_frames": confirmation.get("normal_frames", 10),
        }

    @classmethod
    def from_manager(cls, manager, output_dir):
        roles = ("camera", "depth", "instance_model", "grasp_engine")
        return cls(manager.ctx, *(manager.require(role) for role in roles),
                   output_dir, manager.app_config.get("pipeline"))

    def observe(self, prompt, flush_count=None, run_id=None, label=None):
        return self._observe(prompt, flush_count, run_id, label, False)[0]

    def observe_all(self, prompt, flush_count=None, run_id=None, label=None):
        """Return all target instances from one temporally confirmed frame."""
        return self._observe(prompt, flush_count, run_id, label, True)

    def _observe(self, prompt, flush_count, run_id, label, multiple):
        confirmer = TargetConfirmer(**self.confirmation)
        missing_color = empty_detection = confirmed_frames = missing_depth = 0
        peak_high = peak_normal = 0
        best_scores, last_scores = {}, {}
        for frame in range(self.max_frames):
            self.ctx.color = self.ctx.depth = self.ctx.ir = None
            self.camera.capture(
                self.ctx, discard_frames=flush_count if frame == 0 else 0,
            )
            if self.ctx.color is None:
                missing_color += 1
                confirmer.reset()
                continue
            instances = self.instance_model.predict(self.ctx.color, prompt)
            if not instances:
                empty_detection += 1
            last_scores = {}
            for item in instances:
                last_scores[item.label] = max(
                    last_scores.get(item.label, 0.0), item.score,
                )
                best_scores[item.label] = max(
                    best_scores.get(item.label, 0.0), item.score,
                )
            confirmed = (confirmer.update_all(instances) if multiple else
                         confirmer.update(instances))
            peak_high = max(peak_high, confirmer.high_count)
            peak_normal = max(peak_normal, confirmer.normal_count)
            if confirmed is None:
                continue
            confirmed_frames += 1
            selected, reason = confirmed
            selected = selected if multiple else [selected]
            self.depth.step(self.ctx)
            if self.ctx.depth is None:
                missing_depth += 1
                continue
            shape = self.ctx.color.shape[:2]
            if (self.ctx.depth.shape != shape or
                    any(item.mask.shape != shape for item in selected)):
                raise RuntimeError(
                    "RGB/深度/mask未对齐: "
                    f"rgb={shape}, depth={self.ctx.depth.shape}"
                )
            name = label or prompt
            scores = ", ".join(
                f"{item.label}:{item.score:.3f}" for item in selected
            )
            print(f"[确认] {name}: {reason}, {scores}")
            if self.save_debug:
                try_save("检测图", save_vlm_boxes, self.output_dir,
                         self.ctx.color, [item.box for item in selected],
                         run_id, "yolo_seg")
                for index, item in enumerate(selected):
                    suffix = (run_id if len(selected) == 1 or run_id is None
                              else f"{run_id}_{index}")
                    try_save("分割图", save_seg_mask, self.output_dir,
                             item.mask, suffix)
            return [Observation(
                self.ctx.color, self.ctx.depth, item.box,
                item.score, item.label, item.mask,
            ) for item in selected]
        scores = lambda values: (", ".join(
            f"{name}:{score:.3f}" for name, score in sorted(values.items())
        ) or "无")
        raise PerceptionEmptyError(
            f"{label or prompt}观察{self.max_frames}帧无有效结果: "
            f"无RGB={missing_color}, YOLO空={empty_detection}, "
            f"连续≥{self.confirmation['high_confidence']:.2f}最高="
            f"{peak_high}/{self.confirmation['high_frames']}, "
            f"连续≥{self.confirmation['normal_confidence']:.2f}最高="
            f"{peak_normal}/{self.confirmation['normal_frames']}, "
            f"最高置信度={scores(best_scores)}, "
            f"末帧={scores(last_scores)}, "
            f"确认后无深度={missing_depth}/{confirmed_frames}"
        )

    def target_point(self, observation):
        return mask_depth_target(
            observation.depth, observation.mask,
            self.grasp_engine.intrinsic, self.locate_max_depth,
            self.min_target_depth_points,
        )

    def generate(self, observation, label=None):
        color, depth, mask = (
            observation.color, observation.depth, observation.mask,
        )

        grasps, info = retry(
            "抓取生成", lambda: self.grasp_engine.predict(
                color, depth, mask=mask, topk=self.predict_topk,
            ),
            lambda result: bool(result and result[0] is not None
                                and len(result[0]) > 0), "无结果",
            attempts=GRASP_ATTEMPTS,
        )
        info = info or {}
        generated = int(info.get("generated_count", len(grasps)))
        target_rejected = int(info.get("target_rejected", 0))
        collision_rejected = int(info.get("collision_rejected", 0))
        prefix = f"{label}: " if label else ""
        print(
            f"[候选] {prefix}模型输出={generated}, "
            f"目标体积淘汰={target_rejected}, "
            f"场景碰撞淘汰={collision_rejected}, "
            f"最终={len(grasps)}（topk上限={self.predict_topk}）"
        )
        return PerceptionResult(observation, grasps, info)
