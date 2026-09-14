"""Stable OBB filtering and multi-view fusion in the robot base frame."""
import numpy as np

from obb import OBB
from registry import register
from transform import camera_frame_to_base


def _proper(matrix):
    u, _, vt = np.linalg.svd(matrix)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def _aligned(rotation, reference):
    result = rotation.copy()
    dots = np.sum(result*reference, axis=0)
    result *= np.where(dots < 0, -1., 1.)
    if np.linalg.det(result) < 0:
        result[:, np.argmin(np.abs(dots))] *= -1
    return result


class StableOBBFusion:
    def __init__(self, hand_eye_r, hand_eye_t, target_samples=10,
                 center_jump=.04, extent_jump=.25, angle_jump_deg=15,
                 min_inlier_ratio=.9):
        self.r_ec = np.asarray(hand_eye_r, float)
        self.t_ec = np.asarray(hand_eye_t, float)
        self.target_samples = int(target_samples)
        self.center_jump, self.extent_jump = center_jump, extent_jump
        self.angle_jump = np.deg2rad(angle_jump_deg)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.reset()

    def reset(self):
        self._samples = []
        self.last_reason = ""

    @property
    def samples(self):
        return tuple(self._samples)

    @property
    def complete(self):
        return len(self._samples) >= self.target_samples

    def to_base(self, obb, ee_pose):
        center, rotation = camera_frame_to_base(
            obb.center, obb.rotation, ee_pose, self.r_ec, self.t_ec,
        )
        return OBB(
            center, rotation, obb.extents.copy(),
            obb.point_count, obb.inlier_ratio,
        )

    def add(self, obb, ee_pose):
        self.last_reason = ""
        if obb.inlier_ratio < self.min_inlier_ratio:
            self._samples = []
            self.last_reason = (f"覆盖率{obb.inlier_ratio:.0%}"
                                f"<{self.min_inlier_ratio:.0%}")
            return False
        sample = self.to_base(obb, ee_pose)
        if self._samples:
            reference = self.fuse(self._samples)
            sample = OBB(sample.center, _aligned(sample.rotation,
                         reference.rotation), sample.extents,
                         sample.point_count, sample.inlier_ratio)
            if not self._compatible(sample, reference):
                # A bad first frame must not anchor the entire collection.
                # Start a new consecutive-stability window from the new frame.
                self._samples = [sample]
                self.last_reason = "中心/尺寸/朝向变化，重新采样"
                return False
        self._samples.append(sample)
        return True

    def _compatible(self, sample, reference):
        relative = np.maximum(reference.extents, 1e-6)
        angles = np.arccos(np.clip(np.abs(np.diag(
            reference.rotation.T @ sample.rotation)), -1., 1.))
        return (np.linalg.norm(sample.center-reference.center)
                <= self.center_jump
                and np.max(np.abs(sample.extents-reference.extents)/relative)
                <= self.extent_jump
                and np.max(angles) <= self.angle_jump)

    def fuse(self, samples=None):
        samples = tuple(self._samples if samples is None else samples)
        if not samples:
            raise ValueError("没有可融合的OBB")
        reference = samples[0].rotation
        rotations = [_aligned(item.rotation, reference) for item in samples]
        return OBB(
            np.median([item.center for item in samples], axis=0),
            _proper(np.mean(rotations, axis=0)),
            np.median([item.extents for item in samples], axis=0),
            sum(item.point_count for item in samples),
            float(np.mean([item.inlier_ratio for item in samples])),
        )


@register("obb_fusion", "stable")
def build_stable_fusion(cfg=None, hw=None, ctx=None, dependencies=None):
    if hw is None:
        raise ValueError("obb_fusion requires hardware hand-eye calibration")
    cfg = cfg or {}
    return StableOBBFusion(
        hw.hand_eye_r, hw.hand_eye_t,
        **{key: cfg[key] for key in (
            "target_samples", "center_jump", "extent_jump", "angle_jump_deg",
            "min_inlier_ratio",
        ) if key in cfg},
    )
