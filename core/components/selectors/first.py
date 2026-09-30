"""Mask-PCA candidate selector component."""
import numpy as np

from perception_errors import PerceptionEmptyError
from registry import register
from saver import save_2d_grasp, try_save

from .image_candidates import prepare_image_candidates, project_grasp


class PCASelector:
    def __init__(self, top_k=8, round_ratio=1.3):
        self.top_k = int(top_k)
        self.round_ratio = float(round_ratio)

    def select(self, color, grasps, intrinsic, output_dir, mask=None):
        images, candidates = prepare_image_candidates(
            color, grasps, intrinsic, output_dir, self.top_k,
        )
        if not candidates:
            raise PerceptionEmptyError("二维几何筛选无候选")
        index = self._choose(candidates, mask, intrinsic)
        try_save("2D抓取图", save_2d_grasp, output_dir, images)
        try_save("PCA结果图", save_2d_grasp, output_dir,
                 images[index:index + 1], subdir="pca_select")
        return candidates[index]

    def _choose(self, candidates, mask, intrinsic):
        if mask is None:
            return 0
        xy = np.argwhere(mask > 0)[:, ::-1].astype(float)
        if len(xy) < 3:
            return 0
        values, vectors = np.linalg.eigh(np.cov(xy, rowvar=False))
        ratio = np.sqrt(values[1] / max(values[0], 1e-9))
        axis = np.array([1.0, 0.0]) if ratio <= self.round_ratio else vectors[:, 0]
        scores = []
        for item in candidates:
            points = project_grasp(
                np.asarray(item["translation"]), np.asarray(item["rotation"]),
                item["width"], item["depth"], intrinsic,
            )
            base = points[2] - points[1]
            scores.append(abs(base @ axis) / max(np.linalg.norm(base), 1e-9))
        return int(np.argmax(scores))


@register("candidate_selector", "pca")
def build_pca_selector(cfg=None, hw=None, ctx=None, dependencies=None):
    cfg = cfg or {}
    return PCASelector(cfg.get("top_k", 8), cfg.get("round_ratio", 1.3))
