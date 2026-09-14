"""Robust, size-prior OBB fitting for masked depth points."""
from itertools import permutations

import numpy as np
from obb import OBB
from registry import register


def _erode(mask, iterations):
    """Tiny dependency-free 3x3 binary erosion."""
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, constant_values=False)
        result = np.logical_and.reduce([
            padded[dy:dy + result.shape[0], dx:dx + result.shape[1]]
            for dy in range(3) for dx in range(3)
        ])
    return result


def _proper(matrix):
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _right_handed(rotation, axis=-1):
    if np.linalg.det(rotation) < 0:
        rotation = rotation.copy()
        rotation[:, axis] *= -1
    return rotation


class PointCloudOBBEstimator:
    """Fit a prior-bounded OBB around the densest target points."""

    def __init__(self, intrinsic, factor_depth=1.0, min_points=100,
                 max_depth=None, robust=False, priors=None, erosion=2,
                 outlier_sigma=5.0, margin=.003, smoothing=.5,
                 max_fit_points=2000, size_tolerance=.25):
        self.intrinsic = np.asarray(intrinsic, dtype=float)
        if self.intrinsic.shape != (3, 3):
            raise ValueError("OBB intrinsic must be a 3x3 camera matrix")
        self.factor_depth = float(factor_depth)
        if self.factor_depth <= 0:
            raise ValueError("OBB factor_depth must be positive")
        self.min_points, self.max_depth = int(min_points), max_depth
        self.robust, self.erosion = bool(robust), int(erosion)
        self.outlier_sigma, self.margin = float(outlier_sigma), float(margin)
        self.smoothing = float(smoothing)
        self.max_fit_points = int(max_fit_points)
        self.size_tolerance = float(size_tolerance)
        self.priors = {
            name: np.asarray(size, dtype=float)
            for name, size in (priors or {}).items()
        }
        self._previous = {}

    def points(self, depth, mask=None, color=None):
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError("OBB depth must be a 2-D image")
        selected = (np.ones(depth.shape, bool) if mask is None
                    else np.asarray(mask, dtype=bool))
        if selected.shape != depth.shape:
            raise ValueError(
                f"OBB mask/depth shape mismatch: {selected.shape}, {depth.shape}"
            )
        if mask is not None and self.erosion:
            eroded = _erode(selected, self.erosion)
            if np.count_nonzero(eroded) >= self.min_points:
                selected = eroded
        valid = selected & np.isfinite(depth) & (depth > 0)
        if self.max_depth is not None:
            valid &= depth <= float(self.max_depth) * self.factor_depth

        v, u = np.nonzero(valid)
        z = depth[v, u].astype(float) / self.factor_depth
        if len(z) >= self.min_points and self.outlier_sigma > 0:
            median = np.median(z)
            mad = np.median(np.abs(z - median))
            if mad > 0:
                keep = np.abs(z - median) <= self.outlier_sigma * 1.4826 * mad
                u, v, z = u[keep], v[keep], z[keep]
        if len(z) < self.min_points:
            raise ValueError(f"OBB有效点数不足: {len(z)} < {self.min_points}")

        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        xyz = np.stack([(u-cx)*z/fx, (v-cy)*z/fy, z], axis=1)
        rgb = None if color is None else np.asarray(color)[v, u]
        return xyz, rgb

    def _geometric_fit(self, points):
        """Use Open3D's minimal box when available; PCA is a portable fallback."""
        if len(points) > self.max_fit_points:
            indices = np.linspace(
                0, len(points)-1, self.max_fit_points, dtype=int,
            )
            points = points[indices]
        try:
            import open3d as o3d
            cloud = o3d.utility.Vector3dVector(points)
            factory = getattr(
                o3d.geometry.OrientedBoundingBox,
                "create_from_points_minimal",
                o3d.geometry.OrientedBoundingBox.create_from_points,
            )
            box = factory(cloud, robust=self.robust)
            return np.asarray(box.R), np.asarray(box.extent)
        except (ImportError, AttributeError, RuntimeError):
            return self._pca_fit(points)

    @staticmethod
    def _pca_fit(points):
        _, rotation = np.linalg.eigh(np.cov(points, rowvar=False))
        rotation = _right_handed(rotation[:, ::-1])
        return rotation, np.ptp(points @ rotation, axis=0)

    def _semantic_axes(self, rotation, extents, label):
        prior = self.priors.get(label)
        if prior is None:
            order = np.argsort(extents)[::-1]
        else:
            order = min(permutations(range(3)), key=lambda p: np.sum(
                np.log(np.maximum(extents[list(p)], 1e-5) / prior) ** 2
            ))
        rotation = rotation[:, order]
        previous = self._previous.get(label)
        if previous is not None:
            dots = np.sum(rotation * previous.rotation, axis=0)
            rotation *= np.where(dots < 0, -1., 1.)
            rotation = _right_handed(rotation, np.argmin(np.abs(dots)))
            rotation = _proper(
                (1-self.smoothing)*previous.rotation + self.smoothing*rotation
            )
        else:
            for axis in range(3):
                dominant = np.argmax(np.abs(rotation[:, axis]))
                if rotation[dominant, axis] < 0:
                    rotation[:, axis] *= -1
            rotation = _right_handed(rotation)
        return rotation

    @staticmethod
    def _dense_bounds(values, width):
        """Return the densest interval no wider than ``width``."""
        values = np.sort(values)
        left = best_left = best_right = 0
        for right in range(len(values)):
            while values[right] - values[left] > width:
                left += 1
            if right-left > best_right-best_left:
                best_left, best_right = left, right
        return values[best_left], values[best_right]

    def _body_points(self, points, rotation, prior):
        local = points @ rotation
        bounds = np.array([
            self._dense_bounds(local[:, axis], prior[axis])
            for axis in range(3)
        ])
        center = bounds.mean(axis=1)
        selected = np.all(
            np.abs(local-center) <= prior/2+self.margin, axis=1,
        )
        return points[selected]

    def _body_fit(self, points, seeds, label):
        """Choose the densest seed, then iteratively refit its body."""
        prior = self.priors[label]
        body, rotation = max(
            ((self._body_points(points, seed, prior), seed) for seed in seeds),
            key=lambda result: len(result[0]),
        )
        for _ in range(5):
            if len(body) < self.min_points:
                break
            raw_rotation, raw_extents = self._pca_fit(body)
            rotation = self._semantic_axes(raw_rotation, raw_extents, label)
            body = self._body_points(points, rotation, prior)
        return body, rotation

    def fit(self, points, label=None):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) < self.min_points:
            raise ValueError(f"OBB有效点数不足: {len(points)} < {self.min_points}")
        raw_rotation, raw_extents = self._geometric_fit(points)
        rotation = self._semantic_axes(raw_rotation, raw_extents, label)
        prior = self.priors.get(label)
        body = points
        if prior is not None:
            pca_rotation, pca_extents = self._pca_fit(points)
            seeds = [rotation, self._semantic_axes(
                pca_rotation, pca_extents, label,
            )]
            previous = self._previous.get(label)
            if previous is not None:
                seeds.append(previous.rotation)
            body, rotation = self._body_fit(points, seeds, label)
        local = body @ rotation
        if prior is None:
            low, high = local.min(axis=0), local.max(axis=0)
        else:
            low = np.quantile(local, .01, axis=0)
            high = np.quantile(local, .99, axis=0)
        span = high-low

        size = span + 2*self.margin
        if prior is not None:
            # All three dimensions follow the observation while the physical
            # prior prevents incomplete depth or outliers producing huge boxes.
            size = np.clip(
                size, prior*(1-self.size_tolerance),
                prior*(1+self.size_tolerance),
            )

        # Place the constrained box where it covers the densest target body.
        bounds = np.array([
            self._dense_bounds(local[:, axis], size[axis])
            for axis in range(3)
        ])
        center_local = bounds.mean(axis=1)
        center = center_local @ rotation.T

        previous = self._previous.get(label)
        if previous is not None:
            alpha = self.smoothing
            center = (1-alpha)*previous.center + alpha*center
            size = (1-alpha)*previous.extents + alpha*size
        inside = np.all(
            np.abs((body-center) @ rotation) <= size/2 + self.margin, axis=1,
        )
        result = OBB(center, rotation, size, len(body), float(inside.mean()))
        self._previous[label] = result
        return result

    def estimate(self, depth, mask=None, label=None):
        points, _ = self.points(depth, mask)
        return self.fit(points, label)

    def reset(self):
        self._previous.clear()


@register("obb_estimator", "point_cloud", requires=("camera", "depth"))
def build_point_cloud_obb(cfg=None, hw=None, ctx=None, dependencies=None):
    camera, depth = dependencies["camera"], dependencies["depth"]
    cfg = cfg or {}
    intrinsic = np.array([
        [camera.color_fx, 0, camera.color_cx],
        [0, camera.color_fy, camera.color_cy], [0, 0, 1],
    ])
    return PointCloudOBBEstimator(
        intrinsic, factor_depth=getattr(depth, "factor_depth", 1.0),
        **{key: cfg[key] for key in (
            "min_points", "max_depth", "robust", "priors", "erosion",
            "outlier_sigma", "margin", "smoothing",
            "max_fit_points", "size_tolerance",
        ) if key in cfg},
    )
