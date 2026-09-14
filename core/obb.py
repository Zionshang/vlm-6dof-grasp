"""Backend-independent OBB and observation-plan data."""
from dataclasses import dataclass

import numpy as np


OBB_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0),
             (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))


@dataclass(frozen=True)
class OBB:
    center: np.ndarray
    rotation: np.ndarray       # columns are semantic object axes
    extents: np.ndarray
    point_count: int
    inlier_ratio: float = 1.0

    @property
    def vertices(self):
        signs = np.array([
            [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1],
        ], dtype=float)
        return self.center + (signs*self.extents*.5) @ self.rotation.T

    def contains(self, points, tolerance=1e-6):
        local = (np.asarray(points).reshape(-1, 3)-self.center) @ self.rotation
        return np.all(np.abs(local) <= self.extents*.5+tolerance, axis=1)


@dataclass(frozen=True)
class ViewPlan:
    label: str
    face: str
    angle: float
    visibility: float
    center: np.ndarray
    ee_poses: tuple
    reason: str = ""
