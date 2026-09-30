"""Backend-neutral 6-DOF grasp candidate containers."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspPose:
    """One scored gripper pose expressed in the input point-cloud frame."""

    pose: np.ndarray
    score: float
    width: float
    height: float
    depth: float

    @property
    def rotation_matrix(self):
        return self.pose[:3, :3]

    @property
    def translation(self):
        return self.pose[:3, 3]


class GraspPoses:
    """Array-backed grasp collection matching the shared pipeline contract."""

    def __init__(self, poses=None, scores=None, widths=0.0, heights=0.02,
                 depths=0.0):
        poses = (np.empty((0, 4, 4), dtype=np.float32) if poses is None
                 else np.asarray(poses, dtype=np.float32))
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"Grasp poses must have shape (N, 4, 4), got {poses.shape}")
        fields = [self._field(value, len(poses), default)
                  for value, default in ((scores, 0.0), (widths, None),
                                         (heights, None), (depths, None))]
        valid = np.isfinite(poses).all(axis=(1, 2))
        for field in fields:
            valid &= np.isfinite(field)
        self.poses, self.scores, self.widths, self.heights, self.depths = (
            poses[valid], *(field[valid] for field in fields)
        )

    @staticmethod
    def _field(value, size, default=None):
        if value is None:
            value = default
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            return np.full(size, float(array), dtype=np.float32)
        array = array.reshape(-1)
        if len(array) != size:
            raise ValueError(f"Grasp field has {len(array)} values for {size} poses")
        return array

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return GraspPose(
                self.poses[index], float(self.scores[index]),
                float(self.widths[index]), float(self.heights[index]),
                float(self.depths[index]),
            )
        return type(self)(
            self.poses[index], self.scores[index], self.widths[index],
            self.heights[index], self.depths[index],
        )

    @property
    def rotation_matrices(self):
        return self.poses[:, :3, :3]

    @property
    def translations(self):
        return self.poses[:, :3, 3]
