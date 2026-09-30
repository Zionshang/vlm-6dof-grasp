"""Common instance-segmentation result types."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    box: list[int]
    score: float
    label: str
    mask: np.ndarray
