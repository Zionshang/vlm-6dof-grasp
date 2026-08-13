"""Mutable latest-frame buffer shared by camera/depth orchestration."""
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class FrameContext:
    color: Optional[np.ndarray] = None      # RGB color-image grid
    depth: Optional[np.ndarray] = None      # 深度(米,color 视角)
    ir: Any = None                           # (ir1, ir2) 立体对(FFS 用)
    state: dict = field(default_factory=dict)   # 自由状态(quit flag、当前 prompt 等)
