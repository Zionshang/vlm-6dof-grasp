"""FrameContext:组件间共享的最新数据(解耦传递)。

Node(camera)写 color/depth/ir;Agent(depth/detector/grasp)读 ctx 再写 ctx;
Handler(visualizer/executor)读 ctx。组件不互相直接依赖,统一经 ctx 传递。
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class FrameContext:
    color: Optional[np.ndarray] = None      # RGB(对齐到 depth 视角)
    depth: Optional[np.ndarray] = None      # 深度(米,color 视角)
    ir: Any = None                           # (ir1, ir2) 立体对(FFS 用)
    grasps: Any = None                       # 最新抓取(GraspGroup 或 None)
    prompt: str = "mug"
    run_id: Optional[str] = None
    state: dict = field(default_factory=dict)   # 自由状态(quit flag、当前 prompt 等)
