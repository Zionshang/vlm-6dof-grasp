"""感知层接口契约与共享数据结构。

这里只做两件事:
  1) Detection:检测结果的数据结构(检测器返回它,pipeline 消费它)。
  2) Detector / Segmenter:可插拔接口的"契约"——声明一个检测器/分割器
     必须实现什么方法。实现者(VLMDetector / FastSAMSegmenter 等)靠鸭子类型
     自动满足,无需继承;它的价值是让人/类型检查器一眼看清接口规范。

换感知后端(VLM <-> YOLO、FastSAM <-> 其他)只需新增一个实现上述协议的类。
"""
from dataclasses import dataclass, field
from typing import Protocol, Optional, List
import numpy as np


@dataclass
class Detection:
    """检测结果:像素框列表 [[x1, y1, x2, y2], ...]。"""
    boxes: List[List[int]]
    scores: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)


class Detector(Protocol):
    """检测器接口:图像 + 文本提示 -> 检测结果。"""
    def detect(self, color: np.ndarray, prompt: str) -> Optional[Detection]: ...


class Segmenter(Protocol):
    """分割器接口:图像 + 框 -> mask。FastSAM / SAM2 / bbox 等可实现。"""
    def segment(self, color: np.ndarray, boxes: List[List[int]]) -> Optional[np.ndarray]: ...
