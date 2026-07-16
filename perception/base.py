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
    """检测器接口:图像 + 文本提示 -> 检测结果。VLM / YOLO / GroundingDINO 等可实现。

    生命周期方法(unload/warmup)用于后端资源管理(如 VLM 与 PyTorch 抢 VRAM);
    不需要这种管理的后端(如 YOLO)可空实现。
    """
    def detect(self, color: np.ndarray, prompt: str) -> Optional[Detection]: ...
    def unload(self) -> None: ...
    def warmup(self) -> None: ...


class Segmenter(Protocol):
    """分割器接口:图像 + 框 -> mask。FastSAM / SAM2 / bbox 等可实现。"""
    def segment(self, color: np.ndarray, boxes: List[List[int]]) -> Optional[np.ndarray]: ...


# =============================================================================
# Registry:配置驱动的可插拔(换后端不改 pipeline)
# =============================================================================
# backend 名 -> 工厂函数 factory(cfg, root, args=None) -> 实现
# 实现者用 @register_detector / @register_segmenter 注册自己的工厂;
# pipeline 只调 build_detector / build_segmenter,不 import 任何具体类。
_DETECTOR_REGISTRY = {}
_SEGMENTER_REGISTRY = {}


def register_detector(name):
    """装饰一个检测器工厂 factory(cfg, root, args=None) -> Detector,注册为 backend=name。"""
    def deco(factory):
        _DETECTOR_REGISTRY[name] = factory
        return factory
    return deco


def register_segmenter(name):
    """装饰一个分割器工厂 factory(cfg, root, args=None) -> Segmenter,注册为 backend=name。"""
    def deco(factory):
        _SEGMENTER_REGISTRY[name] = factory
        return factory
    return deco


def build_detector(backend, cfg, root, args=None):
    """按 backend 名从 registry 取检测器工厂并构建。换检测算法只需注册新工厂 + 改配置。"""
    if backend not in _DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector backend '{backend}'. Registered: {list(_DETECTOR_REGISTRY)}")
    return _DETECTOR_REGISTRY[backend](cfg=cfg, root=root, args=args)


def build_segmenter(backend, cfg, root, args=None):
    """按 backend 名从 registry 取分割器工厂并构建。换分割算法只需注册新工厂 + 改配置。"""
    if backend not in _SEGMENTER_REGISTRY:
        raise ValueError(f"Unknown segmenter backend '{backend}'. Registered: {list(_SEGMENTER_REGISTRY)}")
    return _SEGMENTER_REGISTRY[backend](cfg=cfg, root=root, args=args)
