"""感知层:可插拔的检测(Detector)与分割(Segmenter)。

通过 registry 配置驱动:实现者用 @register_detector / @register_segmenter
注册工厂,pipeline 用 build_detector / build_segmenter 按 backend 名取实现,
换后端无需改动 pipeline。
"""
from .base import Detection, Detector, Segmenter, build_detector, build_segmenter
from . import detectors, segmenters  # 触发各 @register_* 装饰器,填充 registry

__all__ = ["Detection", "Detector", "Segmenter", "build_detector", "build_segmenter"]
