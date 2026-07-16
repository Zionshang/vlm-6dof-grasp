"""可插拔组件包。import 本包触发各子模块的 @register(注册生效)。

阶段 1:骨架占位。阶段 2 起陆续填入子模块,并在下方 import 使注册生效:
    from . import (cameras, depth, detectors, segmenters,
                   grasp_engines, selectors, executors, visualizers, robots)
"""
# 阶段 2 填:每加一个子模块在此 import,确保 @register 在 Manager 构建前执行。
from . import (cameras, robots, executors, grasp_engines,   # noqa: F401
               detectors, selectors, depth, segmenters, visualizers)
