"""EconomicGrasp 抓取引擎组件(包装 economic_grasp.inference.EconomicGraspInference)。

intrinsic 取自 camera 组件的 color 内参(或回退 hw.camera_matrix);
factor_depth 取自 depth 组件声明(FFS=1 / raw=1000),消除硬编码不一致。
依赖 camera、depth 组件先构建。
"""
import numpy as np
from registry import register


@register("grasp_engine", "economic", requires=("camera", "depth"))
def build_economic_grasp(cfg=None, hw=None, ctx=None, dependencies=None):
    from economic_grasp.inference import EconomicGraspInference
    import paths
    ROOT = paths.PROJECT_ROOT
    gcfg = cfg or {}

    cam = dependencies["camera"]
    if cam is not None and hasattr(cam, "color_fx"):
        K = np.array([[cam.color_fx, 0, cam.color_cx],
                      [0, cam.color_fy, cam.color_cy],
                      [0, 0, 1.0]])
    else:
        K = hw.camera_matrix

    depth = dependencies["depth"]
    factor = getattr(depth, "factor_depth", None)
    if factor is None:
        factor = hw.factor_depth if hw is not None else 1000

    return EconomicGraspInference(
        str(ROOT / gcfg.get("checkpoint", "economic_grasp/checkpoint/economicgrasp_epoch10.tar")),
        intrinsic=K, factor_depth=float(factor),
        use_collision=gcfg.get("use_collision", True),
    )
