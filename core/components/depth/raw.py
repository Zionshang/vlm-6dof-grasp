"""Raw 深度组件:把相机硬件 depth(z16 mm)转成米(color 视角)。

step 读 ctx.depth(mm)→ 写 ctx.depth(米)。声明 factor_depth=1000 供 grasp_engine。
"""
from registry import register


@register("depth", "raw")
def build_raw_depth(cfg=None, hw=None, ctx=None, dependencies=None):
    class RawDepth:
        factor_depth = 1.0      # step 已把 ctx.depth 转成米,grasp_engine 用 factor=1

        def step(self, ctx):
            if ctx.depth is not None:
                ctx.depth = (ctx.depth / 1000.0).astype("float32")   # mm → 米
    return RawDepth()
