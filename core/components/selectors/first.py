"""First 抓取选择器组件(复用 grasping.selector.FirstGraspSelector,跳过 VLM 取首个)。"""
from registry import register


@register("selector", "first")
def build_first_selector(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from grasping.selector import FirstGraspSelector
    return FirstGraspSelector()
