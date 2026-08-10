"""First selector plugin: geometric rejection, then first survivor."""
from registry import register


@register("selector", "first")
def build_first_selector(cfg=None, hw=None, ctx=None, dependencies=None):
    from grasp_selector import FirstGraspSelector
    return FirstGraspSelector()
