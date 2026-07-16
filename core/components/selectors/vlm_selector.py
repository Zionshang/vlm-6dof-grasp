"""VLM 抓取选择器组件(复用 grasping.selector.VLMSelector)。"""
from registry import register


@register("selector", "vlm")
def build_vlm_selector(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from grasping.selector import VLMSelector
    import paths
    ROOT = paths.PROJECT_ROOT
    scfg = cfg or {}
    return VLMSelector(
        model_name=scfg.get("model", "qwen3-vl:8b-instruct-q4_K_M"),
        prompts_dir=str(ROOT / "vlm" / "prompts"),
    )
