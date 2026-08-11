"""VLM grasp-selector plugin."""
from registry import register


@register("selector", "vlm")
def build_vlm_selector(cfg=None, hw=None, ctx=None, dependencies=None):
    from grasp_selector import VLMSelector
    import paths
    ROOT = paths.PROJECT_ROOT
    scfg = cfg or {}
    return VLMSelector(
        model_name=scfg.get("model", "qwen3-vl:8b-instruct-q4_K_M"),
        prompts_dir=str(ROOT / "third_party/vlm/prompts"),
    )
