"""VLM 检测器组件(复用 perception.detectors.VLMDetector,改用统一 registry)。"""
from registry import register


@register("detector", "vlm")
def build_vlm_detector(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from perception.detectors import VLMDetector
    import paths
    ROOT = paths.PROJECT_ROOT
    vcfg = cfg or {}
    return VLMDetector(
        model_name=vcfg.get("model", "qwen3-vl:8b-instruct-q4_K_M"),
        template_name=vcfg.get("template", "standard_detection.v2"),
        prompts_dir=str(ROOT / "vlm" / vcfg.get("prompts_dir", "prompts")),
    )
