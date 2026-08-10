"""VLM 检测器组件(复用 perception.detectors.VLMDetector,改用统一 registry)。"""
from registry import register


@register("detector", "vlm", preflight=True)
def build_vlm_detector(cfg=None, hw=None, ctx=None, dependencies=None):
    from perception.detectors import VLMDetector
    import paths
    ROOT = paths.PROJECT_ROOT
    vcfg = cfg or {}
    return VLMDetector(
        model_name=vcfg.get("model", "qwen3-vl:8b-instruct-q4_K_M"),
        template_name=vcfg.get("template", "standard_detection.v2"),
        prompts_dir=str(ROOT / "vlm" / vcfg.get("prompts_dir", "prompts")),
        host=vcfg.get("host"),
        num_ctx=vcfg.get("num_ctx", 4096),
        keep_alive=vcfg.get("keep_alive", 0),
    )
