"""VLM detector adapter and component registration."""
import os
import tempfile
import time
from dataclasses import dataclass, field

import cv2

from registry import register


@dataclass
class Detection:
    boxes: list[list[int]]
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


class VLMDetector:
    """Adapt the vendored VLM static-detection application."""

    def __init__(self, model_name, template_name="standard_detection.v2",
                 prompts_dir="prompts", host=None, num_ctx=4096,
                 keep_alive=0):
        from vlm.src.apps.static_detection import StaticDetectionApp
        self.app = StaticDetectionApp(
            model_name=model_name, template_name=template_name,
            prompts_dir=prompts_dir, host=host, num_ctx=num_ctx,
            keep_alive=keep_alive,
        )

    def check(self):
        self.app.llm_client.check()

    def preflight(self):
        self.check()
        self.unload()

    def detect(self, color, prompt):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        try:
            result = self.app.run(path, prompt)
        finally:
            os.remove(path)
        boxes = result.get("pixel_boxes", [])
        return Detection(boxes) if boxes else None

    def unload(self):
        try:
            print("[VLM] Unloading to free VRAM...")
            self.app.llm_client.unload()
            time.sleep(3)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[VLM] Warning: Failed to unload: {exc}")

    def warmup(self):
        try:
            self.app.llm_client.warmup()
        except Exception as exc:
            print(f"[VLM] Warning: Failed to warmup: {exc}")


@register("detector", "vlm", preflight=True)
def build_vlm_detector(cfg=None, hw=None, ctx=None, dependencies=None):
    import paths
    ROOT = paths.PROJECT_ROOT
    vcfg = cfg or {}
    return VLMDetector(
        model_name=vcfg.get("model", "qwen3-vl:8b-instruct-q4_K_M"),
        template_name=vcfg.get("template", "standard_detection.v2"),
        prompts_dir=str(ROOT / "third_party/vlm" / vcfg.get("prompts_dir", "prompts")),
        host=vcfg.get("host"),
        num_ctx=vcfg.get("num_ctx", 4096),
        keep_alive=vcfg.get("keep_alive", 0),
    )
