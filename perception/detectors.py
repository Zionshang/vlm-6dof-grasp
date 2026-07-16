"""检测器实现:默认 VLM。新增检测器用 @register_detector 注册工厂(见 base.py)。"""
import os
import tempfile
import time
from typing import Optional
import cv2
import numpy as np

from .base import Detection, register_detector


class VLMDetector:
    """适配 vlm.src.apps.static_detection.StaticDetectionApp。"""

    def __init__(self, model_name: str, template_name: str = "standard_detection.v2",
                 prompts_dir: str = "prompts"):
        from vlm.src.apps.static_detection import StaticDetectionApp
        self.app = StaticDetectionApp(
            model_name=model_name,
            template_name=template_name,
            prompts_dir=prompts_dir,
        )

    def detect(self, color: np.ndarray, prompt: str) -> Optional[Detection]:
        # StaticDetectionApp.run 需要图片路径,落盘临时文件(逻辑等价于原 captures 落盘)。
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        try:
            res = self.app.run(tmp_path, prompt)
        finally:
            os.remove(tmp_path)
        boxes = res.get("pixel_boxes", [])
        return Detection(boxes=boxes) if boxes else None

    def unload(self):
        """释放 VLM 占用的 VRAM(如给 PyTorch 抓取模型让路)。"""
        try:
            print("[VLM] Unloading to free VRAM...")
            self.app.llm_client.unload()
            time.sleep(3)  # Wait for VRAM release
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[VLM] Warning: Failed to unload: {e}")

    def warmup(self):
        """预热 VLM(重新加载,避免首次检测延迟)。"""
        try:
            self.app.llm_client.warmup()
        except Exception as e:
            print(f"[VLM] Warning: Failed to warmup: {e}")


# =============================================================================
# Registry 工厂:每个 backend 一个 @register_detector
# 加新检测器 = 新增一个工厂(或在新文件里 @register_detector),pipeline 无需改动。
# =============================================================================

@register_detector("vlm")
def build_vlm_detector(cfg, root, args=None):
    """默认检测器:VLM(qwen3-vl via ollama)。"""
    return VLMDetector(
        model_name=cfg.get("detection_model", "qwen3-vl:8b-instruct-q4_K_M"),
        template_name=cfg.get("template", "standard_detection.v2"),
        prompts_dir=str(root / "vlm" / cfg.get("prompts_dir", "prompts")),
    )
