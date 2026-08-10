"""Detector implementations; component registration lives in core/components."""
import os
import tempfile
import time
from typing import Optional, Union
import cv2
import numpy as np

from .base import Detection


class VLMDetector:
    """适配 vlm.src.apps.static_detection.StaticDetectionApp。"""

    def __init__(self, model_name: str, template_name: str = "standard_detection.v2",
                 prompts_dir: str = "prompts", host: str = None,
                 num_ctx: int = 4096, keep_alive: Union[str, int] = 0):
        from vlm.src.apps.static_detection import StaticDetectionApp
        self.app = StaticDetectionApp(
            model_name=model_name,
            template_name=template_name,
            prompts_dir=prompts_dir,
            host=host,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
        )

    def check(self):
        self.app.llm_client.check()

    def preflight(self):
        """Verify Ollama and evict a model retained by an earlier run."""
        self.check()
        self.unload()

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
