import argparse
import os
import sys
from pathlib import Path

# Fix: Unset proxy environment variables that cause issues with httpx/ollama
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
     if k in os.environ:
         del os.environ[k]

from src.core.config import load_config
from vlm.src.apps.live_detection import LiveDetectionApp

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Ollama VLM - Live Detection Mode")
    args = parser.parse_args()
    
    print("\n=== Live Detection Configuration ===")
    print(f"Model: {config.get('default_model', 'qwen3-vl')}")
    print(f"Template: {config.get('template', 'standard_detection.v2')}")
    print("Target: [Interactive Input]")
    print("====================================\n")

    app = LiveDetectionApp(
        model_name=config.get("default_model", "qwen3-vl"),
        template_name=config.get("template", "standard_detection.v2"),
        prompts_dir=config.get("prompts_dir", "prompts"),
        camera_config=config.get("camera", {}),
        visualization_config=config.get("visualization", {}),
        max_image_width=config.get("max_image_width", 640),
        show_latency=config.get("show_latency", False)
    )
        
    app.run()




if __name__ == "__main__":
    main()
