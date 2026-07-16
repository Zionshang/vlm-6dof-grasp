import sys
from pathlib import Path
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
ROOT = paths.PROJECT_ROOT

from vlm.src.apps.grasp_selection import GraspSelectionApp
from vlm.src.core.config import load_config

if __name__ == "__main__":
    cfg = load_config(str(ROOT / "vlm/config/settings.yaml"))
    
    app = GraspSelectionApp(
        model_name=cfg.get("grasp_selection_model", "qwen3-vl:8b-instruct-q4_K_M"),
        prompts_dir=str(ROOT / "vlm/prompts")
    )

    img_dir = ROOT / "output/2D_grasp"
    if not img_dir.exists():
        print("No output/2D_grasp directory found.")
        sys.exit(1)

    images = sorted(list(img_dir.glob("*.jpg")), key=lambda p: int(p.stem) if p.stem.isdigit() else 999)
    print(f"Analyzing {len(images)} images...")

    res = app.run([str(p) for p in images])
    
    print("\n" + "="*30)
    print(f"Selected ID: {res.get('selected_id')}")
    print(f"Reason:      {res.get('reason')}")
    print("="*30 + "\n")