import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT), str(ROOT / "vlm")])

from vlm.src.apps.grasp_selection import GraspSelectionApp
from vlm.src.core.config import load_config

if __name__ == "__main__":
    cfg = load_config(str(ROOT / "vlm/config/settings.yaml"))
    print(f"Loading Model: {cfg.get('default_model')}")
    
    app = GraspSelectionApp(
        model_name=cfg.get("default_model", "qwen2.5-vl"),
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