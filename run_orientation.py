import sys, os, argparse
from pathlib import Path
import yaml

# Setup paths (ensure vlm modules can be imported)
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT), str(ROOT / "vlm")])

# Clear proxy env vars to fix httpx/ollama socks error
[os.environ.pop(k, None) for k in list(os.environ) if "PROXY" in k.upper()]

from vlm.src.apps.orientation_prediction import OrientationPredictionApp
# from vlm.src.core.config import load_config # Optional, we can just load yaml directly for simplicity

def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run VLM Orientation Prediction")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", required=True, help="Target object name")
    parser.add_argument("--model", help="Override default model name (optional)")
    args = parser.parse_args()

    # Load configuration
    cfg_path = ROOT / "vlm/config/settings.yaml"
    cfg = load_config(str(cfg_path))
    
    # Determined model
    model_name = args.model or cfg.get("default_model", "qwen2.5-vl")
    
    print(f"Initializing Orientation Logic...")
    print(f"Model: {model_name}")
    print(f"Prompt Template: orientation.v1")
    
    # Initialize App
    app = OrientationPredictionApp(
        model_name=model_name,
        template_name="orientation.v1",
        prompts_dir=str(ROOT / "vlm/prompts")
    )
    
    print(f"Analyzing image: {args.image}")
    print(f"Target Object: {args.prompt}")
    
    # Run
    result = app.run(args.image, args.prompt)
    
    if result["success"]:
        angle = result["rotation_angle"]
        print("\n" + "="*40)
        print("ORIENTATION RESULT")
        print("="*40)
        print(f"Object: {args.prompt}")
        print(f"Angle required: {angle} degrees")
        print(f"  (+ for Counter-Clockwise, - for Clockwise)")
        print(f"  (Goal: Align major axis with vertical Y-axis)")
        print("="*40 + "\n")
    else:
        print("\n" + "!"*40)
        print("FAILURE")
        print(f"Error: {result.get('error')}")
        print("!"*40 + "\n")
        
    if "raw_response" in result:
        print("Debug - Raw LLM Response:")
        print(result["raw_response"])

if __name__ == "__main__":
    main()
