import argparse
import os
import sys
from pathlib import Path

# Fix: Unset proxy environment variables that cause issues with httpx/ollama
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
     if k in os.environ:
         del os.environ[k]

from src.core.config import load_config
from src.apps.static_detection import StaticDetectionApp
from src.utils.image_utils import draw_bounding_boxes

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Ollama VLM - Static Image Mode")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--target", type=str, help="Target object description")
    parser.add_argument("--output", type=str, help="Path to save result image (Optional)")
    
    args = parser.parse_args()
    
    image_path = args.image
    target_object = args.target
    
    # Prompt for target if not provided
    if not target_object:
        target_object = input("Enter target object description: ").strip()
        if not target_object:
            return

    print("\n=== Static Detection Configuration ===")
    print(f"Image: {image_path}")
    print(f"Target: {target_object}")
    print(f"Model: {config.get('default_model', 'qwen2.5-vl')}")
    print("======================================\n")

    app = StaticDetectionApp(
        model_name=config.get("default_model", "qwen2.5-vl"),
        template_name=config.get("template", "standard_detection.v2"),
        prompts_dir=config.get("prompts_dir", "prompts")
    )

    result = app.run(image_path, target_object)
    
    if result["success"]:
        print("Raw Response:", result.get("raw_response"))
        print("Detected Boxes (Normalized):", result.get("normalized_boxes"))
        print("Detected Boxes (Pixel):", result.get("pixel_boxes"))
        
        normalized_boxes = result.get("normalized_boxes", [])
        
        if normalized_boxes:
            # Determine Output Path
            final_output_path = args.output
            if not final_output_path:
                output_directory = config.get("output_dir", "output")
                out_dir_path = Path(output_directory)
                out_dir_path.mkdir(parents=True, exist_ok=True)
                src_name = Path(image_path).stem
                final_output_path = str(out_dir_path / f"{src_name}_result.jpg")
                
            vis_config = config.get("visualization", {})
            box_color = vis_config.get("box_color", "red")
            box_width = vis_config.get("box_width", 3)
            
            draw_bounding_boxes(
                image_path, 
                normalized_boxes, 
                final_output_path, 
                labels=[target_object] * len(normalized_boxes),
                color=box_color,
                width=box_width
            )
        else:
            print("No objects detected to visualize.")
            
    else:
        print("Error:", result.get("error"))

if __name__ == "__main__":
    main()
