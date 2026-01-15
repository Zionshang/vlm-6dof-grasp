import sys, os, argparse, cv2, time
from pathlib import Path

# Setup paths & environment
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT), str(ROOT / "vlm")])
[os.environ.pop(k, None) for k in list(os.environ) if "PROXY" in k.upper()]

from vlm.src.core.config import load_config
from vlm.src.apps.static_detection import StaticDetectionApp
from fastsam.segmentor import ImageSegmentor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fastsam", default="fastsam/weight/FastSAM-s.pt")
    args = parser.parse_args()

    # Load Config & Resolve Paths
    cfg = load_config(str(ROOT / "vlm/config/settings.yaml"))

    def resolve(p):
        return (ROOT / p).resolve() if not Path(p).is_absolute() else Path(p)

    img_path = resolve(args.image)
    sam_path = resolve(args.fastsam)
    out_dir = resolve(cfg.get("output_dir", "output"))

    if not img_path.exists():
        return print(f"Error: Image not found at {img_path}")

    # Initialize Models
    try:
        vlm = StaticDetectionApp(
            model_name=cfg.get("default_model", "qwen2.5-vl"),
            template_name=cfg.get("template", "standard_detection.v2"),
            prompts_dir=str(ROOT / "vlm" / cfg.get("prompts_dir", "prompts")),
            show_latency=cfg.get("show_latency", False),
        )
        sam = ImageSegmentor(str(sam_path))
    except Exception as e:
        return print(f"Initialization Error: {e}")


    # Run Pipeline
    print(f"Detecting '{args.prompt}'...")
    t_vlm_start = time.time()
    res = vlm.run(str(img_path), args.prompt)
    print(f"VLM Inference Time: {time.time() - t_vlm_start:.4f}s")

    if not res.get("success") or not res.get("pixel_boxes"):
        return print(f"No objects detected. {res.get('error', '')}")

    print(f"Found {len(res['pixel_boxes'])} objects. Segmenting...")
    t_sam_start = time.time()
    seg_res = sam.segment(str(img_path), res["pixel_boxes"])
    print(f"FastSAM Inference Time: {time.time() - t_sam_start:.4f}s")

    if seg_res:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seg_{img_path.name}"
        cv2.imwrite(str(out_path), seg_res.plot(boxes=True, masks=True))
        print(f"Saved: {out_path}")
    else:
        print("Segmentation failed.")


if __name__ == "__main__":
    main()
