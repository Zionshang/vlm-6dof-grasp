import sys, os, argparse, cv2, time
import numpy as np
import torch
import open3d as o3d
import scipy.io as scio
from pathlib import Path

# Setup paths & environment
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT), str(ROOT / "vlm"), str(ROOT / "economic_grasp")])
[os.environ.pop(k, None) for k in list(os.environ) if "PROXY" in k.upper()]

from vlm.src.core.config import load_config
from vlm.src.apps.static_detection import StaticDetectionApp
from vlm.src.utils.image_utils import make_bbox_mask
from fastsam.segmentor import ImageSegmentor
from inference import EconomicGraspInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing color.png, depth.png, and meta.mat")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fastsam", default="fastsam/weight/FastSAM-s.pt")
    parser.add_argument("--grasp_checkpoint", help="EconomicGrasp checkpoint path.")
    parser.add_argument("--grasp_topk", type=int, default=10)
    parser.add_argument("--no_collision", action="store_true", help="Disable collision detection during grasp generation.")
    args = parser.parse_args()

    cfg = load_config(str(ROOT / "vlm/config/settings.yaml"))
    resolve = lambda p: (ROOT / p).resolve() if not Path(p).is_absolute() else Path(p)

    data_dir = resolve(args.data_dir)
    if not data_dir.exists():
        return print(f"Error: Data directory not found at {data_dir}")

    img_path = data_dir / "color.png"
    depth_path = data_dir / "depth.png"
    meta_path = data_dir / "meta.mat"

    if not img_path.exists():
        return print(f"Error: Color image not found at {img_path}")

    # Output setup
    out_dir = resolve(cfg.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. VLM Detection
    print(f"Detecting '{args.prompt}'...")
    vlm = StaticDetectionApp(
        model_name=cfg.get("default_model", "qwen2.5-vl"),
        template_name=cfg.get("template", "standard_detection.v2"),
        prompts_dir=str(ROOT / "vlm" / cfg.get("prompts_dir", "prompts")),
    )
    vlm_res = vlm.run(str(img_path), args.prompt)
    if not vlm_res.get("pixel_boxes"):
        return print(f"No objects detected.")

    # Save VLM Bbox Result
    img_vis = cv2.imread(str(img_path))
    for box in vlm_res["pixel_boxes"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(img_vis, args.prompt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / "vlm_detection.png"), img_vis)
    print(f"Saved VLM detection to {out_dir / 'vlm_detection.png'}")

    # 2. FastSAM Segmentation
    print(f"Segmenting {len(vlm_res['pixel_boxes'])} objects...")
    sam = ImageSegmentor(str(resolve(args.fastsam)))
    seg_res = sam.segment(str(img_path), vlm_res["pixel_boxes"])

    seg_mask = None
    if getattr(seg_res, "masks", None) is not None:
        mask_data = seg_res.masks.data
        if torch.is_tensor(mask_data):
            mask_data = mask_data.cpu().numpy()
        seg_mask = np.any(mask_data > 0, axis=0)

    # 3. Grasp Generation
    if args.grasp_checkpoint and seg_mask is not None:
        print("Generating grasps...")

        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found at {depth_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found at {meta_path}")

        # Load Data
        color = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth image from {depth_path}")

        # Load Meta (Strict, no defaults)
        meta = scio.loadmat(str(meta_path))
        if "intrinsic_matrix" not in meta:
            raise KeyError(f"Meta file {meta_path} missing 'intrinsic_matrix'")
        if "factor_depth" not in meta:
            raise KeyError(f"Meta file {meta_path} missing 'factor_depth'")

        intrinsic = meta["intrinsic_matrix"]
        factor_depth = float(meta["factor_depth"])

        # Align Mask
        if seg_mask.shape != depth.shape:
            print(f"Resizing mask {seg_mask.shape} -> {depth.shape}")
            seg_mask = cv2.resize(seg_mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            cv2.imwrite(str(out_dir / "segmentation_mask.png"), (seg_mask * 255).astype(np.uint8))
            print(f"Saved segmentation mask to {out_dir / 'segmentation_mask.png'}")

        # Inference
        grasp_engine = EconomicGraspInference(
            str(resolve(args.grasp_checkpoint)), intrinsic=intrinsic, factor_depth=factor_depth, use_collision=not args.no_collision
        )
        gg, data_dict = grasp_engine.predict(color, depth, mask=seg_mask, topk=args.grasp_topk)

        # Visualization
        if len(gg) > 0:
            print("Visualizing... (Close the window to exit)")
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(data_dict["point_clouds"])
            cloud.colors = o3d.utility.Vector3dVector(data_dict["cloud_colors"])
            o3d.visualization.draw_geometries([cloud, *gg.to_open3d_geometry_list()], window_name="Result")
        else:
            print("No grasps found.")


if __name__ == "__main__":
    main()
