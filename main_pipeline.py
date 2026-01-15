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
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fastsam", default="fastsam/weight/FastSAM-s.pt")
    parser.add_argument("--depth", help="Depth image path for grasp generation.")
    parser.add_argument("--meta", help="Meta mat file path for camera intrinsics.")
    parser.add_argument("--grasp_checkpoint", help="EconomicGrasp checkpoint path.")
    parser.add_argument("--grasp_topk", type=int, default=10)
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

    seg_mask = None
    if getattr(seg_res, "masks", None) is not None and seg_res.masks is not None:
        mask_data = seg_res.masks.data
        if mask_data is not None and len(mask_data) > 0:
            if torch.is_tensor(mask_data):
                mask_data = mask_data.cpu().numpy()
            seg_mask = np.any(mask_data > 0, axis=0)
    elif res.get("pixel_boxes"):
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            seg_mask = make_bbox_mask(res.get("pixel_boxes", []), h, w)

    if args.grasp_checkpoint:
        img_dir = img_path.parent
        depth_path = resolve(args.depth) if args.depth else (img_dir / "depth.png")
        meta_path = resolve(args.meta) if args.meta else (img_dir / "meta.mat")

        if not depth_path.exists():
            return print(f"Error: Depth not found at {depth_path}")

        try:
            color_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if color_bgr is None:
                return print(f"Error: Failed to load color image at {img_path}")
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                return print(f"Error: Failed to load depth image at {depth_path}")

            intrinsic = np.array([[631.5, 0, 638.5], [0, 631.2, 367.0], [0, 0, 1]])
            factor_depth = 1000.0
            if meta_path and meta_path.exists():
                meta = scio.loadmat(str(meta_path))
                if "intrinsic_matrix" in meta and "factor_depth" in meta:
                    intrinsic = meta["intrinsic_matrix"]
                    factor_depth = float(meta["factor_depth"])

            grasp_engine = EconomicGraspInference(
                str(resolve(args.grasp_checkpoint)),
                intrinsic=intrinsic,
                factor_depth=factor_depth,
            )
            mask_for_grasp = seg_mask
            gg, data_dict = grasp_engine.predict(
                color=color_rgb,
                depth=depth,
                mask=mask_for_grasp,
                topk=args.grasp_topk,
            )
        except Exception as e:
            return print(f"Grasp Generation Error: {e}")

        if gg and len(gg) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data_dict["point_clouds"])
            pcd.colors = o3d.utility.Vector3dVector(data_dict["cloud_colors"])
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries(
                [pcd, *grippers],
                window_name="EconomicGrasp Result",
            )
        else:
            print("No grasps generated.")


if __name__ == "__main__":
    main()
