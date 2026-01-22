import sys
import os
import cv2
import numpy as np
import torch
import open3d as o3d
from pathlib import Path

# Setup paths & environment
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT), str(ROOT / "vlm"), str(ROOT / "economic_grasp")])
[os.environ.pop(k, None) for k in list(os.environ) if "PROXY" in k.upper()]

from vlm.src.core.config import load_config
from vlm.src.apps.static_detection import StaticDetectionApp
from vlm.src.utils.image_utils import make_bbox_mask
from fastsam.segmentor import ImageSegmentor
from economic_grasp.inference import EconomicGraspInference
import argparse

# =============================================================================
# Global Camera & Default Configuration
# =============================================================================
CAMERA_MATRIX = np.array([
    [435.75600787289994, 0.0, 423.5139606211078],
    [0.0, 435.6741418717883, 243.52287948995928],
    [0.0, 0.0, 1.0]
])
# CAMERA_MATRIX = np.array([[433.89368, 0.00000, 423.08353],
#  [0.00000, 432.80573, 244.67590],
#  [0.00000, 0.00000, 1.00000]])
DIST_COEFFS = np.array([
    -0.061438834574162444, 0.11244487882386699, 
    -0.0008922372006081498, 0.0010226929723920338, 
    -0.09769331639799333
])
FACTOR_DEPTH = 10000


class GraspPipeline:
    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", default="mug", help="Text prompt for detection")
        parser.add_argument("--fastsam", default="fastsam/weight/FastSAM-s.pt")
        parser.add_argument("--config", default="vlm/config/settings.yaml", help="VLM config path")
        parser.add_argument("--grasp_checkpoint", default="economic_grasp/checkpoint/economicgrasp_epoch10.tar", help="EconomicGrasp checkpoint path.")
        parser.add_argument("--grasp_topk", type=int, default=10)
        parser.add_argument("--use_collision", default=True, help="Enable collision detection during grasp generation.")
        parser.add_argument("--use_sam", default=True, help="Use FastSAM for segmentation.")
        parser.add_argument("--output_dir", default="output", help="Directory to save debug images")
        parser.add_argument("--no_vis", action="store_true", help="Disable Open3D visualization")
        return parser

    def __init__(self, args=None):
        # 1. Load Parameters
        if args is None:
            parser = self.get_parser()
            self.args, _ = parser.parse_known_args()
        else:
            self.args = args

        # Resolve paths internally
        self.output_dir = (ROOT / self.args.output_dir)
        self.dirs = {
            "vlm": self.output_dir / "vlm",
            "sam": self.output_dir / "sam",
            # "inputs": self.output_dir / "pipeline_inputs",
            "captures": self.output_dir / "captures"
        }
        # Create directories
        for d in [self.dirs["vlm"], self.dirs["sam"]]:
            d.mkdir(exist_ok=True, parents=True)
        
        self.cfg = load_config(str(ROOT / self.args.config))
        
        # 2. VLM Detection Init
        print(f"[Pipeline] Loading VLM (Model: {self.cfg.get('default_model', 'qwen2.5-vl')})...")
        self.vlm = StaticDetectionApp(
            model_name=self.cfg.get("default_model", "qwen2.5-vl"),
            template_name=self.cfg.get("template", "standard_detection.v2"),
            prompts_dir=str(ROOT / "vlm" / self.cfg.get("prompts_dir", "prompts")),
        )
        
        # 3. Segmentation Init
        self.sam = None
        if self.args.use_sam:
            print("[Pipeline] Loading FastSAM...")
            fastsam_path = str(ROOT / self.args.fastsam)
            self.sam = ImageSegmentor(fastsam_path)
            
        # 4. Grasp Generation Init
        self.grasp_engine = None
        if self.args.grasp_checkpoint:
            print(f"[Pipeline] Loading EconomicGrasp from {self.args.grasp_checkpoint}...")
            self.grasp_engine = EconomicGraspInference(
                str(ROOT / self.args.grasp_checkpoint), 
                intrinsic=CAMERA_MATRIX, 
                factor_depth=FACTOR_DEPTH, 
                use_collision=self.args.use_collision
            )
        
        print("[Pipeline] Initialization Complete.")

    def detect_objects(self, prompt, run_id=None):
        # 1. Input Handling
        img_path = None
        if run_id:
            cap_path = self.dirs["captures"] / f"{run_id}_color.png"
            if cap_path.exists():
                img_path = str(cap_path)
                print(f"[Pipeline] Using existing capture: {cap_path.name}")
        
        # 2. VLM Detection
        vlm_res = self.vlm.run(img_path, prompt)
        pixel_boxes = vlm_res.get("pixel_boxes", [])
        return pixel_boxes, img_path

    def run(self, color, depth, prompt=None, run_id=None):
        # Use prompt from args if not provided
        prompt = prompt or self.args.prompt
        print(f"[Pipeline] Processing prompt: '{prompt}'")

        pixel_boxes, img_path = self.detect_objects(prompt, run_id)
        
        if not pixel_boxes:
            print(f"[Pipeline] No objects found for '{prompt}'.")
            return None, None, None

        # Visualize VLM
        self._visualize_vlm(color, pixel_boxes, run_id)

        # 2. Segmentation
        seg_mask = self._run_segmentation(img_path, pixel_boxes, color.shape)
        
        # Visualize Segmentation
        if seg_mask is not None:
             self._visualize_mask(seg_mask, color.shape, run_id)

        # 3. Grasp Generation
        return self._run_grasping(color, depth, seg_mask)

    def _run_segmentation(self, img_path, boxes, shape):
        h, w = shape[:2]
        if self.args.use_sam and self.sam:
            print(f"[Pipeline] Segmenting {len(boxes)} object(s)...")
            seg_res = self.sam.segment(img_path, boxes)
            if getattr(seg_res, "masks", None) is not None:
                mask_data = seg_res.masks.data
                if torch.is_tensor(mask_data):
                    mask_data = mask_data.cpu().numpy()
                return np.any(mask_data > 0, axis=0) # Merge masks
        else:
            print("[Pipeline] Using BBox mask (No SAM)...")
            return make_bbox_mask(boxes, h, w)
        return None

    def _run_grasping(self, color, depth, mask):
        best_translation = None
        best_rotation_matrix = None
        best_width = None

        if self.grasp_engine and mask is not None:
            print("[Pipeline] Generating grasps...")
            # Align mask
            if mask.shape != depth.shape:
                mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            
            gg, data_dict = self.grasp_engine.predict(color, depth, mask=mask, topk=self.args.grasp_topk)
            
            if len(gg) > 0:
                best = gg[0]
                best_translation = best.translation
                best_rotation_matrix = best.rotation_matrix
                best_width = best.width
                print(f"[Pipeline] Best Grasp Found:\n T: {best_translation}")
                if not self.args.no_vis:
                    self._visualize_grasps(gg, data_dict)
            else:
                print("[Pipeline] No valid grasps found.")
        
        return best_translation, best_rotation_matrix, best_width

    def _visualize_vlm(self, color, boxes, run_id=None):
        img_vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        filename = f"{run_id}_vlm.png" if run_id else "vlm_result.png"
        out_path = self.dirs["vlm"] / filename
        cv2.imwrite(str(out_path), img_vis)
        print(f"Saved VLM visualization to {out_path}")

    def _visualize_mask(self, mask, shape, run_id=None):
        if mask.shape[:2] != shape[:2]:
             vis_mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
             vis_mask = mask.astype(np.uint8)
             
        filename = f"{run_id}_sam.png" if run_id else "seg_result.png"
        out_path = self.dirs["sam"] / filename
        cv2.imwrite(str(out_path), vis_mask * 255)
        print(f"Saved Mask visualization to {out_path}")

    def _visualize_grasps(self, gg, data_dict):
        print("Visualizing Grasps... (Close window to continue)")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(data_dict["point_clouds"])
        cloud.colors = o3d.utility.Vector3dVector(data_dict["cloud_colors"])
        
        # Background (rest) + Best (black)
        geometries = [cloud, *gg[1:].to_open3d_geometry_list()]
        
        best_geo = gg[0].to_open3d_geometry(color=(0, 0, 0))
        geometries.extend(best_geo if isinstance(best_geo, list) else [best_geo])
        
        o3d.visualization.draw_geometries(geometries, window_name="Grasp Results")


if __name__ == "__main__":
    parser = GraspPipeline.get_parser()
    # Add data_dir argument for standalone script usage
    parser.add_argument("--data_dir", required=True, help="Directory containing color.png and depth.png")
    args = parser.parse_args()
    
    # If run as main, we expect data_dir
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist.")
        sys.exit(1)

    print(f"Running GraspPipeline on {data_dir}")
    
    # Load Images
    color_path = data_dir / "color.png"
    depth_path = data_dir / "depth.png"
    
    if not color_path.exists() or not depth_path.exists():
        print("Error: color.png or depth.png not found in data_dir.")
        sys.exit(1)
        
    color = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    
    # Initialize & Run
    pipeline = GraspPipeline(args)
    pipeline.run(color, depth)

