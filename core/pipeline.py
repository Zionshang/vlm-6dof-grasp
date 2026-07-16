import sys
import os
import cv2
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Setup paths & environment
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
ROOT = paths.PROJECT_ROOT
[os.environ.pop(k, None) for k in list(os.environ) if "PROXY" in k.upper()]

from vlm.src.core.config import load_config
from economic_grasp.inference import EconomicGraspInference
from perception import build_detector, build_segmenter  # 可插拔:registry 按 backend 取实现
from saver import save_vlm_boxes, save_seg_mask        # 统一保存
from hardware import HardwareConfig                    # 硬件配置(可插拔)
import argparse

class GraspPipeline:
    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", default="mug", help="Text prompt for detection")
        parser.add_argument("--fastsam", default="fastsam/weight/FastSAM-s.pt")
        parser.add_argument("--config", default="vlm/config/settings.yaml", help="VLM config path")
        parser.add_argument("--grasp_checkpoint", default="economic_grasp/checkpoint/economicgrasp_epoch10.tar", help="EconomicGrasp checkpoint path.")
        parser.add_argument("--grasp_topk", type=int, default=8)
        parser.add_argument("--use_collision", default=True, help="Enable collision detection during grasp generation.")
        parser.add_argument("--use_sam", default=True, help="Use FastSAM for segmentation.")
        parser.add_argument("--output_dir", default="output", help="Directory to save debug images")
        return parser

    def __init__(self, args=None, intrinsic=None, factor_depth=None):
        # 1. Load Parameters
        if args is None:
            parser = self.get_parser()
            self.args, _ = parser.parse_known_args()
        else:
            self.args = args

        # Resolve paths internally
        self.output_dir = (ROOT / self.args.output_dir)

        self.cfg = load_config(str(ROOT / self.args.config))

        # 管线级配置(感知后端选择等),与 vlm 模型配置、硬件 profile 分离
        with open(str(ROOT / "config/pipeline.yaml")) as f:
            self.pipe_cfg = yaml.safe_load(f)

        # 硬件配置(相机内参/深度因子等,可插拔 profile)
        # intrinsic/factor_depth 可由外部覆盖(如离线 demo 用 meta.mat 内参),默认取 profile
        self.hw = HardwareConfig()
        self.camera_matrix = intrinsic if intrinsic is not None else self.hw.camera_matrix
        self.factor_depth = factor_depth if factor_depth is not None else self.hw.factor_depth

        # 2. Detection Init (可插拔:backend 由 config/pipeline.yaml 指定,换算法只改配置)
        detector_backend = self.pipe_cfg["detector"]
        print(f"[Pipeline] Loading detector backend: {detector_backend}")
        self.detector = build_detector(detector_backend, self.cfg, ROOT, self.args)

        # 3. Segmentation Init (可插拔:backend 由 config/pipeline.yaml 指定)
        segmenter_backend = self.pipe_cfg["segmenter"]
        print(f"[Pipeline] Loading segmenter backend: {segmenter_backend}")
        self.segmenter = build_segmenter(segmenter_backend, self.cfg, ROOT, self.args)

        # 4. Grasp Generation Init (直接用 economic_grasp,不套适配器)
        self.grasp_engine = None
        if self.args.grasp_checkpoint:
            print(f"[Pipeline] Loading EconomicGrasp from {self.args.grasp_checkpoint}...")
            self.grasp_engine = EconomicGraspInference(
                str(ROOT / self.args.grasp_checkpoint),
                intrinsic=self.camera_matrix,
                factor_depth=self.factor_depth,
                use_collision=self.args.use_collision
            )

        print("[Pipeline] Initialization Complete.")

    @staticmethod
    def expand_boxes(boxes, shape, scale=1.5):
        h, w = shape[:2]
        return [[
            max(0, int((x1+x2)/2 - (x2-x1)*scale/2)),
            max(0, int((y1+y2)/2 - (y2-y1)*scale/2)),
            min(w, int((x1+x2)/2 + (x2-x1)*scale/2)),
            min(h, int((y1+y2)/2 + (y2-y1)*scale/2))
        ] for x1, y1, x2, y2 in boxes]

    def detect_objects(self, color, prompt):
        """检测(通过可插拔 detector)。返回像素框列表。"""
        detection = self.detector.detect(color, prompt)
        return detection.boxes if detection else []

    def get_target_position(self, depth, color, prompt="object", run_id=None, transform_info=None):
        """
        Locate target 3D position using detection + Depth.
        """
        pixel_boxes = self.detect_objects(color, prompt)
        if not pixel_boxes:
            print(f"[Pipeline] Object '{prompt}' not detected.")
            return None

        # 1. Pixel Center & Depth (Robust Median)
        x1, y1, x2, y2 = pixel_boxes[0]
        u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
        h, w = depth.shape
        u, v = np.clip(u, 0, w-1), np.clip(v, 0, h-1)

        patch_size = 5; half = patch_size // 2
        d_patch = depth[max(0,v-half):min(h,v+half+1), max(0,u-half):min(w,u+half+1)]
        valid = d_patch[d_patch > 0]

        if valid.size == 0:
            print(f"[Pipeline] Invalid depth at center ({u}, {v}).")
            return None

        z_c = np.median(valid) / self.factor_depth # Scale to meters

        # 2. Project to 3D (Camera Frame)
        fx, fy = self.camera_matrix[0,0], self.camera_matrix[1,1]
        cx, cy = self.camera_matrix[0,2], self.camera_matrix[1,2] # 435.756, 435.674 ...

        x_c = (u - cx) * z_c / fx
        y_c = (v - cy) * z_c / fy
        p_cam = np.array([x_c, y_c, z_c])

        # 3. Coordinate Transformation (Required)
        curr_pose, R_he, T_he = transform_info

        T_cam2ee = np.eye(4); T_cam2ee[:3, :3] = R_he; T_cam2ee[:3, 3] = T_he
        T_ee2base = np.eye(4); T_ee2base[:3, :3] = R.from_euler('xyz', curr_pose[3:], False).as_matrix(); T_ee2base[:3, 3] = curr_pose[:3]

        # p_base = T_ee2base @ (T_cam2ee @ p_cam)
        p_base = (T_ee2base @ T_cam2ee @ np.append(p_cam, 1.0))[:3]

        print(f"[Pipeline] Target '{prompt}' Pos (Base): {p_base}")
        return p_base

    def run(self, color, depth, prompt=None, run_id=None, visualize=False):
        # Use prompt from args if not provided
        prompt = prompt or self.args.prompt
        print(f"[Pipeline] Processing prompt: '{prompt}'")

        pixel_boxes = self.detect_objects(color, prompt)
        if not pixel_boxes:
            print(f"[Pipeline] No objects found for '{prompt}'.")
            return None, None, None

        # Save VLM boxes(原始 + 扩展后)
        save_vlm_boxes(self.output_dir, color, pixel_boxes, run_id, tag="origin_vlm")
        pixel_boxes = self.expand_boxes(pixel_boxes, color.shape)
        save_vlm_boxes(self.output_dir, color, pixel_boxes, run_id, tag="vlm")

        # 2. Segmentation(可插拔 segmenter)
        seg_mask = self.segmenter.segment(color, pixel_boxes)
        if seg_mask is not None:
            save_seg_mask(self.output_dir, seg_mask, run_id)

        # 3. Grasp Generation(visualize=True 时弹出 Open3D 可视化)
        return self._run_grasping(color, depth, seg_mask, visualize=visualize)

    def _run_grasping(self, color, depth, mask, visualize=False):
        if self.grasp_engine and mask is not None:
            print("[Pipeline] Generating grasps...")
            if mask.shape != depth.shape:
                mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

            # keep_topk:最终保留/可视化的抓取数(来自 --grasp_topk)
            # predict topk 为候选生成池(100),需大于 keep_topk 以便按方向筛选
            keep_topk = self.args.grasp_topk
            gg, data_dict = self.grasp_engine.predict(color, depth, mask=mask, topk=100)

            if len(gg) > 0:
                keep_inds = []
                for i in range(len(gg)):
                    if len(keep_inds) >= keep_topk: break
                    R = gg[i].rotation_matrix
                    # Cond 1: Approach(X) close to Vertical(Z) (<60 deg)
                    ang_x = np.arccos(np.clip(np.dot(R[:, 0], [0, 0, 1]), -1, 1))
                    # Cond 2: Closing(Y) close to Camera Right(X) (<110 deg)
                    ang_y = np.arccos(np.clip(np.dot(R[:, 1], [1, 0, 0]), -1, 1))

                    if ang_x < np.deg2rad(50) and ang_y < np.deg2rad(100):
                        keep_inds.append(i)

                if keep_inds:
                    gg = gg[keep_inds]
                    print(f"[Pipeline] Filtered to top {len(gg)} grasps.")
                else:
                    print(f"[Pipeline] No grasps met criteria. Using default top {keep_topk}.")
                    gg = gg[:keep_topk]

                if visualize:
                    self._visualize_grasps(gg, data_dict)

                return gg.translations, gg.rotation_matrices, gg.widths
            else:
                print("[Pipeline] No valid grasps found.")

        return None, None, None

    def _visualize_grasps(self, gg, data_dict):
        print("Visualizing Grasps... (Close window to continue)")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(data_dict["point_clouds"])
        cloud.colors = o3d.utility.Vector3dVector(data_dict["cloud_colors"])

        geometries = [cloud]

        # Visualizing top grasp_topk only，and Base Right Finger Root Blue Dot
        top_n = min(len(gg), self.args.grasp_topk)
        for i in range(top_n):
             g = gg[i].to_open3d_geometry(color=(0, 0, 0))
             geometries.extend(g if isinstance(g, list) else [g])

             # Blue Dot at Right Finger Root (+Y side)
             # Assumption: X=0 is Tip, so Base is at -g.depth
             pt = gg[i].translation + gg[i].rotation_matrix @ np.array([-0.025, gg[i].width / 2 + 0.002, 0])
             sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
             sphere.translate(pt)
             sphere.paint_uniform_color([0, 0, 1])
             geometries.append(sphere)

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
