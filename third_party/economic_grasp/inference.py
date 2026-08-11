import os
import numpy as np
import torch

from models.economicgrasp import economicgrasp, pred_decode
from utils.arguments import cfgs
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup
from utils.collision_detector import ModelFreeCollisionDetector


class EconomicGraspInference:
    def __init__(
        self,
        checkpoint_path,
        intrinsic,
        factor_depth,
        device=None,
        use_collision=True,
    ):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.intrinsic = intrinsic
        self.factor_depth = factor_depth
        self.use_collision = use_collision

        # Load params from cfgs
        self.num_points = cfgs.num_point
        self.voxel_size = cfgs.voxel_size
        self.m_point = cfgs.m_point
        self.grasp_max_width = cfgs.grasp_max_width
        self.collision_thresh = cfgs.collision_thresh

        self.net = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        net = economicgrasp(
            seed_feat_dim=512,
            is_training=False,
            num_depth=cfgs.num_depth,
            num_angle=cfgs.num_angle,
            m_point=cfgs.m_point,
            num_view=cfgs.num_view,
            graspness_threshold=cfgs.graspness_threshold,
            grasp_max_width=cfgs.grasp_max_width,
            voxel_size=self.voxel_size,
        )
        net.to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net

    def predict(self, color, depth, mask=None, topk=100):
        # 1. Process Data
        if depth.ndim == 3: depth = depth[..., 0]
        if color.dtype == np.uint8: color = color.astype(np.float32) / 255.0

        camera = CameraInfo(depth.shape[1], depth.shape[0], self.intrinsic[0][0], self.intrinsic[1][1], self.intrinsic[0][2], self.intrinsic[1][2], self.factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # Masking
        workspace_mask = mask if mask is not None else (depth > 0)
        if workspace_mask.shape != depth.shape:
            raise ValueError(f"Mask shape {workspace_mask.shape} does not match depth shape {depth.shape}")
        
        final_mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[final_mask]
        color_masked = color[final_mask]

        if len(cloud_masked) == 0:
            print("Warning: No points in mask!")
            return GraspGroup(), np.zeros((0, 3))

        # Sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        
        cloud_sampled = cloud_masked[idxs].astype(np.float32)
        
        # Prepare Batch
        batch_data = {
            'point_clouds': torch.from_numpy(cloud_sampled).unsqueeze(0).to(self.device),
            'coordinates_for_voxel': [torch.from_numpy(cloud_sampled / self.voxel_size).to(self.device)]
        }

        # 2. Inference
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points, m_point=self.m_point, grasp_max_width=self.grasp_max_width)
        
        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        # 3. Collision Detection
        if self.use_collision and self.collision_thresh > 0:
            mfcdetector = ModelFreeCollisionDetector(cloud_sampled, voxel_size=self.voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
            gg = gg[~collision_mask]

        # 4. Post-processing
        gg = gg.nms()
        gg = gg.sort_by_score()
        if topk is not None:
             gg = gg[:topk]
             
        data_dict = {
            "point_clouds": cloud_sampled, 
            "cloud_colors": color_masked[idxs].astype(np.float32)
        }
        return gg, data_dict

if __name__ == "__main__":
    import argparse
    import scipy.io as scio
    import open3d as o3d
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', default='example_data', help='Data directory')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # Load meta
    meta_path = os.path.join(data_dir, 'meta.mat')
    if not os.path.exists(meta_path):
         raise FileNotFoundError(f"Meta file not found: {meta_path}")
    meta = scio.loadmat(meta_path)
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # Load images
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    
    # Load mask
    workspace_mask_path = os.path.join(data_dir, 'workspace_mask.png')
    workspace_mask = None
    if os.path.exists(workspace_mask_path):
        workspace_mask = np.array(Image.open(workspace_mask_path)) > 0
    
    # Inference
    inference = EconomicGraspInference(
        checkpoint_path=args.checkpoint_path,
        intrinsic=intrinsic,
        factor_depth=factor_depth
    )
    
    gg, data_dict = inference.predict(
        color=color,
        depth=depth,
        mask=workspace_mask
    )
    
    # Visualization
    print('Visualizing... (Close the window to exit)')
    cloud_points = data_dict['point_clouds']
    cloud_colors = data_dict['cloud_colors']
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_points)
    cloud.colors = o3d.utility.Vector3dVector(cloud_colors)
    
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])
