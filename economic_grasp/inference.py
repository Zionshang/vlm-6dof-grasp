import os
import numpy as np
from PIL import Image
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
        num_points=None,
        voxel_size=None,
        m_point=None,
        grasp_max_width=None,
        collision_thresh=None,
    ):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_points = num_points if num_points is not None else cfgs.num_point
        self.voxel_size = voxel_size if voxel_size is not None else cfgs.voxel_size
        self.m_point = m_point if m_point is not None else cfgs.m_point
        self.intrinsic = intrinsic
        self.factor_depth = factor_depth
        self.grasp_max_width = (
            grasp_max_width if grasp_max_width is not None else cfgs.grasp_max_width
        )
        self.collision_thresh = (
            collision_thresh if collision_thresh is not None else cfgs.collision_thresh
        )
        self.net = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required.")
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
        )
        net.to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net

    def _camera_from_intrinsics(self, depth_shape, intrinsic, factor_depth):
        height, width = depth_shape
        return CameraInfo(
            width,
            height,
            intrinsic[0][0],
            intrinsic[1][1],
            intrinsic[0][2],
            intrinsic[1][2],
            factor_depth,
        )

    def _prepare_data(self, color, depth, mask=None, seg_mask=None):
        if depth.ndim == 3:
            depth = depth[..., 0]
        if color.dtype != np.float32 and color.dtype != np.float64:
            color = color.astype(np.float32) / 255.0

        camera = self._camera_from_intrinsics(depth.shape, self.intrinsic, self.factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        depth_mask = depth > 0
        final_mask = depth_mask
        if seg_mask is not None:
            if seg_mask.shape != final_mask.shape:
                seg_mask = np.array(
                    Image.fromarray((seg_mask > 0).astype(np.uint8) * 255).resize(
                        (final_mask.shape[1], final_mask.shape[0]), resample=Image.NEAREST
                    )
                ) > 0
            final_mask = final_mask & seg_mask
        elif mask is not None:
            if mask.shape != final_mask.shape:
                mask = np.array(
                    Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                        (final_mask.shape[1], final_mask.shape[0]), resample=Image.NEAREST
                    )
                ) > 0
            final_mask = final_mask & mask

        cloud_masked = cloud[final_mask]
        color_masked = color[final_mask]
        if len(cloud_masked) == 0:
            raise ValueError("No valid points found after masking.")

        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(
                len(cloud_masked), self.num_points - len(cloud_masked), replace=True
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        data_dict = {
            "point_clouds": cloud_sampled.astype(np.float32),
            "cloud_colors": color_sampled.astype(np.float32),
        }
        batch_data = {
            "point_clouds": torch.from_numpy(data_dict["point_clouds"])
            .unsqueeze(0)
            .to(self.device),
            "coordinates_for_voxel": [
                torch.from_numpy(data_dict["point_clouds"] / self.voxel_size).to(self.device)
            ],
        }
        return data_dict, batch_data


    def predict(self, color, depth, mask=None, use_collision=True, topk=100):
        data_dict, batch_data = self._prepare_data(
            color=color,
            depth=depth,
            mask=mask,
            seg_mask=None,
        )
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(
                end_points, m_point=self.m_point, grasp_max_width=self.grasp_max_width
            )

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        if use_collision and self.collision_thresh > 0:
            mfcdetector = ModelFreeCollisionDetector(
                data_dict["point_clouds"], voxel_size=self.voxel_size
            )
            collision_mask = mfcdetector.detect(
                gg, approach_dist=0.05, collision_thresh=self.collision_thresh
            )
            gg = gg[~collision_mask]

        gg = gg.nms()
        gg = gg.sort_by_score()
        if topk is not None:
            gg = gg[:topk]
        return gg, data_dict
