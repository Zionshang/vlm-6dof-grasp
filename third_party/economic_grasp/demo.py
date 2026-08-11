import os
import sys
import numpy as np
import open3d as o3d
import argparse
import scipy.io as scio
from PIL import Image
import cv2

import torch
from graspnetAPI import GraspGroup
from models.economicgrasp import economicgrasp, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from utils.arguments import cfgs
from utils.vlm_utils import vlm_grasp_visualize, vlm_grasp_visualize_batch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
# sys.path.append(PROJECT_ROOT) # Avoid appending root to prevent module conflict

# Manually add vlm path if needed, but relative imports inside vlm package expect 'vlm' to be top level if running from outside.
# However, inside vlm package, files use 'from src.core ...' which implies 'vlm/src' or 'vlm' folder is a source root.
# Let's add PROJECT_ROOT/vlm to sys.path to allow 'from src.core ...' to work from inside those files
sys.path.append(os.path.join(PROJECT_ROOT, 'vlm'))

from src.apps.grasp_selection import GraspSelectionApp
from src.core.config import load_config

DEFAULT_CHECKPOINT = os.path.join(ROOT_DIR, 'checkpoint', 'economicgrasp_epoch10.tar')
# Correct Camera Matrix (fx, fy, cx, cy are needed)
CAMERA_MATRIX = np.array([
    [435.75600787289994, 0.0, 423.5139606211078],
    [0.0, 435.6741418717883, 243.52287948995928],
    [0.0, 0.0, 1.0]
])

DIST_COEFFS = np.array([
    -0.061438834574162444, 0.11244487882386699, 
    -0.0008922372006081498, 0.0010226929723920338, 
    -0.09769331639799333
])
FACTOR_DEPTH = 10000
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=DEFAULT_CHECKPOINT, help='Model checkpoint path')
args = parser.parse_args()


def get_net(checkpoint_path):
    # Init the model
    net = economicgrasp(
        seed_feat_dim=512,
        is_training=False,
        num_depth=cfgs.num_depth,
        num_angle=cfgs.num_angle,
        m_point=cfgs.m_point,
        num_view=cfgs.num_view,
        graspness_threshold=cfgs.graspness_threshold,
        grasp_max_width=cfgs.grasp_max_width,
        voxel_size=cfgs.voxel_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    # color_path = os.path.join(data_dir, 'color.png')
    # color_img = Image.open(color_path)
    # color_rgb = np.array(color_img)
    # color = color_rgb.astype(np.float32) / 255.0
    color_path  ='/home/jyx/python_ws/vlm-6dof-grasp/output/captures/20260121-115814_color.png'
    color_img = Image.open(color_path)
    color_rgb = np.array(color_img)
    color = color_rgb.astype(np.float32) / 255.0

    depth = np.array(Image.open('/home/jyx/python_ws/vlm-6dof-grasp/output/captures/20260121-115814_depth.png'))
    
    # workspace_mask_path = os.path.join(data_dir, 'workspace_mask.png')
    workspace_mask_path = '/home/jyx/python_ws/vlm-6dof-grasp/output/sam/20260121-115814_sam.png'
    if os.path.exists(workspace_mask_path):
        workspace_mask = np.array(Image.open(workspace_mask_path)) > 0
    else:
        raise FileNotFoundError(f"Workspace mask file not found at {workspace_mask_path}")

    # meta_path = os.path.join(data_dir, 'meta.mat')
    # if os.path.exists(meta_path):
    #     meta = scio.loadmat(meta_path)
    #     intrinsic = meta['intrinsic_matrix']
    #     factor_depth = meta['factor_depth']
    # else:
    #     raise FileNotFoundError(f"Meta file not found at {meta_path}")
    intrinsic = CAMERA_MATRIX
    factor_depth = FACTOR_DEPTH
    # generate cloud
    camera = CameraInfo(848, 480, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    
    end_points = dict()
    cloud_sampled_torch = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled_torch = cloud_sampled_torch.to(device)
    
    end_points['point_clouds'] = cloud_sampled_torch
    end_points['cloud_colors'] = torch.from_numpy(color_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points['coordinates_for_voxel'] = [torch.from_numpy(cloud_sampled / cfgs.voxel_size).to(device)]

    return end_points, cloud, color_rgb, intrinsic

def get_grasps(net, end_points):
    # Forward pass
    print('Running inference...')
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points, m_point=cfgs.m_point, grasp_max_width=cfgs.grasp_max_width)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    print('Running collision detection...')
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    print('Visualizing 3D... (Close the window to exit)')
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir, checkpoint_path):
    net = get_net(checkpoint_path)
    end_points, cloud, color_rgb, intrinsic = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    
    # Process grasps (NMS, Sort, Top-k)
    gg = gg.nms()
    gg.sort_by_score()
    gg = gg[:5]
    
    # 2D Visualization (Batch Mode)
    print('Generating 2D visualizations (Individual Mode)...')
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    
    vis_images, candidates = vlm_grasp_visualize_batch(
        color_bgr, 
        gg.translations, 
        gg.rotation_matrices, 
        gg.widths, 
        intrinsic, 
        top_k=5 # 生成 5 张图
    )
    
    output_dir = os.path.join(os.path.dirname(ROOT_DIR), 'output', '2D_grasp')
    # Clear directory if needed
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, img in enumerate(vis_images):
        save_path = os.path.join(output_dir, f'candidate_{i}.png')
        cv2.imwrite(save_path, img)
        saved_paths.append(save_path)
        print(f'Saved candidate {i} to {save_path}')

    # VLM Selection
    print("Requesting VLM selection...")
    
    # Load VLM Config
    config_path = os.path.join(PROJECT_ROOT, "vlm", "config", "settings.yaml")
    vlm_cfg = load_config(config_path)
    model_name = vlm_cfg.get("default_model", "qwen2.5-vl")
    
    vlm_app = GraspSelectionApp(
        model_name=model_name, 
        prompts_dir=os.path.join(PROJECT_ROOT, "vlm", "prompts")
    )
    result = vlm_app.run(saved_paths)
    
    if result["success"]:
        best_id = result["selected_id"]
        print(f"\n[VLM SELECTED] ID: {best_id}")
        print(f"[REASON] {result['reason']}")
        
        # Highlight best grasp in 3D
        if 0 <= best_id < len(gg):
            best_grasp = gg[best_id] # Assuming index alignment
            # Visualization code will show all, but we printed the choice
    else:
        print(f"[VLM FAILED] {result.get('raw_response')}")

    vis_grasps(gg, cloud)

if __name__=='__main__':
    data_dir = 'example_data'
    demo(data_dir, args.checkpoint_path)
