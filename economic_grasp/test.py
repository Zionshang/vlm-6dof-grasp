import os
import argparse
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

from utils.collision_detector import ModelFreeCollisionDetector
from utils.arguments import cfgs

from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from models.economicgrasp import economicgrasp, pred_decode

def parse_args():
    parser = argparse.ArgumentParser(description='EconomicGrasp testing')
    parser.add_argument('--dataset_root', required=True, help='Dataset root')
    parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
    parser.add_argument('--save_dir', required=True, help='Dir to save outputs')
    parser.add_argument('--test_mode', required=True, help='Mode of the testing (seen, similar, novel)')
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--inference', action='store_true', help='Whether to inference')
    args, _ = parser.parse_known_args()
    return args


args = parse_args()

# ------------ GLOBAL CONFIG ------------
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# Create dataset and dataloader
num_point = cfgs.num_point
batch_size = cfgs.batch_size
voxel_size = cfgs.voxel_size
collision_thresh = cfgs.collision_thresh
if args.test_mode == 'seen':
    TEST_DATASET = GraspNetDataset(args.dataset_root, split='test_seen',
                                   camera=args.camera, num_points=num_point, remove_outlier=True, augment=False,
                                   load_label=False)
elif args.test_mode == 'similar':
    TEST_DATASET = GraspNetDataset(args.dataset_root, split='test_similar',
                                   camera=args.camera, num_points=num_point, remove_outlier=True, augment=False,
                                   load_label=False)
elif args.test_mode == 'novel':
    TEST_DATASET = GraspNetDataset(args.dataset_root, split='test_novel',
                                   camera=args.camera, num_points=num_point, remove_outlier=True, augment=False,
                                   load_label=False)

SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False,
                             num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

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
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load checkpoint
checkpoint = torch.load(args.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)" % (args.checkpoint_path, start_epoch))


# ------ Testing ------------
def inference():
    batch_interval = 20
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            elif 'graph' in key:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points, m_point=cfgs.m_point, grasp_max_width=cfgs.grasp_max_width)

        # Save results for evaluation
        for i in range(batch_size):
            data_idx = batch_idx * batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # collision detection
            if collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(args.save_dir, SCENE_LIST[data_idx], args.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / batch_interval))
            tic = time.time()


def evaluate_seen():
    ge = GraspNetEval(root=args.dataset_root, camera=args.camera, split='test')
    # In test time, we will select top-10 grasps for each objects (sorted by our predicted score).
    # Then, for all the grasp, we will further select the top-50 grasps for evaluation.
    res, ap = ge.eval_seen(args.save_dir, proc=6)
    save_dir = os.path.join(args.save_dir, 'ap_{}_seen.npy'.format(args.camera))
    np.save(save_dir, res)
    print(f"seen testing, AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")


def evaluate_similar():
    ge = GraspNetEval(root=args.dataset_root, camera=args.camera, split='test')
    # In test time, we will select top-10 grasps for each objects (sorted by our predicted score).
    # Then, for all the grasp, we will further select the top-50 grasps for evaluation.
    res, ap = ge.eval_similar(args.save_dir, proc=6)
    save_dir = os.path.join(args.save_dir, 'ap_{}_similar.npy'.format(args.camera))
    np.save(save_dir, res)
    print(f"similar testing, AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")


def evaluate_novel():
    ge = GraspNetEval(root=args.dataset_root, camera=args.camera, split='test')
    # In test time, we will select top-10 grasps for each objects (sorted by our predicted score).
    # Then, for all the grasp, we will further select the top-50 grasps for evaluation.
    res, ap = ge.eval_novel(args.save_dir, proc=6)
    save_dir = os.path.join(args.save_dir, 'ap_{}_novel.npy'.format(args.camera))
    np.save(save_dir, res)
    print(f"novel testing, AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")


if __name__ == '__main__':
    if args.inference:
        inference()
    if args.test_mode == 'seen':
        evaluate_seen()
    elif args.test_mode == 'similar':
        evaluate_similar()
    elif args.test_mode == 'novel':
        evaluate_novel()
