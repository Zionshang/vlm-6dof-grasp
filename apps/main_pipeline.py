"""D435i 相机 → (可选 FFS 深度优化)→ 彩色点云 → EconomicGrasp 抓取 → o3d 可视化。

--use_ffs true(默认): D435iStereo(IR 立体)→ FFS 优化深度 → align 彩色 → 抓取
--use_ffs false:       D435i(硬件 depth)→ 直接转米 → 抓取(不走 FFS,用于对比/排查)

抓取推理在单子线程(FFS+grasp 串联),与主线程 o3d(OpenGL)隔离。
"""
import sys
import os
import time
import argparse
import threading
import yaml
import numpy as np
import torch
import cv2
import open3d as o3d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
ROOT = paths.PROJECT_ROOT
[os.environ.pop(k, None) for k in list(os.environ) if "PROXY" in k.upper()]

# FFS 代码路径 + triton CC(仅 use_ffs=True 时真正用到,但先准备好)
_FFS_DIR = str((ROOT / "Fast-FoundationStereo").resolve())
if _FFS_DIR not in sys.path:
    sys.path.insert(0, _FFS_DIR)
if not os.environ.get('CC'):
    _cc = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'bin', 'x86_64-conda-linux-gnu-gcc')
    if os.path.exists(_cc):
        os.environ['CC'] = _cc

AMP_DTYPE = torch.float16

from camera import RealSenseD435i, RealSenseD435iStereo
from economic_grasp.inference import EconomicGraspInference
from utils.data_utils import create_point_cloud_from_depth_image, CameraInfo

DEPTH_MAX_M = 3.0


# ---------------- 点云 / 抓取几何(两条路径共用) ----------------

def make_point_cloud(color, depth_m, cam_info):
    """深度(米)+ CameraInfo(scale=1)→ 彩色点云(米)。"""
    cloud = create_point_cloud_from_depth_image(depth_m, cam_info, organized=True)
    valid = (depth_m > 0) & (depth_m < DEPTH_MAX_M)
    pts = cloud[valid].astype(np.float64)
    cols = color[valid].astype(np.float64) / 255.0
    return pts, cols


def build_grasp_geoms(gg):
    geoms = []
    if gg is None or len(gg) == 0:
        return geoms
    for i in range(len(gg)):
        g = gg[i].to_open3d_geometry(color=(0, 0, 0))
        geoms.extend(g if isinstance(g, list) else [g])
    return geoms


# ---------------- FFS 路径(depth 优化) ----------------

def align_ir_to_color(depth_ir, cam):
    H, W = depth_ir.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth_ir > 0
    z = depth_ir
    x = (u - cam.ir_cx) * z / cam.ir_fx
    y = (v - cam.ir_cy) * z / cam.ir_fy
    pts = np.stack([x, y, z], -1).reshape(-1, 3)
    pts_c = pts @ cam.ir_to_color_R.T + cam.ir_to_color_T
    zc = pts_c[:, 2]
    uc = pts_c[:, 0] / zc * cam.color_fx + cam.color_cx
    vc = pts_c[:, 1] / zc * cam.color_fy + cam.color_cy
    uci = np.round(uc).astype(np.int32)
    vci = np.round(vc).astype(np.int32)
    m = valid.reshape(-1) & (zc > 0) & (uci >= 0) & (uci < W) & (vci >= 0) & (vci < H)
    depth_color = np.zeros(H * W, dtype=np.float32)
    zc_m, uci_m, vci_m = zc[m], uci[m], vci[m]
    order = np.argsort(-zc_m)
    idx = (vci_m[order] * W + uci_m[order]).astype(np.int64)
    depth_color[idx] = zc_m[order]
    return depth_color.reshape(H, W)


def load_ffs_model(model_dir, valid_iters, max_disp, scale, W, H):
    from core.utils.utils import InputPadder
    with open(f'{os.path.dirname(model_dir)}/cfg.yaml') as ff:
        cfg = yaml.safe_load(ff)
    cfg['valid_iters'] = valid_iters
    cfg['max_disp'] = max_disp
    cfg['scale'] = scale
    print(f"[main] Loading FFS from {model_dir}")
    model = torch.load(model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()
    Wh, Hh = int(W * scale), int(H * scale)
    z = torch.zeros(1, 3, Hh, Wh, device='cuda')
    padder = InputPadder(z.shape, divis_by=32, force_square=False)
    z0, z1 = padder.pad(z, z)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(z0, z1, iters=valid_iters, test_mode=True, optimize_build_volume='pytorch1')
    torch.cuda.synchronize()
    print("[main] FFS warmup done.")
    return model


def ffs_forward(model, ir1, ir2, cam, scale, valid_iters):
    from core.utils.utils import InputPadder
    img0 = np.stack([ir1, ir1, ir1], -1)
    img1 = np.stack([ir2, ir2, ir2], -1)
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
    Hw, Ww = img0.shape[:2]
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(img0, img1, iters=valid_iters, test_mode=True, optimize_build_volume='pytorch1')
    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(Hw, Ww).clip(1e-6, None)
    depth_ffs = cam.ir_fx * scale * cam.baseline / disp
    depth_ffs = cv2.resize(depth_ffs, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST)
    return depth_ffs.astype(np.float32)


# ---------------- worker(单子线程串联,避免双 worker 并发 CUDA 崩) ----------------

class _BaseWorker:
    """submit/take 框架。子类实现 _process(frame) -> (depth_m, color, gg|None)。"""
    def _process(self, frame):
        raise NotImplementedError

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._result = None
        self._running = True
        self._traced = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self._running:
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                torch.cuda.empty_cache()
                res = self._process(frame)
            except Exception as e:
                print(f"[worker] 异常: {str(e).splitlines()[0]}")
                if not self._traced:
                    self._traced = True
                    import traceback
                    traceback.print_exc()
                res = None
            with self._lock:
                self._result = res

    def submit(self, frame):
        with self._lock:
            self._frame = frame

    def take(self):
        with self._lock:
            r = self._result
            self._result = None
            return r

    def stop(self):
        self._running = False
        self.thread.join(timeout=2.0)


class FFSGraspWorker(_BaseWorker):
    """FFS on: IR→FFS→depth→align→predict(串联)。frame = (color, ir1, ir2)。"""
    def __init__(self, model, cam, scale, valid_iters, engine, depth_max=DEPTH_MAX_M):
        self.model, self.cam, self.scale, self.valid_iters = model, cam, scale, valid_iters
        self.engine, self.depth_max = engine, depth_max
        super().__init__()

    def _process(self, frame):
        color, ir1, ir2 = frame
        depth_ir = ffs_forward(self.model, ir1, ir2, self.cam, self.scale, self.valid_iters)
        depth_color = align_ir_to_color(depth_ir, self.cam)
        mask = (depth_color > 0) & (depth_color < self.depth_max)
        gg = None
        if int(mask.sum()) >= self.engine.num_points:
            gg, _ = self.engine.predict(color, depth_color, mask=mask, topk=None)
        return depth_color, color, gg


class HardwareGraspWorker(_BaseWorker):
    """FFS off: 硬件 depth(mm)→ 米 → predict。frame = (color, depth_mm)。"""
    def __init__(self, engine, depth_max=DEPTH_MAX_M):
        self.engine, self.depth_max = engine, depth_max
        super().__init__()

    def _process(self, frame):
        color, depth_mm = frame
        depth_m = (depth_mm / 1000.0).astype(np.float32)   # mm → 米(硬件 depth 已 align color)
        mask = (depth_m > 0) & (depth_m < self.depth_max)
        gg = None
        if int(mask.sum()) >= self.engine.num_points:
            gg, _ = self.engine.predict(color, depth_m, mask=mask, topk=None)
        return depth_m, color, gg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ffs", type=lambda x: x.lower() == "true", default=True,
                        help="true=FFS 优化深度; false=相机硬件 depth 直通")
    parser.add_argument("--ffs_model",
                        default="Fast-FoundationStereo/weights/20-26-39/model_best_bp2_serialize.pth")
    parser.add_argument("--ffs_scale", type=float, default=0.5)
    parser.add_argument("--ffs_valid_iters", type=int, default=4)
    parser.add_argument("--ffs_max_disp", type=int, default=192)
    parser.add_argument("--grasp_checkpoint",
                        default="economic_grasp/checkpoint/economicgrasp_epoch10.tar")
    parser.add_argument("--use_collision", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--profile",
                        default="config/hardware/realsense_d435i.yaml",
                        help="相机内参 yaml(FFS off 时用其 color 内参)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # 相机 + 内参(两条路径不同)
    if args.use_ffs:
        cam = RealSenseD435iStereo(width=args.width, height=args.height, fps=args.fps)
        color_fx, color_fy, color_cx, color_cy = cam.color_fx, cam.color_fy, cam.color_cx, cam.color_cy
    else:
        cam = RealSenseD435i(width=args.width, height=args.height, fps=args.fps)
        with open(ROOT / args.profile) as f:
            cam_cfg = yaml.safe_load(f)["camera"]
        K = np.array(cam_cfg["intrinsic"], dtype=float)
        color_fx, color_fy, color_cx, color_cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam_info = CameraInfo(args.width, args.height, color_fx, color_fy, color_cx, color_cy, 1.0)
    K_color = np.array([[color_fx, 0, color_cx], [0, color_fy, color_cy], [0, 0, 1.0]], dtype=float)

    # EconomicGrasp(depth 已是米 → factor_depth=1,两条路径一致)
    print(f"[main] Loading EconomicGrasp from {args.grasp_checkpoint} ...")
    engine = EconomicGraspInference(str(ROOT / args.grasp_checkpoint),
                                    intrinsic=K_color, factor_depth=1.0,
                                    use_collision=args.use_collision)
    print("[main] EconomicGrasp ready.")

    # worker
    if args.use_ffs:
        ffs_model = load_ffs_model(str(ROOT / args.ffs_model), args.ffs_valid_iters,
                                   args.ffs_max_disp, args.ffs_scale, args.width, args.height)
        worker = FFSGraspWorker(ffs_model, cam, args.ffs_scale, args.ffs_valid_iters, engine, DEPTH_MAX_M)
    else:
        ffs_model = None
        worker = HardwareGraspWorker(engine, DEPTH_MAX_M)
    print(f"[main] use_ffs={args.use_ffs}")

    # o3d
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"D435i Grasp (use_ffs={args.use_ffs})  [Q] quit")
    pcd = o3d.geometry.PointCloud()
    pcd_added = False
    grasp_geoms = []
    state = {"quit": False}

    def cb_quit(_vis):
        state["quit"] = True
        return False

    vis.register_key_callback(ord('Q'), cb_quit)
    vis.register_key_callback(ord('q'), cb_quit)

    print("\n[main] 实时抓取中(单 worker)。  [Q / Esc / 关窗] 退出\n")
    try:
        while not state["quit"]:
            if args.use_ffs:
                color, ir1, ir2 = cam.get_stereo_frames()
                if color is None or ir1 is None or ir2 is None:
                    if not vis.poll_events():
                        break
                    vis.update_renderer()
                    continue
                worker.submit((color, ir1, ir2))
            else:
                color, depth_mm = cam.get_frames()
                if color is None or depth_mm is None:
                    if not vis.poll_events():
                        break
                    vis.update_renderer()
                    continue
                worker.submit((color, depth_mm))

            res = worker.take()
            if res is not None:
                depth_m, color_f, gg = res
                pts, cols = make_point_cloud(color_f, depth_m, cam_info)
                if len(pts):
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.colors = o3d.utility.Vector3dVector(cols)
                    if not pcd_added:
                        vis.add_geometry(pcd, reset_bounding_box=True)
                        pcd_added = True
                    else:
                        vis.update_geometry(pcd)
                if gg is not None:
                    new_geoms = build_grasp_geoms(gg)
                    for g in grasp_geoms:
                        vis.remove_geometry(g, reset_bounding_box=False)
                    grasp_geoms = new_geoms
                    for g in grasp_geoms:
                        vis.add_geometry(g, reset_bounding_box=False)
                    print(f"[main] 更新抓取: {len(grasp_geoms)} 组位姿")

            if not vis.poll_events():
                break
            vis.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        vis.destroy_window()
        cam.release()
        print("[main] 已退出。")


if __name__ == "__main__":
    main()
