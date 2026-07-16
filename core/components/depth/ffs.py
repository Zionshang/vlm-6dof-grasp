"""FFS 深度优化组件:IR 立体对 → FFS → disp → depth(IR 视角)→ align 到 color。

提供 ffs_depth(ir1, ir2, cam) -> depth_color(米,color 视角);factor_depth=1。
流程照搬原 apps/main_pipeline.py(load_ffs_model + ffs_forward + align_ir_to_color),
供 main_pipeline 迁移到 Handler 后复用。不内置 worker —— 异步编排由 Handler 负责。
"""
import os
import sys
import numpy as np
import torch
import cv2
import yaml
from registry import register

AMP_DTYPE = torch.float16
_FFS_BOOTED = False


def _boot_ffs_env(root):
    """把 FFS 代码目录加入 sys.path(torch.load 反序列化需要 FFS 网络类)+ 设 CC(triton)。"""
    global _FFS_BOOTED
    if _FFS_BOOTED:
        return
    ffs_dir = str((root / "Fast-FoundationStereo").resolve())
    if ffs_dir not in sys.path:
        sys.path.insert(0, ffs_dir)
    if not os.environ.get("CC"):
        cc = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                          "bin", "x86_64-conda-linux-gnu-gcc")
        if os.path.exists(cc):
            os.environ["CC"] = cc
    _FFS_BOOTED = True


def align_ir_to_color(depth_ir, cam):
    """left-IR 视角深度(米)→ 彩色视角深度(米),z-buffer 取最近。"""
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


@register("depth", "ffs")
def build_ffs_depth(ctx=None, cfg=None, hw=None, manager=None, **kw):
    import paths
    ROOT = paths.PROJECT_ROOT
    _boot_ffs_env(ROOT)
    from core.utils.utils import InputPadder   # FFS core(FFS_DIR 已在 sys.path)

    fcfg = cfg or {}
    model_dir = str(ROOT / fcfg.get("model", "Fast-FoundationStereo/weights/20-26-39/model_best_bp2_serialize.pth"))
    scale = float(fcfg.get("scale", 0.5))
    valid_iters = int(fcfg.get("valid_iters", 4))
    max_disp = int(fcfg.get("max_disp", 192))
    W, H = int(fcfg.get("width", 640)), int(fcfg.get("height", 480))

    with open(f"{os.path.dirname(model_dir)}/cfg.yaml") as ff:
        mcfg = yaml.safe_load(ff)
    mcfg.update(valid_iters=valid_iters, max_disp=max_disp, scale=scale)
    print(f"[FFS] loading {model_dir}")
    model = torch.load(model_dir, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()

    # warmup(吸收 torch.compile 首次编译)
    Wh, Hh = int(W * scale), int(H * scale)
    z = torch.zeros(1, 3, Hh, Wh, device="cuda")
    padder0 = InputPadder(z.shape, divis_by=32, force_square=False)
    z0, z1 = padder0.pad(z, z)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(z0, z1, iters=valid_iters, test_mode=True, optimize_build_volume="pytorch1")
    torch.cuda.synchronize()
    print("[FFS] warmup done")

    class FFSDepth:
        factor_depth = 1.0   # 输出已是米

        def ffs_depth(self, ir1, ir2, cam):
            img0 = np.stack([ir1, ir1, ir1], -1)
            img1 = np.stack([ir2, ir2, ir2], -1)
            img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
            img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
            Hw, Ww = img0.shape[:2]
            t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
            t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(t0.shape, divis_by=32, force_square=False)
            t0, t1 = padder.pad(t0, t1)
            with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
                disp = model.forward(t0, t1, iters=valid_iters, test_mode=True, optimize_build_volume="pytorch1")
            disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(Hw, Ww).clip(1e-6, None)
            depth_ir = (cam.ir_fx * scale * cam.baseline / disp).astype(np.float32)
            depth_ir = cv2.resize(depth_ir, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST)
            return align_ir_to_color(depth_ir, cam)

    return FFSDepth()
