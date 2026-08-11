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
from registry import register

AMP_DTYPE = torch.float16
_FFS_BOOTED = False


def _boot_ffs_env(root):
    """把 FFS 代码目录加入 sys.path(torch.load 反序列化需要 FFS 网络类)+ 设 CC(triton)。"""
    global _FFS_BOOTED
    if _FFS_BOOTED:
        return
    ffs_dir = str((root / "third_party/Fast-FoundationStereo").resolve())
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
    ir_h, ir_w = depth_ir.shape
    color_h, color_w = int(cam.height), int(cam.width)
    if (ir_h, ir_w) != (color_h, color_w):
        raise ValueError(
            "FFS depth size must match the calibrated left-IR stream: "
            f"depth_ir={(ir_h, ir_w)}, stream={(color_h, color_w)}"
        )
    u, v = np.meshgrid(np.arange(ir_w), np.arange(ir_h))
    valid = (depth_ir > 0) & np.isfinite(depth_ir)
    z = depth_ir[valid]
    x = (u[valid] - cam.ir_cx) * z / cam.ir_fx
    y = (v[valid] - cam.ir_cy) * z / cam.ir_fy
    pts = np.stack([x, y, z], axis=-1)
    pts_c = pts @ cam.ir_to_color_R.T + cam.ir_to_color_T
    zc = pts_c[:, 2]
    uc = pts_c[:, 0] / zc * cam.color_fx + cam.color_cx
    vc = pts_c[:, 1] / zc * cam.color_fy + cam.color_cy
    uci = np.round(uc).astype(np.int32)
    vci = np.round(vc).astype(np.int32)
    inside = ((zc > 0) & np.isfinite(zc) & np.isfinite(uc) & np.isfinite(vc)
              & (uci >= 0) & (uci < color_w)
              & (vci >= 0) & (vci < color_h))
    depth_color = np.full(color_h * color_w, np.inf, dtype=np.float32)
    idx = (vci[inside] * color_w + uci[inside]).astype(np.int64)
    # Several IR points can project to one color pixel.  Keep the nearest
    # surface explicitly; repeated advanced-index assignment is not a z-buffer.
    np.minimum.at(depth_color, idx, zc[inside].astype(np.float32))
    depth_color[~np.isfinite(depth_color)] = 0.0
    return depth_color.reshape(color_h, color_w)


@register("depth", "ffs", requires=("camera",))
def build_ffs_depth(cfg=None, hw=None, ctx=None, dependencies=None):
    import paths
    ROOT = paths.PROJECT_ROOT
    _boot_ffs_env(ROOT)
    from core.utils.utils import InputPadder   # FFS core(FFS_DIR 已在 sys.path)

    fcfg = cfg or {}
    cam = dependencies["camera"]
    model_dir = str(ROOT / fcfg.get("model", "third_party/Fast-FoundationStereo/weights/20-26-39/model_best_bp2_serialize.pth"))
    scale = float(fcfg.get("scale", 0.5))
    valid_iters = int(fcfg.get("valid_iters", 4))
    max_disp = int(fcfg.get("max_disp", 192))
    W = int(fcfg.get("width", cam.width))
    H = int(fcfg.get("height", cam.height))

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

        def __init__(self, cam):
            self.cam = cam

        def step(self, ctx):
            """读 ctx.ir(left/right IR)→ FFS → align → 写 ctx.depth(米,color 视角)。

            对齐:ffs_depth 输出对齐 left IR(stream1);align_ir_to_color 用 stream1 IR 内参
            deproject + stream1→color 外参 transform + color 内参 project,left IR→color 准确。
            """
            if ctx.ir is None:
                return
            ir1, ir2 = ctx.ir
            ctx.depth = self.ffs_depth(ir1, ir2, self.cam)

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

    return FFSDepth(cam)
