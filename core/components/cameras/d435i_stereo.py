"""D435i 立体相机组件(4 流:color+IR1+IR2+depth),供 FFS。

step 写 ctx.color + ctx.ir=(ir1,ir2)(FFS depth agent 据此算 ctx.depth)。
对外暴露 color/IR 内参、IR→color 外参、baseline,供 depth(ffs) 组件读取。
"""
from registry import register


@register("camera", "d435i_stereo")
def build_d435i_stereo_camera(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from camera import RealSenseD435iStereo
    cfg = cfg or {}
    cam = RealSenseD435iStereo(width=cfg.get("width", 640), height=cfg.get("height", 480),
                               fps=cfg.get("fps", 30))

    class D435iStereoCamera:
        # 透传分辨率 / 内参 / 外参 / baseline(FFS depth agent 需要)
        width, height = cam.width, cam.height
        color_fx, color_fy = cam.color_fx, cam.color_fy
        color_cx, color_cy = cam.color_cx, cam.color_cy
        ir_fx, ir_fy = cam.ir_fx, cam.ir_fy
        ir_cx, ir_cy = cam.ir_cx, cam.ir_cy
        ir_to_color_R, ir_to_color_T = cam.ir_to_color_R, cam.ir_to_color_T
        baseline = cam.baseline

        def step(self, ctx):
            color, ir1, ir2 = cam.get_stereo_frames()
            if color is not None:
                ctx.color = color
            if ir1 is not None and ir2 is not None:
                ctx.ir = (ir1, ir2)
        def release(self):
            cam.release()
    return D435iStereoCamera()
