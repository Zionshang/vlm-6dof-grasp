"""D435i 相机组件(640x480 rgb8+z16,depth align color,mm)。"""
from registry import register


@register("camera", "d435i")
def build_d435i_camera(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from camera import RealSenseD435i
    cfg = cfg or {}
    cam = RealSenseD435i(width=cfg.get("width", 640), height=cfg.get("height", 480),
                         fps=cfg.get("fps", 30))

    class D435iCamera:
        color_fx, color_fy = cam.color_fx, cam.color_fy
        color_cx, color_cy = cam.color_cx, cam.color_cy
        width, height = cam.width, cam.height
        def step(self, ctx):
            color, depth = cam.get_frames()
            if color is not None:
                ctx.color = color
            if depth is not None:
                ctx.depth = depth            # z16 mm(待 depth agent 转米)
        def release(self):
            cam.release()
    return D435iCamera()
