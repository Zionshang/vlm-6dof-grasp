"""D405 相机组件(包装 core.camera.RealSenseD405,848x480 rgb8+z16,depth 已 align color,mm)。"""
from registry import register


@register("camera", "d405")
def build_d405_camera(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from camera import RealSenseD405
    cam = RealSenseD405()

    class D405Camera:
        def step(self, ctx):
            color, depth = cam.get_frames()
            if color is not None:
                ctx.color = color
            if depth is not None:
                ctx.depth = depth            # z16 mm(待 depth agent 转米)
        def release(self):
            cam.release()
    return D405Camera()
