"""RealSense component adapters; camera algorithms remain in core.camera."""
from registry import register


class _CameraAdapter:
    def __init__(self, driver, discard_frames, calibration, optional=False):
        self.driver = driver
        self.discard_frames = int(discard_frames)
        for name in calibration:
            if not optional or hasattr(driver, name):
                setattr(self, name, getattr(driver, name))

    def capture(self, ctx, discard_frames=None):
        """Discard stale frames, then publish one fresh frame set to ctx."""
        count = self.discard_frames if discard_frames is None else int(discard_frames)
        for _ in range(max(0, count)):
            self._read()
        self.step(ctx)

    def close(self):
        self.driver.release()


class RGBDCamera(_CameraAdapter):
    def __init__(self, driver, discard_frames=5):
        super().__init__(driver, discard_frames, (
            "width", "height", "color_fx", "color_fy", "color_cx", "color_cy",
        ), optional=True)

    def _read(self):
        return self.driver.get_frames()

    def step(self, ctx):
        color, depth = self._read()
        if color is not None:
            ctx.color = color
        if depth is not None:
            ctx.depth = depth


class StereoCamera(_CameraAdapter):
    _CALIBRATION = (
        "width", "height", "color_fx", "color_fy", "color_cx", "color_cy",
        "ir_fx", "ir_fy", "ir_cx", "ir_cy", "ir_to_color_R",
        "ir_to_color_T", "baseline",
    )

    def __init__(self, driver, discard_frames=5):
        super().__init__(driver, discard_frames, self._CALIBRATION)

    def _read(self):
        return self.driver.get_stereo_frames()

    def step(self, ctx):
        color, ir1, ir2 = self._read()
        if color is not None:
            ctx.color = color
        if ir1 is not None and ir2 is not None:
            ctx.ir = (ir1, ir2)


def _wrap(driver, cfg, stereo=False):
    adapter = StereoCamera if stereo else RGBDCamera
    return adapter(driver, discard_frames=cfg.get("discard_frames", 5))


@register("camera", "d405")
def build_d405(cfg=None, hw=None, ctx=None, dependencies=None):
    from camera import RealSenseD405
    return _wrap(RealSenseD405(), cfg)


@register("camera", "d435i")
def build_d435i(cfg=None, hw=None, ctx=None, dependencies=None):
    from camera import RealSenseD435i
    return _wrap(RealSenseD435i(
        width=cfg.get("width", 640), height=cfg.get("height", 480),
        fps=cfg.get("fps", 30),
    ), cfg)


@register("camera", "d405_stereo")
def build_d405_stereo(cfg=None, hw=None, ctx=None, dependencies=None):
    from camera import RealSenseD405Stereo
    return _wrap(RealSenseD405Stereo(
        width=cfg.get("width", 848), height=cfg.get("height", 480),
        fps=cfg.get("fps", 30),
    ), cfg, stereo=True)


@register("camera", "d435i_stereo")
def build_d435i_stereo(cfg=None, hw=None, ctx=None, dependencies=None):
    from camera import RealSenseD435iStereo
    return _wrap(RealSenseD435iStereo(
        width=cfg.get("width", 640), height=cfg.get("height", 480),
        fps=cfg.get("fps", 30),
    ), cfg, stereo=True)
