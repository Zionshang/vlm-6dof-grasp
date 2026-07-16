import numpy as np
import pyrealsense2 as rs


class _RealSenseBase:
    name = "RealSense"

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print(f"{self.name} Camera started.")

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            frames = self.align.process(frames)
            color = np.asanyarray(frames.get_color_frame().get_data())
            depth = np.asanyarray(frames.get_depth_frame().get_data())
            return color, depth
        except Exception:
            return None, None

    def release(self):
        self.pipeline.stop()


class RealSenseD405(_RealSenseBase):
    name = "D405"


class RealSenseD435i(_RealSenseBase):
    name = "D435i"