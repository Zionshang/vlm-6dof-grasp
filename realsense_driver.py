import pyrealsense2 as rs
import numpy as np

class RealSenseD405:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print("Camera started.")
    
    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            color = np.asanyarray(aligned_frames.get_color_frame().get_data())
            depth = np.asanyarray(aligned_frames.get_depth_frame().get_data())
            return color, depth
        except Exception:
            return None, None

    def release(self):
        self.pipeline.stop()
