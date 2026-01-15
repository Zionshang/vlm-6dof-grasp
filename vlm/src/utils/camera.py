import pyrealsense2 as rs
import numpy as np
from PIL import Image
import time

class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30, warmup=1.0):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            
            # Reduce latency by setting queue size to 1
            # Note: This might require different handling depending on pyrealsense2 version, 
            # but usually pipeline config handles streams. 
            # Ideally we set frame_queue size on the sensor, but for simplicity we'll just start.
            
            self.pipeline.start(self.config)
            # Warmup
            if warmup > 0:
                time.sleep(warmup)
            print("RealSense Camera initialized.")
        except Exception as e:
            print(f"Error initializing RealSense: {e}")
            self.pipeline = None
            
    def capture(self, save_path="temp_capture.jpg", max_width=None):
        if not self.pipeline:
            raise RuntimeError("Camera not initialized")

        # Discard old frames to get the freshest one (simple flush)
        # Reducing to 2 frames to speed up capture
        for _ in range(2):
            self.pipeline.try_wait_for_frames(timeout_ms=10)

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Could not capture frame")
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Create PIL Image
        img = Image.fromarray(color_image)
        
        # Resize if requested
        if max_width:
            target_width = max_width 
            if img.width > target_width:
                ratio = target_width / float(img.width)
                new_height = int((float(img.height) * float(ratio)))
                img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

        img.save(save_path)
        return save_path

    def get_frame(self):
        """Returns the current frame as a numpy array (RGB)"""
        if not self.pipeline:
            return None
            
        # Try to get new frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
            
        return np.asanyarray(color_frame.get_data())

    def release(self):
        if self.pipeline:
            self.pipeline.stop()
