"""相机层:可插拔。

定义 Camera 协议与默认的 RealSense 实现。换相机(其他型号 / 文件回放 / 仿真)
只需新增一个实现 Camera 接口的类,入口与 pipeline 无需改动。
"""
from typing import Protocol, Optional, Tuple
import numpy as np


class Camera(Protocol):
    """相机接口:获取对齐的 (color, depth) 帧,以及释放资源。"""
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: ...
    def release(self) -> None: ...


class RealSenseD405:
    """RealSense D405 相机驱动(原 realsense_driver.py 逻辑,原样保留)。"""

    def __init__(self):
        import pyrealsense2 as rs
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


class RealSenseD435i:
    """RealSense D435i 相机驱动。

    彩色(rgb8)/ 深度(z16)流默认 640x480 @ 30,深度对齐到彩色——内参对应
    config/hardware/realsense_d435i.yaml(640x480 彩色流)。改分辨率/帧率时,
    需在该 yaml 重测内参。
    """

    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(self.config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.color_fx, self.color_fy = intr.fx, intr.fy
        self.color_cx, self.color_cy = intr.ppx, intr.ppy
        self.width, self.height = width, height
        self.align = rs.align(rs.stream.color)
        print("Camera (D435i) started.")

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


def _ir_to_uint8(img):
    """IR 帧(uint8/uint16)归一化到 uint8 灰度,供 FFS 网络输入。

    参考 third_party/Fast-FoundationStereo/scripts/run_realsense_demo.py:ir_to_uint8。
    """
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    mask = img > 0
    if mask.any():
        mn, mx = np.percentile(img[mask], [1, 99])
        img = np.clip((img - mn) / (mx - mn + 1e-6), 0, 1) * 255
    return img.astype(np.uint8)


def _realsense_rotation_matrix(extrinsics):
    """Convert librealsense's column-major flat rotation to a NumPy matrix."""
    return np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3).T


class RealSenseD435iStereo:
    """RealSense D435i 立体驱动:开 color + 左右 IR + depth 四流,供 Fast-FoundationStereo。

    start 时缓存 color / left-IR 内参、IR→color 外参、stereo baseline,供 FFS 深度计算
    (depth = ir_fx*scale*baseline/disp)与「IR 深度→彩色对齐」使用。IR 流不做 align
    (FFS 需要 raw 立体对)。参考 third_party/Fast-FoundationStereo/scripts/run_realsense_demo.py。
    """

    def __init__(self, width=640, height=480, fps=30, device_name="D435i"):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        self.config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(self.config)
        self.width, self.height = width, height

        color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        ir_intr = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
        self.color_fx, self.color_fy = color_intr.fx, color_intr.fy
        self.color_cx, self.color_cy = color_intr.ppx, color_intr.ppy
        self.ir_fx, self.ir_fy = ir_intr.fx, ir_intr.fy
        self.ir_cx, self.ir_cy = ir_intr.ppx, ir_intr.ppy

        extr = profile.get_stream(rs.stream.infrared, 1).get_extrinsics_to(profile.get_stream(rs.stream.color))
        self.ir_to_color_R = _realsense_rotation_matrix(extr)
        self.ir_to_color_T = np.array(extr.translation, dtype=np.float64)

        dev = profile.get_device()
        depth_sensor = next((s for s in dev.query_sensors() if s.is_depth_sensor()), None)
        raw = depth_sensor.get_option(rs.option.stereo_baseline) if depth_sensor else 0.0
        self.baseline = raw / 1000.0 if raw > 0.5 else raw  # mm→m 或已是 m
        print(f"Camera ({device_name} stereo) started | color_fx={self.color_fx:.2f} "
              f"ir_fx={self.ir_fx:.2f} baseline={self.baseline:.4f}m")

    def get_stereo_frames(self):
        """返回 (color_rgb, ir1, ir2)。IR 归一化 uint8;color rgb8 直接 RGB。"""
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        color = np.asanyarray(frames.get_color_frame().get_data())  # rgb8 → RGB
        ir1 = np.asanyarray(frames.get_infrared_frame(1).get_data())
        ir2 = np.asanyarray(frames.get_infrared_frame(2).get_data())
        return color, _ir_to_uint8(ir1), _ir_to_uint8(ir2)

    def release(self):
        self.pipeline.stop()


class RealSenseD405Stereo(RealSenseD435iStereo):
    """D405 color + stereo IR streams for FFS depth reconstruction."""

    def __init__(self, width=848, height=480, fps=30):
        super().__init__(width=width, height=height, fps=fps, device_name="D405")
