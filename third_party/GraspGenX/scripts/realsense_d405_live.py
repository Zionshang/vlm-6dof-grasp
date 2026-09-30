#!/usr/bin/env python3
"""RealSense D405 -> GraspGenX (Piper) -> Open3D live viewer."""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.checkpoint_io import load_model_cfg


# Target-object ROI in the RealSense optical frame (metres):
# +X right, +Y down, +Z forward.
X_MIN, X_MAX = -0.20, 0.20
Y_MIN, Y_MAX = -0.25, 0.25
Z_MIN, Z_MAX = 0.10, 0.65
NUM_GRASPS, MAX_INPUT_POINTS = 100, 5000


def camera_cloud(depth_frame, color_frame, depth_scale):
    """Back-project valid pixels inside the target ROI to camera-frame points."""
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    bgr = np.asanyarray(color_frame.get_data())
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    v, u = np.mgrid[: depth.shape[0], : depth.shape[1]]
    z = depth
    x = (u - intr.ppx) * z / intr.fx
    y = (v - intr.ppy) * z / intr.fy
    keep = (
        np.isfinite(z)
        & (z > Z_MIN) & (z < Z_MAX)
        & (x > X_MIN) & (x < X_MAX)
        & (y > Y_MIN) & (y < Y_MAX)
    )
    return np.stack((x[keep], y[keep], z[keep]), 1), bgr[keep, ::-1] / 255.0


def grasp_lines(grasps, scores, local_points):
    """Make one Piper control-point wireframe per grasp pose."""
    points, lines, colors = [], [], []
    for pose, score in zip(grasps, scores):
        p = local_points @ pose[:3, :3].T + pose[:3, 3]
        base = len(points)
        points.extend(p)
        lines.extend([[base + j, base + j + 1] for j in range(len(p) - 1)])
        colors.extend([[1.0 - score, score, 0.0]] * (len(p) - 1))
    return points, lines, colors


def main():
    ckpt = get_checkpoints_version_dir()
    cfg = load_model_cfg(ckpt / "gen", ckpt / "dis")
    sampler = GraspGenXSampler(cfg, "piper_hand")
    gripper = sampler.get_gripper_info()
    local_lines = np.asarray(gripper.control_points_visualization[0])[:, :3]

    pipe, rs_cfg = rs.pipeline(), rs.config()
    rs_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(rs_cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.depth)

    vis = o3d.visualization.Visualizer()
    vis.create_window("GraspGenX | D405 | piper_hand", 1280, 720)
    cloud, wire = o3d.geometry.PointCloud(), o3d.geometry.LineSet()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(gripper.visual_mesh.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(gripper.visual_mesh.faces))
    mesh.paint_uniform_color([0.1, 0.5, 1.0])
    vis.add_geometry(cloud)
    vis.add_geometry(wire)
    vis.add_geometry(mesh)
    vis.get_render_option().point_size = 2.0
    first = True

    try:
        while vis.poll_events():
            frames = align.process(pipe.wait_for_frames())
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if not depth or not color:
                continue
            points, rgb = camera_cloud(depth, color, scale)
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(rgb)
            vis.update_geometry(cloud)
            if len(points) < 100:
                vis.update_renderer()
                continue

            if len(points) > MAX_INPUT_POINTS:  # bounds GraspGenX's KNN cost
                ids = np.random.choice(len(points), MAX_INPUT_POINTS, replace=False)
                model_points = points[ids]
            else:
                model_points = points

            (grasp_t, score_t), = GraspGenXSampler.run_inference_batch(
                [model_points.astype(np.float32)], sampler,
                grasp_threshold=-1.0, num_grasps=NUM_GRASPS,
                topk_num_grasps=NUM_GRASPS, remove_outliers=True,
            )
            grasps = grasp_t.detach().cpu().numpy()
            scores = np.clip(score_t.detach().cpu().numpy(), 0.0, 1.0)
            p, l, c = grasp_lines(grasps, scores, local_lines)
            wire.points = o3d.utility.Vector3dVector(p)
            wire.lines = o3d.utility.Vector2iVector(l)
            wire.colors = o3d.utility.Vector3dVector(c)
            vis.update_geometry(wire)

            if len(grasps):
                best = grasps[int(scores.argmax())]
                vertices = np.asarray(gripper.visual_mesh.vertices)
                vertices = vertices @ best[:3, :3].T + best[:3, 3]
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.compute_vertex_normals()
                vis.update_geometry(mesh)
                print(f"\rpoints={len(points):6d} grasps={len(grasps):3d} best={scores.max():.3f}", end="")
            if first:
                vis.reset_view_point(True)
                first = False
            vis.update_renderer()
    finally:
        pipe.stop()
        vis.destroy_window()


if __name__ == "__main__":
    main()
