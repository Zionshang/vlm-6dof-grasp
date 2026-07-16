"""o3d 可视化组件:实时彩色点云 + 抓取位姿叠加(主线程渲染)。

update_cloud(color, depth_m)/update_grasps(gg)/poll()/render() 由 Handler 在主线程调用。
点云内参取自 camera 组件的 color 内参。
"""
import numpy as np
import open3d as o3d
from registry import register

DEPTH_MAX_M = 3.0   # 与原 main_pipeline 的 make_point_cloud DEPTH_MAX_MM(3000mm)一致


@register("visualizer", "o3d")
def build_o3d_visualizer(ctx=None, cfg=None, hw=None, manager=None, **kw):
    vcfg = cfg or {}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=vcfg.get("title", "Grasp"))
    pcd = o3d.geometry.PointCloud()
    state = {"added": False}
    grasp_geoms = []

    class O3DVisualizer:
        def update_cloud(self, color, depth_m):
            cam = manager.get("camera")
            fx, fy, cx, cy = cam.color_fx, cam.color_fy, cam.color_cx, cam.color_cy
            H, W = depth_m.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            valid = (depth_m > 0) & (depth_m < DEPTH_MAX_M)
            z = depth_m
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts = np.stack([x, y, z], -1)[valid].astype(np.float64)
            cols = color[valid].astype(np.float64) / 255.0
            if len(pts):
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)
                if not state["added"]:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    state["added"] = True
                else:
                    vis.update_geometry(pcd)

        def update_grasps(self, gg):
            nonlocal grasp_geoms
            for g in grasp_geoms:
                vis.remove_geometry(g, reset_bounding_box=False)
            grasp_geoms = []
            if gg is not None and len(gg) > 0:
                for i in range(len(gg)):
                    g = gg[i].to_open3d_geometry(color=(0, 0, 0))
                    grasp_geoms.extend(g if isinstance(g, list) else [g])
                for g in grasp_geoms:
                    vis.add_geometry(g, reset_bounding_box=False)

        def poll(self):
            return vis.poll_events()

        def render(self):
            vis.update_renderer()

        def release(self):
            vis.destroy_window()

    return O3DVisualizer()
