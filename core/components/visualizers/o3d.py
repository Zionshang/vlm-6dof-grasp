"""o3d 可视化组件:实时彩色点云 + 抓取位姿叠加(主线程渲染)。

update_cloud(color, depth_m)/update_grasps(gg)/poll()/render() 由 Handler 在主线程调用。
点云内参取自 camera 组件的 color 内参。
"""
import numpy as np
import open3d as o3d
from registry import register
from visualization_data import grasp_geometries, point_cloud_arrays


@register("visualizer", "o3d", requires=("camera",))
def build_o3d_visualizer(cfg=None, hw=None, ctx=None, dependencies=None):
    vcfg = cfg or {}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=vcfg.get("title", "Grasp"))
    pcd = o3d.geometry.PointCloud()
    state = {"added": False}
    grasp_geoms = []
    camera = dependencies["camera"]

    class O3DVisualizer:
        def update_cloud(self, color, depth_m):
            intrinsic = np.array([[camera.color_fx, 0, camera.color_cx],
                                  [0, camera.color_fy, camera.color_cy],
                                  [0, 0, 1.0]])
            pts, cols = point_cloud_arrays(color, depth_m, intrinsic)
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
            grasp_geoms = grasp_geometries(gg)
            for geometry in grasp_geoms:
                vis.add_geometry(geometry, reset_bounding_box=False)

        def poll(self):
            return vis.poll_events()

        def render(self):
            vis.update_renderer()

        def close(self):
            vis.destroy_window()

    return O3DVisualizer()
