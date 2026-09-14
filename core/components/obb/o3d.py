"""Open3D preview backend for a fused OBB and planned gripper poses."""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from obb import OBB_EDGES
from registry import register
from transform import pose_transform


def gripper_lines(pose, width, finger_length=.08):
    transform = pose_transform(pose)
    position, rotation = transform[:3, 3], transform[:3, :3]
    x_axis, y_axis = rotation[:, 0], rotation[:, 1]
    left, right = position-width/2*y_axis, position+width/2*y_axis
    points = np.array([left, right, left+finger_length*x_axis,
                       right+finger_length*x_axis, position-.05*x_axis,
                       position])
    return points, ((0, 1), (0, 2), (1, 3), (4, 5))


def direction_arrow(start, end, color):
    import open3d as o3d
    vector = np.asarray(end, float)-np.asarray(start, float)
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return None
    length = min(.06, .7*norm)
    direction = vector/norm
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=.004, cone_radius=.008,
        cylinder_height=.7*length, cone_height=.3*length,
    )
    rotation = Rotation.align_vectors(
        direction, [[0., 0., 1.]],
    )[0].as_matrix()
    arrow.rotate(rotation, center=np.zeros(3))
    arrow.translate(np.asarray(end, float)-direction*length)
    return arrow.paint_uniform_color(color)


@register("view_plan_visualizer", "o3d")
def build_view_plan_visualizer(cfg=None, hw=None, ctx=None, dependencies=None):
    import open3d as o3d

    def lines(points, edges, color):
        geometry = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector2iVector(edges),
        )
        geometry.colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(edges), 1)),
        )
        return geometry

    class ViewPlanVisualizer:
        def show(self, obb, plan, width, seconds=30, start_pose=None):
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="OBB View Plan Preview")
            vis.add_geometry(lines(obb.vertices, OBB_EDGES, [1., .35, .05]))
            poses = np.asarray(plan.ee_poses, dtype=float).reshape(-1, 6)
            path = np.asarray(
                ([np.asarray(start_pose, dtype=float)[:3]
                  if start_pose is not None else []]
                 + [pose[:3] for pose in poses]),
                dtype=float,
            )
            if len(path) > 1:
                vis.add_geometry(lines(
                    np.asarray(path),
                    [(i, i+1) for i in range(len(path)-1)], [.2, .7, 1.],
                ))
                arrow = direction_arrow(path[-2], path[-1], [.1, .5, 1.])
                if arrow is not None:
                    vis.add_geometry(arrow)
            for index, pose in enumerate(poses):
                points, edges = gripper_lines(pose, width)
                color = ([.2, 1., .3] if index < len(poses)-1
                         else [1., .1, .1])
                vis.add_geometry(lines(points, edges, color))
            vis.get_render_option().line_width = 5.
            vis.reset_view_point(True)
            deadline = time.monotonic()+seconds
            while time.monotonic() < deadline and vis.poll_events():
                vis.update_renderer()
                time.sleep(.02)
            vis.destroy_window()

    return ViewPlanVisualizer()
