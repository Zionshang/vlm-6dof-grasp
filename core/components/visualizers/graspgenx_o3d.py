"""Open3D rendering for GraspGenX gripper-conditioned grasp poses."""
from functools import partial

from registry import register
from visualization_data import graspgenx_geometries
from .o3d import create_o3d_visualizer


@register("visualizer", "graspgenx_o3d",
          requires=("camera", "grasp_engine"))
def build_graspgenx_o3d(cfg=None, hw=None, ctx=None, dependencies=None):
    engine = dependencies["grasp_engine"]
    gripper = getattr(engine, "gripper_info", None)
    if gripper is None:
        raise ValueError("graspgenx_o3d requires a GraspGenX grasp engine")
    vcfg = cfg or {}
    renderer = partial(
        graspgenx_geometries,
        gripper=gripper,
        show_top_mesh=bool(vcfg.get("show_top_mesh", True)),
        top_mesh_color=vcfg.get("top_mesh_color", (0.1, 0.5, 1.0)),
    )
    return create_o3d_visualizer(vcfg, dependencies["camera"], renderer)
