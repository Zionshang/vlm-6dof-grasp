"""GraspGenX adapter for segmented RGB-D object point clouds."""
import sys

import numpy as np

from grasp_candidates import GraspPoses
from registry import register


MIN_TARGET_POINTS_IN_GRASP = 10


def _canonicalize_grasp_twist(poses):
    """Use jaw symmetry to keep local-Z twist within +/-90 deg."""
    poses = np.asarray(poses).copy()
    rotations = poses[:, :3, :3]
    approach = rotations[:, :, 2]
    reference = np.broadcast_to([1.0, 0.0, 0.0], approach.shape).copy()
    reference -= np.sum(reference * approach, axis=1, keepdims=True) * approach
    degenerate = np.linalg.norm(reference, axis=1) < 1e-6
    fallback = np.broadcast_to([0.0, 1.0, 0.0], approach.shape)
    reference[degenerate] = (
        fallback[degenerate]
        - np.sum(
            fallback[degenerate] * approach[degenerate], axis=1, keepdims=True,
        ) * approach[degenerate]
    )
    flip = np.sum(rotations[:, :, 0] * reference, axis=1) < 0.0
    rotations[flip, :, :2] *= -1.0
    return poses


class GraspGenXEngine:
    """Convert instance masks to camera-frame clouds and run GraspGenX."""

    def __init__(self, sampler, intrinsic, factor_depth, planner,
                 num_grasps=200, grasp_threshold=-1.0,
                 min_input_points=100, max_input_points=5000,
                 min_depth_m=0.0, max_depth_m=3.0, use_collision=True,
                 collision_threshold=0.02, max_scene_points=8192,
                 num_collision_samples=2000, collision_batch_size=16,
                 planner_options=None):
        self.sampler = sampler
        self.intrinsic = np.asarray(intrinsic, dtype=np.float32)
        self.factor_depth = float(factor_depth)
        self.planner = planner
        self.num_grasps = int(num_grasps)
        self.grasp_threshold = float(grasp_threshold)
        self.min_input_points = int(min_input_points)
        self.max_input_points = int(max_input_points)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.use_collision = bool(use_collision)
        self.collision_threshold = float(collision_threshold)
        self.max_scene_points = int(max_scene_points)
        self.collision_batch_size = int(collision_batch_size)
        self.planner_options = dict(planner_options or {})

        self.gripper_info = sampler.get_gripper_info()
        self.gripper_width = float(self.gripper_info.width)
        self.gripper_depth = float(self.gripper_info.depth)
        self.gripper_surface_points = None
        if self.use_collision:
            import trimesh

            sampled, _ = trimesh.sample.sample_surface(
                self.gripper_info.collision_mesh,
                int(num_collision_samples),
            )
            self.gripper_surface_points = np.asarray(sampled, dtype=np.float32)

    def _candidates(self, poses=None, scores=None):
        return GraspPoses(
            poses, scores, widths=self.gripper_width,
            depths=self.gripper_depth,
        )

    def _instance_point_clouds(self, depth, masks):
        """Build target clouds and target-excluded scene clouds once per frame."""
        from graspgenx.utils.scene_loaders import depth_to_camera_xyz

        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"GraspGenX depth must be HxW, got {depth.shape}")
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim != 3 or masks.shape[1:] != depth.shape:
            raise ValueError(
                f"GraspGenX masks/depth mismatch: {masks.shape}/{depth.shape}"
            )

        depth_m = depth.astype(np.float32, copy=False) / self.factor_depth
        xyz = depth_to_camera_xyz(depth_m, self.intrinsic)
        valid_depth = (
            np.isfinite(depth_m) & (depth_m > self.min_depth_m) &
            (depth_m < self.max_depth_m)
        )
        object_pcs, scene_pcs = [], []
        for instance_mask in masks:
            points = xyz[valid_depth & instance_mask]
            if len(points) < self.min_input_points:
                continue
            if self.max_input_points > 0 and len(points) > self.max_input_points:
                indices = np.random.choice(
                    len(points), self.max_input_points, replace=False,
                )
                points = points[indices]
            object_pcs.append(np.ascontiguousarray(points, dtype=np.float32))
            if self.use_collision:
                scene = xyz[valid_depth & ~instance_mask]
                if self.max_scene_points > 0 and len(scene) > self.max_scene_points:
                    indices = np.random.choice(
                        len(scene), self.max_scene_points, replace=False,
                    )
                    scene = scene[indices]
                scene_pcs.append(np.ascontiguousarray(scene, dtype=np.float32))
        return object_pcs, scene_pcs

    def _collision_filter(self, results, scene_pcs):
        if not self.use_collision:
            return results, 0

        from graspgenx.utils.collision_filter import filter_colliding_grasps

        filtered, rejected = [], 0
        for scene, (poses, scores, tags, metadata) in zip(scene_pcs, results):
            keep = filter_colliding_grasps(
                scene_pc=scene,
                grasp_poses=poses,
                collision_threshold=self.collision_threshold,
                gripper_surface_points=self.gripper_surface_points,
                batch_size=self.collision_batch_size,
            )
            rejected += len(poses) - int(np.count_nonzero(keep))
            filtered.append((
                poses[keep], scores[keep],
                [tag for tag, accepted in zip(tags, keep) if accepted], metadata,
            ))
        return filtered, rejected

    def _target_filter(self, results, object_pcs):
        """Keep poses whose configured gripper volume contains mask points."""
        lower, upper = np.asarray(self.gripper_info.grasp_volume)
        filtered, rejected = [], 0
        for points, (poses, scores, tags, metadata) in zip(object_pcs, results):
            keep = []
            for pose in poses:
                local = (points - pose[:3, 3]) @ pose[:3, :3]
                keep.append(np.count_nonzero(
                    np.all((local >= lower) & (local <= upper), axis=1)
                ) >= MIN_TARGET_POINTS_IN_GRASP)
            keep = np.asarray(keep, dtype=bool)
            rejected += len(poses) - int(np.count_nonzero(keep))
            filtered.append((
                poses[keep], scores[keep],
                [tag for tag, accepted in zip(tags, keep) if accepted], metadata,
            ))
        return filtered, rejected

    def predict(self, color, depth, mask=None, topk=100):
        del color  # GraspGenX currently consumes geometry, not RGB features.
        if mask is None:
            mask = np.ones(np.asarray(depth).shape[:2], dtype=bool)
        object_pcs, scene_pcs = self._instance_point_clouds(depth, mask)
        if not object_pcs:
            return self._candidates(), {
                "point_clouds": [], "generated_count": 0,
                "target_rejected": 0, "collision_rejected": 0,
                "output_count": 0,
            }

        from graspgenx.samplers import run_planner_on_batch

        results = run_planner_on_batch(
            object_pcs, self.sampler, planner=self.planner,
            grasp_threshold=self.grasp_threshold,
            num_grasps=self.num_grasps,
            # Keep the configured pool intact for geometric filtering. Passing
            # -1 here triggers GraspGenX's implicit 100-candidate cap.
            topk_num_grasps=self.num_grasps,
            **self.planner_options,
        )
        generated_count = sum(len(item[0]) for item in results)
        results, target_rejected = self._target_filter(results, object_pcs)
        results, collision_rejected = self._collision_filter(results, scene_pcs)
        pose_groups = [item[0] for item in results if len(item[0])]
        score_groups = [item[1] for item in results if len(item[0])]
        if not pose_groups:
            return self._candidates(), {
                "point_clouds": object_pcs,
                "generated_count": generated_count,
                "target_rejected": target_rejected,
                "collision_rejected": collision_rejected,
                "output_count": 0,
            }

        poses = _canonicalize_grasp_twist(
            np.concatenate(pose_groups).astype(np.float32, copy=False)
        )
        scores = np.concatenate(score_groups).astype(np.float32, copy=False)
        order = np.argsort(scores)[::-1]
        if topk is not None:
            order = order[:int(topk)]
        candidates = self._candidates(poses[order], scores[order])
        return candidates, {
            "point_clouds": object_pcs,
            "generated_count": generated_count,
            "target_rejected": target_rejected,
            "collision_rejected": collision_rejected,
            "output_count": len(candidates),
        }


def _runtime_intrinsic(camera, hw):
    if camera is not None and hasattr(camera, "color_fx"):
        return np.array([
            [camera.color_fx, 0, camera.color_cx],
            [0, camera.color_fy, camera.color_cy],
            [0, 0, 1.0],
        ])
    if hw is not None and hw.camera_matrix is not None:
        return hw.camera_matrix
    raise ValueError("GraspGenX requires runtime or configured camera intrinsics")


@register("grasp_engine", "graspgenx", requires=("camera", "depth"))
def build_graspgenx(cfg=None, hw=None, ctx=None, dependencies=None):
    del ctx
    import paths

    gcfg = cfg or {}
    root = paths.PROJECT_ROOT
    source_dir = root / gcfg.get("source", "third_party/GraspGenX")
    source_text = str(source_dir.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    checkpoint_dir = root / gcfg.get(
        "checkpoint_dir",
        "third_party/GraspGenX/ext/graspgenx_checkpoints/release",
    )
    gen_dir, dis_dir = checkpoint_dir / "gen", checkpoint_dir / "dis"
    if not gen_dir.is_dir() or not dis_dir.is_dir():
        raise FileNotFoundError(
            f"GraspGenX checkpoint root must contain gen/ and dis/: {checkpoint_dir}"
        )

    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg

    model_cfg = load_model_cfg(
        gen_dir, dis_dir, gcfg.get("gen_checkpoint"),
        gcfg.get("dis_checkpoint"),
    )
    assets_dir = root / gcfg.get("assets_dir", "third_party/GraspGenX/assets")
    sampler = GraspGenXSampler(
        model_cfg, gripper_name=gcfg.get("gripper", "piper_hand"),
        assets_dir=str(assets_dir),
        use_tensorrt=bool(gcfg.get("use_tensorrt", False)),
        tensorrt_precision=gcfg.get("tensorrt_precision", "fp32"),
    )

    depth = dependencies["depth"]
    factor = getattr(depth, "factor_depth", None)
    if factor is None:
        factor = hw.factor_depth if hw is not None else None
    if not factor:
        raise ValueError("GraspGenX depth component must declare factor_depth")

    planner_keys = (
        "moe_num_yaws", "moe_z_offsets_cm", "moe_outlier_threshold",
        "moe_outlier_k", "moe_obb_mode", "moe_skip_obb_rule",
        "moe_obb_density", "moe_obb_position_spacing_cm",
    )
    return GraspGenXEngine(
        sampler, _runtime_intrinsic(dependencies["camera"], hw), factor,
        planner=gcfg.get("planner", "graspmoe"),
        num_grasps=gcfg.get("num_grasps", 200),
        grasp_threshold=gcfg.get("grasp_threshold", -1.0),
        min_input_points=gcfg.get("min_input_points", 100),
        max_input_points=gcfg.get("max_input_points", 5000),
        min_depth_m=gcfg.get("min_depth_m", 0.0),
        max_depth_m=gcfg.get("max_depth_m", 3.0),
        use_collision=gcfg.get("use_collision", True),
        collision_threshold=gcfg.get("collision_threshold", 0.02),
        max_scene_points=gcfg.get("max_scene_points", 8192),
        num_collision_samples=gcfg.get("num_collision_samples", 2000),
        collision_batch_size=gcfg.get("collision_batch_size", 16),
        planner_options={key: gcfg[key] for key in planner_keys if key in gcfg},
    )
