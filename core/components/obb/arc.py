"""Pure, fixed-OBB arc planning; perception and motion stay outside."""
import numpy as np
from scipy.spatial.transform import Rotation

from obb import ViewPlan
from registry import register
from transform import camera_pose_to_ee, camera_to_base_transform


def _unit(vector):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("方向无效")
    return vector/norm


def _angle(first, second):
    return float(np.arccos(np.clip(first @ second, -1., 1.)))


def _toward(start, target, angle):
    total = _angle(start, target)
    if total < 1e-8 or angle >= total:
        return target
    return _unit(np.sin(total-angle)/np.sin(total)*start
                 + np.sin(angle)/np.sin(total)*target)


def _look_at(position, target, x_reference):
    z_axis = _unit(target-position)
    x_axis = x_reference-z_axis*(x_reference @ z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        reference = np.array([0., 0., 1.])
        if abs(reference @ z_axis) > .95:
            reference = np.array([0., 1., 0.])
        x_axis = np.cross(reference, z_axis)
    x_axis = _unit(x_axis)
    return np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))


def _nearby_euler(rotation, reference):
    rpy = Rotation.from_matrix(rotation).as_euler("xyz")
    return rpy + 2*np.pi*np.round((reference-rpy)/(2*np.pi))


def _rotation_step(start, target, fraction):
    delta = Rotation.from_matrix(start.T @ target).as_rotvec()
    return start @ Rotation.from_rotvec(fraction*delta).as_matrix()


class OBBArcViewAdjuster:
    """Return a precomputed safe arc, or a no-motion plan when none is safe."""

    FACES = {"red_bag": (("ac", 1),),
             "lunch_box": (("bc", 0), ("ac", 1))}

    def __init__(self, hand_eye_r, hand_eye_t, workspace=None,
                 max_angle_deg=60, step_deg=10, max_move=.15):
        self.r_ec = np.asarray(hand_eye_r, dtype=float)
        self.t_ec = np.asarray(hand_eye_t, dtype=float)
        self.workspace = workspace
        self.max_angle = np.deg2rad(min(float(max_angle_deg), 60.))
        self.step_angle = np.deg2rad(float(step_deg))
        self.max_move = float(max_move)
        if (self.r_ec.shape != (3, 3) or self.t_ec.shape != (3,)
                or min(self.max_angle, self.step_angle, self.max_move) <= 0):
            raise ValueError("view_adjust配置无效")

    def _candidate(self, obb, ee_pose, label, face, axis, sign, frame):
        p_be = np.asarray(ee_pose[:3], float)
        camera_to_base = camera_to_base_transform(
            ee_pose, self.r_ec, self.t_ec,
        )
        r_bc, p_bc = camera_to_base[:3, :3], camera_to_base[:3, 3]
        if frame == "base":
            center, r_bo = obb.center, obb.rotation
        else:
            center = p_bc+r_bc @ obb.center
            r_bo = r_bc @ obb.rotation
        start, normal = _unit(p_bc-center), sign*r_bo[:, axis]
        if start @ normal < -1e-6:  # Never orbit toward the hidden backside.
            return None

        distance = np.linalg.norm(p_bc-center)
        radius = lambda view: .5*np.sum(obb.extents*np.abs(r_bo.T @ view))
        clearance = distance-radius(start)
        if clearance <= 0:
            return None

        def camera_at(theta):
            view = _toward(start, normal, theta)
            return center + (clearance+radius(view))*view

        def motion_at(theta):
            p_camera = camera_at(theta)
            r_camera = _look_at(p_camera, center, r_bc[:, 0])
            p_ee, _ = camera_pose_to_ee(
                p_camera, r_camera, self.r_ec, self.t_ec,
            )
            return max(np.linalg.norm(p_camera-p_bc), np.linalg.norm(p_ee-p_be))

        travel = min(_angle(start, normal), self.max_angle)
        if motion_at(travel) > self.max_move:
            low, high = 0., travel
            for _ in range(20):
                middle = (low+high)/2
                if motion_at(middle) <= self.max_move:
                    low = middle
                else:
                    high = middle
            travel = low

        poses, x_reference = [], r_bc[:, 0]
        previous_rpy = np.asarray(ee_pose[3:], float)

        def append_pose(p_camera, r_camera):
            nonlocal previous_rpy
            p_ee, r_ee = camera_pose_to_ee(
                p_camera, r_camera, self.r_ec, self.t_ec,
            )
            pose = np.r_[p_ee, _nearby_euler(r_ee, previous_rpy)]
            if (not np.all(np.isfinite(pose))
                    or np.linalg.norm(p_ee-p_be) > self.max_move
                    or self.workspace and not self.workspace(*p_ee)):
                return False
            poses.append(pose)
            previous_rpy = pose[3:]
            return True

        # First rotate smoothly around the fixed camera centre until its
        # optical axis points at the OBB.  This removes the initial combined
        # translation/rotation jump before the orbit begins.
        aligned = _look_at(p_bc, center, x_reference)
        align_angle = Rotation.from_matrix(r_bc.T @ aligned).magnitude()
        align_count = int(np.ceil(align_angle/self.step_angle))
        fractions = (np.linspace(1/align_count, 1, align_count)
                     if align_count else ())
        for fraction in fractions:
            if not append_pose(p_bc, _rotation_step(r_bc, aligned, fraction)):
                return None
        x_reference = aligned[:, 0]

        count = int(np.ceil(travel/self.step_angle))
        for theta in np.linspace(travel/count, travel, count) if count else ():
            p_camera = camera_at(theta)
            r_camera = _look_at(p_camera, center, x_reference)
            if not append_pose(p_camera, r_camera):
                return None
            x_reference = r_camera[:, 0]
        visibility = abs(_toward(start, normal, travel) @ normal)
        return ViewPlan(label, face, travel, visibility, center, tuple(poses))

    def plan(self, obb, ee_pose, label, frame="camera"):
        """Consume one OBB result; this method never invokes perception or motion."""
        if label not in self.FACES:
            raise ValueError(f"view_adjust不支持类别: {label}")
        if frame not in ("camera", "base"):
            raise ValueError(f"view_adjust不支持坐标系: {frame}")
        if (len(ee_pose) != 6 or not np.all(np.isfinite(ee_pose))
                or not np.all(np.asarray(obb.extents) > 0)):
            raise ValueError("view_adjust输入位姿或OBB无效")
        plans = [plan for face, axis in self.FACES[label]
                 for sign in (-1, 1)
                 if (plan := self._candidate(
                     obb, ee_pose, label, face, axis, sign, frame)) is not None]
        if not plans:
            return ViewPlan(label, "current", 0., 0., np.asarray(obb.center), (),
                            "没有安全可行的圆弧，保持当前视角")

        def score(plan):
            endpoint = plan.ee_poses[-1] if plan.ee_poses else ee_pose
            lower_bonus = (ee_pose[2]-endpoint[2]
                           if label == "lunch_box" else 0.)
            return 2*plan.visibility-.5*np.linalg.norm(
                endpoint[:3]-ee_pose[:3]) + .25*lower_bonus
        return max(plans, key=score)


@register("view_adjust", "obb_arc")
def build_obb_arc(cfg=None, hw=None, ctx=None, dependencies=None):
    if hw is None:
        raise ValueError("view_adjust requires hardware hand-eye calibration")
    cfg = cfg or {}
    return OBBArcViewAdjuster(
        hw.hand_eye_r, hw.hand_eye_t, workspace=hw.in_workspace,
        **{key: cfg[key] for key in
           ("max_angle_deg", "step_deg", "max_move") if key in cfg},
    )
