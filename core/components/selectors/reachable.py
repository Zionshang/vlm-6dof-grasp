"""Pinocchio/Pink reachability component."""
import numpy as np

from registry import register
from transform import PIPER_FLANGE_T_TCP, pose_transform


class ReachabilitySelector:
    """Precheck observation poses and select safe multiview grasps."""

    def __init__(self, urdf, solver="quadprog", joint_limit_margin_deg=1.0,
                 singular_threshold=1e-4, risk_tolerance=0.05,
                 max_joint_risk=1.0, view_joint_limit_margin_deg=5.0):
        import pinocchio as pin
        import qpsolvers

        if solver not in qpsolvers.available_solvers:
            raise RuntimeError(
                f"IK求解器{solver}不可用，可用项: {qpsolvers.available_solvers}"
            )
        model = pin.buildModelFromUrdf(str(urdf))
        keep = {model.getJointId(f"link{i}_joint") for i in range(1, 7)}
        locked = [i for i in range(1, model.njoints) if i not in keep]
        self.model = pin.buildReducedModel(model, locked, pin.neutral(model))
        self.solver = solver
        self.joint_limit_margin = np.deg2rad(float(joint_limit_margin_deg))
        if self.joint_limit_margin < 0:
            raise ValueError("关节限位安全余量不能小于0")
        self.singular_threshold = float(singular_threshold)
        self.risk_tolerance = float(risk_tolerance)
        self.max_joint_risk = float(max_joint_risk)
        self.view_joint_limit_margin = np.deg2rad(
            float(view_joint_limit_margin_deg)
        )

    def _check_joint_limits(self, joints, margin=None):
        joints = np.asarray(joints, dtype=float).reshape(-1)
        margin = self.joint_limit_margin if margin is None else float(margin)
        lower = np.asarray(self.model.lowerPositionLimit) + margin
        upper = np.asarray(self.model.upperPositionLimit) - margin
        if joints.shape != lower.shape:
            raise RuntimeError(
                f"IK关节数量错误: {joints.size}，期望{lower.size}"
            )
        exceeded = np.flatnonzero(
            ~np.isfinite(joints) | (joints < lower) | (joints > upper)
        )
        if exceeded.size:
            details = ", ".join(
                f"J{i + 1}={np.rad2deg(joints[i]):.1f}°"
                f"∉[{np.rad2deg(lower[i]):.1f}°,{np.rad2deg(upper[i]):.1f}°]"
                for i in exceeded
            )
            raise RuntimeError(f"IK关节超限: {details}")

    def _joint_risks(self, current, reach):
        lower = np.asarray(self.model.lowerPositionLimit)
        span = np.asarray(self.model.upperPositionLimit) - lower
        current_ratio = (np.asarray(current) - lower) / span
        reach_ratio = (np.asarray(reach) - lower) / span
        motion = np.abs(reach_ratio - current_ratio)
        limit = np.abs(2.0 * reach_ratio - 1.0)
        return motion, limit, motion + limit - motion * limit

    def _extremely_singular(self, joints):
        import pinocchio as pin

        jacobian = pin.computeFrameJacobian(
            self.model, self.model.createData(), np.asarray(joints),
            self.model.getFrameId("piperL_flange_link"), pin.LOCAL,
        )
        jacobian[:3] /= .10
        sigma_min = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
        return (not np.isfinite(sigma_min)
                or sigma_min < self.singular_threshold), sigma_min

    def select_observation_pose(self, candidates, current_joints):
        """Return the widest observation pose passing IK and safety checks."""
        lower = np.asarray(self.model.lowerPositionLimit)
        upper = np.asarray(self.model.upperPositionLimit)
        checks = []
        for candidate in candidates:
            item = {
                "angle_deg": candidate["angle_deg"],
                "pose": np.asarray(candidate["pose"]).tolist(),
                "reachable": False, "reason": None,
            }
            joints, seed = self._observation_ik(
                candidate["pose"], current_joints,
            )
            item["ik_seed"] = seed
            if joints is None:
                item["reason"] = "ik_failed_all_seeds"
            else:
                margin = np.minimum(joints - lower, upper - joints)
                item["min_margin_deg"] = float(np.rad2deg(margin.min()))
                singular, item["sigma_min"] = self._extremely_singular(joints)
                if singular:
                    item["reason"] = "extreme_singularity"
                else:
                    item["reachable"] = True
                    checks.append(item)
                    return candidate, checks
            checks.append(item)
        return None, checks

    def _observation_ik(self, pose, current):
        """Try the live state first, then one stable Piper viewing branch."""
        joints = self._ik(pose, current, self.view_joint_limit_margin)
        if joints is not None:
            return joints, "current"
        pose = np.asarray(pose, dtype=float)
        seed = np.deg2rad([0.0, 80.0, -23.0, 0.0, 0.0, 0.0])
        seed[0] = np.arctan2(pose[1], pose[0])
        joints = self._ik(pose, seed, self.view_joint_limit_margin)
        return joints, "front_nominal" if joints is not None else None

    def select_multiview(self, candidates, reference_joints):
        """Evaluate explicit approach/reach poses and select by joint risk."""
        reference = np.asarray(reference_joints, dtype=float)
        evaluations = []
        for candidate in candidates:
            item = dict(candidate)
            item.update(status="rejected", reason=None, reachable=False)
            reach = np.asarray(candidate["tcp_pose"], dtype=float)
            approach = np.asarray(candidate["approach_pose"], dtype=float)

            approach_joints = self._ik(approach, reference)
            if approach_joints is None:
                item["reason"] = "approach_ik_failed"
                evaluations.append(item)
                continue
            reach_joints = self._ik(reach, approach_joints)
            if reach_joints is None:
                item["reason"] = "reach_ik_failed"
                evaluations.append(item)
                continue
            singular, sigma_min = self._extremely_singular(reach_joints)
            item["sigma_min"] = sigma_min
            if singular:
                item["reason"] = "extreme_singularity"
                evaluations.append(item)
                continue

            motion, limit, risks = self._joint_risks(reference, reach_joints)
            worst = int(np.argmax(risks))
            item.update(
                status="reachable", reachable=True,
                approach_pose=approach.tolist(),
                approach_joints=np.asarray(approach_joints).tolist(),
                reach_joints=np.asarray(reach_joints).tolist(),
                motion=motion.tolist(), limit=limit.tolist(),
                joint_risks=risks.tolist(), risk=float(risks[worst]),
                worst_joint=worst + 1,
                motion_worst=float(motion[worst]),
                limit_worst=float(limit[worst]),
            )
            evaluations.append(item)

        reachable = [item for item in evaluations if item["reachable"]]
        if not reachable:
            return None, evaluations, None, None
        risk_min = min(item["risk"] for item in reachable)
        if risk_min >= self.max_joint_risk:
            for item in reachable:
                item.update(status="risk_rejected",
                            reason="joint_risk_above_limit")
            return None, evaluations, risk_min, self.max_joint_risk
        risk_limit = min(risk_min + self.risk_tolerance,
                         self.max_joint_risk)
        pool = [item for item in reachable
                if item["risk"] < self.max_joint_risk
                and item["risk"] <= risk_limit]
        for item in pool:
            item["status"] = "risk_pool"
        selected = max(pool, key=lambda item: (
            item["score"], -item["risk"], item["view"] == "front",
        ))
        selected["status"] = "selected"
        return selected, evaluations, risk_min, risk_limit

    def _ik(self, target_tcp, current, joint_limit_margin=None):
        import pinocchio as pin
        import pink
        from pink.tasks import FrameTask, PostureTask

        current = np.asarray(current, dtype=float).copy()
        configuration = pink.Configuration(
            self.model, self.model.createData(), current,
        )
        base = configuration.get_transform_frame_to_world("piperL_base_link")
        flange = pose_transform(target_tcp) @ np.linalg.inv(PIPER_FLANGE_T_TCP)
        task = FrameTask("piperL_flange_link", 1.0, 1.0)
        task.set_target(base * pin.SE3(flange[:3, :3], flange[:3, 3]))
        posture = PostureTask(1e-3)
        posture.set_target(current)
        for _ in range(100):
            error = task.compute_error(configuration)
            if (np.linalg.norm(error[:3]) < .010 and
                    np.linalg.norm(error[3:]) < np.deg2rad(5)):
                try:
                    self._check_joint_limits(
                        configuration.q, joint_limit_margin,
                    )
                except RuntimeError:
                    return None
                return configuration.q.copy()
            velocity = pink.solve_ik(
                configuration, [task, posture], .01, self.solver,
            )
            configuration.integrate_inplace(velocity, .01)
            try:
                configuration.check_limits(safety_break=True)
            except pink.exceptions.NotWithinConfigurationLimits:
                return None
        return None


@register("reachability", "pink")
def build_reachability_selector(cfg=None, hw=None, ctx=None,
                                dependencies=None):
    import paths

    cfg = cfg or {}
    return ReachabilitySelector(
        paths.PROJECT_ROOT / cfg.get(
            "urdf", "grq20_v2d5_piperL_gripper.urdf",
        ),
        solver=cfg.get("solver", "quadprog"),
        joint_limit_margin_deg=float(cfg.get("joint_limit_margin_deg", 1.0)),
        singular_threshold=float(cfg.get("singular_threshold", 1e-4)),
        risk_tolerance=float(cfg.get("risk_tolerance", 0.05)),
        max_joint_risk=float(cfg.get("max_joint_risk", 1.0)),
        view_joint_limit_margin_deg=float(
            cfg.get("view_joint_limit_margin_deg", 5.0)
        ),
    )
