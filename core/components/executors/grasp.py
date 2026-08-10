"""抓取执行器组件(包装 core.grasp_executor.GraspExecutor)。

依赖 robot 组件先构建;grip_max 统一取自 hardware profile(消除原 run_realtime/run_grasp_lcm 不一致)。
"""
from registry import register


@register("executor", "grasp", requires=("robot",))
def build_grasp_executor(cfg=None, hw=None, ctx=None, dependencies=None):
    from grasp_executor import GraspExecutor, GraspStep
    client = dependencies["robot"]
    grip_max = hw.gripper_max_width if hw is not None else 0.085
    steps = [GraspStep(**step) for step in cfg.get("steps", [])]
    if cfg.get("require_home") and (not steps or not steps[-1].use_home_pose):
        raise ValueError("Grasp sequence must end with a HOME pose step")
    return GraspExecutor(client, hw, grip_max, steps)
