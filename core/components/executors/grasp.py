"""抓取执行器组件(包装 core.grasp_executor.GraspExecutor)。

依赖 robot 组件先构建;grip_max 统一取自 hardware profile(消除原 run_realtime/run_grasp_lcm 不一致)。
"""
from registry import register


@register("executor", "grasp")
def build_grasp_executor(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from grasp_executor import GraspExecutor
    client = manager.get("robot")
    grip_max = hw.gripper_max_width if hw is not None else 0.085
    return GraspExecutor(client, hw, grip_max)
