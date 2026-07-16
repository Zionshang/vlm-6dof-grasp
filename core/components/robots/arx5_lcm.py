"""Arx5 LCM 机械臂组件(包装 core.robot_client.make_robot_client)。

client 本身带 reset_to_home(),Manager.release_resources 会调用。
"""
from registry import register


@register("robot", "arx5_lcm")
def build_arx5_robot(ctx=None, cfg=None, hw=None, manager=None, **kw):
    from robot_client import make_robot_client
    return make_robot_client(hw)
