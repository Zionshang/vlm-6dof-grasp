"""ARX5 LCM robot plugin."""
from registry import register


@register("robot", "arx5_lcm")
def build_arx5_robot(cfg=None, hw=None, ctx=None, dependencies=None):
    from communication.lcm.lcm_client import Arx5LcmClient
    from robot_client import SafeRobotClient

    client = Arx5LcmClient(
        url="", address=hw.lcm_arm_address,
        port=hw.lcm_arm_port, ttl=hw.lcm_arm_ttl,
    )
    return SafeRobotClient(client)
