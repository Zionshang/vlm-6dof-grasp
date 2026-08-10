"""Piper LCM robot plugin backed by agx_control's ArmLcmClient."""
from registry import register


@register("robot", "piper_lcm")
def build_piper_robot(cfg=None, hw=None, ctx=None, dependencies=None):
    from robot_client import PiperLcmRobotClient, SafeRobotClient
    # Do not send HOME until the application has verified ARM_STATE feedback.
    return SafeRobotClient(
        PiperLcmRobotClient.from_hardware(hw), safe_stop_enabled=False,
    )
