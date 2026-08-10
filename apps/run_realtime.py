"""run_realtime 薄入口:VLM 检测 → 抓取 → 机械臂执行(键盘触发,GraspManager 编排)。

组件组合由 config/apps/realtime.yaml 声明;键盘业务在 apps/handlers/keyboard_grasp.py。
依赖:pynput(键盘)、ollama(VLM)、arx5 LCM(机械臂)。
"""
import sys
import os
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
ROOT = paths.PROJECT_ROOT
[os.environ.pop(k, None) for k in list(os.environ) if "PROXY" in k.upper()]

from manager import GraspManager
from hardware import HardwareConfig
from apps.handlers.keyboard_grasp import KeyboardGraspHandler


def main():
    cfg = yaml.safe_load(open(ROOT / "config/apps/realtime.yaml"))
    hw = HardwareConfig()
    m = GraspManager(cfg, hw=hw)
    if not m.handshake([("Robot connection", lambda: m.get("robot") is not None)]):
        m.release_resources()
        raise SystemExit("[Error] Component handshake failed")
    try:
        handler = KeyboardGraspHandler(m, hw, prompt="mug", output_dir=ROOT / "output")
        handler.on_start()
        m.run(handler)
    except BaseException:
        m.release_resources()
        raise


if __name__ == "__main__":
    main()
