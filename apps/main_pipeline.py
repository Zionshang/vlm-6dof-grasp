"""D435i → depth(ffs/raw)→ EconomicGrasp → o3d(GraspManager 编排,薄入口)。

组件由 config/apps/main_pipeline_{ffs,raw}.yaml 声明(--use_ffs 选);业务编排(单 worker
cam→depth→grasp + o3d 主线程)在 apps/handlers/realtime_vis.py。
换组件(相机/深度/抓取/可视化)只改 config 的 backend 名。
"""
import sys
import os
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
ROOT = paths.PROJECT_ROOT
[os.environ.pop(k, None) for k in list(os.environ) if "PROXY" in k.upper()]

from manager import GraspManager
from apps.handlers.realtime_vis import RealtimeVisHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ffs", type=lambda x: x.lower() == "true", default=True,
                        help="true=FFS 优化深度(main_pipeline_ffs.yaml); false=硬件 depth(main_pipeline_raw.yaml)")
    args = parser.parse_args()

    cfg_name = "main_pipeline_ffs.yaml" if args.use_ffs else "main_pipeline_raw.yaml"
    cfg = yaml.safe_load(open(ROOT / "config/apps" / cfg_name))
    m = GraspManager(cfg)   # 按 config build camera/depth/grasp_engine/visualizer
    if not m.handshake():
        m.release_resources()
        raise SystemExit("[Error] Camera handshake failed")
    try:
        handler = RealtimeVisHandler(
            m.require("depth"), m.require("grasp_engine"),
            m.require("camera"), m.require("visualizer"),
        )
        m.run(handler)
    except BaseException:
        m.release_resources()
        raise


if __name__ == "__main__":
    main()
