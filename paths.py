"""
项目路径基础设施 (Project path bootstrap).

作用:
    在任意子目录的脚本/模块顶部 ``import paths`` 之后,项目根目录及各子模块目录
    (core / vlm / economic_grasp) 会被自动加入 ``sys.path``,从而保证跨目录的
    import 始终可用,无需每个脚本各自维护 sys.path。

用法:
    在入口脚本顶部按如下方式引导即可(先把项目根加入 sys.path,才能 import 本文件):

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import paths
        ROOT = paths.PROJECT_ROOT   # 兼容旧代码中 ROOT 指向项目根的约定
"""
import sys
from pathlib import Path

# 项目根目录(本文件所在目录即为根)
PROJECT_ROOT = Path(__file__).resolve().parent

# 需要注册进 sys.path 的目录清单:
#   - 项目根:          供 fastsam 等顶层包以及 paths 自身被 import
#   - core/:           自研核心库(pipeline / transform / camera)
#   - vlm/:            VLM 子模块,供 ``from vlm.src.... import ...``
#   - economic_grasp/: EconomicGrasp 子模块,供 ``from economic_grasp.... import ...``
_PATH_ENTRIES = [
    PROJECT_ROOT,
    PROJECT_ROOT / "core",
    PROJECT_ROOT / "vlm",
    PROJECT_ROOT / "economic_grasp",
]

for _entry in _PATH_ENTRIES:
    _entry_str = str(_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

# 清自研代码 __pycache__,确保每次启动跑最新 .py(避免旧缓存版本/旧数据)
# 只清 core/apps/perception/grasping;外部库(economic_grasp/vlm/fastsam 等)不清,免得重编译慢
import shutil
for _sub in ("core", "apps", "perception", "grasping"):
    for _d in (PROJECT_ROOT / _sub).rglob("__pycache__"):
        shutil.rmtree(_d, ignore_errors=True)
