"""可插拔组件包(按需 import)。

GraspManager._build_components 根据 config 的 role 动态 import 对应子模块
(如 `components.cameras`)触发其 @register;不用的角色(detectors/segmenters 等
重依赖)不会被加载。新增角色子模块只需建目录 + @register,无需改本文件。
"""
