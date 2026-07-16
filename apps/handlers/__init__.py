"""各 app 的业务 Handler(用 manager.components 编排)。

app 显式 `from handlers.<name> import <Handler>`,不经此 __init__ 导入 ——
避免 import 本包就触发 pynput / lcm 等重依赖(只在对应 app 用到时才加载)。
"""
