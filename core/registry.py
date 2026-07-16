"""统一组件注册表:可插拔组件的注册 + 构建(扩展 perception/base.py 的 register/build 思路)。

每类组件(role)一个 dict:backend 名 -> 工厂 factory(ctx, cfg, hw, manager, **kw)。
组件用 @register(role, name) 装饰注册;Manager 按 config 声明的 backend 名 build。
换组件(相机 / 深度算法 / 抓取引擎 …)只改 config,不动代码。
"""
from typing import Any

_REGISTRIES: dict = {}   # role -> {backend_name -> factory}


def _registry(role: str) -> dict:
    if role not in _REGISTRIES:
        _REGISTRIES[role] = {}
    return _REGISTRIES[role]


def register(role: str, name: str):
    """装饰器:把工厂注册为 (role, name)。

    工厂签名:factory(ctx, cfg=None, hw=None, manager=None, **kw) -> 组件实例。
    """
    def deco(factory):
        _registry(role)[name] = factory
        return factory
    return deco


def build(role: str, name: str, ctx=None, cfg=None, hw=None, manager=None, **kw):
    """按 (role, name) 取工厂并构建。未知 name 报错并列出已注册项。"""
    reg = _registry(role)
    if name not in reg:
        raise ValueError(f"Unknown {role} backend '{name}'. Registered: {list(reg)}")
    return reg[name](ctx=ctx, cfg=cfg, hw=hw, manager=manager, **kw)


def registered(role: str) -> list:
    """列出某 role 下已注册的 backend 名(调试用)。"""
    return list(_registry(role))
