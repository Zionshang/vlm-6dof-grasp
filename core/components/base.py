"""组件角色契约(Node / Agent / Handler)+ 生命周期钩子。

鸭子类型,不强制继承;Protocol 只为让人/类型检查器一眼看清接口规范。
所有组件可选实现 release():由 GraspManager.release_resources() 在退出时统一调用。
"""
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from context import FrameContext


class Component(Protocol):
    """所有组件公共约定:可选 release()(Manager 退出时统一调)。"""
    def release(self) -> None: ...


class Node(Protocol):
    """数据源 / 硬件(相机、机械臂)。step(ctx) 推进一拍:取数据写 ctx(相机)
    或读 ctx 写硬件(机械臂)。需握手(首帧 / 连接就绪)。"""
    def step(self, ctx: "FrameContext") -> None: ...


class Agent(Protocol):
    """算力(depth / detector / segmenter / grasp_engine / selector)。step(ctx) 读 ctx、写 ctx。
    可同步(step),也可异步(submit/take,内置 worker 规避 CUDA/OpenGL 争用)。"""
    def step(self, ctx: "FrameContext") -> None: ...


class Handler(Protocol):
    """调度 / IO(visualizer / executor / comm)。step(ctx, components) 每帧/每事件动作,
    由 GraspManager.run() 循环调用。"""
    def step(self, ctx: "FrameContext", components: dict) -> None: ...
