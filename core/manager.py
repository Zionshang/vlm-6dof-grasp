"""GraspManager:统一编排器(参考 ROS BaseManager)。

职责:① 按 config build + 注册组件;② handshake 等组件就绪;
③ run(handler) 主循环;④ release_resources 统一释放资源。
apps 变薄成「读 config → GraspManager(config) → handshake → run(handler)」。
"""
import time
from registry import build
from context import FrameContext


class GraspManager:
    def __init__(self, app_config: dict, hw=None):
        self.hw = hw
        self.ctx = FrameContext()
        self.components: dict = {}           # role -> 组件实例
        self._build_components(app_config)

    def _build_components(self, cfg: dict):
        """按 config['components'] 的 backend 名 build + 注册组件。"""
        import components   # 触发各组件 @register(把 factory 登记进 registry)
        for role, spec in (cfg.get("components") or {}).items():
            spec = spec or {}
            name = spec.get("backend")
            if not name:
                continue
            self.components[role] = build(role, name, ctx=self.ctx, cfg=spec,
                                          hw=self.hw, manager=self)

    def get(self, role):
        """按 role 取组件实例(不存在返回 None)。"""
        return self.components.get(role)

    def handshake(self, rules=None, timeout=10.0):
        """等组件就绪。默认:有 camera 则等首帧(ctx.color 非 None)。
        rules=[(label, predicate)] 追加自定义规则(如机械臂连接)。"""
        cam = self.components.get("camera")
        rules = list(rules or [])
        if cam is not None:
            rules.append(("Camera first frame", lambda: self.ctx.color is not None))
        if not rules:
            return True
        print(f"[Manager] handshake: {[lbl for lbl, _ in rules]}")
        t0, pending = time.time(), list(rules)
        while pending and time.time() - t0 < timeout:
            if cam is not None and hasattr(cam, "step"):
                try:
                    cam.step(self.ctx)
                except Exception:
                    pass
            pending = [(lbl, p) for lbl, p in pending if not p()]
            if pending:
                time.sleep(0.1)
        if pending:
            print(f"[Manager] handshake TIMEOUT: {[lbl for lbl, _ in pending]}")
        else:
            print("[Manager] handshake OK")
        return not pending

    def run(self, handler, freq_hz=None):
        """主循环:循环调 handler.step(ctx, components),直到 ctx.state['quit'] 或 Ctrl-C。
        退出时 finally 统一释放资源。"""
        try:
            while not self.ctx.state.get("quit"):
                handler.step(self.ctx, self.components)
                if freq_hz:
                    time.sleep(1.0 / freq_hz)
        except KeyboardInterrupt:
            pass
        finally:
            getattr(handler, "release", lambda: None)()   # handler 持有的 worker 等收尾
            self.release_resources()

    def release_resources(self):
        """遍历组件统一释放:按优先级调 release/stop/destroy_window/reset_to_home(命中一个即停)。"""
        print("[Manager] releasing resources...")
        for role, comp in self.components.items():
            for method in ("release", "stop", "destroy_window", "reset_to_home"):
                fn = getattr(comp, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as e:
                        print(f"[Manager] {role}.{method} failed: {e}")
                    break
