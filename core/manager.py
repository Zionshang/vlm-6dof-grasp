"""Dependency-aware component container and lifecycle manager."""
import importlib
import time

from context import FrameContext
from registry import backend, build


_ROLE_MODULES = {
    "camera": "cameras",
    "depth": "depth",
    "detector": "detectors",
    "segmenter": "segmenters",
    "grasp_engine": "grasp_engines",
    "selector": "selectors",
    "executor": "executors",
    "visualizer": "visualizers",
    "dashboard": "dashboards",
    "robot": "robots",
}


class GraspManager:
    """Resolve configured components and own their lifecycle.

    Component factories receive only declared dependencies. Backends marked as
    preflight are prepared before CUDA-heavy components are initialized.
    """

    @classmethod
    def from_yaml(cls, path, **kwargs):
        import yaml
        with open(path) as stream:
            return cls(yaml.safe_load(stream), **kwargs)

    def __init__(self, app_config: dict, hw=None, initial_components=None,
                 eager=True):
        self.app_config = app_config or {}
        self.hw = hw
        self.ctx = FrameContext()
        self.specs = dict(self.app_config.get("components") or {})
        robot_spec = self.specs.get("robot") or {}
        hardware_robot = getattr(hw, "robot_kind", None)
        if (hardware_robot and robot_spec.get("backend")
                and robot_spec["backend"] != hardware_robot):
            raise ValueError(
                f"Robot backend mismatch: app={robot_spec['backend']}, "
                f"hardware={hardware_robot}"
            )
        self.components = dict(initial_components or {})
        self._build_order = list(self.components)
        self._building = []
        self._closed = False
        self.handshake_error = None
        if eager:
            try:
                self.initialize()
            except BaseException:
                self.release_resources()
                raise

    def _load_backend(self, role, name):
        if backend(role, name) is None:
            module = _ROLE_MODULES.get(role, role + "s")
            module_name = f"components.{module}"
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if exc.name != module_name:
                    raise
                raise ValueError(
                    f"Unknown {role} backend '{name}': no plugin package "
                    f"'{module_name}'"
                ) from None
        item = backend(role, name)
        if item is None:
            raise ValueError(f"Unknown {role} backend '{name}'")
        return item

    def _configured_backend(self, role):
        spec = self.specs.get(role)
        if not spec or not spec.get("backend"):
            raise KeyError(f"Required component role '{role}' is not configured")
        return spec, spec["backend"]

    def require(self, role):
        """Return a required component, constructing dependencies on demand."""
        if role in self.components:
            return self.components[role]
        spec, name = self._configured_backend(role)
        if role in self._building:
            cycle = " -> ".join(self._building + [role])
            raise RuntimeError(f"Component dependency cycle: {cycle}")

        item = self._load_backend(role, name)
        self._building.append(role)
        try:
            deps = {dep: self.require(dep) for dep in item.requires}
            component = build(
                role, name, cfg=spec, hw=self.hw, ctx=self.ctx,
                dependencies=deps,
            )
        finally:
            self._building.pop()
        self.components[role] = component
        self._build_order.append(role)
        return component

    def get(self, role, default=None):
        """Return an already built component without triggering construction."""
        return self.components.get(role, default)

    def initialize(self, roles=None):
        """Preflight lightweight backends, then eagerly build non-lazy roles."""
        specs = self.specs if roles is None else {
            role: self.specs[role] for role in roles
        }
        preflight_roles = []
        for role, spec in specs.items():
            if not spec or not spec.get("backend"):
                continue
            item = self._load_backend(role, spec["backend"])
            if item.preflight:
                preflight_roles.append(role)

        for role in preflight_roles:
            component = self.require(role)
            prepare = getattr(component, "preflight", None)
            if callable(prepare):
                prepare()

        for role, spec in specs.items():
            if spec and spec.get("backend") and not spec.get("lazy", False):
                self.require(role)
        return self

    def handshake(self, rules=None, timeout=10.0):
        """Wait for explicit readiness rules and the first camera frame."""
        camera = self.get("camera")
        pending = list(rules or [])
        if camera is not None:
            pending.append(("Camera first frame", lambda: self.ctx.color is not None))
        if not pending:
            return True

        print(f"[Manager] handshake: {[label for label, _ in pending]}")
        deadline = time.monotonic() + timeout
        last_camera_error = None
        while pending and time.monotonic() < deadline:
            if camera is not None and hasattr(camera, "step"):
                try:
                    camera.step(self.ctx)
                except Exception as exc:
                    last_camera_error = exc
            pending = [(label, check) for label, check in pending if not check()]
            if pending:
                time.sleep(0.1)
        if pending:
            detail = f"; camera error: {last_camera_error}" if last_camera_error else ""
            print(f"[Manager] handshake TIMEOUT: {[x[0] for x in pending]}{detail}")
        else:
            print("[Manager] handshake OK")
        self.handshake_error = last_camera_error
        return not pending

    def run(self, handler, freq_hz=None):
        try:
            while not self.ctx.state.get("quit"):
                handler.step(self.ctx, self.components)
                if freq_hz:
                    time.sleep(1.0 / freq_hz)
        except KeyboardInterrupt:
            pass
        finally:
            close = getattr(handler, "close", None)
            if callable(close):
                close()
            self.release_resources()

    def release_resources(self):
        """Safe-stop hardware first, then close resources in reverse order."""
        if self._closed:
            return
        self._closed = True
        print("[Manager] releasing resources...")
        self._call_lifecycle("safe_stop", self._build_order)
        self._call_lifecycle("close", reversed(self._build_order))

    def _call_lifecycle(self, method, roles):
        for role in roles:
            component = self.components.get(role)
            callback = getattr(component, method, None)
            if callable(callback):
                try:
                    callback()
                except Exception as exc:
                    print(f"[Manager] {role}.{method} failed: {exc}")

    close = release_resources

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release_resources()
