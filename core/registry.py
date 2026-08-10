"""Single registry for all pluggable component backends."""
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class ComponentBackend:
    factory: Callable
    requires: tuple[str, ...] = ()
    preflight: bool = False


_REGISTRIES: dict[str, dict[str, ComponentBackend]] = {}


def register(role: str, name: str, *, requires: Iterable[str] = (),
             preflight: bool = False):
    """Register a backend and declare its component dependencies."""
    def decorate(factory):
        registry = _REGISTRIES.setdefault(role, {})
        if name in registry and registry[name].factory is not factory:
            raise ValueError(f"Duplicate component backend: {role}.{name}")
        registry[name] = ComponentBackend(
            factory=factory, requires=tuple(requires), preflight=preflight,
        )
        return factory
    return decorate


def backend(role: str, name: str) -> ComponentBackend | None:
    return _REGISTRIES.get(role, {}).get(name)


def build(role: str, name: str, *, cfg=None, hw=None, ctx=None,
          dependencies=None):
    """Build one registered backend without exposing the Manager to it."""
    item = backend(role, name)
    if item is None:
        names = sorted(_REGISTRIES.get(role, {}))
        raise ValueError(
            f"Unknown {role} backend '{name}'. Registered: {names}"
        )
    return item.factory(
        cfg=cfg or {}, hw=hw, ctx=ctx,
        dependencies=dependencies or {},
    )


def registered(role: str) -> list[str]:
    return sorted(_REGISTRIES.get(role, {}))
