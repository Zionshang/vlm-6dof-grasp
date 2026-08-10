# Component plugins

Each subdirectory is one replaceable role. A backend registers one factory in
the shared `core/registry.py` registry; application code selects it by name in
`config/apps/*.yaml`.

| Role | Runtime contract | Typical dependencies |
|---|---|---|
| `camera` | `step(frame)`, `close()` | none |
| `depth` | `step(frame)`, `factor_depth` | camera for stereo FFS |
| `detector` | `detect(image, prompt)` | none |
| `segmenter` | `segment(image, boxes)` | none |
| `grasp_engine` | `predict(image, depth, mask, topk)` | camera, depth |
| `selector` | `select(...)` | none |
| `executor` | `run_sequence(...)` | robot |
| `visualizer` | update/poll/render/close | camera |
| `robot` | RobotClient plus `safe_stop()` | none |

Factories must not receive or import `GraspManager`. Declare dependencies in
the decorator and consume them from the injected `dependencies` mapping:

```python
@register("depth", "example", requires=("camera",))
def build_example(cfg, hw, ctx, dependencies):
    return ExampleDepth(dependencies["camera"], cfg)
```

Use `preflight=True` only for lightweight checks that must happen before heavy
CUDA components are initialized. Use `lazy: true` in app YAML for resources
such as optional GUI visualizers.
