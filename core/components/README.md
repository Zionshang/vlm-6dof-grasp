# Component plugins

Subdirectories group related replaceable backends. Each backend registers its
role and factory in `core/registry.py`; application code selects roles
independently in `config/apps/*.yaml`.

| Role | Runtime contract | Typical dependencies |
|---|---|---|
| `camera` | `step(frame)`, `close()` | none |
| `depth` | `step(frame)`, `factor_depth` | camera for stereo FFS |
| `detector` | `detect(image, prompt)` | none |
| `segmenter` | `segment(image, boxes)` | none |
| `obb_estimator` | `estimate(depth, mask, label)` | camera, depth |
| `obb_fusion` | stable filtering and base-frame fusion | none |
| `view_adjust` | `plan(obb, ee_pose, label)` | none |
| `view_plan_visualizer` | `show(obb, plan, width, seconds, start_pose)` | none |
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
