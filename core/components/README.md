# Component plugins

Subdirectories group related replaceable backends. Each backend registers its
role and factory in `core/registry.py`; application code selects roles
independently in `config/apps/*.yaml`.

| Role | Runtime contract | Typical dependencies |
|---|---|---|
| `camera` | `step(frame)`, `close()` | none |
| `depth` | `step(frame)`, `factor_depth` | camera for stereo FFS |
| `detector` | `detect(image, prompt)` | none |
| `segmenter` | `segment(image, boxes)`, optional `segment_instances(...)` | none |
| `instance_model` | `predict(image, target) -> instances` | none or detector+segmenter |
| `grasp_engine` | `predict(image, depth, mask, topk)` | camera, depth |
| `candidate_selector` | choose one legacy image candidate | none |
| `reachability` | check explicit observation/approach/reach poses with IK and risk selection | none |
| `executor` | resolve configured step poses and `run_sequence(...)` | robot |
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

Motion-step geometry belongs to the executor. Applications may call
`executor.pose_for_step(grasp_pose, step_name)` to prepare an explicit pose for
reachability checks; reachability backends must not inspect executor steps or
reimplement configured offsets. Reachability only evaluates poses and never
sends robot commands.
