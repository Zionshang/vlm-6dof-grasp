# VLM-6DoF-Grasp

Config-driven 6DoF grasping with RealSense, FFS, unified instance perception,
pluggable grasp generators, and pluggable robot backends.

## Architecture

- `config/hardware/`: physical calibration, robot poses, workspace bounds and
  communication parameters.
- `config/apps/`: component composition and application-level pipeline options.
- `core/components/`: backend plugins registered by `(role, backend)`.
- `core/manager.py`: dependency resolution, preflight and lifecycle ownership.
- `core/grasp_perception.py`: shared instance observation, depth acquisition,
  grasp generation, and expected-empty-result retry orchestration.
- `third_party/`: vendored algorithm projects (`Fast-FoundationStereo`,
  `fastsam`, `EfficientSAM`, `economic_grasp` and `vlm`); adapters stay under
  `core/components/`.
- `apps/`: application workflows and event/transport entrypoints.

Component factories declare dependencies in the single `core/registry.py`
registry. They receive only their configuration, hardware profile, frame
context and declared dependencies; they do not depend on the Manager.

## Ollama

Install Ollama and download the configured detector model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:8b-instruct-q4_K_M
sudo systemctl enable --now ollama
```

`config/apps/grasp_lcm.yaml` limits the VLM context to 4096 and sets
`keep_alive: 0`. Manager preflight checks the exact model and unloads an old
resident instance before loading CUDA-heavy components.

## Formal entrypoints

Piper+D405 three-view grasping:

```bash
python apps/piper_run_atec.py --target can
```

The application verifies ARM_STATE, returns home, starts D405+FFS+YOLO26-seg,
confirms a target from model confidence, locates it from robust mask depth,
collects front/left/right GraspGenX candidates, applies IK reachability
selection, asks for terminal confirmation, then executes
approach → reach → grasp → lift → home → release.
Every Cartesian step is feedback-verified; failures return the robot home.

The same command opens `http://127.0.0.1:8765` automatically. Its local web
dashboard mirrors the terminal log in real time, separates component details,
shows images produced by the current run, and renders an interactive point
cloud. Dashboard and pipeline settings live in
`config/apps/piper_run_atec.yaml`.

Task-LCM grasp service (requires a hardware profile with task LCM, drop pose
and grasp policy configured, and an app YAML selecting the matching robot
backend):

```bash
python apps/run_grasp_lcm.py \
  --hardware-profile config/hardware/<profile>.yaml \
  --app-config config/apps/<matching-grasp-app>.yaml
```

The current `grasp_lcm.yaml` selects `piper_lcm`; Piper task-LCM channels,
drop pose and the formal service policy are intentionally still unset.

D435i live grasp visualization:

```bash
python apps/main_pipeline.py --use_ffs true
python apps/main_pipeline.py --use_ffs false
```

D405 perception-only, key-triggered YOLO grasp visualization (no robot):

```bash
conda run --no-capture-output -n economicgrasp \
  python apps/d405_yolo_grasp_realtime.py --target all
```

The bundled YOLO26 checkpoint detects `watermelon`, `can`, `lunch_box`, and
`red_bag`. Pass one class name (or comma-separated names) to `--target` to
restrict detection. The 2D window continuously shows YOLO boxes. Focus that
window and press `1` to run FFS, EfficientSAM and EconomicGrasp once; the mask
overlay and Open3D scene update when inference finishes. Press `Q`/`Esc`, or
close the Open3D window, to exit.

Keyboard-triggered X5 realtime grasping:

```bash
python apps/run_realtime.py
```

## Adding a backend

Implement the role's existing domain interface, add a factory under the
matching `core/components/<role>/` package, and register it:

```python
@register("depth", "my_depth", requires=("camera",))
def build_my_depth(cfg, hw, ctx, dependencies):
    return MyDepth(camera=dependencies["camera"], **cfg)
```

Then select `backend: my_depth` in an app YAML. No application workflow or
Manager branch should be added.

## Framework tests

The framework regression suite uses only the Python standard library runner:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Hardware programs are intentionally not started by the automated test suite.
