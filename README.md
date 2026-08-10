# VLM-6DoF-Grasp

Config-driven 6DoF grasping with RealSense, FFS, VLM/FastSAM,
EconomicGrasp and pluggable robot backends.

## Architecture

- `config/hardware/`: physical calibration, robot poses, workspace bounds and
  communication parameters.
- `config/apps/`: component composition and application-level pipeline options.
- `core/components/`: backend plugins registered by `(role, backend)`.
- `core/manager.py`: dependency resolution, preflight and lifecycle ownership.
- `core/grasp_perception.py`: the shared
  detect → segment → generate → select chain.
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

Piper+D405 feedback-verified grasp test:

```bash
python apps/piper_run_test.py --prompt orange
```

The test verifies ARM_STATE, returns home, starts D405+FFS, moves to the
configured observation pose, runs VLM+FastSAM+EconomicGrasp, selects the first
geometrically valid grasp, executes the configured five-step sequence and
visualizes the candidates. Every Cartesian/gripper step is feedback-verified;
any failure after robot-state verification returns home.

Task-LCM grasp service (requires a hardware profile with task LCM, drop pose
and grasp policy configured, and an app YAML selecting the matching robot
backend):

```bash
python apps/run_grasp_lcm.py \
  --hardware-profile config/hardware/<profile>.yaml \
  --app-config config/apps/<matching-grasp-app>.yaml
```

The current `grasp_lcm.yaml` selects `piper_lcm`; Piper task-LCM channels,
drop pose and the formal service policy are intentionally still unset. The
standalone `piper_run_test.py` sequence is configured separately in its app
YAML.

D435i live grasp visualization:

```bash
python apps/main_pipeline.py --use_ffs true
python apps/main_pipeline.py --use_ffs false
```

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
