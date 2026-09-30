# Piper + D405 grasping environment and Jetson deployment

This note covers `config/apps/piper_run_atec.yaml`, not every optional backend
in this repository. The matching pip dependencies are in `requirements.txt`.
Choose the Jetson model and JetPack release before selecting PyTorch. A single
x86_64 `pip freeze` cannot serve as an ARM64 CUDA lock file.

## Local environment evidence (2026-09-30)

- `graspgenx` uses Python 3.10.20 on x86_64, PyTorch 2.1.0+cu121,
  torchvision 0.16.0, and NumPy 1.26.4. `pip freeze` contains x86_64-local
  wheel paths, NVIDIA cu121 pip runtime wheels, and an editable checkout of
  `GraspGenX`; do not copy those entries to Jetson.
- No dedicated environment installation log was found in the project or in
  the nearby log files searched. Shell history shows checking `nvidia-smi`
  and `nvcc --version` before activating `graspgenx`, but not a reproducible
  install transcript.
- Conda revision history shows a 2026-09-15 adjustment from NumPy 2.2.6 to
  1.26.4, Pinocchio 4.1.0 to 3.9.0, and Pink 4.3.0 to 3.5.0. The current
  environment also has `qpsolvers` 4.13.0 and `quadprog` 0.1.13.
- The configured GraspGenX release checkpoints use `ptv3vanilla`, which does
  not need spconv, torch-scatter or torch-cluster. The installed
  `pointnet2_ops` refers to a different local checkout; the current
  checkpoints do not select the PointNet++ backbone. Do not migrate that
  x86_64 extension wheel. GraspGenX may still emit an optional PointNet++
  import warning.
- `pip check` in the existing workstation environment reports unrelated
  optional-package metadata conflicts: `pyrender` expects PyOpenGL 3.1.0
  while 3.1.5 is installed, and `urdfpy` expects old `networkx`/`pycollada`
  versions. The current inference path tolerates missing `pyrender`; neither
  `pyrender` nor `urdfpy` is in the deployment requirements.

## Install order on Jetson

1. Confirm `uname -m` is `aarch64`, check the JetPack/L4T version, and use a
   Python 3.10 environment. GraspGenX declares Python >=3.10; older JetPack
   images with only Python 3.8 need a matching Python and PyTorch build or a
   suitable container. NVIDIA's matrix lists PyTorch 2.4 for JetPack 6.0 and
   2.5 for JetPack 6.1, within GraspGenX's declared `torch>=2.1,<2.7` range.
   For JetPack 6.2 the listed NVIDIA containers use PyTorch 2.7 or 2.8 and
   have no standalone wheel in that matrix; this is outside GraspGenX's
   declared range and needs a separate end-to-end validation before use.
2. Install the CUDA-enabled **torch and matching torchvision** builds for the
   exact JetPack version using NVIDIA's compatibility matrix/install guide,
   or start from an NVIDIA container that provides the matching pair.
   Verify `torch.cuda.is_available()` before installing the other packages.
   Do not install the x86_64 `torch==2.1.0+cu121` or `nvidia-*-cu12` wheels
   shown in this workstation's `pip freeze`.
3. Install the IK stack at the locally used versions, preferably from
   conda-forge in the same environment:

   ```bash
   conda install -c conda-forge "pinocchio=3.9" "pink=3.5" \
     "qpsolvers=4.13" "quadprog=0.1.13" "numpy=1.26.4" "scipy=1.15.2"
   ```

   Confirm `import pinocchio, pink, qpsolvers, quadprog` and that
   `"quadprog" in qpsolvers.available_solvers`. The configured reachability
   backend explicitly requests `quadprog`.
4. From this repository root, install `requirements.txt` **after** the
   platform's PyTorch build. Then install the vendored GraspGenX source with
   `--no-deps` so its broad upstream dependency list cannot replace the
   JetPack-matched torch build:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install --no-deps -e third_party/GraspGenX
   ```

   The robot client imports `communication.lcm.arm_lcm_client` from the
   `robot.driver_root` path in `config/hardware/piper_d405.yaml`. Copy the
   matching `agx_control` source to Jetson and update that path, or install
   the same source into this environment. The robot-side LCM server and
   client must use the **same generated `ArmState` definition**; changing
   its fields changes the LCM fingerprint and otherwise drops state frames.
5. Provide the model/data assets referenced by the YAML: the FFS `.pth`,
   YOLO `.pt`, GraspGenX `gen/` and `dis/` checkpoints, `piper_hand` assets,
   and the Piper URDF. These are files, not pip packages. Update hardware
   calibration, robot driver path and LCM network URL for the Jetson setup.

## Jetson checks before enabling arm motion

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import cv2, pyrealsense2, lcm, pinocchio, pink, qpsolvers, quadprog; print('quadprog' in qpsolvers.available_solvers)"
python -c "import ultralytics, timm, graspgenx, open3d; print('imports OK')"
python -m pip check  # diagnostic; see the optional-package conflicts above
```

Test the D405 stream, FFS warm-up and a GraspGenX prediction on Jetson before
running `apps/piper_run_atec.py`. FFS contains `torch.compile` functions and
the current application calls the CUDA model directly; successful imports do
not prove that its kernels compile on Jetson. The FFS checkpoint is loaded
using `torch.load(..., weights_only=False)`, so test deserialization with the
chosen JetPack PyTorch version. Start the LCM server and check that the client
receives `ARM_STATE` with `arm_status` and an updated `target_utime` before
commanding a grasp.

The current `pyrealsense2` version has a Python 3.10 aarch64 wheel, but camera
access may still need Jetson-specific librealsense/udev setup. Open3D 0.20
has a Python 3.10 aarch64 wheel for glibc >=2.35; its ARM wheel is CPU-only.
The web dashboard uses Flask/Plotly and does not require Open3D CUDA support.

## Sources for platform-specific choices

- NVIDIA: [PyTorch installation](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
  and [JetPack compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html).
- RealSense: [Jetson installation guide](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md).
- Open3D: [ARM support](https://github.com/isl-org/Open3D/blob/main/docs/arm.rst)
  and [0.20.0 aarch64 wheel](https://pypi.org/project/open3d/0.20.0/).
