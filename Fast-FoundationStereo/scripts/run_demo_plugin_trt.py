#!/usr/bin/env python3
"""Run the FFSGWCVolume-plugin TensorRT engine from Python.

This is the Python equivalent of cpp/src/ffs_single_depth_inference.cpp plus
cpp/app/main.cpp for the single-engine plugin path:

  - engine inputs: left, right
  - engine output: disp
  - input tensors are RGB, CHW, float32, raw 0-255 values
  - resize is aspect-ratio-preserving with right/bottom replicate padding
  - disparity is cropped, nearest-neighbor upsampled, and scaled back to the
    original input-image pixel units
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

code_dir = Path(__file__).resolve().parent
repo_dir = code_dir.parent
sys.path.append(str(repo_dir))

from Utils import set_logging_format, set_seed, vis_disparity
from build_plugin_trt import (
    PLUGIN_NAME,
    find_default_plugin_library,
    find_plugin_creator,
    load_plugin_library,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Fast-FoundationStereo single TensorRT plugin engine."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=repo_dir / "engine1_plug",
        help="Directory containing fast_foundationstereo.engine and one YAML config.",
    )
    parser.add_argument(
        "--engine_file",
        "--model_file",
        dest="engine_file",
        type=Path,
        default=None,
        help="Explicit TensorRT engine path. Defaults to <model_dir>/fast_foundationstereo.engine.",
    )
    parser.add_argument(
        "--config_file",
        type=Path,
        default=None,
        help="Explicit YAML config path. Defaults to the single YAML in the engine directory.",
    )
    parser.add_argument(
        "--plugin_lib",
        type=Path,
        default=None,
        help="Path to libffs_gwc_plugin.so. Defaults to cpp/build/libffs_gwc_plugin.so.",
    )
    parser.add_argument("--left_file", type=Path, default=repo_dir / "demo_data" / "left.png")
    parser.add_argument("--right_file", type=Path, default=repo_dir / "demo_data" / "right.png")
    parser.add_argument(
        "--intrinsic_file",
        type=Path,
        default=repo_dir / "demo_data" / "K.txt",
        help="Text file with 3x3 K on line 1 and baseline in meters on line 2.",
    )
    parser.add_argument("--out_dir", type=Path, default=repo_dir / "output_plugin_trt")
    return parser.parse_args()


def resolve_engine_path(model_dir: Path, engine_file: Path | None) -> Path:
    path = engine_file if engine_file is not None else model_dir / "fast_foundationstereo.engine"
    if not path.exists():
        raise FileNotFoundError(f"TensorRT engine does not exist: {path}")
    return path


def resolve_config_path(engine_path: Path, model_dir: Path, config_file: Path | None) -> Path:
    if config_file is not None:
        if not config_file.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_file}")
        return config_file

    search_dir = model_dir if model_dir.exists() else engine_path.parent
    yaml_files = sorted(
        p for p in search_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".yaml", ".yml")
    )
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config found in: {search_dir}")
    if len(yaml_files) > 1:
        names = " ".join(str(p) for p in yaml_files)
        raise RuntimeError(
            f"Expected exactly one YAML config in {search_dir}, found {len(yaml_files)}: {names}"
        )
    return yaml_files[0]


def load_intrinsics(path: Path) -> tuple[np.ndarray, float]:
    with path.open("r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise RuntimeError("intrinsic file must contain K on line 1 and baseline on line 2")
    k = np.array(list(map(float, lines[0].strip().split())), dtype=np.float32).reshape(3, 3)
    baseline = float(lines[1])
    if not (k[0, 0] > 0.0 and baseline > 0.0):
        raise RuntimeError("invalid focal length or baseline in intrinsic file")
    return k, baseline


def load_rgb(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def resize_uniform_and_pad_rgb(img: np.ndarray, target_h: int, target_w: int):
    src_h, src_w = img.shape[:2]
    if src_h == target_h and src_w == target_w:
        return img.copy(), target_h, target_w

    scale = min(float(target_w) / src_w, float(target_h) / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))

    scaled = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        scaled,
        0,
        target_h - scaled_h,
        0,
        target_w - scaled_w,
        cv2.BORDER_REPLICATE,
    )
    return np.ascontiguousarray(padded), scaled_h, scaled_w


def tensor_from_rgb_255(img: np.ndarray) -> torch.Tensor:
    arr = img.astype(np.float32, copy=False)
    return torch.as_tensor(arr, device="cuda").permute(2, 0, 1).unsqueeze(0).contiguous()


def cpp_nearest_resize(src: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    src_h, src_w = src.shape
    xs = np.rint((np.arange(dst_w, dtype=np.float32) + 0.5) * src_w / dst_w - 0.5)
    ys = np.rint((np.arange(dst_h, dtype=np.float32) + 0.5) * src_h / dst_h - 0.5)
    xs = np.clip(xs.astype(np.int64), 0, src_w - 1)
    ys = np.clip(ys.astype(np.int64), 0, src_h - 1)
    return src[ys[:, None], xs[None, :]]


def postprocess_disparity(
    raw_disp: np.ndarray,
    input_h: int,
    input_w: int,
    model_h: int,
    model_w: int,
    scaled_h: int,
    scaled_w: int,
) -> np.ndarray:
    disp = raw_disp.reshape(model_h, model_w).astype(np.float32, copy=False)
    disp = np.maximum(disp, 0.0)

    if input_h == model_h and input_w == model_w:
        return np.ascontiguousarray(disp)

    cropped = disp[:scaled_h, :scaled_w]
    upsampled = cpp_nearest_resize(cropped, input_h, input_w)
    return np.ascontiguousarray(upsampled * (float(input_w) / scaled_w), dtype=np.float32)


def disparity_to_depth_cpp(disp: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    d = disp.astype(np.float32, copy=True)
    xs = np.arange(d.shape[1], dtype=np.float32)[None, :]
    d[(xs - d) < 0.0] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.float32(fx * baseline_m) / d
    return np.ascontiguousarray(depth, dtype=np.float32)


def save_float_matrix_bin(path: Path, values: np.ndarray) -> None:
    arr = np.ascontiguousarray(values, dtype=np.float32)
    dims = np.array(arr.shape[:2], dtype=np.int32)
    with path.open("wb") as f:
        dims.tofile(f)
        arr.tofile(f)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    safe = np.zeros_like(depth, dtype=np.float32)
    safe[valid] = depth[valid]
    min_val = float(safe[valid].min())
    max_val = float(safe[valid].max())
    if max_val <= min_val:
        max_val = min_val + 1.0

    depth_u8 = np.clip((safe - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colored = colored_bgr[..., ::-1]
    colored[~valid] = 0
    return colored


class PluginTensorRTRunner:
    def __init__(self, engine_path: Path, plugin_lib: Path):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        load_plugin_library(str(plugin_lib))
        if not find_plugin_creator(trt):
            raise RuntimeError(f"{PLUGIN_NAME} plugin creator is not registered after loading {plugin_lib}")

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context: {engine_path}")

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        missing = [name for name in ("left", "right", "disp") if name not in names]
        if missing:
            raise RuntimeError(
                f"Plugin engine must expose tensors named left, right, and disp. "
                f"Missing {missing}; found {names}"
            )

    def _torch_dtype(self, trt_dtype):
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        if hasattr(trt.DataType, "BF16"):
            mapping[trt.DataType.BF16] = torch.bfloat16
        if trt_dtype not in mapping:
            raise RuntimeError(f"Unsupported TensorRT dtype: {trt_dtype}")
        return mapping[trt_dtype]

    def infer(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        inputs = {"left": left, "right": right}
        for name, tensor in list(inputs.items()):
            expected = self._torch_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                tensor = tensor.to(expected)
            inputs[name] = tensor.contiguous()
            self.context.set_input_shape(name, tuple(inputs[name].shape))

        out_shape = tuple(self.context.get_tensor_shape("disp"))
        out_dtype = self._torch_dtype(self.engine.get_tensor_dtype("disp"))
        output = torch.empty(out_shape, device="cuda", dtype=out_dtype)

        self.context.set_tensor_address("left", int(inputs["left"].data_ptr()))
        self.context.set_tensor_address("right", int(inputs["right"].data_ptr()))
        self.context.set_tensor_address("disp", int(output.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT enqueue failed")
        return output


def main() -> int:
    args = parse_args()
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    engine_path = resolve_engine_path(args.model_dir, args.engine_file)
    config_path = resolve_config_path(engine_path, args.model_dir, args.config_file)
    plugin_lib = args.plugin_lib or find_default_plugin_library()
    if plugin_lib is None or not plugin_lib.exists():
        raise FileNotFoundError(
            "Could not find libffs_gwc_plugin.so. Build cpp first or pass "
            "--plugin_lib /path/to/libffs_gwc_plugin.so"
        )

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    model_h, model_w = [int(v) for v in cfg["image_size"]]

    left = load_rgb(args.left_file)
    right = load_rgb(args.right_file)
    if left.shape != right.shape:
        raise RuntimeError(f"left/right image size mismatch: {left.shape} vs {right.shape}")
    input_h, input_w = left.shape[:2]

    logging.info(f"Engine: {engine_path}")
    logging.info(f"Plugin: {plugin_lib}")
    logging.info(f"Config: {config_path}")
    logging.info(f"Input images: {input_w}x{input_h}")
    logging.info(f"Model target resolution: {model_w}x{model_h}")

    left_model, scaled_h, scaled_w = resize_uniform_and_pad_rgb(left, model_h, model_w)
    right_model, right_scaled_h, right_scaled_w = resize_uniform_and_pad_rgb(right, model_h, model_w)
    if (scaled_h, scaled_w) != (right_scaled_h, right_scaled_w):
        raise RuntimeError("left/right images produced different resize scales")
    if (input_h, input_w) != (model_h, model_w):
        logging.info(
            f"Uniform resize+pad: {input_w}x{input_h} -> "
            f"{scaled_w}x{scaled_h} inside {model_w}x{model_h}"
        )

    runner = PluginTensorRTRunner(engine_path, plugin_lib)
    t_left = tensor_from_rgb_255(left_model)
    t_right = tensor_from_rgb_255(right_model)

    logging.info("Running TensorRT inference")
    disp_raw_t = runner.infer(t_left, t_right)
    torch.cuda.current_stream().synchronize()
    logging.info("Inference done")

    disp_raw = disp_raw_t.float().detach().cpu().numpy()
    disp = postprocess_disparity(
        disp_raw,
        input_h=input_h,
        input_w=input_w,
        model_h=model_h,
        model_w=model_w,
        scaled_h=scaled_h,
        scaled_w=scaled_w,
    )

    if not np.isfinite(disp).any():
        raise RuntimeError("Model produced no finite disparity values")

    k, baseline = load_intrinsics(args.intrinsic_file)
    depth = disparity_to_depth_cpp(disp, float(k[0, 0]), baseline)

    disparity_path = args.out_dir / "disparity.bin"
    depth_bin_path = args.out_dir / "depth_meter.bin"
    depth_npy_path = args.out_dir / "depth_meter.npy"
    disp_vis_path = args.out_dir / "disp_vis.png"
    depth_vis_path = args.out_dir / "depth_vis.png"

    save_float_matrix_bin(disparity_path, disp)
    save_float_matrix_bin(depth_bin_path, depth)
    np.save(depth_npy_path, depth)

    disp_color = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
    disp_vis = np.concatenate([left, right, disp_color], axis=1)
    imageio.imwrite(disp_vis_path, disp_vis)
    imageio.imwrite(depth_vis_path, colorize_depth(depth))

    logging.info(f"Saved: {disparity_path}")
    logging.info(f"Saved: {depth_bin_path}")
    logging.info(f"Saved: {depth_npy_path}")
    logging.info(f"Saved: {disp_vis_path}")
    logging.info(f"Saved: {depth_vis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
