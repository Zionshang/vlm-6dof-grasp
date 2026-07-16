#!/usr/bin/env python3
"""Build a TensorRT engine from the FFSGWCVolume plugin ONNX.

This mirrors cpp/app/build_single_engine.cpp in Python. The ONNX parser still
needs the custom FFSGWCVolume plugin creator registered before parsing, so the
shared plugin library is auto-detected from cpp/build or can be passed with
--plugin_lib.
"""

import argparse
import ctypes
import os
from pathlib import Path


PLUGIN_NAME = "FFSGWCVolume"
PLUGIN_VERSION = "1"
_LOADED_PLUGIN_LIBS = []


def find_default_plugin_library() -> Path | None:
    repo_dir = Path(__file__).resolve().parents[1]
    candidates = [
        repo_dir / "cpp" / "build" / "libffs_gwc_plugin.so",
        repo_dir / "cpp" / "build" / "lib" / "libffs_gwc_plugin.so",
        repo_dir / "cpp" / "build" / "Release" / "libffs_gwc_plugin.so",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_plugin_library(path: str) -> None:
    """Load an optional shared library that registers FFSGWCVolume."""
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    _LOADED_PLUGIN_LIBS.append(lib)

    # The current C++ code registers via ffs_depth::registerFFSGWCPlugin().
    # A loadable Python plugin library should expose an extern "C" wrapper with
    # one of these names so ctypes can call it without C++ name mangling.
    for symbol in ("ffs_register_gwc_plugin", "registerFFSGWCPlugin"):
        try:
            fn = getattr(lib, symbol)
        except AttributeError:
            continue
        fn.restype = ctypes.c_bool
        if not fn():
            raise RuntimeError(f"{symbol}() returned false for {path}")
        return

    # Some TensorRT plugin libraries register creators during library load. That
    # is not true for this repo's current static C++ helper, but allow it here.


def find_plugin_creator(trt) -> bool:
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator(PLUGIN_NAME, PLUGIN_VERSION, "")
    return creator is not None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from an ONNX graph containing FFSGWCVolume."
    )
    parser.add_argument("plugin_onnx", type=Path, help="Path to plugin ONNX file")
    parser.add_argument("output_engine", type=Path, help="Path to write TensorRT engine")
    parser.add_argument(
        "--plugin_lib",
        type=Path,
        default=None,
        help=(
            "Shared library that registers FFSGWCVolume. Defaults to "
            "cpp/build/libffs_gwc_plugin.so when present. Pure Python cannot "
            "provide this repo's CUDA plugin implementation."
        ),
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable FP16 builder flag. Default allows FP16 when supported.",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=4096,
        help="TensorRT workspace memory limit in MiB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.plugin_onnx.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {args.plugin_onnx}")
    args.output_engine.parent.mkdir(parents=True, exist_ok=True)

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    plugin_lib = args.plugin_lib or find_default_plugin_library()
    if plugin_lib is not None:
        if not plugin_lib.exists():
            raise FileNotFoundError(f"Plugin library does not exist: {plugin_lib}")
        load_plugin_library(str(plugin_lib))

    if not find_plugin_creator(trt):
        raise RuntimeError(
            f"{PLUGIN_NAME} plugin creator is not registered. "
            "Build/load a shared library for cpp/src/gwc_volume_plugin.cpp and pass "
            "--plugin_lib, or use cpp/build/ffs_build_single_engine."
        )

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    parsed = False
    if hasattr(parser, "parse_from_file"):
        parsed = parser.parse_from_file(str(args.plugin_onnx))
    else:
        parsed = parser.parse(args.plugin_onnx.read_bytes())
    if not parsed:
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"failed to parse ONNX: {args.plugin_onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_mb) * 1024 * 1024
    )
    if not args.fp32 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network failed")

    args.output_engine.write_bytes(bytes(serialized))
    precision = "FP32" if args.fp32 else "FP16 allowed"
    print(f"Built engine: {args.output_engine}")
    print(f"Precision: {precision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
