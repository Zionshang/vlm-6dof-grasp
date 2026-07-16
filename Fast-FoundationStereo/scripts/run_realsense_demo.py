# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Real-time RealSense D4xx demo: shows three live panels side by side --
#   [RGB color] | [camera hardware depth] | [Fast-FoundationStereo depth]
# The model runs on the hardware-rectified left/right infrared stereo pair.
#
# Usage (after setting up the `ffs` env and downloading weights):
#   python scripts/run_realsense_demo.py \
#       --model_dir weights/20-26-39/model_best_bp2_serialize.pth \
#       --scale 0.5 --valid_iters 4
# Press q or ESC to quit.

import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import argparse, logging, time
import numpy as np
import cv2
import torch
import yaml
from omegaconf import OmegaConf

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed

import pyrealsense2 as rs


def depth_to_color(d, zfar):
  """Turbo-colorize a depth map (meters) on a fixed [0, zfar] range so that
  different depth sources (hardware vs model) are directly comparable."""
  valid = (d > 0) & (d < zfar)
  norm = np.zeros_like(d, dtype=np.float32)
  norm[valid] = np.clip(d[valid] / zfar, 0, 1)
  vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  vis[~valid] = 0
  return vis


def ir_to_uint8(img):
  """Normalize an IR frame (uint8 or uint16) to uint8 grayscale for the net."""
  if img.dtype == np.uint8:
    return img
  img = img.astype(np.float32)
  mask = img > 0
  if mask.any():
    mn, mx = np.percentile(img[mask], [1, 99])
    img = np.clip((img - mn) / (mx - mn + 1e-6), 0, 1) * 255
  return img.astype(np.uint8)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', default=f'{code_dir}/../weights/20-26-39/model_best_bp2_serialize.pth', type=str)
  parser.add_argument('--scale', default=0.5, type=float, help='downscale factor for model input (smaller = faster)')
  parser.add_argument('--valid_iters', type=int, default=4, help='refinement updates (fewer = faster)')
  parser.add_argument('--max_disp', type=int, default=192)
  parser.add_argument('--zfar', type=float, default=10.0, help='max depth (m) for visualization')
  parser.add_argument('--width', type=int, default=640)
  parser.add_argument('--height', type=int, default=480)
  parser.add_argument('--fps', type=int, default=30)
  parser.add_argument('--baseline', type=float, default=-1.0, help='override stereo baseline in meters (-1: read from device)')
  parser.add_argument('--serial', type=str, default='', help='optional camera serial number')
  parser.add_argument('--display_width', type=int, default=480, help='per-panel width in the output window')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  # ---- Load model (mirrors scripts/run_demo.py) ----
  with open(f'{os.path.dirname(args.model_dir)}/cfg.yaml', 'r') as ff:
    cfg = yaml.safe_load(ff)
  cfg['valid_iters'] = args.valid_iters
  cfg['max_disp'] = args.max_disp
  cfg['scale'] = args.scale
  args_cfg = OmegaConf.create(cfg)
  logging.info(f"Loading model from {args.model_dir}")
  model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
  model.args.valid_iters = args.valid_iters
  model.args.max_disp = args.max_disp
  model.cuda().eval()

  # ---- RealSense pipeline ----
  pipeline = rs.pipeline()
  config = rs.config()
  if args.serial:
    config.enable_device(args.serial)
  W, H, FPS = args.width, args.height, args.fps
  config.enable_stream(rs.stream.color,  W, H, rs.format.bgr8, FPS)
  config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)  # left IR  = img0
  config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)  # right IR = img1
  config.enable_stream(rs.stream.depth,  W, H, rs.format.z16, FPS)
  profile = pipeline.start(config)
  align = rs.align(rs.stream.color)  # align depth onto color frame for display

  dev = profile.get_device()
  depth_sensor = next(s for s in dev.query_sensors() if s.is_depth_sensor())
  # fx/cx/cy of the left-IR (= depth reference) frame, used for depth = fx*baseline/disp
  intr = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
  fx, cx, cy = intr.fx, intr.ppx, intr.ppy
  logging.info(f"IR intrinsics: fx={fx:.2f} cx={cx:.2f} cy={cy:.2f}")

  if args.baseline > 0:
    baseline = args.baseline
  else:
    try:
      raw = depth_sensor.get_option(rs.option.stereo_baseline)
      baseline = raw / 1000.0 if raw > 0.5 else raw  # heuristic: ~50 -> mm, ~0.05 -> m
      logging.info(f"stereo_baseline raw={raw} -> interpreted as {baseline:.4f} m (override with --baseline if wrong)")
    except Exception as e:
      baseline = 0.05
      logging.warning(f"could not read stereo_baseline ({e}); using default {baseline} m (override with --baseline)")
  logging.info(f"baseline={baseline:.4f} m  (tip: if model depth looks globally scaled vs hardware depth, fix units with --baseline)")

  # ---- Warm up (first forward is slow due to torch.compile) ----
  # Match the real per-frame scaled shape so no recompilation happens in the loop.
  logging.info("Warming up the model (first run compiles kernels)...")
  Wh, Hh = int(W * args.scale), int(H * args.scale)
  z = torch.zeros(1, 3, Hh, Wh, device='cuda')
  padder0 = InputPadder(z.shape, divis_by=32, force_square=False)
  z0, z1 = padder0.pad(z, z)
  with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = model.forward(z0, z1, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
  torch.cuda.synchronize()
  logging.info("Warmup done. Starting live stream (press q / ESC to quit).")

  fps_ema, t_prev = 0.0, time.time()
  try:
    while True:
      frames = pipeline.wait_for_frames()
      aligned = align.process(frames)  # color + depth share a frame here (for display)

      color = np.asanyarray(aligned.get_color_frame().get_data())          # BGR uint8
      depth_hw = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32) / 1000.0  # mm -> m
      left_ir = ir_to_uint8(np.asanyarray(frames.get_infrared_frame(1).get_data()))   # original L/R IR for stereo
      right_ir = ir_to_uint8(np.asanyarray(frames.get_infrared_frame(2).get_data()))
      if color.ndim == 2:  # safety
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)

      # Build stereo pair (grayscale IR -> 3 channels), downscale, to tensor
      img0 = np.stack([left_ir, left_ir, left_ir], -1)
      img1 = np.stack([right_ir, right_ir, right_ir], -1)
      s = args.scale
      img0 = cv2.resize(img0, fx=s, fy=s, dsize=None)
      img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
      Hw, Ww = img0.shape[:2]
      t0 = img0.copy()
      img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
      img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

      padder = InputPadder(img0.shape, divis_by=32, force_square=False)
      img0, img1 = padder.pad(img0, img1)
      with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
      disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(Hw, Ww).clip(1e-6, None)

      # disp is in pixels of the *scaled* image, so the matching focal length is
      # fx*scale (focal length scales with resolution). depth = f_eff*baseline/disp.
      depth_ffs = fx * args.scale * baseline / disp  # meters; disp>0 guaranteed by clip
      depth_ffs = depth_ffs.astype(np.float32)

      # ---- Three-panel visualization (common display height) ----
      dh = args.display_width
      def panel(img, label):
        im = cv2.resize(img, (dh, int(dh * img.shape[0] / img.shape[1])))
        cv2.putText(im, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return im
      p_color = panel(color, "RGB")
      p_hw = panel(depth_to_color(depth_hw, args.zfar), "HW depth (m)")
      p_ffs = panel(depth_to_color(depth_ffs, args.zfar), "FFS depth (m)")
      canvas = np.hstack([p_color, p_hw, p_ffs])

      now = time.time()
      dt = now - t_prev
      t_prev = now
      if dt > 0:
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else (1.0 / dt)
      cv2.putText(canvas, f"FPS {fps_ema:5.1f}  iters={args.valid_iters} scale={args.scale}", (8, canvas.shape[0] - 12),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

      cv2.imshow('Fast-FoundationStereo | RGB | HW depth | FFS depth', canvas)
      key = cv2.waitKey(1) & 0xFF
      if key in (ord('q'), 27):
        break
  finally:
    pipeline.stop()
    cv2.destroyAllWindows()
