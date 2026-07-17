"""main_pipeline 业务 Handler:主线程取帧 + worker(depth+grasp)→ o3d。

分工(和旧版单 worker 一致,避免 worker 跨线程碰 RealSense 导致启动慢):
  - 主线程(handler.step):cam.step 取帧(单线程 RealSense)+ worker.submit + viz 渲染。
  - worker 子线程:depth.step(ffs/raw)+ grasp.predict(纯 CUDA,不碰相机)。

depth 组件统一 step(ctx)(ffs: IR→FFS→align→color;raw: mm→米),handler 不区分 FFS 开关。
"""
import time
import threading
from context import FrameContext

DEPTH_MAX_M = 3.0


class _VisWorker:
    """子线程:depth.step + grasp.predict(不碰相机)。submit(frame) / take(depth,color,gg)。"""

    def __init__(self, depth, grasp, depth_max=DEPTH_MAX_M):
        self.depth = depth
        self.grasp = grasp
        self.depth_max = depth_max
        self._lock = threading.Lock()
        self._frame = None
        self._result = None
        self._running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="vis")
        self.thread.start()

    def _loop(self):
        import torch
        while self._running:
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                time.sleep(0.005)
                continue
            color, ir, depth_mm = frame
            # 用 local ctx,不和主线程 ctx 共享(depth.step 写 local)
            lctx = FrameContext(color=color, ir=ir, depth=depth_mm)
            res = None
            try:
                torch.cuda.empty_cache()
                self.depth.step(lctx)   # ffs: ctx.ir→depth; raw: ctx.depth(mm)→米
                if lctx.depth is not None and lctx.color is not None:
                    mask = (lctx.depth > 0) & (lctx.depth < self.depth_max)
                    gg = None
                    if int(mask.sum()) >= self.grasp.num_points:
                        gg, _ = self.grasp.predict(lctx.color, lctx.depth, mask=mask, topk=None)
                    res = (lctx.depth, lctx.color, gg)
            except Exception as e:
                print(f"[vis-worker] {str(e).splitlines()[0]}")
            with self._lock:
                self._result = res

    def submit(self, frame):
        with self._lock:
            self._frame = frame

    def take(self):
        with self._lock:
            r = self._result
            self._result = None
            return r

    def stop(self):
        self._running = False
        self.thread.join(timeout=2.0)


class RealtimeVisHandler:
    """主线程 cam.step 取帧 + worker(depth+grasp)+ o3d 渲染。"""

    def __init__(self, depth, grasp_engine, cam, viz, ctx):
        self.depth = depth
        self.grasp = grasp_engine
        self.cam = cam
        self.viz = viz
        self.ctx = ctx
        self.worker = _VisWorker(depth, grasp_engine)

    def step(self, ctx, components):
        # 1. 主线程取帧(单线程 RealSense,避免 worker 跨线程 wait 慢)
        self.cam.step(ctx)
        if ctx.color is not None:
            self.worker.submit((ctx.color, ctx.ir, ctx.depth))

        # 2. 取 worker 最新结果(可能 None:首帧/推理未完成)
        res = self.worker.take()
        if res is not None:
            depth_m, color, gg = res
            ctx.depth = depth_m
            self.viz.update_cloud(color, depth_m)
            if gg is not None:
                self.viz.update_grasps(gg)

        # 3. 渲染(每帧,保持窗口响应)
        if not self.viz.poll():
            ctx.state["quit"] = True
        self.viz.render()

    def release(self):
        self.worker.stop()
