"""main_pipeline 业务 Handler:主线程取帧 + worker(depth+grasp)→ o3d。

分工(和旧版单 worker 一致,避免 worker 跨线程碰 RealSense 导致启动慢):
  - 主线程(handler.step):cam.step 取帧(单线程 RealSense)+ worker.submit + viz 渲染。
  - worker 子线程:depth.step(ffs/raw)+ grasp.predict(纯 CUDA,不碰相机)。

depth 组件统一 step(ctx)(ffs: IR→FFS→align→color;raw: mm→米),handler 不区分 FFS 开关。
"""
from context import FrameContext
from worker import AsyncWorker

DEPTH_MAX_M = 3.0


class RealtimeVisHandler:
    """主线程 cam.step 取帧 + worker(depth+grasp)+ o3d 渲染。"""

    def __init__(self, depth, grasp_engine, cam, viz):
        self.depth = depth
        self.grasp = grasp_engine
        self.cam = cam
        self.viz = viz
        self.worker = AsyncWorker(self._process_frame, name="vis-worker")

    def _process_frame(self, frame):
        color, ir, depth_mm = frame
        local = FrameContext(color=color, ir=ir, depth=depth_mm)
        self.depth.step(local)
        if local.depth is None or local.color is None:
            return None
        mask = (local.depth > 0) & (local.depth < DEPTH_MAX_M)
        grasps = None
        if int(mask.sum()) >= self.grasp.num_points:
            grasps, _ = self.grasp.predict(
                local.color, local.depth, mask=mask, topk=None,
            )
        return local.depth, local.color, grasps

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

    def close(self):
        self.worker.stop()
