"""main_pipeline 业务 Handler:camera → FFS+grasp(单 worker 串联)→ o3d(主线程)。

单 worker 串联(FFS 完再 grasp):避免 FFS 与 EconomicGrasp 两个 CUDA 线程并发争用
(并发会触发 CUDA illegal memory access,CUDA_LAUNCH_BLOCKING=1 串行化后消失)。
代价:点云更新频率 = FFS+grasp 耗时(~0.5s/帧),但稳。
"""
from worker import AsyncWorker

DEPTH_MAX_M = 3.0   # 与原 main_pipeline 的 DEPTH_MAX_MM(3000mm)一致


class RealtimeVisHandler:
    def __init__(self, ffs, grasp_engine, cam, viz):
        self.cam = cam
        self.viz = viz
        self.ffs = ffs
        self.grasp_engine = grasp_engine
        self.worker = AsyncWorker(self._ffs_then_grasp, name="ffs+grasp")

    def _ffs_then_grasp(self, inp):
        """同一子线程内:FFS → grasp(串联,不并发)。返回 (depth_color, gg|None)。"""
        ir1, ir2, color = inp
        depth = self.ffs.ffs_depth(ir1, ir2, self.cam)
        mask = (depth > 0) & (depth < DEPTH_MAX_M)
        gg = None
        if int(mask.sum()) >= self.grasp_engine.num_points:
            gg, _ = self.grasp_engine.predict(color, depth, mask=mask, topk=None)
        return depth, gg

    def step(self, ctx, components):
        self.cam.step(ctx)
        if ctx.ir is not None and ctx.color is not None:
            self.worker.submit((ctx.ir[0], ctx.ir[1], ctx.color))

        res = self.worker.take()
        if res is not None:
            depth, gg = res
            ctx.depth = depth
            self.viz.update_cloud(ctx.color, depth)
            if gg is not None:
                self.viz.update_grasps(gg)

        if not self.viz.poll():
            ctx.state["quit"] = True
        self.viz.render()

    def release(self):
        self.worker.stop()
