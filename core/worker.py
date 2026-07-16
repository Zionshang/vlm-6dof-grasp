"""通用异步 worker:子线程跑 fn(submit/take 模式)。

用途:把 CUDA 推理(FFS / EconomicGrasp)放到子线程,与主线程 o3d(OpenGL)隔离,
规避两者同线程同 GPU 争用导致的 pointnet2 gather 越界崩溃。
"""
import threading
import time


class AsyncWorker:
    def __init__(self, fn, name="worker"):
        self.fn = fn
        self.name = name
        self._lock = threading.Lock()
        self._in = None
        self._out = None
        self._running = True
        self._traced = False      # 首次异常打印完整 traceback(调试 CUDA 越界定位)
        self.thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self.thread.start()

    def _loop(self):
        import torch
        while self._running:
            with self._lock:
                inp = self._in
                self._in = None
            if inp is None:
                time.sleep(0.005)
                continue
            try:
                torch.cuda.empty_cache()
                out = self.fn(inp)
            except Exception as e:
                print(f"[{self.name}] {str(e).splitlines()[0]}")
                if not self._traced:
                    self._traced = True
                    import traceback
                    traceback.print_exc()
            else:
                with self._lock:
                    self._out = out

    def submit(self, inp):
        """投递最新输入(覆盖未处理的旧输入,始终处理最新帧)。"""
        with self._lock:
            self._in = inp

    def take(self):
        """取回最新输出;无新输出返回 None(调用方跳过本次更新)。"""
        with self._lock:
            o = self._out
            self._out = None
            return o

    def stop(self):
        self._running = False
        self.thread.join(timeout=2.0)
