"""run_realtime 业务 Handler:键盘事件驱动抓取(h=grasp / space=home / n=prompt / q=quit)。

逻辑等价于原 run_realtime.RealtimeGraspController(键盘 listener + cv2 预览 + action)。
依赖 pynput;抓取动作走 GraspAction。
"""
import time
import cv2
from pynput import keyboard
from grasp_action import GraspAction


class KeyboardGraspHandler:
    def __init__(self, cam, detector, segmenter, grasp_engine, selector, executor, robot, hw,
                 prompt="mug"):
        self.cam = cam
        self.action = GraspAction(detector, segmenter, grasp_engine, selector, executor, robot, hw)
        self.robot = robot
        self.prompt = prompt
        self._ctx = None
        self.key_actions = {
            keyboard.KeyCode.from_char("h"): self._grasp,
            keyboard.KeyCode.from_char("q"): self._quit,
            keyboard.KeyCode.from_char("n"): self._new_prompt,
            keyboard.Key.space: self._home,
        }
        self.key_pressed = {k: False for k in self.key_actions}
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def on_start(self):
        """由 app 在 handshake 后、run 前调用:启动 listener + 机械臂就位。"""
        self._listener.start()
        self.robot.reset_to_home()

    def step(self, ctx, components):
        self._ctx = ctx
        cam = components.get("camera") or self.cam
        cam.step(ctx)                                    # 取 color + depth(mm)
        depth = components.get("depth")
        if depth is not None and hasattr(depth, "step"):
            depth.step(ctx)                              # raw: mm → 米

        if ctx.color is not None:
            cv2.imshow("Realtime Grasp (h grasp / q quit / n prompt / space home)",
                       cv2.cvtColor(ctx.color, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        for key, fn in self.key_actions.items():
            if self.key_pressed.get(key, False):
                fn(ctx, components)
                if key != keyboard.KeyCode.from_char("q"):
                    self._wait_release(key)
        time.sleep(0.01)

    def _on_press(self, key):
        if key in self.key_pressed:
            self.key_pressed[key] = True

    def _on_release(self, key):
        if key in self.key_pressed:
            self.key_pressed[key] = False

    def _wait_release(self, key):
        """阻塞至按键释放,期间持续 flush 相机(避免流滞后)。"""
        while self.key_pressed.get(key, False) and self._ctx is not None:
            self.cam.step(self._ctx)
            time.sleep(0.05)

    def _grasp(self, ctx, components):
        if ctx.color is None or ctx.depth is None:
            return
        print(f"[grasp] '{self.prompt}' ...")
        ok, reason = self.action.run(ctx.color, ctx.depth, self.prompt)
        print(f"[grasp] {reason}" + (" OK" if ok else " FAILED"))

    def _home(self, ctx, components):
        print("[home]")
        self.robot.reset_to_home()

    def _new_prompt(self, ctx, components):
        try:
            p = input("New prompt > ").strip()
            if p:
                self.prompt = p
                print(f"prompt = {p}")
        except EOFError:
            pass

    def _quit(self, ctx, components):
        ctx.state["quit"] = True

    def release(self):
        try:
            self._listener.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
