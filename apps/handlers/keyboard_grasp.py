"""Keyboard event loop for the single-view realtime grasp application."""
import time
import cv2
import numpy as np

from grasp_executor import GraspStep
from grasp_perception import GraspPerception
from transform import convert_new


DEFAULT_STEPS = [
    GraspStep("approach", gripper="max", offset=(0.0, 0.0, 0.05), preview=1.3, wait=1.5),
    GraspStep("reach", gripper="max", preview=0.5, wait=0.7),
    GraspStep("grasp", gripper="target", preview=0.5, wait=0.8),
    GraspStep("lift", gripper="target", offset=(0.0, 0.0, 0.06), preview=0.5, wait=0.8),
    GraspStep("home", gripper="target", use_home_pose=True, preview=1.5, wait=1.5),
    GraspStep("reopen", gripper="max", use_home_pose=True, preview=0.5),
]


class KeyboardGraspHandler:
    def __init__(self, manager, hw, prompt="mug", output_dir="output"):
        from pynput import keyboard

        self.keyboard = keyboard
        self.cam = manager.require("camera")
        self.depth = manager.require("depth")
        self.perception = GraspPerception(manager, output_dir)
        self.executor = manager.require("executor")
        self.robot = manager.require("robot")
        self.hw = hw
        self.target_width_offset = float(
            (manager.app_config.get("pipeline") or {}).get(
                "target_width_offset", -0.05,
            )
        )
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
        self.depth.step(ctx)                             # raw: mm → 米

        if ctx.color is not None:
            cv2.imshow("Realtime Grasp (h grasp / q quit / n prompt / space home)",
                       cv2.cvtColor(ctx.color, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        for key, fn in self.key_actions.items():
            if self.key_pressed.get(key, False):
                fn(ctx, components)
                if key != self.keyboard.KeyCode.from_char("q"):
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
        grasps = self.perception.generate(ctx.color, ctx.depth, self.prompt)
        selected = self.perception.select(ctx.color, grasps)
        if selected is None:
            print("[grasp] detection/grasp generation failed FAILED")
            return
        state = self.robot.get_state()
        if not state:
            print("[grasp] robot state unavailable FAILED")
            return
        command = convert_new(
            np.asarray(selected["translation"]), np.asarray(selected["rotation"]),
            state["ee_pose"], self.hw.hand_eye_r, self.hw.hand_eye_t,
            selected["depth"],
        )
        if not self.hw.in_workspace(*command[:3]):
            print(f"[grasp] out of workspace {command[:3]} FAILED")
            return
        width = max(0.0, selected["width"] + self.target_width_offset)
        ok, reason = self.executor.run_sequence(command, width, DEFAULT_STEPS)
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

    def close(self):
        try:
            self._listener.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
