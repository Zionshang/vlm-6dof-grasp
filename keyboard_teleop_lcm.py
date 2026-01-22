import time
from queue import Queue
from pynput import keyboard
import click
import numpy as np

from communication.lcm.lcm_client import Arx5LcmClient


def start_keyboard_teleop(
    client: Arx5LcmClient,
    pos_speed: float,
    ori_speed: float,
    gripper_speed: float,
    cmd_dt: float,
    preview_time: float,
    gripper_width: float,
) -> None:
    target_pose_6d = client.get_state()["ee_pose"].copy()
    target_gripper_pos = float(client.get_state()["gripper_pos"])
    window_size = 5
    keyboard_queue = Queue(window_size)

    print("Teleop tracking started.")

    key_pressed = {
        keyboard.Key.up: False,  # +x
        keyboard.Key.down: False,  # -x
        keyboard.Key.left: False,  # +y
        keyboard.Key.right: False,  # -y
        keyboard.Key.page_up: False,  # +z
        keyboard.Key.page_down: False,  # -z
        keyboard.KeyCode.from_char("q"): False,  # +roll
        keyboard.KeyCode.from_char("a"): False,  # -roll
        keyboard.KeyCode.from_char("w"): False,  # +pitch
        keyboard.KeyCode.from_char("s"): False,  # -pitch
        keyboard.KeyCode.from_char("e"): False,  # +yaw
        keyboard.KeyCode.from_char("d"): False,  # -yaw
        keyboard.KeyCode.from_char("r"): False,  # open gripper
        keyboard.KeyCode.from_char("f"): False,  # close gripper
        keyboard.Key.space: False,  # reset to home
    }

    def on_press(key):
        if key in key_pressed:
            key_pressed[key] = True

    def on_release(key):
        if key in key_pressed:
            key_pressed[key] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def get_filtered_keyboard_output(key_pressed_map: dict) -> np.ndarray:
        state = np.zeros(6, dtype=np.float64)
        if key_pressed_map[keyboard.Key.up]:
            state[0] = 1
        if key_pressed_map[keyboard.Key.down]:
            state[0] = -1
        if key_pressed_map[keyboard.Key.left]:
            state[1] = 1
        if key_pressed_map[keyboard.Key.right]:
            state[1] = -1
        if key_pressed_map[keyboard.Key.page_up]:
            state[2] = 1
        if key_pressed_map[keyboard.Key.page_down]:
            state[2] = -1
        if key_pressed_map[keyboard.KeyCode.from_char("q")]:
            state[3] = 1
        if key_pressed_map[keyboard.KeyCode.from_char("a")]:
            state[3] = -1
        if key_pressed_map[keyboard.KeyCode.from_char("w")]:
            state[4] = 1
        if key_pressed_map[keyboard.KeyCode.from_char("s")]:
            state[4] = -1
        if key_pressed_map[keyboard.KeyCode.from_char("e")]:
            state[5] = 1
        if key_pressed_map[keyboard.KeyCode.from_char("d")]:
            state[5] = -1

        if keyboard_queue.maxsize > 0 and keyboard_queue._qsize() == keyboard_queue.maxsize:
            keyboard_queue._get()

        keyboard_queue.put(state)

        return np.mean(np.array(list(keyboard_queue.queue)), axis=0)

    start_time = time.monotonic()
    loop_cnt = 0
    while True:
        state = client.get_state()
        ee_pose = state["ee_pose"]
        print(
            f"Time elapsed: {time.monotonic() - start_time:.03f}s, "
            f"x: {ee_pose[0]:.03f}, y: {ee_pose[1]:.03f}, z: {ee_pose[2]:.03f}",
            end="\r",
        )

        state_cmd = get_filtered_keyboard_output(key_pressed)
        key_open = key_pressed[keyboard.KeyCode.from_char("r")]
        key_close = key_pressed[keyboard.KeyCode.from_char("f")]
        key_space = key_pressed[keyboard.Key.space]

        if key_space:
            client.reset_to_home()
            target_pose_6d = client.get_state()["ee_pose"].copy()
            target_gripper_pos = 0.0
            loop_cnt = 0
            start_time = time.monotonic()
            continue
        elif key_open and not key_close:
            gripper_cmd = 1
        elif key_close and not key_open:
            gripper_cmd = -1
        else:
            gripper_cmd = 0

        target_pose_6d[:3] += state_cmd[:3] * pos_speed * cmd_dt
        target_pose_6d[3:] += state_cmd[3:] * ori_speed * cmd_dt
        target_gripper_pos += gripper_cmd * gripper_speed * cmd_dt
        if target_gripper_pos >= gripper_width:
            target_gripper_pos = gripper_width
        elif target_gripper_pos <= 0:
            target_gripper_pos = 0

        loop_cnt += 1
        while time.monotonic() < start_time + loop_cnt * cmd_dt:
            pass

        client.set_ee_pose(
            target_pose_6d,
            gripper_pos=target_gripper_pos,
            preview_time=preview_time,
        )


@click.command()
@click.option("--address", default="239.255.76.67", help="LCM multicast address")
@click.option("--port", type=int, default=7667, help="LCM multicast port")
@click.option("--ttl", type=int, default=1, help="LCM multicast TTL")
@click.option("--pos-speed", type=float, default=0.2, help="Position speed scale (m/s)")
@click.option("--ori-speed", type=float, default=0.5, help="Orientation speed scale (rad/s)")
@click.option("--gripper-speed", type=float, default=0.04, help="Gripper speed scale (m/s)")
@click.option("--cmd-dt", type=float, default=0.01, help="Command interval (s)")
@click.option("--preview-time", type=float, default=0.1, help="Preview time (s)")
@click.option("--gripper-width", type=float, default=0.9, help="Max gripper width (m)")
def main(
    address: str,
    port: int,
    ttl: int,
    pos_speed: float,
    ori_speed: float,
    gripper_speed: float,
    cmd_dt: float,
    preview_time: float,
    gripper_width: float,
) -> None:
    client = Arx5LcmClient(url="", address=address, port=port, ttl=ttl)
    client.reset_to_home()

    np.set_printoptions(precision=4, suppress=True)
    try:
        start_keyboard_teleop(
            client,
            pos_speed=pos_speed,
            ori_speed=ori_speed,
            gripper_speed=gripper_speed,
            cmd_dt=cmd_dt,
            preview_time=preview_time,
            gripper_width=gripper_width,
        )
    except KeyboardInterrupt:
        print("Teleop is terminated. Resetting to home.")
        client.reset_to_home()
        client.set_to_damping()


if __name__ == "__main__":
    main()
