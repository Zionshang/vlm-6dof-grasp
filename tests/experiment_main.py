import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT)])

from inference_pipeline import GraspPipeline


class RealSenseBase:
    model = ""

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self._serial())
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        print(f"{self.model} started.")

    def _serial(self):
        for dev in rs.context().query_devices():
            name = dev.get_info(rs.camera_info.name).lower()
            if self.model.lower() in name:
                return dev.get_info(rs.camera_info.serial_number)
        raise RuntimeError(f"{self.model} not found")

    def get_frames(self):
        try:
            frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=1000))
            color = np.asanyarray(frames.get_color_frame().get_data())
            depth = np.asanyarray(frames.get_depth_frame().get_data())
            return color, depth
        except Exception:
            return None, None

    def release(self):
        self.pipeline.stop()


class RealSenseD405(RealSenseBase):
    model = "D405"


class RealSenseD435i(RealSenseBase):
    model = "D435"


def ts():
    return time.strftime("%Y%m%d-%H%M%S")


def save_rgb(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

def save_depth(path_raw, path_vis, depth):
    path_raw.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path_raw), depth)
    valid = depth[depth > 0]
    if valid.size == 0:
        cv2.imwrite(str(path_vis), np.zeros((*depth.shape, 3), dtype=np.uint8))
        return
    lo, hi = np.percentile(valid, [2, 98])
    depth_norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_u8[depth == 0] = 0
    cv2.imwrite(str(path_vis), cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO))


def pca_long_axis(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 6:
        return None
    points = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors, _ = cv2.PCACompute2(points, mean=np.empty((0)))
    cx, cy = mean[0]
    vx, vy = eigenvectors[0]
    if vy < 0 or (abs(vy) < 1e-6 and vx < 0):
        vx, vy = -vx, -vy
    return (float(cx), float(cy)), (float(vx), float(vy)), math.degrees(math.atan2(vx, vy))


def detect_segment(pipeline, color, prompt, out_dir):
    img_path = out_dir / "rgb.png"
    save_rgb(img_path, color)
    boxes = pipeline.vlm.run(str(img_path), prompt).get("pixel_boxes", [])
    if not boxes:
        return None, None
    boxes = pipeline.expand_boxes(boxes, color.shape)
    mask = pipeline._run_segmentation(str(img_path), boxes, color.shape)
    return boxes, mask


def save_pca_visualization(color, mask, boxes, out_path):
    h, w = color.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_u8 = (mask > 0).astype(np.uint8)
    pca_res = pca_long_axis(mask_u8)

    vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    overlay = vis.copy()
    overlay[mask_u8 > 0] = (0, 255, 120)
    vis = cv2.addWeighted(vis, 0.45, overlay, 0.55, 0)

    box = None
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (95, 245, 110), 2)
        box = box or (x1, y1, x2, y2)

    angle_text = "angle: N/A"
    if pca_res is not None:
        (cx, cy), (vx, vy), angle_deg = pca_res
        c = (int(round(cx)), int(round(cy)))
        axis_len = max(60, min(h, w) // 5)
        dx, dy = int(round(vx * axis_len)), int(round(vy * axis_len))
        cv2.arrowedLine(vis, (c[0], c[1] - axis_len), (c[0], c[1] + axis_len), (235, 235, 235), 2, cv2.LINE_AA, tipLength=0.04)
        cv2.arrowedLine(vis, (c[0] - dx, c[1] - dy), (c[0] + dx, c[1] + dy), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.08)
        cv2.circle(vis, c, 5, (255, 255, 255), -1, cv2.LINE_AA)
        angle_text = f"angle: {angle_deg:+.1f}"

    if box is not None:
        x1, y1, _, _ = box
        scale, thick, px, py = 0.78, 2, 10, 8
        (tw, th), base = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        tx1, ty2 = max(0, x1), max(th + base + py * 2, y1 - 6)
        ty1, tx2 = max(0, ty2 - th - base - py * 2), min(w, tx1 + tw + px * 2)
        cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
        cv2.putText(vis, angle_text, (tx1 + px, ty2 - base - py), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis)
    return pca_res


def generate_grasps(pipeline, color, depth, mask):
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    gg, data = pipeline.grasp_engine.predict(color, depth, mask=mask, topk=100)
    if len(gg) == 0:
        return None, None
    keep = []
    for i in range(len(gg)):
        if len(keep) >= 8:
            break
        r = gg[i].rotation_matrix
        if np.arccos(np.clip(np.dot(r[:, 0], [0, 0, 1]), -1, 1)) < np.deg2rad(50) and np.arccos(np.clip(np.dot(r[:, 1], [1, 0, 0]), -1, 1)) < np.deg2rad(100):
            keep.append(i)
    return (gg[keep] if keep else gg[:8]), data


def visualize_o3d_auto(gg, data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data["point_clouds"])
    cloud.colors = o3d.utility.Vector3dVector(data["cloud_colors"])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Grasp Results")
    vis.add_geometry(cloud)
    for i in range(min(len(gg), 4)):
        g = gg[i].to_open3d_geometry(color=(0, 0, 0))
        for geom in (g if isinstance(g, list) else [g]):
            vis.add_geometry(geom)

    last, idx = 0.0, 0
    while vis.poll_events():
        vis.update_renderer()
        now = time.time()
        if now - last >= 3.0:
            vis.capture_screen_image(str(out_dir / f"{idx:03d}.png"))
            idx += 1
            last = now
        time.sleep(0.02)
    vis.destroy_window()


def project_grasp_to_2d(trans, rot, width, intrinsic, depth=0.04):
    t_grasp = np.eye(4)
    t_grasp[:3, :3] = rot
    t_grasp[:3, 3] = trans

    t_align = np.eye(4)
    t_align[:3, 3] = [0.04, 0, 0]
    t_final = t_grasp @ t_align

    hw = width / 2
    points_g = np.array([[0, -hw, 0], [-depth, -hw, 0], [-depth, hw, 0], [0, hw, 0]]).T
    points_c = t_final[:3, :3] @ points_g + t_final[:3, 3].reshape(3, 1)

    z = points_c[2]
    z[z == 0] = 1e-3
    u = intrinsic[0, 0] * points_c[0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_c[1] / z + intrinsic[1, 2]
    return np.stack([u, v], axis=1).astype(int)


def draw_id_label(vis_img, pts, idx):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    label = f"ID: {idx}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x = int((min_x + max_x) // 2 - tw // 2)
    y = int(min_y - 40)
    if y - th < 0:
        y = int(max_y + th + 40)
    x = max(2, min(x, vis_img.shape[1] - tw - 2))
    cv2.rectangle(vis_img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (255, 255, 255), -1)
    cv2.putText(vis_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)


def save_2d_grasps(pipeline, color, gg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    trans = np.asarray(gg.translations)
    rot = np.asarray(gg.rotation_matrices)
    width = np.asarray(gg.widths)
    intrinsic = pipeline.grasp_engine.intrinsic

    for i in range(min(len(trans), 4)):
        if trans[i][2] <= 0:
            continue
        vis = color.copy()
        pts = project_grasp_to_2d(trans[i], rot[i], width[i], intrinsic)
        center_base = np.mean(pts[1:3], axis=0).astype(int)
        center_tip = np.mean([pts[0], pts[3]], axis=0).astype(int)

        cv2.line(vis, tuple(pts[0]), tuple(pts[1]), (255, 0, 0), 3)
        cv2.line(vis, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), 3)
        cv2.line(vis, tuple(pts[2]), tuple(pts[3]), (255, 0, 0), 3)
        cv2.arrowedLine(vis, tuple(center_base), tuple(center_tip), (0, 0, 255), 3, tipLength=0.35)
        draw_id_label(vis, pts, i)
        cv2.imwrite(str(out_dir / f"{i}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main():
    parser = GraspPipeline.get_parser()
    parser.set_defaults(no_vis=True)
    args = parser.parse_args()

    prompt_name = "_".join(args.prompt.strip().split()) or "object"
    result_dir = ROOT / "experiment_result" / f"{prompt_name}_{ts()}"
    result_dir.mkdir(parents=True, exist_ok=True)
    pipeline = GraspPipeline(args)
    d435, d405 = RealSenseD435i(), RealSenseD405()
    print(f"result: {result_dir}")
    print("q: save D435i | w: save D405 RGB+PCA | k: grasp data | Esc: quit")

    try:
        while True:
            c435, _ = d435.get_frames()
            c405, d405_depth = d405.get_frames()
            if c435 is not None:
                cv2.imshow("D435i", cv2.cvtColor(c435, cv2.COLOR_RGB2BGR))
            if c405 is not None:
                cv2.imshow("D405", cv2.cvtColor(c405, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == ord("q") and c435 is not None:
                out = result_dir / f"435_{ts()}"
                save_rgb(out / "rgb.png", c435)
                print(f"saved {out}")
            if key == ord("w") and c405 is not None:
                out = result_dir / f"405_{ts()}"
                save_rgb(out / "rgb.png", c405)
                boxes, mask = detect_segment(pipeline, c405, args.prompt, out)
                if mask is not None:
                    save_pca_visualization(c405, mask, boxes, out / "pca.png")
                print(f"saved {out}")
            if key == ord("k") and c405 is not None and d405_depth is not None:
                stamp = ts()
                pic_dir = result_dir / f"grasp_pic_{stamp}"
                boxes, mask = detect_segment(pipeline, c405, args.prompt, pic_dir)
                if mask is None:
                    print("no mask")
                    continue
                save_rgb(pic_dir / "rgb.png", c405)
                save_depth(pic_dir / "depth_raw.png", pic_dir / "depth_vis.png", d405_depth)
                cv2.imwrite(str(pic_dir / "sam.png"), (cv2.resize(mask.astype(np.uint8), (c405.shape[1], c405.shape[0]), interpolation=cv2.INTER_NEAREST) * 255))
                gg, data = generate_grasps(pipeline, c405, d405_depth, mask)
                if gg is None:
                    print("no grasp")
                    continue
                visualize_o3d_auto(gg, data, result_dir / f"grasp_o3d_{stamp}")
                save_2d_grasps(pipeline, c405, gg, result_dir / f"grasp_2Dgrasp_{stamp}")
                print(f"saved grasp data {stamp}")
    finally:
        d435.release()
        d405.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
