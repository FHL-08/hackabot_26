import argparse
import json
import math
import select
import socket
import time

import cv2
import numpy as np

from camera_geometry import (
    CalibrationError,
    apply_calibration_to_frame,
    center_and_radius_from_points,
    load_arena_calibration,
    marker_theta_from_warp_corners,
    warp_points,
)


CAMERA_INDEX = 0
TARGET_MARKER_ID = 0
USE_CALIBRATION = True   # Set True to use arena_calibration.json warping/undistortion
SHOW_BOTH_FEEDS = False   # Show raw + calibrated preview windows side by side
# Display-only zoom for warped preview
FIT_CALIBRATED_DISPLAY_TO_CONTENT = False
CALIBRATED_DISPLAY_THRESHOLD = 8
CALIBRATED_DISPLAY_MARGIN_PX = 10

# Arena mapping (live warped u,v -> arena x,y).
# x_cm = -CM_PER_PX_X * (v_live - ARENA_ORIGIN_V_PX)
# y_cm = -CM_PER_PX_Y * (u_live - ARENA_ORIGIN_U_PX)
USE_CUSTOM_ARENA_ORIGIN = True
ARENA_ORIGIN_U_PX = 621.68
ARENA_ORIGIN_V_PX = 418.20
CM_PER_PX_X = 0.3256260775073436
CM_PER_PX_Y = 0.28330199259559835
THETA_CONVENTION = "rad_ccw_from_x_up_0_to_2pi"
AXIS_CONVENTION = "x_up_y_left"
COORD_FRAME_NAME = "arena_center_xy"
COORD_UNITS = "cm"

SEND_ALL_VISIBLE_MARKERS = True
# Obstacles are manually measured/fixed.
INCLUDE_OBSTACLES_IN_MARKER_PACKET = False
# Obstacles are manually measured/fixed.
ENABLE_LOCAL_OBSTACLE_OVERLAY = False

SERVER_IP = "127.0.0.1"   # this machine
SERVER_PORT = 5000

# seconds between sends (30Hz)
SEND_INTERVAL = 1.0 / 10.0
PREDICT_HOLD_SECONDS = 0.35
LOST_REPORT_INTERVAL = 0.25
PRINT_INTERVAL = 0.50

DESIRED_CAMERA_FPS = 60
MANUAL_EXPOSURE = -6
FORCE_MANUAL_EXPOSURE = False
AUTO_LOCK_IF_TARGET_MISSING = True

WINDOW_NAME = "Tracker"
WINDOW_NAME_RAW = "Tracker Raw"
WINDOW_NAME_CALIBRATED = "Tracker Calibrated"


class TcpJsonClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=3)
        self.sock.settimeout(0.5)

    def send_packet(self, packet):
        msg = json.dumps(packet) + "\n"
        self.sock.sendall(msg.encode("utf-8"))

    def recv_packet(self):
        while "\n" not in self.buffer:
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("Connection closed by receiver")
            self.buffer += data.decode("utf-8")

        line, self.buffer = self.buffer.split("\n", 1)
        return json.loads(line)

    def drain_packets(self, max_packets=10):
        packets = []

        while len(packets) < max_packets:
            if "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                line = line.strip()
                if line:
                    packets.append(json.loads(line))
                continue

            ready, _, _ = select.select([self.sock], [], [], 0)
            if not ready:
                break

            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("Connection closed by receiver")
            self.buffer += data.decode("utf-8")

        return packets

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


class PoseEstimator:
    def __init__(self, alpha=0.72, beta=0.35, max_speed=5000.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_speed = float(max_speed)

        self.x = 0.0
        self.y = 0.0
        self.r = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vr = 0.0

        self.last_update = None
        self.last_measurement = None

    def has_state(self):
        return self.last_update is not None

    def _clamp(self, value, limit):
        return max(-limit, min(limit, value))

    def update(self, measurement, now):
        mx, my, mr = measurement

        if not self.has_state():
            self.x, self.y, self.r = float(mx), float(my), float(mr)
            self.vx = self.vy = self.vr = 0.0
            self.last_update = now
            self.last_measurement = now
            return self.x, self.y, self.r

        dt = max(1e-3, min(0.2, now - self.last_update))

        pred_x = self.x + self.vx * dt
        pred_y = self.y + self.vy * dt
        pred_r = self.r + self.vr * dt

        err_x = float(mx) - pred_x
        err_y = float(my) - pred_y
        err_r = float(mr) - pred_r

        self.x = pred_x + self.alpha * err_x
        self.y = pred_y + self.alpha * err_y
        self.r = pred_r + self.alpha * err_r

        self.vx = self._clamp(
            self.vx + (self.beta * err_x) / dt, self.max_speed)
        self.vy = self._clamp(
            self.vy + (self.beta * err_y) / dt, self.max_speed)
        self.vr = self._clamp(
            self.vr + (self.beta * err_r) / dt, self.max_speed)

        self.last_update = now
        self.last_measurement = now
        return self.x, self.y, self.r

    def predict(self, now):
        if not self.has_state():
            return None

        dt = max(0.0, min(0.2, now - self.last_update))
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.r += self.vr * dt
        self.last_update = now
        return self.x, self.y, self.r

    def age_since_measurement(self, now):
        if self.last_measurement is None:
            return float("inf")
        return max(0.0, now - self.last_measurement)


def configure_capture(cap):
    # Best-effort low-latency capture settings.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_CAMERA_FPS)

    if FORCE_MANUAL_EXPOSURE:
        if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if hasattr(cv2, "CAP_PROP_EXPOSURE"):
            cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)


def get_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    # More tolerant settings for blur/fast motion.
    if hasattr(parameters, "adaptiveThreshWinSizeMin"):
        parameters.adaptiveThreshWinSizeMin = 3
    if hasattr(parameters, "adaptiveThreshWinSizeMax"):
        parameters.adaptiveThreshWinSizeMax = 31
    if hasattr(parameters, "adaptiveThreshWinSizeStep"):
        parameters.adaptiveThreshWinSizeStep = 4
    if hasattr(parameters, "minMarkerPerimeterRate"):
        parameters.minMarkerPerimeterRate = 0.015
    if hasattr(parameters, "maxMarkerPerimeterRate"):
        parameters.maxMarkerPerimeterRate = 4.0
    if hasattr(parameters, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(parameters, "cornerRefinementWinSize"):
        parameters.cornerRefinementWinSize = 5
    if hasattr(parameters, "cornerRefinementMaxIterations"):
        parameters.cornerRefinementMaxIterations = 50

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    def detect(gray):
        return aruco.detectMarkers(gray, dictionary, parameters=parameters)

    return detect


def extract_marker_points(corners):
    return corners.reshape(4, 2).astype(np.float32)


def draw_marker_overlay(display, points_xy, marker_id, cx, cy, radius):
    pts_i = np.asarray(points_xy, dtype=np.float32).astype(int)
    center_i = (int(cx), int(cy))

    cv2.polylines(display, [pts_i], True, (0, 255, 0), 2)
    cv2.circle(display, center_i, 4, (0, 0, 255), -1)
    cv2.circle(display, center_i, int(radius), (255, 0, 0), 2)

    label = f"id={marker_id} x={cx:.1f} y={cy:.1f} r={radius:.1f}"
    cv2.putText(
        display,
        label,
        (center_i[0] + 10, center_i[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def compute_non_black_fit_transform(image, threshold=8, margin_px=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = image.shape[:2]
        return {
            "crop_x0": 0,
            "crop_y0": 0,
            "crop_w": int(w),
            "crop_h": int(h),
            "out_w": int(w),
            "out_h": int(h),
            "scale_x": 1.0,
            "scale_y": 1.0,
        }

    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(image.shape[1], x + w + margin_px)
    y1 = min(image.shape[0], y + h + margin_px)
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))

    out_w = int(image.shape[1])
    out_h = int(image.shape[0])
    return {
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_w": crop_w,
        "crop_h": crop_h,
        "out_w": out_w,
        "out_h": out_h,
        "scale_x": float(out_w) / float(crop_w),
        "scale_y": float(out_h) / float(crop_h),
    }


def apply_fit_transform_for_display(image, transform):
    x0 = int(transform["crop_x0"])
    y0 = int(transform["crop_y0"])
    w = int(transform["crop_w"])
    h = int(transform["crop_h"])
    out_w = int(transform["out_w"])
    out_h = int(transform["out_h"])

    cropped = image[y0:y0 + h, x0:x0 + w]
    if cropped.size == 0:
        return image
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def warp_to_display_xy(u_warp, v_warp, transform):
    u_disp = (float(u_warp) -
              float(transform["crop_x0"])) * float(transform["scale_x"])
    v_disp = (float(v_warp) -
              float(transform["crop_y0"])) * float(transform["scale_y"])
    return u_disp, v_disp


def display_to_warp_xy(u_disp, v_disp, transform):
    u_warp = float(u_disp) / \
        float(transform["scale_x"]) + float(transform["crop_x0"])
    v_warp = float(v_disp) / \
        float(transform["scale_y"]) + float(transform["crop_y0"])
    return u_warp, v_warp


def get_arena_origin_uv(pose_width, pose_height):
    if USE_CUSTOM_ARENA_ORIGIN:
        return float(ARENA_ORIGIN_U_PX), float(ARENA_ORIGIN_V_PX)
    u0 = (float(pose_width) - 1.0) * 0.5
    v0 = (float(pose_height) - 1.0) * 0.5
    return u0, v0


def uv_to_arena_px(u_live, v_live, origin_u, origin_v):
    arena_x_px = -(float(v_live) - float(origin_v))
    arena_y_px = -(float(u_live) - float(origin_u))
    return arena_x_px, arena_y_px


def arena_px_to_cm(x_px, y_px):
    return float(x_px) * float(CM_PER_PX_X), float(y_px) * float(CM_PER_PX_Y)


def draw_heading_arrow(display, u, v, theta_rad, length_px=40, color=(255, 255, 0)):
    # Arena frame heading (x up+, y left+) -> image frame delta (u right+, v down+)
    du = -math.sin(float(theta_rad)) * float(length_px)
    dv = -math.cos(float(theta_rad)) * float(length_px)
    p0 = (int(round(u)), int(round(v)))
    p1 = (int(round(u + du)), int(round(v + dv)))
    cv2.arrowedLine(display, p0, p1, color, 2, tipLength=0.25)


def open_capture(camera_index):
    # Backend index mapping on Windows can differ (e.g., DSHOW vs default/MSMF).
    # Try default first so it matches the behavior seen in simple preview scripts.
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        return cap
    cap.release()

    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            return cap
        cap.release()

    # Return a closed capture object for consistent upstream error handling.
    return cv2.VideoCapture(camera_index)


def run_coordinate_sanity_checks(pose_width, pose_height, origin_u, origin_v):
    # Selected origin must map to arena origin.
    x0, y0 = uv_to_arena_px(origin_u, origin_v, origin_u, origin_v)
    if abs(x0) > 1e-6 or abs(y0) > 1e-6:
        raise RuntimeError(
            f"Coordinate sanity failed at origin: x={x0} y={y0}")

    # Up in image should increase +x; left in image should increase +y.
    xu, yu = uv_to_arena_px(origin_u, origin_v - 10.0, origin_u, origin_v)
    xl, yl = uv_to_arena_px(origin_u - 10.0, origin_v, origin_u, origin_v)
    if xu <= 0.0:
        raise RuntimeError(
            f"Coordinate sanity failed: up did not map to +x (x={xu}, y={yu})")
    if yl <= 0.0:
        raise RuntimeError(
            f"Coordinate sanity failed: left did not map to +y (x={xl}, y={yl})")

    if not (0.0 <= float(origin_u) <= float(pose_width - 1)):
        print(
            f"[TRACKER] Warning: origin_u={origin_u:.2f} is outside pose width [0, {pose_width - 1}]"
        )
    if not (0.0 <= float(origin_v) <= float(pose_height - 1)):
        print(
            f"[TRACKER] Warning: origin_v={origin_v:.2f} is outside pose height [0, {pose_height - 1}]"
        )

    # Theta convention: up=0, left=pi/2, down=pi, right=3pi/2.
    cardinals = {
        "up": (np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64), 0.0),
        "left": (np.array([[-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]], dtype=np.float64), math.pi / 2.0),
        "down": (np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], dtype=np.float64), math.pi),
        "right": (np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=np.float64), 1.5 * math.pi),
    }
    for name, (corners, expected) in cardinals.items():
        theta = marker_theta_from_warp_corners(corners)
        err = abs((theta - expected + math.pi) % (2.0 * math.pi) - math.pi)
        if err > 1e-6:
            raise RuntimeError(
                f"Theta sanity failed for {name}: theta={theta:.6f}, expected={expected:.6f}, err={err:.6f}"
            )


def validate_display_transform_math(transform):
    # Round-trip check for crop+resize transform.
    sample_points = [
        (float(transform["crop_x0"]), float(transform["crop_y0"])),
        (float(transform["crop_x0"] + transform["crop_w"] - 1),
         float(transform["crop_y0"])),
        (float(transform["crop_x0"]), float(
            transform["crop_y0"] + transform["crop_h"] - 1)),
        (
            float(transform["crop_x0"] + 0.5 * (transform["crop_w"] - 1)),
            float(transform["crop_y0"] + 0.5 * (transform["crop_h"] - 1)),
        ),
    ]
    max_err = 0.0
    for u, v in sample_points:
        ud, vd = warp_to_display_xy(u, v, transform)
        ur, vr = display_to_warp_xy(ud, vd, transform)
        err = max(abs(ur - u), abs(vr - v))
        max_err = max(max_err, err)
    return max_err


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send tracker marker packets to laptop_server auto mode"
    )
    parser.add_argument(
        "--server-ip",
        default=SERVER_IP,
        help="Destination IP for laptop_server tracker listener",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=SERVER_PORT,
        help="Destination TCP port for laptop_server tracker listener",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    need_calibration = USE_CALIBRATION or SHOW_BOTH_FEEDS

    if need_calibration:
        try:
            calibration = load_arena_calibration()
        except CalibrationError as e:
            raise RuntimeError(
                f"{e}\n"
                f"Run calibration first:\n"
                f"python -m src.scripts.vision.calibrate_arena_camera --camera-index {CAMERA_INDEX}"
            )
        print("[TRACKER] Calibration loaded")
    else:
        calibration = None
        print("[TRACKER] Tracking mode: uncalibrated (raw camera)")

    if USE_CALIBRATION:
        if calibration is None:
            raise RuntimeError(
                "USE_CALIBRATION=True requires arena calibration")
        print("[TRACKER] Coordinate source: calibrated warped frame")
        pose_frame_name = COORD_FRAME_NAME
        pose_w, pose_h = calibration["warp_size"]
        u0, v0 = get_arena_origin_uv(pose_w, pose_h)
        run_coordinate_sanity_checks(pose_w, pose_h, u0, v0)
        print("[TRACKER] Coordinate sanity checks passed")
    else:
        print("[TRACKER] Coordinate source: raw camera frame")
        pose_frame_name = "camera_raw_custom_xy" if USE_CUSTOM_ARENA_ORIGIN else "camera_raw_center_xy"
        pose_w = pose_h = 0
        u0 = v0 = 0.0

    base_frame_metadata = {
        "coord_frame": pose_frame_name,
        "axis_convention": AXIS_CONVENTION,
        "theta_convention": THETA_CONVENTION,
        "units": COORD_UNITS,
        "cm_per_px_x": float(CM_PER_PX_X),
        "cm_per_px_y": float(CM_PER_PX_Y),
        "arena_origin_u_px": float(u0),
        "arena_origin_v_px": float(v0),
        "arena_origin_mode": "custom" if USE_CUSTOM_ARENA_ORIGIN else "center",
        "theta_range": "[0,2pi)",
    }
    if calibration is not None:
        warp_w, warp_h = calibration["warp_size"]
        base_frame_metadata.update(
            {
                "warp_width": int(warp_w),
                "warp_height": int(warp_h),
            }
        )
    print(
        f"[TRACKER] Scale constants: cm_per_px_x={CM_PER_PX_X:.6f}, "
        f"cm_per_px_y={CM_PER_PX_Y:.6f}"
    )
    print(
        f"[TRACKER] Arena origin: u0={u0:.2f}, v0={v0:.2f} "
        f"({('custom' if USE_CUSTOM_ARENA_ORIGIN else 'center')})"
    )

    if SHOW_BOTH_FEEDS:
        print("[TRACKER] Display mode: side-by-side raw + calibrated")
    elif USE_CALIBRATION:
        print("[TRACKER] Display mode: calibrated only")
    else:
        print("[TRACKER] Display mode: raw only")

    print(f"[TRACKER] Target server: {args.server_ip}:{args.server_port}")
    client = TcpJsonClient(args.server_ip, args.server_port)
    client.connect()

    hello_packet = {
        "type": "hello",
        "message": "tracker_connected"
    }
    client.send_packet(hello_packet)
    hello_ack = client.recv_packet()
    print("[TRACKER] Handshake ACK:", hello_ack)

    cap = open_capture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
    configure_capture(cap)

    detect_markers = get_detector()

    estimator = PoseEstimator()
    active_target_id = int(TARGET_MARKER_ID)
    autolock_message_until = 0.0
    seq = 0
    last_send_time = 0.0
    last_lost_report_time = 0.0
    last_print_time = 0.0
    display_transform_checked = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[TRACKER] Failed to read frame")
                break

            now = time.time()
            raw_display = frame.copy()
            calibrated_display = (
                apply_calibration_to_frame(
                    frame, calibration) if calibration is not None else None
            )
            pose_frame = calibrated_display if (
                USE_CALIBRATION and calibrated_display is not None) else frame
            if not USE_CALIBRATION and (pose_w <= 0 or pose_h <= 0):
                pose_h, pose_w = pose_frame.shape[:2]
                u0, v0 = get_arena_origin_uv(pose_w, pose_h)
                run_coordinate_sanity_checks(pose_w, pose_h, u0, v0)
                print("[TRACKER] Coordinate sanity checks passed")
            active_display = (
                calibrated_display if (
                    USE_CALIBRATION and calibrated_display is not None) else raw_display
            )

            display_transform = None
            if USE_CALIBRATION and FIT_CALIBRATED_DISPLAY_TO_CONTENT and calibrated_display is not None:
                display_transform = compute_non_black_fit_transform(
                    calibrated_display,
                    threshold=CALIBRATED_DISPLAY_THRESHOLD,
                    margin_px=CALIBRATED_DISPLAY_MARGIN_PX,
                )
                if not display_transform_checked:
                    max_err = validate_display_transform_math(
                        display_transform)
                    if max_err > 1e-4:
                        raise RuntimeError(
                            f"Display transform round-trip failed (max_err={max_err:.6f} px)"
                        )
                    print(
                        f"[TRACKER] Display transform sanity passed (max_err={max_err:.6f}px)")
                    display_transform_checked = True

            frame_metadata = dict(base_frame_metadata)
            frame_metadata.update(
                {
                    "pose_width": int(pose_w),
                    "pose_height": int(pose_h),
                    "arena_origin_u_px": float(u0),
                    "arena_origin_v_px": float(v0),
                    "arena_center_u": float(u0),
                    "arena_center_v": float(v0),
                }
            )
            if display_transform is not None:
                frame_metadata["display_transform"] = {
                    "type": "crop_resize",
                    "crop_x0": int(display_transform["crop_x0"]),
                    "crop_y0": int(display_transform["crop_y0"]),
                    "crop_w": int(display_transform["crop_w"]),
                    "crop_h": int(display_transform["crop_h"]),
                    "out_w": int(display_transform["out_w"]),
                    "out_h": int(display_transform["out_h"]),
                    "scale_x": float(display_transform["scale_x"]),
                    "scale_y": float(display_transform["scale_y"]),
                }

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray)

            target_measurement = None
            visible_ids = []
            fallback_measurement = None
            fallback_radius = -1.0
            bot_poses_payload = []

            if ids is not None:
                ids = ids.flatten()

                for c, marker_id in zip(corners, ids):
                    marker_id_int = int(marker_id)
                    marker_points = extract_marker_points(c)
                    raw_points = marker_points
                    raw_cx, raw_cy, raw_radius = center_and_radius_from_points(
                        raw_points)
                    draw_marker_overlay(
                        raw_display,
                        raw_points,
                        marker_id_int,
                        raw_cx,
                        raw_cy,
                        raw_radius,
                    )

                    warped_points = None
                    if calibrated_display is not None:
                        warped_points = warp_points(marker_points, calibration)
                        warped_cx, warped_cy, warped_radius = center_and_radius_from_points(
                            warped_points)
                        draw_marker_overlay(
                            calibrated_display,
                            warped_points,
                            marker_id_int,
                            warped_cx,
                            warped_cy,
                            warped_radius,
                        )

                    pose_points = (
                        warped_points if (
                            USE_CALIBRATION and warped_points is not None) else marker_points
                    )
                    cx, cy, radius = center_and_radius_from_points(pose_points)
                    arena_x_px, arena_y_px = uv_to_arena_px(cx, cy, u0, v0)
                    arena_x_cm, arena_y_cm = arena_px_to_cm(
                        arena_x_px, arena_y_px)
                    theta_rad = marker_theta_from_warp_corners(pose_points)

                    bot_pose = {
                        "marker_id": marker_id_int,
                        "x_cm": float(arena_x_cm),
                        "y_cm": float(arena_y_cm),
                        "x_arena_mm": float(arena_x_cm) * 10.0,
                        "y_arena_mm": float(arena_y_cm) * 10.0,
                        "theta_rad": float(theta_rad),
                        "x_px": float(arena_x_px),
                        "y_px": float(arena_y_px),
                        "u_pose_px": float(cx),
                        "v_pose_px": float(cy),
                        "tracking_state": "measured",
                        "confidence": 1.0,
                        "age_ms": 0,
                    }
                    if display_transform is not None:
                        u_disp, v_disp = warp_to_display_xy(
                            cx, cy, display_transform)
                        bot_pose["u_display_px"] = float(u_disp)
                        bot_pose["v_display_px"] = float(v_disp)

                    bot_poses_payload.append(bot_pose)
                    visible_ids.append(marker_id_int)

                    draw_heading_arrow(
                        active_display, cx, cy, theta_rad, length_px=max(16.0, radius * 1.25))
                    cv2.putText(
                        active_display,
                        f"id={marker_id_int} x={arena_x_cm:.1f}cm y={arena_y_cm:.1f}cm th={theta_rad:.2f}",
                        (int(cx) + 10, int(cy) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )

                    if marker_id_int == active_target_id:
                        target_measurement = (cx, cy, radius)
                    if radius > fallback_radius:
                        fallback_radius = radius
                        fallback_measurement = (
                            marker_id_int, (cx, cy, radius))

                if (
                    target_measurement is None
                    and AUTO_LOCK_IF_TARGET_MISSING
                    and not estimator.has_state()
                    and fallback_measurement is not None
                ):
                    active_target_id, target_measurement = fallback_measurement
                    autolock_message_until = now + 3.0
                    print(
                        f"[TRACKER] Auto-locked target marker id={active_target_id}")

            bot_poses_payload.sort(key=lambda m: int(m["marker_id"]))

            if target_measurement is not None:
                x, y, radius = estimator.update(target_measurement, now)
                tracking_state = "measured"
                confidence = 1.0
            else:
                predicted = estimator.predict(now)
                if predicted is None:
                    x = y = radius = None
                    tracking_state = "lost"
                    confidence = 0.0
                else:
                    x, y, radius = predicted
                    age = estimator.age_since_measurement(now)
                    if age <= PREDICT_HOLD_SECONDS:
                        tracking_state = "predicted"
                        confidence = max(
                            0.1, 1.0 - (age / PREDICT_HOLD_SECONDS))
                    else:
                        tracking_state = "lost"
                        confidence = 0.0

            age_seconds = estimator.age_since_measurement(now)
            age_ms = int(
                age_seconds * 1000) if age_seconds != float("inf") else -1

            legacy_marker_id = int(active_target_id)
            legacy_x = x
            legacy_y = y
            legacy_radius = radius
            legacy_tracking_state = tracking_state
            legacy_confidence = float(confidence)
            legacy_age_ms = int(age_ms)
            if (target_measurement is None and bot_poses_payload) or (legacy_x is None and bot_poses_payload):
                first = bot_poses_payload[0]
                legacy_marker_id = int(first["marker_id"])
                legacy_x = float(first["u_pose_px"])
                legacy_y = float(first["v_pose_px"])
                legacy_radius = 0.0
                legacy_tracking_state = "measured"
                legacy_confidence = 1.0
                legacy_age_ms = 0
            if legacy_x is None or legacy_y is None or legacy_radius is None:
                legacy_x = 0.0
                legacy_y = 0.0
                legacy_radius = 0.0
                legacy_tracking_state = "lost"
                legacy_confidence = 0.0
                legacy_age_ms = -1

            legacy_x_arena_px, legacy_y_arena_px = uv_to_arena_px(
                legacy_x, legacy_y, u0, v0)
            legacy_x_cm, legacy_y_cm = arena_px_to_cm(
                legacy_x_arena_px, legacy_y_arena_px)

            if x is not None and y is not None and radius is not None:
                color = (0, 255, 0) if tracking_state == "measured" else (
                    0, 255, 255)
                if tracking_state == "lost":
                    color = (0, 0, 255)

                cv2.circle(active_display, (int(x), int(y)), 6, color, -1)
                cv2.circle(active_display, (int(x), int(y)),
                           int(max(radius, 1.0)), color, 2)

                state_label = (
                    f"state={tracking_state} conf={confidence:.2f} age_ms={age_ms} "
                    f"x={legacy_x_cm:.1f}cm y={legacy_y_cm:.1f}cm"
                )
                cv2.putText(
                    active_display,
                    state_label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            else:
                cv2.putText(
                    active_display,
                    f"state=lost target_id={active_target_id} visible_ids={visible_ids}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                active_display,
                f"visible_markers={len(bot_poses_payload)} ids={visible_ids}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if now < autolock_message_until:
                cv2.putText(
                    active_display,
                    f"AUTO LOCKED marker id={active_target_id}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            should_send = False
            if bot_poses_payload:
                should_send = (now - last_send_time) >= SEND_INTERVAL
            elif x is not None and y is not None and radius is not None:
                if tracking_state in ("measured", "predicted"):
                    should_send = (now - last_send_time) >= SEND_INTERVAL
                else:
                    should_send = (
                        now - last_lost_report_time) >= LOST_REPORT_INTERVAL

            if should_send:
                packet = {
                    "type": "marker_data",
                    "seq": seq,
                    "timestamp": now,
                    "marker_id": int(legacy_marker_id),
                    "x": float(legacy_x_cm),
                    "y": float(legacy_y_cm),
                    "radius": 0.0,
                    "x_cm": float(legacy_x_cm),
                    "y_cm": float(legacy_y_cm),
                    "x_arena_mm": float(legacy_x_cm) * 10.0,
                    "y_arena_mm": float(legacy_y_cm) * 10.0,
                    "x_px": float(legacy_x_arena_px),
                    "y_px": float(legacy_y_arena_px),
                    "u_pose_px": float(legacy_x),
                    "v_pose_px": float(legacy_y),
                    "tracking_state": legacy_tracking_state,
                    "confidence": float(legacy_confidence),
                    "age_ms": int(legacy_age_ms),
                    **frame_metadata
                }
                if SEND_ALL_VISIBLE_MARKERS:
                    packet["bot_poses"] = bot_poses_payload
                    packet["markers"] = bot_poses_payload
                client.send_packet(packet)

                seq += 1
                last_send_time = now
                if tracking_state == "lost":
                    last_lost_report_time = now

            try:
                client.drain_packets(max_packets=10)
            except ConnectionError:
                print("[TRACKER] Connection closed by receiver")
                break

            if now - last_print_time >= PRINT_INTERVAL:
                if x is None:
                    print("[TRACKER] state=lost (no pose yet)")
                else:
                    print(
                        f"[TRACKER] state={tracking_state} conf={confidence:.2f} age_ms={age_ms} "
                        f"x_cm={legacy_x_cm:.2f} y_cm={legacy_y_cm:.2f}"
                    )
                last_print_time = now

            if SHOW_BOTH_FEEDS and calibrated_display is not None:
                calibrated_preview = calibrated_display
                if display_transform is not None:
                    calibrated_preview = apply_fit_transform_for_display(
                        calibrated_display, display_transform)
                cv2.imshow(WINDOW_NAME_RAW, raw_display)
                cv2.imshow(WINDOW_NAME_CALIBRATED, calibrated_preview)
            else:
                single_view = active_display
                if USE_CALIBRATION and calibrated_display is not None and display_transform is not None:
                    single_view = apply_fit_transform_for_display(
                        calibrated_display, display_transform)
                cv2.imshow(WINDOW_NAME, single_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.close()


if __name__ == "__main__":
    main()
