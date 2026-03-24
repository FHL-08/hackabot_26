import cv2
import math
import numpy as np
import socket
import json
import time

from src.camera_geometry import (
    CalibrationError,
    apply_calibration_to_frame,
    coord_frame_metadata,
    load_arena_calibration,
)

CAMERA_INDEX = 0
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
SEND_INTERVAL = 0.20

WINDOW_FRAME = "Colour Cluster Detector"

ENABLE_NETWORK = True

MIN_AREA = 500

OPEN_KERNEL_SIZE = 5
CLOSE_KERNEL_SIZE = 9

BLUR_SIZE = 5
RADIUS_MODE = "equivalent_area"

COLOUR_CONFIGS = [
    {
        "name": "red",
        "ranges": [
            ((7, 110, 170), (13, 155, 225)),
        ],
        "draw_colour": (0, 0, 255),
    },
]


class TcpJsonClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=3)
        self.sock.settimeout(3)

    def send_packet(self, packet):
        message = json.dumps(packet) + "\n"
        self.sock.sendall(message.encode("utf-8"))

    def recv_packet(self):
        while "\n" not in self.buffer:
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("Connection closed by receiver")
            self.buffer += data.decode("utf-8")

        line, self.buffer = self.buffer.split("\n", 1)
        return json.loads(line)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


def build_colour_mask(hsv_frame, colour_ranges):
    full_mask = None

    for lower, upper in colour_ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower_np, upper_np)

        if full_mask is None:
            full_mask = mask
        else:
            full_mask = cv2.bitwise_or(full_mask, mask)

    return full_mask


def clean_mask(mask):
    open_kernel = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
    close_kernel = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return cleaned


def find_clusters_from_mask(mask, colour_name):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    clusters = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue

        cx, cy = centroids[label_id]

        radius = float(math.sqrt(float(area) / math.pi))

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        clusters.append({
            "colour_name": colour_name,
            "x": float(cx),
            "y": float(cy),
            "radius": radius,
            "radius_mode": RADIUS_MODE,
            "area": float(area),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
        })

    return clusters


def main():
    try:
        calibration = load_arena_calibration()
    except CalibrationError as e:
        raise RuntimeError(
            f"{e}\n"
            f"Run calibration first:\n"
            f"python .\\calibrate_arena_camera.py --camera-index {CAMERA_INDEX}"
        )

    frame_metadata = coord_frame_metadata(calibration)

    client = None

    if ENABLE_NETWORK:
        client = TcpJsonClient(SERVER_IP, SERVER_PORT)
        client.connect()

        hello_packet = {
            "type": "hello",
            "message": "colour_cluster_detector_connected"
        }
        client.send_packet(hello_packet)
        hello_ack = client.recv_packet()
        print("[COLOUR] Handshake ACK:", hello_ack)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    last_send_time = 0.0
    seq = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[COLOUR] Failed to read frame")
            break

        corrected_frame = apply_calibration_to_frame(frame, calibration)

        if BLUR_SIZE > 1:
            blurred = cv2.GaussianBlur(
                corrected_frame, (BLUR_SIZE, BLUR_SIZE), 0)
        else:
            blurred = corrected_frame

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        display = corrected_frame.copy()
        all_clusters = []

        for colour_cfg in COLOUR_CONFIGS:
            colour_name = colour_cfg["name"]
            draw_colour = colour_cfg["draw_colour"]
            colour_ranges = colour_cfg["ranges"]

            raw_mask = build_colour_mask(hsv, colour_ranges)
            mask = clean_mask(raw_mask)

            clusters = find_clusters_from_mask(mask, colour_name)

            for cluster in clusters:
                cx = cluster["x"]
                cy = cluster["y"]
                radius = cluster["radius"]
                area = cluster["area"]

                cv2.circle(display, (int(cx), int(cy)),
                           int(radius), draw_colour, 2)
                cv2.circle(display, (int(cx), int(cy)), 4, (255, 255, 255), -1)

                label = f"{colour_name} x={cx:.1f} y={cy:.1f} r={radius:.1f} a={area:.0f}"
                cv2.putText(
                    display,
                    label,
                    (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                all_clusters.append({
                    "obstacle_id": len(all_clusters),
                    "colour": colour_name,
                    "x": cx,
                    "y": cy,
                    "radius": radius,
                    "radius_mode": RADIUS_MODE,
                    "area": area,
                })

        cv2.putText(
            display,
            f"Detected colour clusters: {len(all_clusters)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "q = quit",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        now = time.time()
        if ENABLE_NETWORK and now - last_send_time >= SEND_INTERVAL:
            packet = {
                "type": "obstacle_data",
                "seq": seq,
                "timestamp": now,
                "obstacles": all_clusters,
                "obstacle_radius_mode": RADIUS_MODE,
                **frame_metadata,
            }

            try:
                client.send_packet(packet)
                ack = client.recv_packet()
                print(
                    f"[COLOUR] Sent seq={seq} count={len(all_clusters)} | ACK={ack}")
            except Exception as e:
                print(f"[COLOUR] Lost connection to receiver: {e}")
                break

            seq += 1
            last_send_time = now

        cv2.imshow(WINDOW_FRAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
