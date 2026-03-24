#!/usr/bin/env python3
"""
Quick webcam ArUco tracker:
- Detects marker IDs
- Shows marker front direction (toward top edge TL->TR midpoint)
- Prints orientation in radians/degrees

Angle convention:
- theta = 0 points up in the image
- theta increases anticlockwise
- theta range is [0, 2pi)
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.camera_geometry import marker_theta_from_warp_corners


def get_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    def detect(gray):
        return aruco.detectMarkers(gray, dictionary, parameters=parameters)

    return detect


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick ArUco ID + front-orientation viewer")
    parser.add_argument("--camera-index", type=int,
                        default=0, help="Camera index (default: 0)")
    parser.add_argument("--target-id", type=int, default=None,
                        help="Only report this marker id")
    parser.add_argument("--print-interval", type=float,
                        default=0.2, help="Seconds between console prints")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    detect_markers = get_detector()
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray)

            readings = []
            if ids is not None:
                ids = ids.flatten()
                for marker_corners, marker_id_raw in zip(corners, ids):
                    marker_id = int(marker_id_raw)
                    if args.target_id is not None and marker_id != args.target_id:
                        continue

                    pts = marker_corners.reshape(4, 2).astype(np.float32)
                    center = np.mean(pts, axis=0)
                    top_mid = 0.5 * (pts[0] + pts[1])

                    theta_rad = float(marker_theta_from_warp_corners(pts))
                    theta_deg = float(np.degrees(theta_rad))

                    readings.append((marker_id, theta_rad, theta_deg))

                    cv2.polylines(frame, [pts.astype(int)],
                                  True, (0, 255, 0), 2)
                    cv2.circle(frame, tuple(center.astype(int)),
                               4, (0, 0, 255), -1)
                    cv2.arrowedLine(
                        frame,
                        tuple(center.astype(int)),
                        tuple(top_mid.astype(int)),
                        (0, 255, 255),
                        2,
                        tipLength=0.25,
                    )
                    cv2.putText(
                        frame,
                        f"id={marker_id} th={theta_rad:.3f} rad ({theta_deg:.1f} deg)",
                        (int(center[0]) + 10, int(center[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

            now = time.time()
            if now - last_print >= args.print_interval:
                if readings:
                    for marker_id, theta_rad, theta_deg in readings:
                        print(
                            f"[MARKER] id={marker_id} theta_rad={theta_rad:.6f} theta_deg={theta_deg:.2f}")
                else:
                    if args.target_id is None:
                        print("[MARKER] none detected")
                    else:
                        print(
                            f"[MARKER] target id={args.target_id} not detected")
                last_print = now

            cv2.imshow("Quick Marker Orientation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
