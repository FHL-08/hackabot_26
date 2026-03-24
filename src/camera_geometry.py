import json
import math
import os

import cv2
import numpy as np


DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__),
    "generated",
    "calibration",
    "arena_calibration.json",
)

FALLBACK_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "arena_calibration.json",
)


class CalibrationError(RuntimeError):
    pass


def _validate_matrix(name, value, shape):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise CalibrationError(
            f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def load_arena_calibration(path=DEFAULT_CALIBRATION_PATH):
    resolved_path = path
    if (
        path == DEFAULT_CALIBRATION_PATH
        and not os.path.exists(path)
        and os.path.exists(FALLBACK_CALIBRATION_PATH)
    ):
        resolved_path = FALLBACK_CALIBRATION_PATH

    if not os.path.exists(resolved_path):
        raise CalibrationError(f"Calibration file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    required = ("camera_matrix", "dist_coeffs",
                "homography", "warp_size", "image_size")
    missing = [k for k in required if k not in payload]
    if missing:
        raise CalibrationError(
            f"Calibration file missing keys: {', '.join(missing)}")

    camera_matrix = _validate_matrix(
        "camera_matrix", payload["camera_matrix"], (3, 3))
    homography = _validate_matrix("homography", payload["homography"], (3, 3))
    dist_coeffs = np.asarray(
        payload["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    warp_size = payload["warp_size"]
    if not isinstance(warp_size, list) or len(warp_size) != 2:
        raise CalibrationError("warp_size must be [width, height]")
    warp_w, warp_h = int(warp_size[0]), int(warp_size[1])
    if warp_w <= 0 or warp_h <= 0:
        raise CalibrationError("warp_size values must be positive")

    image_size = payload["image_size"]
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise CalibrationError("image_size must be [width, height]")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "homography": homography,
        "warp_size": (warp_w, warp_h),
        "image_size": (int(image_size[0]), int(image_size[1])),
        "raw": payload,
    }


def undistort_frame(frame, calibration):
    return cv2.undistort(frame, calibration["camera_matrix"], calibration["dist_coeffs"])


def warp_frame(undistorted_frame, calibration):
    warp_w, warp_h = calibration["warp_size"]
    return cv2.warpPerspective(
        undistorted_frame,
        calibration["homography"],
        (warp_w, warp_h),
        flags=cv2.INTER_LINEAR,
    )


def apply_calibration_to_frame(frame, calibration):
    undistorted = undistort_frame(frame, calibration)
    return warp_frame(undistorted, calibration)


def warp_points(points_xy, calibration):
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(
        points,
        calibration["camera_matrix"],
        calibration["dist_coeffs"],
        P=calibration["camera_matrix"],
    )
    warped_points = cv2.perspectiveTransform(
        undistorted_points, calibration["homography"])
    return warped_points.reshape(-1, 2)


def center_and_radius_from_points(points_xy):
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    center = pts.mean(axis=0)
    distances = np.sqrt(np.sum((pts - center) ** 2, axis=1))
    radius = float(np.mean(distances)) if len(distances) else 0.0
    return float(center[0]), float(center[1]), radius


def coord_frame_metadata(calibration):
    warp_w, warp_h = calibration["warp_size"]
    return {
        "coord_frame": "arena_topdown_px",
        "warp_width": int(warp_w),
        "warp_height": int(warp_h),
    }


def warp_center(calibration):
    warp_w, warp_h = calibration["warp_size"]
    # Pixel-center convention keeps symmetry for even/odd dimensions.
    u0 = (float(warp_w) - 1.0) * 0.5
    v0 = (float(warp_h) - 1.0) * 0.5
    return u0, v0


def warp_xy_to_arena_xy(u, v, calibration):
    """
    Convert warped-image pixel coordinates (u right+, v down+) to arena coordinates:
      +x up, +y left, origin at warped frame center.
    """
    u0, v0 = warp_center(calibration)
    arena_x = -(float(v) - v0)
    arena_y = -(float(u) - u0)
    return arena_x, arena_y


def warp_points_to_arena(points_uv, calibration):
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    u0, v0 = warp_center(calibration)
    arena = np.empty_like(pts, dtype=np.float64)
    arena[:, 0] = -(pts[:, 1] - v0)  # x
    arena[:, 1] = -(pts[:, 0] - u0)  # y
    return arena


def arena_xy_to_warp_xy(arena_x, arena_y, calibration):
    u0, v0 = warp_center(calibration)
    u = u0 - float(arena_y)
    v = v0 - float(arena_x)
    return u, v


def arena_units_from_px(arena_x_px, arena_y_px, mm_per_px):
    scale = float(mm_per_px)
    return float(arena_x_px) * scale, float(arena_y_px) * scale


def normalize_angle_0_2pi(theta_rad):
    two_pi = 2.0 * math.pi
    return float(theta_rad % two_pi)


def marker_theta_from_warp_corners(marker_corners_warp):
    """
    Compute marker heading in arena frame.
    Zero is +x (up in image), increases anticlockwise, range [0, 2pi).
    """
    pts = np.asarray(marker_corners_warp, dtype=np.float64).reshape(4, 2)

    center_u = float(np.mean(pts[:, 0]))
    center_v = float(np.mean(pts[:, 1]))

    # ArUco corners are ordered TL, TR, BR, BL in marker coordinates.
    top_mid_u = float(0.5 * (pts[0, 0] + pts[1, 0]))
    top_mid_v = float(0.5 * (pts[0, 1] + pts[1, 1]))

    du = top_mid_u - center_u  # +right
    dv = top_mid_v - center_v  # +down

    hx = -dv  # arena +x is image up
    hy = -du  # arena +y is image left

    theta = math.atan2(hy, hx)
    return normalize_angle_0_2pi(theta)
