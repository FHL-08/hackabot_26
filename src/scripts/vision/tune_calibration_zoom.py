import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.camera_geometry import DEFAULT_CALIBRATION_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune calibration JSON for wider (zoomed-out) undistorted view."
    )
    parser.add_argument(
        "--input", default=DEFAULT_CALIBRATION_PATH, help="Input calibration JSON")
    parser.add_argument(
        "--output", default=DEFAULT_CALIBRATION_PATH, help="Output calibration JSON")
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=0.8,
        help="Multiply fx/fy by this factor (<1 zooms out, >1 zooms in).",
    )
    parser.add_argument(
        "--recenter",
        action="store_true",
        help="Recenter principal point to image center after zoom tuning.",
    )
    parser.add_argument(
        "--arena-scale",
        type=float,
        default=1.0,
        help=(
            "Scale arena_corners_px around their center before rebuilding homography. "
            ">1.0 zooms OUT in warped top-down view, <1.0 zooms IN."
        ),
    )
    return parser.parse_args()


def load_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_payload(path, payload):
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    payload = load_payload(in_path)
    k = np.asarray(payload["camera_matrix"], dtype=np.float64)

    image_w, image_h = payload["image_size"]
    zoom = float(args.zoom_factor)
    arena_scale = float(args.arena_scale)

    if zoom <= 0.0:
        raise RuntimeError("zoom-factor must be > 0")
    if arena_scale <= 0.0:
        raise RuntimeError("arena-scale must be > 0")

    fx_old, fy_old = k[0, 0], k[1, 1]
    cx_old, cy_old = k[0, 2], k[1, 2]

    fx_new = fx_old * zoom
    fy_new = fy_old * zoom

    if args.recenter:
        cx_new = (float(image_w) - 1.0) * 0.5
        cy_new = (float(image_h) - 1.0) * 0.5
    else:
        cx_new = cx_old
        cy_new = cy_old

    k_new = np.array(
        [[fx_new, 0.0, cx_new], [0.0, fy_new, cy_new], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Keep arena mapping self-consistent by scaling saved corner points into the tuned camera space.
    corners = np.asarray(payload.get("arena_corners_px", []), dtype=np.float32)
    if corners.shape == (4, 2):
        sx = fx_new / fx_old
        sy = fy_new / fy_old
        corners_new = corners.copy()
        corners_new[:, 0] = (corners[:, 0] - cx_old) * sx + cx_new
        corners_new[:, 1] = (corners[:, 1] - cy_old) * sy + cy_new

        # Optional user-controlled top-down zoom for the warped output.
        # Expanding corners means "capture a larger area" into the same warp_size.
        if arena_scale != 1.0:
            arena_center = corners_new.mean(axis=0)
            corners_new = (corners_new - arena_center) * \
                arena_scale + arena_center

        warp_w, warp_h = payload["warp_size"]
        dst = np.array(
            [[0.0, 0.0], [warp_w - 1.0, 0.0], [warp_w -
                                               1.0, warp_h - 1.0], [0.0, warp_h - 1.0]],
            dtype=np.float32,
        )
        h_new = cv2.getPerspectiveTransform(
            corners_new.astype(np.float32), dst)

        payload["arena_corners_px"] = corners_new.tolist()
        payload["homography"] = h_new.tolist()

    payload["camera_matrix"] = k_new.tolist()
    payload["zoom_tuned_from"] = str(in_path)
    payload["zoom_factor"] = zoom
    payload["arena_zoom_scale"] = arena_scale

    save_payload(out_path, payload)
    print(f"[ZOOM] Saved tuned calibration: {out_path}")
    print(
        "[ZOOM] fx/fy "
        f"{fx_old:.2f}/{fy_old:.2f} -> {fx_new:.2f}/{fy_new:.2f} "
        f"(zoom_factor={zoom:.3f})"
    )
    if arena_scale != 1.0:
        print(
            f"[ZOOM] arena_corners_px scaled by {arena_scale:.3f} for warped-view zoom")


if __name__ == "__main__":
    main()
