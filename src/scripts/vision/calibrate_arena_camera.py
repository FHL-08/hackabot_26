import argparse
import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from src.camera_geometry import DEFAULT_CALIBRATION_PATH


WINDOW_CHESSBOARD = "Chessboard Calibration"
WINDOW_CORNERS = "Arena Corner Calibration"
WINDOW_WARPED = "Arena Warped Preview"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate camera distortion and arena perspective warp."
    )
    parser.add_argument("--camera-index", type=int,
                        default=0, help="OpenCV camera index")
    parser.add_argument("--board-cols", type=int, default=9,
                        help="Chessboard inner corners across")
    parser.add_argument("--board-rows", type=int, default=6,
                        help="Chessboard inner corners down")
    parser.add_argument("--square-size-mm", type=float,
                        default=25.0, help="Chessboard square size in mm")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of chessboard captures")
    parser.add_argument("--warp-width", type=int, default=1200,
                        help="Top-down arena width in pixels")
    parser.add_argument("--warp-height", type=int, default=800,
                        help="Top-down arena height in pixels")
    parser.add_argument(
        "--warn-reproj-error",
        type=float,
        default=2.5,
        help="Warn when reprojection error exceeds this value",
    )
    parser.add_argument(
        "--max-reproj-error",
        type=float,
        default=4.0,
        help="Fail calibration when reprojection error exceeds this value",
    )
    parser.add_argument(
        "--allow-high-error",
        action="store_true",
        help="Allow saving calibration even if reprojection error is high",
    )
    parser.add_argument("--output", default=DEFAULT_CALIBRATION_PATH,
                        help="Calibration JSON output path")
    return parser.parse_args()


def build_object_points(board_cols, board_rows, square_size_mm):
    obj = np.zeros((board_rows * board_cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    obj *= square_size_mm
    return obj


def collect_chessboard_samples(cap, board_size, square_size_mm, target_samples):
    board_cols, board_rows = board_size
    obj_template = build_object_points(board_cols, board_rows, square_size_mm)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    object_points = []
    image_points = []
    image_size = None

    print("[CALIB] Chessboard stage")
    print("[CALIB] Hold pattern steady, press SPACE when corners are detected.")
    print("[CALIB] Press q to cancel.")

    while len(object_points) < target_samples:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, board_size, None)

        preview = frame.copy()
        if found:
            refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(preview, board_size, refined, found)
        else:
            refined = None

        cv2.putText(
            preview,
            f"Samples: {len(object_points)}/{target_samples}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            "SPACE capture | q cancel",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_CHESSBOARD, preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise RuntimeError("Calibration cancelled by user")

        if key == ord(" ") and found and refined is not None:
            object_points.append(obj_template.copy())
            image_points.append(refined)
            print(
                f"[CALIB] Captured sample {len(object_points)}/{target_samples}")

    cv2.destroyWindow(WINDOW_CHESSBOARD)
    return object_points, image_points, image_size


def calibrate_intrinsics(object_points, image_points, image_size):
    reprojection_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    if camera_matrix is None or dist_coeffs is None:
        raise RuntimeError("Camera calibration failed")

    print(f"[CALIB] Reprojection error: {reprojection_error:.4f}")
    return reprojection_error, camera_matrix, dist_coeffs


def collect_arena_corners(cap, camera_matrix, dist_coeffs, warp_size):
    warp_w, warp_h = warp_size
    clicked = []
    homography = None
    undistorted_size = None

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((float(x), float(y)))
            print(f"[CALIB] Corner {len(clicked)} = ({x}, {y})")

    cv2.namedWindow(WINDOW_CORNERS, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_CORNERS, on_mouse)
    cv2.namedWindow(WINDOW_WARPED, cv2.WINDOW_NORMAL)

    print("[CALIB] Arena corner stage")
    print("[CALIB] Click corners in order: top-left, top-right, bottom-right, bottom-left")
    print("[CALIB] Press r to reset points, s to save once 4 points are set, q to cancel.")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")

        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        undistorted_size = (undistorted.shape[1], undistorted.shape[0])
        preview = undistorted.copy()

        for idx, (x, y) in enumerate(clicked, start=1):
            point = (int(x), int(y))
            cv2.circle(preview, point, 6, (0, 255, 0), -1)
            cv2.putText(
                preview,
                str(idx),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            preview,
            "Order: TL, TR, BR, BL | r reset | s save | q cancel",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if len(clicked) == 4:
            src = np.array(clicked, dtype=np.float32)
            dst = np.array(
                [
                    [0.0, 0.0],
                    [warp_w - 1.0, 0.0],
                    [warp_w - 1.0, warp_h - 1.0],
                    [0.0, warp_h - 1.0],
                ],
                dtype=np.float32,
            )
            homography = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(
                undistorted, homography, (warp_w, warp_h))
            cv2.putText(
                preview,
                "4 corners set. Press s to save.",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW_WARPED, warped)
        else:
            blank = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Set 4 corners to preview warp",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_WARPED, blank)

        cv2.imshow(WINDOW_CORNERS, preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            raise RuntimeError("Calibration cancelled by user")
        if key == ord("r"):
            clicked.clear()
            homography = None
        if key == ord("s") and len(clicked) == 4 and homography is not None:
            cv2.destroyWindow(WINDOW_CORNERS)
            cv2.destroyWindow(WINDOW_WARPED)
            return homography, clicked, undistorted_size


def save_calibration(
    path,
    reprojection_error,
    camera_matrix,
    dist_coeffs,
    homography,
    image_size,
    warp_size,
    board_cols,
    board_rows,
    square_size_mm,
    arena_corners,
):
    output_dir = Path(path).expanduser().resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "homography": homography.tolist(),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "warp_size": [int(warp_size[0]), int(warp_size[1])],
        "board": {
            "cols": int(board_cols),
            "rows": int(board_rows),
            "square_size_mm": float(square_size_mm),
        },
        "reprojection_error": float(reprojection_error),
        "corner_order": "top_left, top_right, bottom_right, bottom_left",
        "arena_corners_px": [[float(x), float(y)] for x, y in arena_corners],
        "created_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[CALIB] Saved calibration to: {path}")


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    try:
        board_size = (args.board_cols, args.board_rows)
        object_points, image_points, image_size = collect_chessboard_samples(
            cap,
            board_size=board_size,
            square_size_mm=args.square_size_mm,
            target_samples=args.samples,
        )

        reprojection_error, camera_matrix, dist_coeffs = calibrate_intrinsics(
            object_points,
            image_points,
            image_size=image_size,
        )

        if reprojection_error > args.warn_reproj_error:
            print(
                f"[CALIB] Warning: reprojection error {reprojection_error:.4f} "
                f"exceeds warning threshold {args.warn_reproj_error:.2f}"
            )

        if reprojection_error > args.max_reproj_error and not args.allow_high_error:
            raise RuntimeError(
                f"Reprojection error {reprojection_error:.4f} exceeds max threshold "
                f"{args.max_reproj_error:.2f}. Re-capture samples or rerun with "
                "--allow-high-error."
            )

        homography, arena_corners, image_size = collect_arena_corners(
            cap,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            warp_size=(args.warp_width, args.warp_height),
        )

        save_calibration(
            path=args.output,
            reprojection_error=reprojection_error,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            homography=homography,
            image_size=image_size,
            warp_size=(args.warp_width, args.warp_height),
            board_cols=args.board_cols,
            board_rows=args.board_rows,
            square_size_mm=args.square_size_mm,
            arena_corners=arena_corners,
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
