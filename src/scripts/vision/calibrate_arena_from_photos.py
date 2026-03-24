import argparse
import datetime
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera_geometry import DEFAULT_CALIBRATION_PATH


WINDOW_CAPTURE = "Calibration Photo Capture"
WINDOW_CORNERS = "Arena Corner Selection (Photo)"
WINDOW_WARPED = "Arena Warped Preview"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture calibration photos, then calibrate from images in a folder."
    )
    parser.add_argument("--camera-index", type=int,
                        default=0, help="OpenCV camera index")
    parser.add_argument(
        "--image-dir",
        default="src/generated/calibration_photos",
        help="Folder used to save/load calibration photos",
    )
    parser.add_argument(
        "--filename-prefix",
        default="chessboard_",
        help="Only use images whose filename starts with this prefix (empty = all)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Use at most this many images for calibration (0 = all)",
    )
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help="When --max-images is set, use the newest images instead of oldest",
    )
    parser.add_argument("--board-cols", type=int, default=11,
                        help="Chessboard inner corners across")
    parser.add_argument("--board-rows", type=int, default=7,
                        help="Chessboard inner corners down")
    parser.add_argument("--square-size-mm", type=float,
                        default=25.0, help="Chessboard square size in mm")
    parser.add_argument(
        "--target-captures",
        type=int,
        default=50,
        help="Capture stage target count for chessboard photos",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.1,
        help="Seconds between automatic photo captures",
    )
    parser.add_argument(
        "--min-valid-samples",
        type=int,
        default=8,
        help="Minimum valid chessboard photos required for calibration",
    )
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
    parser.add_argument(
        "--arena-image",
        default="arena_reference.png",
        help="Arena photo filename (relative to image-dir) or absolute path for corner clicking",
    )
    parser.add_argument(
        "--reuse-homography-from",
        default=DEFAULT_CALIBRATION_PATH,
        help="Fallback calibration JSON to reuse homography if arena image is unavailable",
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=1.0,
        help=(
            "Undistortion alpha in [0,1]. 1.0 keeps max field of view (less zoom), "
            "0.0 crops more."
        ),
    )
    parser.add_argument("--capture-only", action="store_true",
                        help="Only capture photos, do not calibrate")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only calibrate from existing photos")
    parser.add_argument("--output", default=DEFAULT_CALIBRATION_PATH,
                        help="Calibration JSON output path")
    return parser.parse_args()


def build_object_points(board_cols, board_rows, square_size_mm):
    obj = np.zeros((board_rows * board_cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    obj *= square_size_mm
    return obj


def list_image_paths(image_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(image_dir.glob(ext))
    paths.sort()
    return paths


def select_image_paths(paths, filename_prefix="", max_images=0, use_latest=False):
    selected = paths
    if filename_prefix:
        selected = [p for p in selected if p.name.startswith(filename_prefix)]

    if max_images and max_images > 0 and len(selected) > max_images:
        if use_latest:
            selected = selected[-max_images:]
        else:
            selected = selected[:max_images]

    return selected


def capture_photos(image_dir, camera_index, board_size, target_captures, capture_interval):
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    board_cols, board_rows = board_size
    capture_count = 0

    print("[CAPTURE] Photo stage (automatic)")
    print(f"[CAPTURE] Auto-capturing every {capture_interval:.2f} seconds")
    print("[CAPTURE] A: save arena reference image")
    print("[CAPTURE] Q: finish capture stage")

    last_capture_ts = 0.0
    latest_frame = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")
            latest_frame = frame

            preview = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (board_cols, board_rows), None)
            if found:
                cv2.drawChessboardCorners(
                    preview, (board_cols, board_rows), corners, found)

            cv2.putText(
                preview,
                f"Saved chessboard photos: {capture_count}/{target_captures}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                f"Corners detected: {'yes' if found else 'no'}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                "AUTO save chessboard | A save arena | Q done",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            now_ts = time.monotonic()
            if (now_ts - last_capture_ts) >= max(0.01, float(capture_interval)):
                ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                path = image_dir / f"chessboard_{ts}.png"
                cv2.imwrite(str(path), frame)
                capture_count += 1
                last_capture_ts = now_ts
                print(
                    f"[CAPTURE] Saved: {path.name} | corners_detected={found}")

            cv2.imshow(WINDOW_CAPTURE, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("a"):
                path = image_dir / "arena_reference.png"
                cv2.imwrite(str(path), frame)
                print(f"[CAPTURE] Saved arena reference: {path.name}")

            if capture_count >= target_captures:
                print("[CAPTURE] Target capture count reached.")
                break

        # Ensure we always have an arena reference image after capture.
        arena_ref = image_dir / "arena_reference.png"
        if latest_frame is not None and not arena_ref.exists():
            cv2.imwrite(str(arena_ref), latest_frame)
            print(f"[CAPTURE] Auto-saved arena reference: {arena_ref.name}")
    finally:
        cap.release()
        cv2.destroyWindow(WINDOW_CAPTURE)


def load_chessboard_samples_from_folder(
    image_dir,
    board_size,
    square_size_mm,
    filename_prefix="",
    max_images=0,
    use_latest=False,
):
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

    image_paths = list_image_paths(image_dir)
    image_paths = select_image_paths(
        image_paths,
        filename_prefix=filename_prefix,
        max_images=max_images,
        use_latest=use_latest,
    )
    if not image_paths:
        raise RuntimeError(f"No images found in folder: {image_dir}")

    used = 0
    skipped = 0
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[CALIB] Skip unreadable image: {path.name}")
            skipped += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(
                f"[CALIB] Skip size mismatch: {path.name} size={current_size} expected={image_size}")
            skipped += 1
            continue

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not found:
            skipped += 1
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(obj_template.copy())
        image_points.append(refined)
        used += 1

    print(f"[CALIB] Candidate photos considered: {len(image_paths)}")
    print(f"[CALIB] Chessboard photos used: {used}")
    print(f"[CALIB] Photos skipped: {skipped}")

    if used == 0 or image_size is None:
        raise RuntimeError(
            "No valid chessboard detections found in folder photos")

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


def build_output_camera_matrix(camera_matrix, dist_coeffs, image_size, alpha):
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha_clamped,
        image_size,
    )
    print(f"[CALIB] Undistort alpha: {alpha_clamped:.2f}")
    return new_camera_matrix


def resolve_arena_image_path(image_dir, arena_image):
    arena_path = Path(arena_image)
    if not arena_path.is_absolute():
        arena_path = image_dir / arena_image

    if arena_path.exists():
        return arena_path

    return None


def load_homography_from_file(path):
    source = Path(path)
    if not source.exists():
        raise RuntimeError(f"Homography fallback file not found: {source}")

    with open(source, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "homography" not in payload:
        raise RuntimeError(f"Fallback file missing homography: {source}")

    homography = np.asarray(payload["homography"], dtype=np.float32)
    if homography.shape != (3, 3):
        raise RuntimeError(
            f"Invalid homography shape in {source}: {homography.shape}")

    arena_corners = payload.get("arena_corners_px")
    if not isinstance(arena_corners, list) or len(arena_corners) != 4:
        arena_corners = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]

    print(f"[CALIB] Reusing homography from: {source}")
    return homography, arena_corners


def collect_arena_corners_from_image(
    image_path,
    camera_matrix,
    dist_coeffs,
    output_camera_matrix,
    warp_size,
):
    warp_w, warp_h = warp_size
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read arena image: {image_path}")

    undistorted = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, output_camera_matrix)
    image_size = (undistorted.shape[1], undistorted.shape[0])
    clicked = []
    homography = None

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((float(x), float(y)))
            print(f"[CALIB] Corner {len(clicked)} = ({x}, {y})")

    cv2.namedWindow(WINDOW_CORNERS, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_CORNERS, on_mouse)
    cv2.namedWindow(WINDOW_WARPED, cv2.WINDOW_NORMAL)

    print(f"[CALIB] Arena corner image: {image_path}")
    print("[CALIB] Click corners in order: top-left, top-right, bottom-right, bottom-left")
    print("[CALIB] Press r to reset points, s to save, q to cancel.")

    try:
        while True:
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
                return homography, clicked, image_size
    finally:
        cv2.destroyWindow(WINDOW_CORNERS)
        cv2.destroyWindow(WINDOW_WARPED)


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
    source_folder,
    undistort_alpha,
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
        "undistort_alpha": float(undistort_alpha),
        "corner_order": "top_left, top_right, bottom_right, bottom_left",
        "arena_corners_px": [[float(x), float(y)] for x, y in arena_corners],
        "source_photo_folder": str(source_folder),
        "created_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[CALIB] Saved calibration to: {path}")


def main():
    args = parse_args()
    if args.capture_only and args.calibrate_only:
        raise RuntimeError(
            "Use only one of --capture-only or --calibrate-only")

    image_dir = Path(args.image_dir)
    board_size = (args.board_cols, args.board_rows)

    if not args.calibrate_only:
        capture_photos(
            image_dir=image_dir,
            camera_index=args.camera_index,
            board_size=board_size,
            target_captures=args.target_captures,
            capture_interval=args.capture_interval,
        )

    if args.capture_only:
        print("[CALIB] Capture stage complete. Calibration not run (--capture-only).")
        return

    object_points, image_points, image_size = load_chessboard_samples_from_folder(
        image_dir=image_dir,
        board_size=board_size,
        square_size_mm=args.square_size_mm,
        filename_prefix=args.filename_prefix,
        max_images=args.max_images,
        use_latest=args.use_latest,
    )

    if len(object_points) < args.min_valid_samples:
        raise RuntimeError(
            f"Only {len(object_points)} valid chessboard photos found, need at least "
            f"{args.min_valid_samples}. Capture more photos and rerun."
        )

    reprojection_error, camera_matrix, dist_coeffs = calibrate_intrinsics(
        object_points=object_points,
        image_points=image_points,
        image_size=image_size,
    )
    output_camera_matrix = build_output_camera_matrix(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        alpha=args.undistort_alpha,
    )

    if reprojection_error > args.warn_reproj_error:
        print(
            f"[CALIB] Warning: reprojection error {reprojection_error:.4f} exceeds "
            f"warning threshold {args.warn_reproj_error:.2f}"
        )

    if reprojection_error > args.max_reproj_error and not args.allow_high_error:
        raise RuntimeError(
            f"Reprojection error {reprojection_error:.4f} exceeds max threshold "
            f"{args.max_reproj_error:.2f}. Capture better photos or rerun with "
            "--allow-high-error."
        )

    arena_image_path = resolve_arena_image_path(image_dir, args.arena_image)
    if arena_image_path is not None:
        homography, arena_corners, image_size = collect_arena_corners_from_image(
            image_path=arena_image_path,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            output_camera_matrix=output_camera_matrix,
            warp_size=(args.warp_width, args.warp_height),
        )
    else:
        print(
            "[CALIB] Arena image not found. Skipping corner click stage and reusing "
            "existing homography."
        )
        homography, arena_corners = load_homography_from_file(
            args.reuse_homography_from)

    save_calibration(
        path=args.output,
        reprojection_error=reprojection_error,
        camera_matrix=output_camera_matrix,
        dist_coeffs=dist_coeffs,
        homography=homography,
        image_size=image_size,
        warp_size=(args.warp_width, args.warp_height),
        board_cols=args.board_cols,
        board_rows=args.board_rows,
        square_size_mm=args.square_size_mm,
        arena_corners=arena_corners,
        source_folder=image_dir,
        undistort_alpha=args.undistort_alpha,
    )


if __name__ == "__main__":
    main()
