import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze all calibration photos, detect chessboard corners, and save "
            "per-photo visual overlays into separate folders."
        )
    )
    parser.add_argument(
        "--image-dir",
        default="src/generated/calibration_photos",
        help="Folder containing input photos",
    )
    parser.add_argument(
        "--output-dir",
        default="src/generated/calibration_photos_analysis",
        help="Folder to write one overlay image per input photo",
    )
    parser.add_argument(
        "--export-success-dir",
        default="successful_photos",
        help=(
            "Folder to copy raw images with successful detection. "
            "Relative paths are resolved under --output-dir."
        ),
    )
    parser.add_argument("--board-cols", type=int, default=11,
                        help="Chessboard inner corners across")
    parser.add_argument("--board-rows", type=int, default=7,
                        help="Chessboard inner corners down")
    return parser.parse_args()


def list_images(image_dir):
    paths = []
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            paths.append(path)
    return paths


def make_output_overlay_path(output_dir, image_name):
    stem = Path(image_name).stem
    candidate = output_dir / f"{stem}_overlay.png"
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        candidate = output_dir / f"{stem}_overlay_{idx}.png"
        if not candidate.exists():
            return candidate
        idx += 1


def detect_standard(gray, board_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def detect_sb(gray, board_size):
    if not hasattr(cv2, "findChessboardCornersSB"):
        return False, None

    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, board_size, flags)
    if not found:
        return False, None
    return True, corners


def draw_corner_grid(image, corners, board_cols, board_rows):
    pts = corners.reshape(board_rows, board_cols, 2).astype(np.int32)
    out = image.copy()

    # Draw horizontal and vertical inner-grid lines.
    for r in range(board_rows):
        for c in range(board_cols - 1):
            p1 = tuple(pts[r, c])
            p2 = tuple(pts[r, c + 1])
            cv2.line(out, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)

    for c in range(board_cols):
        for r in range(board_rows - 1):
            p1 = tuple(pts[r, c])
            p2 = tuple(pts[r + 1, c])
            cv2.line(out, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)

    # Draw board border using outermost detected inner corners.
    tl = tuple(pts[0, 0])
    tr = tuple(pts[0, board_cols - 1])
    br = tuple(pts[board_rows - 1, board_cols - 1])
    bl = tuple(pts[board_rows - 1, 0])
    cv2.polylines(
        out, [np.array([tl, tr, br, bl], dtype=np.int32)], True, (0, 255, 0), 2)

    # Draw corner points and indices.
    flat = corners.reshape(-1, 2)
    for idx, p in enumerate(flat):
        x, y = int(p[0]), int(p[1])
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
        if idx in (0, board_cols - 1, len(flat) - board_cols, len(flat) - 1):
            cv2.putText(
                out,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return out


def draw_overlay(image, board_size, corners, method_name, found):
    overlay = image.copy()
    h, w = overlay.shape[:2]

    if found and corners is not None:
        cv2.drawChessboardCorners(overlay, board_size, corners, True)
        overlay = draw_corner_grid(
            overlay, corners, board_size[0], board_size[1])
        status = f"{method_name}: DETECTED"
        color = (0, 220, 0)
    else:
        status = f"{method_name}: NOT DETECTED"
        color = (0, 0, 255)

    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        status,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return overlay


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_dir = Path(args.export_success_dir)
    if not success_dir.is_absolute():
        success_dir = output_dir / success_dir
    success_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise RuntimeError(f"Image directory does not exist: {image_dir}")

    board_size = (args.board_cols, args.board_rows)
    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    summary = {
        "image_dir": str(image_dir.resolve()),
        "board_size": {"cols": args.board_cols, "rows": args.board_rows},
        "success_dir": str(success_dir.resolve()),
        "total_images": len(image_paths),
        "detected_standard": 0,
        "detected_sb": 0,
        "detected_any": 0,
        "images": [],
    }

    print(f"[ANALYZE] Processing {len(image_paths)} images from: {image_dir}")
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[ANALYZE] Skip unreadable: {image_path.name}")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        std_found, std_corners = detect_standard(gray, board_size)
        sb_found, sb_corners = detect_sb(gray, board_size)

        if std_found:
            summary["detected_standard"] += 1
        if sb_found:
            summary["detected_sb"] += 1
        if std_found or sb_found:
            summary["detected_any"] += 1

        # Prefer SB if available, else standard.
        if sb_found:
            best_method = "sb"
            best_found = True
            best_corners = sb_corners
        elif std_found:
            best_method = "standard"
            best_found = True
            best_corners = std_corners
        else:
            best_method = "none"
            best_found = False
            best_corners = None

        best_overlay = draw_overlay(
            frame, board_size, best_corners, f"best:{best_method}", best_found)
        out_png = make_output_overlay_path(output_dir, image_path.name)
        cv2.imwrite(str(out_png), best_overlay)

        copied_success_path = None
        if best_found:
            copied_success_path = success_dir / image_path.name
            shutil.copy2(str(image_path), str(copied_success_path))

        per_image = {
            "image_name": image_path.name,
            "image_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "standard_found": bool(std_found),
            "sb_found": bool(sb_found),
            "best_method": best_method,
            "best_found": bool(best_found),
            "overlay_png": str(out_png.resolve()),
            "copied_success_raw": str(copied_success_path.resolve()) if copied_success_path else None,
        }
        summary["images"].append(per_image)

        print(
            f"[ANALYZE] {image_path.name} | "
            f"standard={std_found} sb={sb_found} best={best_method}"
        )

    write_json(output_dir / "summary.json", summary)
    print(f"[ANALYZE] Done. Summary: {output_dir / 'summary.json'}")
    print(
        f"[ANALYZE] Successful raw images copied to: {success_dir.resolve()}")
    print(
        f"[ANALYZE] detected_any={summary['detected_any']}/{summary['total_images']} "
        f"standard={summary['detected_standard']} sb={summary['detected_sb']}"
    )


if __name__ == "__main__":
    main()
