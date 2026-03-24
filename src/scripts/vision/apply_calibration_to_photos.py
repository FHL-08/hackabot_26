import argparse
from pathlib import Path

import cv2
import numpy as np

from src.camera_geometry import DEFAULT_CALIBRATION_PATH, apply_calibration_to_frame, load_arena_calibration


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply arena calibration to raw photos and save visual warp previews."
    )
    parser.add_argument(
        "--image-dir",
        default="src/generated/calibration_photos",
        help="Input folder with raw images",
    )
    parser.add_argument(
        "--output-dir",
        default="src/generated/calibration_warp_preview",
        help="Output folder for warped preview images",
    )
    parser.add_argument(
        "--calibration",
        default=DEFAULT_CALIBRATION_PATH,
        help="Path to arena calibration JSON",
    )
    parser.add_argument(
        "--output-mode",
        choices=("side_by_side", "warped_only"),
        default="side_by_side",
        help="Save side-by-side panel or only the warped view image",
    )
    parser.add_argument(
        "--fit-warp-content",
        action="store_true",
        help="Display-only: crop non-black warp content and scale it to fill the right panel.",
    )
    parser.add_argument(
        "--content-threshold",
        type=int,
        default=8,
        help="Threshold used to detect non-black warped pixels.",
    )
    parser.add_argument(
        "--content-margin",
        type=int,
        default=10,
        help="Extra pixels around detected warped content before resizing.",
    )
    return parser.parse_args()


def list_images(image_dir: Path):
    files = []
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    return files


def make_side_by_side(raw, warped):
    raw_h, raw_w = raw.shape[:2]
    warped_h, warped_w = warped.shape[:2]

    target_h = max(raw_h, warped_h)
    raw_s = raw if raw_h == target_h else cv2.resize(
        raw, (int(raw_w * target_h / raw_h), target_h))
    warped_s = warped if warped_h == target_h else cv2.resize(
        warped, (int(warped_w * target_h / warped_h), target_h))

    pad = np.zeros((target_h, 12, 3), dtype=np.uint8)
    panel = np.hstack([raw_s, pad, warped_s])

    cv2.putText(panel, "RAW", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "CALIBRATED + WARPED",
        (raw_s.shape[1] + 24, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def fit_non_black_for_display(image, threshold=8, margin_px=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(image.shape[1], x + w + margin_px)
    y1 = min(image.shape[0], y + h + margin_px)
    cropped = image[y0:y1, x0:x1]
    if cropped.size == 0:
        return image
    return cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise RuntimeError(f"Input image folder not found: {image_dir}")

    calibration = load_arena_calibration(args.calibration)
    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    saved = 0
    for image_path in image_paths:
        raw = cv2.imread(str(image_path))
        if raw is None:
            print(f"[WARP] Skip unreadable: {image_path.name}")
            continue

        warped = apply_calibration_to_frame(raw, calibration)
        warped_view = warped
        if args.fit_warp_content:
            warped_view = fit_non_black_for_display(
                warped,
                threshold=args.content_threshold,
                margin_px=args.content_margin,
            )
        out_name = f"{image_path.stem}_warp_preview.png"
        out_path = output_dir / out_name
        if args.output_mode == "warped_only":
            cv2.imwrite(str(out_path), warped_view)
        else:
            panel = make_side_by_side(raw, warped_view)
            cv2.imwrite(str(out_path), panel)
        saved += 1

    print(f"[WARP] Input images: {len(image_paths)}")
    print(f"[WARP] Saved previews: {saved}")
    print(f"[WARP] Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
