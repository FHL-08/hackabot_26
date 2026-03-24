import argparse
import csv
import datetime
import os
from pathlib import Path

import cv2

WINDOW_NAME = "Coordinate Click Logger"


def compute_non_black_fit_transform(image, threshold=8, margin_px=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    h, w = image.shape[:2]
    if coords is None:
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

    x, y, bw, bh = cv2.boundingRect(coords)
    x0 = max(0, x - int(margin_px))
    y0 = max(0, y - int(margin_px))
    x1 = min(w, x + bw + int(margin_px))
    y1 = min(h, y + bh + int(margin_px))

    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    out_w = int(w)
    out_h = int(h)

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
        return image.copy()
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def source_to_display_xy(u_src, v_src, transform):
    u_disp = (float(u_src) -
              float(transform["crop_x0"])) * float(transform["scale_x"])
    v_disp = (float(v_src) -
              float(transform["crop_y0"])) * float(transform["scale_y"])
    return u_disp, v_disp


def display_to_source_xy(u_disp, v_disp, transform):
    u_src = float(u_disp) / \
        float(transform["scale_x"]) + float(transform["crop_x0"])
    v_src = float(v_disp) / \
        float(transform["scale_y"]) + float(transform["crop_y0"])
    return u_src, v_src


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def uv_to_arena_px(u_live, v_live, origin_u, origin_v):
    arena_x_px = -(float(v_live) - float(origin_v))
    arena_y_px = -(float(u_live) - float(origin_u))
    return arena_x_px, arena_y_px


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load one calibrated arena image, click points, and log coordinates."
    )
    parser.add_argument("--image", required=True,
                        help="Path to calibrated arena image")
    parser.add_argument(
        "--cm-per-px-x",
        type=float,
        default=0.3256260775073436,
        help="Scale for arena x axis (cm per pixel in loaded image)",
    )
    parser.add_argument(
        "--cm-per-px-y",
        type=float,
        default=0.28330199259559835,
        help="Scale for arena y axis (cm per pixel in loaded image)",
    )
    parser.add_argument(
        "--origin-u-px",
        type=float,
        default=621.68,
        help="Arena origin u in live/source image pixels (used unless --use-center-origin).",
    )
    parser.add_argument(
        "--origin-v-px",
        type=float,
        default=418.20,
        help="Arena origin v in live/source image pixels (used unless --use-center-origin).",
    )
    parser.add_argument(
        "--use-center-origin",
        action="store_true",
        help="Use source image center as arena origin instead of --origin-u-px/--origin-v-px.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("src", "generated", "clicked_coordinates.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--fit-warp-content",
        action="store_true",
        help="Show a fit-to-content preview while still logging clicks in live (source warp) coordinates.",
    )
    parser.add_argument(
        "--content-threshold",
        type=int,
        default=8,
        help="Threshold used to detect non-black warped pixels for fit preview.",
    )
    parser.add_argument(
        "--content-margin",
        type=int,
        default=10,
        help="Extra pixels around detected non-black warped pixels for fit preview.",
    )
    return parser.parse_args()


def count_existing_rows(csv_path):
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    return max(0, total - 1)


def main():
    args = parse_args()

    if args.cm_per_px_x <= 0 or args.cm_per_px_y <= 0:
        raise RuntimeError("cm-per-px-x and cm-per-px-y must both be > 0")

    image_path = Path(args.image)
    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    source_image = cv2.imread(str(image_path))
    if source_image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    src_h, src_w = source_image.shape[:2]
    if args.fit_warp_content:
        fit_transform = compute_non_black_fit_transform(
            source_image,
            threshold=args.content_threshold,
            margin_px=args.content_margin,
        )
        display_image = apply_fit_transform_for_display(
            source_image, fit_transform)
    else:
        fit_transform = {
            "crop_x0": 0,
            "crop_y0": 0,
            "crop_w": int(src_w),
            "crop_h": int(src_h),
            "out_w": int(src_w),
            "out_h": int(src_h),
            "scale_x": 1.0,
            "scale_y": 1.0,
        }
        display_image = source_image.copy()

    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    point_index = count_existing_rows(csv_path)
    clicked_points = []
    disp_h, disp_w = display_image.shape[:2]
    frame_shape = (disp_h, disp_w)
    if args.use_center_origin:
        origin_u = (float(src_w) - 1.0) * 0.5
        origin_v = (float(src_h) - 1.0) * 0.5
        origin_mode = "center"
    else:
        origin_u = float(args.origin_u_px)
        origin_v = float(args.origin_v_px)
        origin_mode = "custom"

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "index",
            "timestamp_utc",
            "image_path",
            "u_disp_px",
            "v_disp_px",
            "u_live_px",
            "v_live_px",
            "u_px",
            "v_px",
            "x_px",
            "y_px",
            "x_cm",
            "y_cm",
            "origin_u_px",
            "origin_v_px",
            "origin_mode",
            "source_width",
            "source_height",
            "display_width",
            "display_height",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        def on_mouse(event, x, y, _flags, _userdata):
            nonlocal point_index
            if event != cv2.EVENT_LBUTTONDOWN or frame_shape is None:
                return

            u_live, v_live = display_to_source_xy(x, y, fit_transform)
            u_live = clamp(u_live, 0.0, float(src_w - 1))
            v_live = clamp(v_live, 0.0, float(src_h - 1))

            x_px, y_px = uv_to_arena_px(u_live, v_live, origin_u, origin_v)
            x_cm = float(x_px) * float(args.cm_per_px_x)
            y_cm = float(y_px) * float(args.cm_per_px_y)

            ts = datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            row = {
                "index": int(point_index),
                "timestamp_utc": ts,
                "image_path": str(image_path),
                "u_disp_px": float(x),
                "v_disp_px": float(y),
                "u_live_px": float(u_live),
                "v_live_px": float(v_live),
                "u_px": float(u_live),
                "v_px": float(v_live),
                "x_px": float(x_px),
                "y_px": float(y_px),
                "x_cm": float(x_cm),
                "y_cm": float(y_cm),
                "origin_u_px": float(origin_u),
                "origin_v_px": float(origin_v),
                "origin_mode": origin_mode,
                "source_width": int(src_w),
                "source_height": int(src_h),
                "display_width": int(disp_w),
                "display_height": int(disp_h),
            }
            writer.writerow(row)
            csv_file.flush()

            clicked_points.append(
                {
                    "u_disp": int(x),
                    "v_disp": int(y),
                    "u_live": float(u_live),
                    "v_live": float(v_live),
                    "idx": int(point_index),
                    "x_cm": float(x_cm),
                    "y_cm": float(y_cm),
                }
            )
            print(
                f"[CLICK] idx={point_index} "
                f"disp=({x:.1f},{y:.1f}) "
                f"live=({u_live:.2f},{v_live:.2f}) "
                f"x_cm={x_cm:.2f} y_cm={y_cm:.2f}"
            )
            point_index += 1

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        print("[LOGGER] Left-click to log a coordinate.")
        print("[LOGGER] Press 'c' to clear on-screen click markers (CSV remains).")
        print("[LOGGER] Press 'u' to undo last point (CSV remains).")
        print("[LOGGER] Press 'q' to quit.")
        print(f"[LOGGER] Loaded image: {image_path.resolve()}")
        print(f"[LOGGER] Writing to: {csv_path.resolve()}")
        print(
            f"[LOGGER] Scale: cm_per_px_x={args.cm_per_px_x:.6f}, "
            f"cm_per_px_y={args.cm_per_px_y:.6f}"
        )
        print(
            f"[LOGGER] Origin ({origin_mode}): u0={origin_u:.2f}, v0={origin_v:.2f}"
        )
        if args.fit_warp_content:
            print(
                "[LOGGER] Fit preview ON (clicks map back to source warp/live coordinates)."
            )
            print(
                "[LOGGER] IMPORTANT: for true live mapping, use an unfitted warped image as --image."
            )

        try:
            while True:
                display = display_image.copy()

                center_u, center_v = source_to_display_xy(
                    origin_u, origin_v, fit_transform)
                cv2.drawMarker(
                    display,
                    (int(round(center_u)), int(round(center_v))),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                )

                for point in clicked_points:
                    u = int(point["u_disp"])
                    v = int(point["v_disp"])
                    idx = int(point["idx"])
                    x_cm = float(point["x_cm"])
                    y_cm = float(point["y_cm"])
                    u_live = float(point["u_live"])
                    v_live = float(point["v_live"])

                    cv2.circle(display, (u, v), 4, (0, 255, 0), -1)
                    cv2.putText(
                        display,
                        f"{idx}: ({x_cm:.1f},{y_cm:.1f})cm live=({u_live:.1f},{v_live:.1f})",
                        (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        2,
                    )

                cv2.putText(
                    display,
                    "LMB=log  c=clear markers  u=undo marker  q=quit",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    display,
                    f"cm_per_px_x={args.cm_per_px_x:.5f}  cm_per_px_y={args.cm_per_px_y:.5f}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"origin={origin_mode} (u0={origin_u:.1f}, v0={origin_v:.1f}) source={src_w}x{src_h}",
                    (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    clicked_points.clear()
                if key == ord("u") and clicked_points:
                    clicked_points.pop()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
