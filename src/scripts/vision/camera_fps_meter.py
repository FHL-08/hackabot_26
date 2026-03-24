import argparse
import collections
import time

import cv2


DEFAULT_CAMERA_INDEX = 0
DEFAULT_WINDOW_NAME = "Camera FPS Meter"
DEFAULT_WARMUP_SECONDS = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Measure camera FPS with live overlay.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--window-name",
        default=DEFAULT_WINDOW_NAME,
        help='Display window name (default: "Camera FPS Meter")',
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=DEFAULT_WARMUP_SECONDS,
        help="Seconds to ignore at start before collecting FPS stats (default: 1.0)",
    )
    return parser.parse_args()


def draw_lines(frame, lines):
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        y += 28


def main():
    args = parse_args()
    warmup_seconds = max(0.0, float(args.warmup_seconds))

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    start_time = time.perf_counter()
    warmup_end = start_time + warmup_seconds

    last_frame_time = None
    rolling_samples = collections.deque()  # (timestamp, instantaneous_fps)

    total_frames = 0
    measured_frames = 0
    measured_start_time = None
    measured_end_time = None

    min_fps = float("inf")
    max_fps = 0.0
    fps_sum = 0.0
    fps_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[FPS] Failed to read frame from camera.")
                break

            now = time.perf_counter()
            total_frames += 1

            instantaneous_fps = None
            if last_frame_time is not None:
                dt = now - last_frame_time
                if dt > 0:
                    instantaneous_fps = 1.0 / dt
            last_frame_time = now

            in_measurement = now >= warmup_end and instantaneous_fps is not None
            if in_measurement:
                if measured_start_time is None:
                    measured_start_time = now

                measured_end_time = now
                measured_frames += 1
                fps_sum += instantaneous_fps
                fps_count += 1
                min_fps = min(min_fps, instantaneous_fps)
                max_fps = max(max_fps, instantaneous_fps)

                rolling_samples.append((now, instantaneous_fps))
                while rolling_samples and (now - rolling_samples[0][0] > 1.0):
                    rolling_samples.popleft()
            else:
                while rolling_samples and (now - rolling_samples[0][0] > 1.0):
                    rolling_samples.popleft()

            rolling_fps = 0.0
            if rolling_samples:
                rolling_fps = sum(sample for _, sample in rolling_samples) / len(rolling_samples)

            session_avg_fps = (fps_sum / fps_count) if fps_count > 0 else 0.0
            measured_elapsed = 0.0
            if measured_start_time is not None:
                measured_elapsed = now - measured_start_time

            lines = [f"Camera index: {args.camera_index}"]
            if now < warmup_end:
                lines.append(f"Warmup: {warmup_end - now:.2f}s")
            else:
                lines.append("Warmup: complete")

            lines.append(
                f"Instant FPS: {instantaneous_fps:.2f}" if instantaneous_fps is not None else "Instant FPS: --"
            )
            lines.append(f"Rolling FPS (1s): {rolling_fps:.2f}")
            lines.append(f"Session Avg FPS: {session_avg_fps:.2f}")
            lines.append(f"Measured frames: {measured_frames}")
            lines.append(f"Measured elapsed: {measured_elapsed:.2f}s")
            lines.append("Press q to quit")

            draw_lines(frame, lines)
            cv2.imshow(args.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if fps_count == 0:
        print("[FPS] No FPS samples collected. Increase runtime or reduce warmup seconds.")
        print(f"[FPS] Total frames read: {total_frames}")
        return

    elapsed_seconds = 0.0
    if measured_start_time is not None and measured_end_time is not None:
        elapsed_seconds = measured_end_time - measured_start_time

    print("[FPS] Summary")
    print(f"[FPS] Camera index: {args.camera_index}")
    print(f"[FPS] Warmup seconds: {warmup_seconds:.2f}")
    print(f"[FPS] Total frames read: {total_frames}")
    print(f"[FPS] Measured frames: {measured_frames}")
    print(f"[FPS] Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"[FPS] Min FPS: {min_fps:.2f}")
    print(f"[FPS] Avg FPS: {fps_sum / fps_count:.2f}")
    print(f"[FPS] Max FPS: {max_fps:.2f}")


if __name__ == "__main__":
    main()
