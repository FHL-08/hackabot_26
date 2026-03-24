"""Generate ArUco marker PNG files."""

import argparse
from pathlib import Path

import cv2


def parse_args():
    """Parse marker generation CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker images")
    parser.add_argument(
        "--out-dir", default="src/generated/markers", help="Output directory")
    parser.add_argument("--pixels", type=int, default=147,
                        help="Marker image size in pixels")
    parser.add_argument("--start-id", type=int,
                        default=0, help="First marker ID")
    parser.add_argument("--end-id", type=int, default=9,
                        help="Last marker ID (inclusive)")
    return parser.parse_args()


def main():
    """Generate markers for the selected ID range."""
    args = parse_args()
    if args.pixels <= 0:
        raise RuntimeError("--pixels must be > 0")
    if args.end_id < args.start_id:
        raise RuntimeError("--end-id must be >= --start-id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    for marker_id in range(args.start_id, args.end_id + 1):
        img = aruco.generateImageMarker(
            dictionary, int(marker_id), int(args.pixels))
        path = out_dir / f"aruco_{marker_id}.png"
        cv2.imwrite(str(path), img)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
