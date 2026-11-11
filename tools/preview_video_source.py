"""Preview and characterize a webcam, virtual phone camera or stream URL."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument(
        "--rotation",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
    )
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/camera_preview/latest_frame.jpg"),
    )
    return parser.parse_args()


def normalized_source(value: str) -> int | str:
    stripped = value.strip()
    return int(stripped) if stripped.isdigit() else stripped


def orient(
    frame,
    rotation: int,
    mirror: bool,
):
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mirror:
        frame = cv2.flip(frame, 1)
    return frame


def main() -> None:
    args = parse_args()
    source = normalized_source(args.source)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente {source!r}")

    started = time.perf_counter()
    frames = 0
    failures = 0
    latest = None
    while time.perf_counter() - started < args.duration_seconds:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            continue
        latest = orient(frame, args.rotation, args.mirror)
        frames += 1
        if not args.no_display:
            cv2.imshow("Phone/webcam source preview - Q to stop", latest)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.perf_counter() - started
    reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if latest is None:
        raise RuntimeError("La fuente se abrio, pero no entrego frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), latest)
    result = {
        "source": source,
        "frames": frames,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "effective_fps": frames / max(elapsed, 1e-9),
        "reported_width": reported_width,
        "reported_height": reported_height,
        "reported_fps": reported_fps,
        "output_frame": str(args.output),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
