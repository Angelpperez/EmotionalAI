"""Measure face, eye and pupil detection quality for a video source."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from eye_tracking_backends import create_eye_tracker, is_placeholder_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument(
        "--detector",
        choices=("haar", "mediapipe"),
        default="haar",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/camera_detection_quality"),
    )
    return parser.parse_args()


def normalized_source(value: str) -> int | str:
    stripped = value.strip()
    return int(stripped) if stripped.isdigit() else stripped


def summarise(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    source = normalized_source(args.source)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente {source!r}")
    tracker = create_eye_tracker(args.detector)

    started = time.perf_counter()
    frames = 0
    read_failures = 0
    placeholder_frames = 0
    face_frames = 0
    one_eye_frames = 0
    two_eye_frames = 0
    one_pupil_frames = 0
    two_pupil_frames = 0
    face_widths: list[float] = []
    face_heights: list[float] = []
    eye_widths: list[float] = []
    eye_heights: list[float] = []
    processing_ms: list[float] = []
    eye_openness: list[float] = []
    normalized_iris_x: list[float] = []
    normalized_iris_y: list[float] = []
    best_frame = None
    best_pupil_count = -1

    while time.perf_counter() - started < args.duration_seconds:
        ok, frame = cap.read()
        if not ok:
            read_failures += 1
            continue
        frames += 1
        if is_placeholder_frame(frame):
            placeholder_frames += 1
            if best_frame is None:
                best_frame = frame.copy()
            continue
        processing_start = time.perf_counter_ns()
        detection = tracker.process(frame)
        annotated = detection.display
        pupil_count = detection.pupil_count
        valid_eye_count = detection.eye_count
        if detection.face_detected and detection.face_bbox is not None:
            face_frames += 1
            _, _, width, height = detection.face_bbox
            face_widths.append(float(width))
            face_heights.append(float(height))
        for _, _, eye_width, eye_height in detection.eye_bboxes:
            eye_widths.append(float(eye_width))
            eye_heights.append(float(eye_height))
        eye_openness.extend(detection.eye_openness)
        normalized_iris_x.extend(
            value[0] for value in detection.normalized_iris_offsets
        )
        normalized_iris_y.extend(
            value[1] for value in detection.normalized_iris_offsets
        )

        if valid_eye_count >= 1:
            one_eye_frames += 1
        if valid_eye_count >= 2:
            two_eye_frames += 1
        if pupil_count >= 1:
            one_pupil_frames += 1
        if pupil_count >= 2:
            two_pupil_frames += 1
        if pupil_count > best_pupil_count:
            best_pupil_count = pupil_count
            best_frame = annotated
        processing_ms.append(
            (time.perf_counter_ns() - processing_start) / 1_000_000.0
        )

    elapsed = time.perf_counter() - started
    cap.release()
    tracker.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = args.output_dir / "best_annotated_frame.jpg"
    if best_frame is not None:
        cv2.imwrite(str(annotated_path), best_frame)

    def percentage(count: int) -> float:
        return 100.0 * count / max(1, frames)

    result = {
        "source": source,
        "detector": tracker.metadata,
        "requested_resolution": [args.width, args.height],
        "frames": frames,
        "read_failures": read_failures,
        "placeholder_frames": placeholder_frames,
        "placeholder_frame_percent": percentage(placeholder_frames),
        "valid_video_source": placeholder_frames < max(1, frames // 2),
        "elapsed_seconds": elapsed,
        "effective_fps": frames / max(elapsed, 1e-9),
        "face_frames": face_frames,
        "face_frame_percent": percentage(face_frames),
        "one_eye_frames": one_eye_frames,
        "one_eye_frame_percent": percentage(one_eye_frames),
        "two_eye_frames": two_eye_frames,
        "two_eye_frame_percent": percentage(two_eye_frames),
        "one_pupil_frames": one_pupil_frames,
        "one_pupil_frame_percent": percentage(one_pupil_frames),
        "two_pupil_frames": two_pupil_frames,
        "two_pupil_frame_percent": percentage(two_pupil_frames),
        "face_width_pixels": summarise(face_widths),
        "face_height_pixels": summarise(face_heights),
        "eye_width_pixels": summarise(eye_widths),
        "eye_height_pixels": summarise(eye_heights),
        "eye_openness_ratio": summarise(eye_openness),
        "normalized_iris_offset_x": summarise(normalized_iris_x),
        "normalized_iris_offset_y": summarise(normalized_iris_y),
        "detector_processing_ms": summarise(processing_ms),
        "best_annotated_frame": (
            str(annotated_path) if best_frame is not None else None
        ),
    }
    (args.output_dir / "detection_quality.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
