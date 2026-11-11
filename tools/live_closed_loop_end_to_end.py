"""Measure the webcam -> model -> UDP -> Blender -> ACK closed loop.

The experiment uses the existing six-feature live-compatible VREED model and
records acquisition, eye processing, inference, transport, Blender
application, adaptation transitions and session stability.

The reported processing latency starts when the most recent webcam read begins.
The temporal decision delay is reported separately because the classifier uses
a sliding eye-motion window (2 seconds by default).
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import statistics
import time
import uuid
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from eye_tracking_backends import create_eye_tracker, is_placeholder_frame
from live_vreed_eye_state_udp import (
    FEATURE_NAMES,
    compute_features_from_angles,
    find_latest_live_model,
    predict_state,
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarise(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "mean_ms": statistics.fmean(finite),
        "median_ms": statistics.median(finite),
        "std_ms": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min_ms": min(finite),
        "p95_ms": percentile(finite, 0.95),
        "max_ms": max(finite),
    }


def describe(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "p95": percentile(finite, 0.95),
        "max": max(finite),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("outputs/emotion_benchmark"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--camera-source",
        default=None,
        help=(
            "OpenCV source: numeric camera index or stream URL. "
            "Overrides --camera."
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--min-window-samples", type=int, default=10)
    parser.add_argument(
        "--max-eye-sample-age-seconds",
        type=float,
        default=0.25,
    )
    parser.add_argument("--smoothing-windows", type=int, default=3)
    parser.add_argument("--send-interval", type=float, default=0.5)
    parser.add_argument("--ack-timeout", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument(
        "--camera-startup-timeout-seconds",
        type=float,
        default=30.0,
        help=(
            "Maximum time to wait for a virtual phone camera to replace its "
            "placeholder frame with live video. The experiment clock starts "
            "only after live video is available."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--rotation",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
    )
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--eye-detector",
        choices=("haar", "mediapipe"),
        default="haar",
        help="Ocular-localization backend. The launcher defaults to mediapipe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/closed_loop/end_to_end_run"),
    )
    return parser.parse_args()


def receive_matching_ack(
    sock: socket.socket,
    message_id: str,
) -> tuple[dict, int]:
    while True:
        payload, _ = sock.recvfrom(8192)
        received_ns = time.perf_counter_ns()
        ack = json.loads(payload.decode("utf-8"))
        if ack.get("message_id") == message_id:
            return ack, received_ns


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wait_for_active_video(
    cap: cv2.VideoCapture,
    args: argparse.Namespace,
) -> int:
    started = time.perf_counter()
    placeholder_frames = 0
    while time.perf_counter() - started < args.camera_startup_timeout_seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        if not is_placeholder_frame(frame):
            if not args.no_display and placeholder_frames:
                cv2.destroyWindow("Waiting for Android camera")
            return placeholder_frames
        placeholder_frames += 1
        if not args.no_display:
            display = frame.copy()
            cv2.putText(
                display,
                "Active la camara conectada en Android",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 220, 255),
                2,
            )
            cv2.putText(
                display,
                "El cronometro comenzara al recibir video real",
                (30, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )
            cv2.imshow("Waiting for Android camera", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
    raise RuntimeError(
        "La camara virtual no entrego video real dentro de "
        f"{args.camera_startup_timeout_seconds:.0f} segundos."
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model or find_latest_live_model(args.benchmark_root)
    artifact = joblib.load(model_path)
    if artifact.get("target") != "stress3":
        raise ValueError(f"Expected stress3 model, got {artifact.get('target')}")
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature list is not live-compatible")
    eye_tracker = create_eye_tracker(args.eye_detector)

    source: int | str
    if args.camera_source is None:
        source = args.camera
    elif args.camera_source.strip().isdigit():
        source = int(args.camera_source.strip())
    else:
        source = args.camera_source.strip()
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video {source!r}")
    reported_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        startup_placeholder_frames = wait_for_active_video(cap, args)
    except BaseException:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        eye_tracker.close()
        raise

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(args.ack_timeout)

    angle_buffer: deque[tuple[float, int]] = deque()
    state_buffer: deque[str] = deque(maxlen=max(1, args.smoothing_windows))
    rows: list[dict[str, object]] = []
    frame_read_ms: list[float] = []
    eye_processing_ms: list[float] = []
    frame_count = 0
    camera_failures = 0
    placeholder_frames = startup_placeholder_frames
    consecutive_placeholder_frames = 0
    frames_with_eye = 0
    messages_sent = 0
    messages_acknowledged = 0
    ack_errors = 0
    difficulty_transitions = 0
    visual_change_events = 0
    changed_objects_total = 0
    previous_difficulty: str | None = None
    predicted_states: Counter[str] = Counter()
    completed_reason = "duration_completed"

    session_start_wall = time.time()
    session_start_ns = time.perf_counter_ns()
    deadline_ns = session_start_ns + int(args.duration_seconds * 1_000_000_000)
    last_send_ns = 0

    try:
        while time.perf_counter_ns() < deadline_ns:
            read_start_ns = time.perf_counter_ns()
            ok, frame = cap.read()
            read_end_ns = time.perf_counter_ns()
            frame_read_ms.append((read_end_ns - read_start_ns) / 1_000_000.0)
            frame_count += 1
            if not ok:
                camera_failures += 1
                if camera_failures >= 10:
                    completed_reason = "camera_failure_limit"
                    break
                continue
            if is_placeholder_frame(frame):
                placeholder_frames += 1
                consecutive_placeholder_frames += 1
                if consecutive_placeholder_frames >= 30:
                    completed_reason = "camera_placeholder_stream"
                    break
                continue
            consecutive_placeholder_frames = 0

            frame = cv2.flip(frame, 1)
            if args.no_mirror:
                frame = cv2.flip(frame, 1)
            if args.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif args.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            eye_start_ns = time.perf_counter_ns()
            eye_result = eye_tracker.process(frame)
            angles = eye_result.angles
            display = eye_result.display
            eye_end_ns = time.perf_counter_ns()
            eye_processing_ms.append(
                (eye_end_ns - eye_start_ns) / 1_000_000.0
            )
            if angles:
                frames_with_eye += 1
                angle_buffer.append((float(np.mean(angles)), read_end_ns))

            state_text = "warming-up/no-eye"
            score_text = ""
            now_ns = time.perf_counter_ns()
            cutoff_ns = now_ns - int(args.window_seconds * 1_000_000_000)
            while (
                len(angle_buffer) > 1
                and angle_buffer[0][1] < cutoff_ns
            ):
                angle_buffer.popleft()
            observed_window_seconds = (
                (angle_buffer[-1][1] - angle_buffer[0][1])
                / 1_000_000_000.0
                if len(angle_buffer) >= 2
                else 0.0
            )
            latest_eye_sample_age_seconds = (
                (now_ns - angle_buffer[-1][1]) / 1_000_000_000.0
                if angle_buffer
                else float("inf")
            )
            window_ready = (
                len(angle_buffer) >= args.min_window_samples
                and observed_window_seconds >= args.window_seconds * 0.90
                and latest_eye_sample_age_seconds
                <= args.max_eye_sample_age_seconds
            )
            should_send = (
                window_ready
                and now_ns - last_send_ns
                >= int(args.send_interval * 1_000_000_000)
            )
            if should_send:
                feature_start_ns = time.perf_counter_ns()
                observed_eye_fps = (
                    (len(angle_buffer) - 1) / observed_window_seconds
                )
                features = compute_features_from_angles(
                    [value for value, _ in angle_buffer],
                    fps=observed_eye_fps,
                )
                feature_end_ns = time.perf_counter_ns()
                inference_start_ns = time.perf_counter_ns()
                raw_state, score, probabilities = predict_state(
                    artifact, features
                )
                inference_end_ns = time.perf_counter_ns()
                state_buffer.append(raw_state)
                smoothed_state = Counter(state_buffer).most_common(1)[0][0]
                predicted_states[smoothed_state] += 1

                message_id = (
                    f"e2e-{messages_sent:06d}-{uuid.uuid4().hex[:8]}"
                )
                sent_ns = time.perf_counter_ns()
                payload = {
                    "message_id": message_id,
                    "state": smoothed_state,
                    "raw_state": raw_state,
                    "score": score,
                    "probabilities": probabilities,
                    "timestamp": time.time(),
                    "sent_perf_ns": sent_ns,
                    "frame_read_start_perf_ns": read_start_ns,
                    "frame_read_end_perf_ns": read_end_ns,
                    "oldest_window_frame_perf_ns": angle_buffer[0][1],
                    "feature_start_perf_ns": feature_start_ns,
                    "feature_end_perf_ns": feature_end_ns,
                    "inference_start_perf_ns": inference_start_ns,
                    "inference_end_perf_ns": inference_end_ns,
                    "source": "live_closed_loop_end_to_end",
                }
                sock.sendto(
                    json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                    (args.host, args.port),
                )
                messages_sent += 1
                last_send_ns = sent_ns
                row: dict[str, object] = {
                    "message_index": messages_sent - 1,
                    "message_id": message_id,
                    "wall_timestamp": payload["timestamp"],
                    "raw_state": raw_state,
                    "smoothed_state": smoothed_state,
                    "score": score,
                    "probabilities_json": json.dumps(
                        probabilities or {},
                        separators=(",", ":"),
                    ),
                    "valid_eye_samples": len(angle_buffer),
                    "observed_eye_window_seconds": observed_window_seconds,
                    "observed_eye_fps": observed_eye_fps,
                    "latest_eye_sample_age_seconds": (
                        latest_eye_sample_age_seconds
                    ),
                    "frame_read_ms": (
                        read_end_ns - read_start_ns
                    )
                    / 1_000_000.0,
                    "eye_processing_ms": (
                        eye_end_ns - eye_start_ns
                    )
                    / 1_000_000.0,
                    "feature_computation_ms": (
                        feature_end_ns - feature_start_ns
                    )
                    / 1_000_000.0,
                    "inference_ms": (
                        inference_end_ns - inference_start_ns
                    )
                    / 1_000_000.0,
                    "success": False,
                    **dict(zip(FEATURE_NAMES, features)),
                }
                try:
                    ack, client_received_ns = receive_matching_ack(
                        sock, message_id
                    )
                    receiver_received_ns = int(ack["received_perf_ns"])
                    applied_ns = int(ack["applied_perf_ns"])
                    difficulty = str(ack.get("difficulty"))
                    changed_objects = int(
                        ack.get("changed_tagged_objects", 0)
                    )
                    transition = (
                        previous_difficulty is not None
                        and difficulty != previous_difficulty
                    )
                    if transition:
                        difficulty_transitions += 1
                    if changed_objects > 0:
                        visual_change_events += 1
                        changed_objects_total += changed_objects
                    previous_difficulty = difficulty
                    messages_acknowledged += 1
                    row.update(
                        {
                            "success": True,
                            "difficulty": difficulty,
                            "difficulty_level": ack.get("difficulty_level"),
                            "difficulty_transition": transition,
                            "changed_tagged_objects": changed_objects,
                            "udp_send_to_receive_ms": (
                                receiver_received_ns - sent_ns
                            )
                            / 1_000_000.0,
                            "blender_apply_ms": (
                                applied_ns - receiver_received_ns
                            )
                            / 1_000_000.0,
                            "udp_round_trip_ms": (
                                client_received_ns - sent_ns
                            )
                            / 1_000_000.0,
                            "latest_frame_to_blender_apply_ms": (
                                applied_ns - read_end_ns
                            )
                            / 1_000_000.0,
                            "latest_frame_to_ack_ms": (
                                client_received_ns - read_end_ns
                            )
                            / 1_000_000.0,
                            "oldest_window_frame_to_apply_ms": (
                                applied_ns - angle_buffer[0][1]
                            )
                            / 1_000_000.0,
                            "model_to_ack_ms": (
                                client_received_ns - inference_start_ns
                            )
                            / 1_000_000.0,
                        }
                    )
                except (
                    socket.timeout,
                    OSError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                ) as exc:
                    ack_errors += 1
                    row["error"] = str(exc)
                rows.append(row)
                state_text = (
                    f"{smoothed_state} -> {row.get('difficulty', 'no ACK')}"
                )
                score_text = f"score={score:.2f}"

            if not args.no_display:
                remaining = max(
                    0.0,
                    (deadline_ns - time.perf_counter_ns()) / 1_000_000_000.0,
                )
                cv2.putText(
                    display,
                    state_text,
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"{score_text} remaining={remaining:.0f}s",
                    (15, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Closed-loop doctoral latency experiment", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    completed_reason = "user_stopped"
                    break
    except KeyboardInterrupt:
        completed_reason = "keyboard_interrupt"
    finally:
        cap.release()
        sock.close()
        eye_tracker.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    session_end_ns = time.perf_counter_ns()
    session_duration_seconds = (
        session_end_ns - session_start_ns
    ) / 1_000_000_000.0
    write_csv(args.output_dir / "end_to_end_events.csv", rows)
    successful = [row for row in rows if bool(row.get("success"))]

    def metric(name: str) -> dict[str, float | int]:
        return summarise(
            [float(row[name]) for row in successful if name in row]
        )

    summary = {
        "experiment": (
            "camera -> ocular localization -> eye features -> stress3 model "
            "-> UDP -> Blender difficulty -> ACK"
        ),
        "model": str(model_path),
        "eye_detector": eye_tracker.metadata,
        "camera_source": source,
        "requested_frame_width": args.width,
        "requested_frame_height": args.height,
        "reported_frame_width": reported_frame_width,
        "reported_frame_height": reported_frame_height,
        "rotation_degrees": args.rotation,
        "mirrored": not args.no_mirror,
        "requested_duration_seconds": args.duration_seconds,
        "session_duration_seconds": session_duration_seconds,
        "session_start_wall": session_start_wall,
        "completion_reason": completed_reason,
        "completed_requested_duration": (
            completed_reason == "duration_completed"
            and session_duration_seconds >= args.duration_seconds * 0.99
        ),
        "frames_captured": frame_count,
        "camera_read_failures": camera_failures,
        "camera_placeholder_frames": placeholder_frames,
        "camera_startup_placeholder_frames": startup_placeholder_frames,
        "camera_startup_timeout_seconds": (
            args.camera_startup_timeout_seconds
        ),
        "frames_with_valid_eye_detection": frames_with_eye,
        "valid_eye_frame_percent": (
            100.0 * frames_with_eye / max(1, frame_count)
        ),
        "effective_capture_fps": frame_count / max(session_duration_seconds, 1e-9),
        "messages_sent": messages_sent,
        "messages_acknowledged": messages_acknowledged,
        "ack_errors": ack_errors,
        "packet_loss_percent": (
            100.0
            * (messages_sent - messages_acknowledged)
            / max(1, messages_sent)
        ),
        "predicted_state_counts": dict(predicted_states),
        "adaptation_commands_applied": messages_acknowledged,
        "difficulty_transitions": difficulty_transitions,
        "visual_change_events": visual_change_events,
        "changed_tagged_objects_total": changed_objects_total,
        "temporal_window_seconds": args.window_seconds,
        "minimum_temporal_window_samples": args.min_window_samples,
        "maximum_eye_sample_age_seconds": (
            args.max_eye_sample_age_seconds
        ),
        "frame_read": summarise(frame_read_ms),
        "eye_processing": summarise(eye_processing_ms),
        "observed_eye_window_seconds": describe(
            [
                float(row["observed_eye_window_seconds"])
                for row in successful
                if "observed_eye_window_seconds" in row
            ]
        ),
        "observed_eye_sample_rate_fps": describe(
            [
                float(row["observed_eye_fps"])
                for row in successful
                if "observed_eye_fps" in row
            ]
        ),
        "latest_eye_sample_age_seconds": describe(
            [
                float(row["latest_eye_sample_age_seconds"])
                for row in successful
                if "latest_eye_sample_age_seconds" in row
            ]
        ),
        "feature_computation": metric("feature_computation_ms"),
        "inference": metric("inference_ms"),
        "udp_send_to_receive": metric("udp_send_to_receive_ms"),
        "blender_apply": metric("blender_apply_ms"),
        "udp_round_trip": metric("udp_round_trip_ms"),
        "latest_frame_to_blender_apply": metric(
            "latest_frame_to_blender_apply_ms"
        ),
        "latest_frame_to_ack": metric("latest_frame_to_ack_ms"),
        "oldest_window_frame_to_apply": metric(
            "oldest_window_frame_to_apply_ms"
        ),
        "model_to_ack": metric("model_to_ack_ms"),
        "latency_interpretation": {
            "latest_frame_to_blender_apply": (
                "Software processing latency from completion of the latest "
                "webcam frame read to difficulty application in Blender."
            ),
            "oldest_window_frame_to_apply": (
                "Observed decision delay including accumulation of the "
                "temporal eye-motion window."
            ),
            "excludes": (
                "Physical display refresh, photon latency and human "
                "perceptual response."
            ),
        },
    }
    (args.output_dir / "end_to_end_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
