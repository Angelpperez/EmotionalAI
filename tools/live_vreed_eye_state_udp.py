"""Live webcam eye-state inference with UDP output for Blender.

This runtime intentionally uses the live-compatible VREED feature subset:

    Num_of_Fixations, Mean_Fixation_Duration, SD_Fixation_Duration,
    Num_of_Saccade, Mean_Saccade_Amplitude, Mean_Saccade_Direction

It loads a model artifact produced by
``tools/benchmark_vreed_affective_states.py`` and sends ``low``, ``medium`` or
``high`` states to the Blender UDP controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np


FEATURE_NAMES = [
    "Num_of_Fixations",
    "Mean_Fixation_Duration",
    "SD_Fixation_Duration",
    "Num_of_Saccade",
    "Mean_Saccade_Amplitude",
    "Mean_Saccade_Direction",
]

STATE_TO_SCORE = {
    "low": 0.15,
    "medium": 0.55,
    "high": 0.9,
}


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_pupil_in_eye(eye_gray: np.ndarray):
    eye_eq = cv2.equalizeHist(eye_gray)
    med = cv2.medianBlur(eye_eq, 7)
    blur_soft = cv2.GaussianBlur(eye_eq, (0, 0), 1)
    sharp = cv2.addWeighted(med, 1.2, blur_soft, -0.5, 0)
    edges = cv2.Canny(blur_soft, 100, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    inv = 255 - opened
    _, seg = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 40 or len(contour) < 5:
        return None, None

    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (major_axis, minor_axis), _ = ellipse
    radius = int((major_axis + minor_axis) / 4)
    return (int(cx), int(cy)), radius


def compute_features_from_angles(angles: list[float], fps: float) -> list[float]:
    values = np.asarray(angles, dtype=float)
    if values.size < 2:
        return [0, 0.0, 0.0, 0, 0.0, 0.0]

    diffs = np.diff(values)
    th_fix = 2.0
    th_sacc = 5.0

    fixation_durations = []
    current_len = 1
    for diff in diffs:
        if abs(diff) <= th_fix:
            current_len += 1
        else:
            if current_len > 1:
                fixation_durations.append(current_len / fps)
            current_len = 1
    if current_len > 1:
        fixation_durations.append(current_len / fps)

    if fixation_durations:
        mean_fix = float(np.mean(fixation_durations))
        sd_fix = float(np.std(fixation_durations))
    else:
        mean_fix = 0.0
        sd_fix = 0.0

    sacc_mask = np.abs(diffs) > th_sacc
    sacc_amps = np.abs(diffs[sacc_mask])
    sacc_dirs = np.sign(diffs[sacc_mask])
    if sacc_amps.size:
        mean_amp = float(np.mean(sacc_amps))
        mean_dir = float(np.mean(sacc_dirs))
    else:
        mean_amp = 0.0
        mean_dir = 0.0

    return [
        int(len(fixation_durations)),
        mean_fix,
        sd_fix,
        int(sacc_amps.size),
        mean_amp,
        mean_dir,
    ]


def find_latest_live_model(output_root: Path) -> Path:
    latest_path = output_root / "latest_run.txt"
    if not latest_path.exists():
        raise FileNotFoundError(f"No latest benchmark run found at {latest_path}")
    run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
    candidates = sorted(
        (run_dir / "models" / "stress3").glob("eye_legacy6_live_compatible__*.joblib"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No live-compatible stress3 model found in {run_dir}")
    return candidates[0]


def send_udp(sock: socket.socket, host: str, port: int, state: str, score: float, probabilities=None) -> None:
    payload = {
        "state": state,
        "score": score,
        "timestamp": time.time(),
        "source": "live_vreed_eye_state_udp",
    }
    if probabilities is not None:
        payload["probabilities"] = probabilities
    sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), (host, port))


def predict_state(artifact: dict, features: list[float]) -> tuple[str, float, dict[str, float] | None]:
    model = artifact["pipeline"]
    x = np.asarray(features, dtype=float).reshape(1, -1)
    prediction = str(model.predict(x)[0])
    probabilities = None
    score = STATE_TO_SCORE.get(prediction, 0.5)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        classes = [str(item) for item in model.classes_]
        probabilities = {cls: float(prob) for cls, prob in zip(classes, probs)}
        score = float(probabilities.get(prediction, score))
    return prediction, score, probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="Model artifact .joblib")
    parser.add_argument("--benchmark-root", type=Path, default=Path("outputs/emotion_benchmark"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--smoothing-windows", type=int, default=3)
    parser.add_argument("--send-interval", type=float, default=0.5)
    parser.add_argument("--log", type=Path, default=Path("outputs/emotion_runtime/live_vreed_eye_state_log.csv"))
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or find_latest_live_model(args.benchmark_root)
    artifact = joblib.load(model_path)
    if artifact.get("target") != "stress3":
        raise ValueError(f"Expected stress3 model, got {artifact.get('target')}")
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature list is not live-compatible with this runtime")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara")

    window_frames = max(2, int(args.fps * args.window_seconds))
    angle_buffer: deque[float] = deque(maxlen=window_frames)
    state_buffer: deque[str] = deque(maxlen=max(1, args.smoothing_windows))
    last_send = 0.0

    with args.log.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "valid_eye_angles",
                *FEATURE_NAMES,
                "raw_state",
                "smoothed_state",
                "score",
                "probabilities_json",
                "udp_host",
                "udp_port",
                "model_path",
            ],
        )
        if f.tell() == 0:
            writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            angles_this_frame: list[float] = []

            for (x, y, w, h) in faces[:1]:
                roi_gray = gray[y : y + h, x : x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 20))
                candidates = []
                for (ex, ey, ew, eh) in eyes:
                    if ey + eh / 2 > 0.6 * h:
                        continue
                    ratio = ew / float(eh)
                    if 0.7 <= ratio <= 3.0:
                        candidates.append((ex, ey, ew, eh))
                candidates = sorted(sorted(candidates, key=lambda item: -item[2])[:2], key=lambda item: item[0])

                for (ex, ey, ew, eh) in candidates:
                    eye_gray = roi_gray[ey : ey + eh, ex : ex + ew]
                    center, radius = detect_pupil_in_eye(eye_gray)
                    if center is None or radius is None:
                        continue
                    if radius < 3 or radius > min(ew, eh) // 2:
                        continue

                    cx_global = x + ex + center[0]
                    cy_global = y + ey + center[1]
                    eye_center_x = x + ex + ew // 2
                    eye_center_y = y + ey + eh // 2
                    vx = cx_global - eye_center_x
                    vy = cy_global - eye_center_y
                    angle = float(np.degrees(np.arctan2(-vy, vx)))
                    angles_this_frame.append(angle)

                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1)
                    cv2.circle(frame, (cx_global, cy_global), 3, (0, 255, 255), -1)

            if angles_this_frame:
                angle_buffer.append(float(np.mean(angles_this_frame)))

            if len(angle_buffer) == window_frames:
                features = compute_features_from_angles(list(angle_buffer), fps=args.fps)
                raw_state, score, probabilities = predict_state(artifact, features)
                state_buffer.append(raw_state)
                smoothed_state = Counter(state_buffer).most_common(1)[0][0]
                now = time.time()
                if now - last_send >= args.send_interval:
                    send_udp(sock, args.host, args.port, smoothed_state, score, probabilities)
                    last_send = now
                    row = {
                        "timestamp": now,
                        "valid_eye_angles": len(angle_buffer),
                        **dict(zip(FEATURE_NAMES, features)),
                        "raw_state": raw_state,
                        "smoothed_state": smoothed_state,
                        "score": score,
                        "probabilities_json": json.dumps(probabilities or {}, separators=(",", ":")),
                        "udp_host": args.host,
                        "udp_port": args.port,
                        "model_path": str(model_path),
                    }
                    writer.writerow(row)
                    f.flush()

                cv2.putText(
                    frame,
                    f"state={smoothed_state} raw={raw_state} score={score:.2f}",
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            if not args.no_display:
                cv2.imshow("Live VREED eye state -> Blender UDP", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
