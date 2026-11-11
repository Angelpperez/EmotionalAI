"""Measure live-compatible VREED model inference through Blender ACK."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import statistics
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd


FEATURE_NAMES = [
    "Num_of_Fixations",
    "Mean_Fixation_Duration",
    "SD_Fixation_Duration",
    "Num_of_Saccade",
    "Mean_Saccade_Amplitude",
    "Mean_Saccade_Direction",
]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarise(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "outputs/emotion_benchmark/run_20260616_034800/models/stress3/"
            "eye_legacy6_live_compatible__LogReg_balanced.joblib"
        ),
    )
    parser.add_argument(
        "--eye-features",
        type=Path,
        default=Path(
            "EmotionRecognition/archive/04 Eye Tracking Data/"
            "02 Eye Tracking Data (Features Extracted)/"
            "EyeTracking_FeaturesExtracted.csv"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--messages", type=int, default=30)
    parser.add_argument("--interval", type=float, default=0.03)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/closed_loop/model_blender_latency_run"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    if artifact.get("target") != "stress3":
        raise ValueError(f"Expected stress3 model, got {artifact.get('target')}")
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Artifact is not compatible with the six live eye features")

    table = pd.read_csv(args.eye_features)
    features = table[FEATURE_NAMES].astype(float)
    if args.messages > len(features):
        raise ValueError(f"Requested {args.messages} messages from {len(features)} rows")

    sample_indices = [
        round(index * (len(features) - 1) / max(1, args.messages - 1))
        for index in range(args.messages)
    ]
    model = artifact["pipeline"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(args.timeout)
    rows = []

    for message_index, sample_index in enumerate(sample_indices):
        x = features.iloc[[sample_index]].to_numpy(dtype=float)
        inference_start_ns = time.perf_counter_ns()
        state = str(model.predict(x)[0])
        probabilities = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            probabilities = {
                str(label): float(value)
                for label, value in zip(model.classes_, probs)
            }
        inference_end_ns = time.perf_counter_ns()

        message_id = f"model-{message_index:04d}-{uuid.uuid4().hex[:8]}"
        sent_perf_ns = time.perf_counter_ns()
        payload = {
            "message_id": message_id,
            "state": state,
            "score": None if probabilities is None else probabilities.get(state),
            "probabilities": probabilities,
            "timestamp": time.time(),
            "sent_perf_ns": sent_perf_ns,
            "inference_start_perf_ns": inference_start_ns,
            "inference_end_perf_ns": inference_end_ns,
            "source": "model_to_blender_latency_probe",
        }
        sock.sendto(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            (args.host, args.port),
        )

        row = {
            "message_index": message_index,
            "sample_index": sample_index,
            "message_id": message_id,
            "predicted_state": state,
            "inference_ms": (inference_end_ns - inference_start_ns) / 1_000_000.0,
            "success": False,
        }
        try:
            raw, _ = sock.recvfrom(4096)
            client_received_ns = time.perf_counter_ns()
            ack = json.loads(raw.decode("utf-8"))
            if ack.get("message_id") != message_id:
                raise ValueError("ACK message mismatch")
            receiver_received_ns = int(ack["received_perf_ns"])
            applied_ns = int(ack["applied_perf_ns"])
            row.update(
                {
                    "success": True,
                    "difficulty": ack.get("difficulty"),
                    "difficulty_level": ack.get("difficulty_level"),
                    "udp_send_to_receive_ms": (
                        receiver_received_ns - sent_perf_ns
                    )
                    / 1_000_000.0,
                    "blender_apply_ms": (
                        applied_ns - receiver_received_ns
                    )
                    / 1_000_000.0,
                    "udp_round_trip_ms": (
                        client_received_ns - sent_perf_ns
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
            ConnectionResetError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            row["error"] = str(exc)
        rows.append(row)
        time.sleep(args.interval)

    sock.close()
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_dir / "model_blender_latency_messages.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["success"]]
    state_counts = {}
    for row in successful:
        state_counts[row["predicted_state"]] = (
            state_counts.get(row["predicted_state"], 0) + 1
        )
    summary = {
        "model": str(args.model),
        "feature_source": str(args.eye_features),
        "messages_sent": len(rows),
        "messages_acknowledged": len(successful),
        "packet_loss_percent": 100.0
        * (len(rows) - len(successful))
        / max(1, len(rows)),
        "predicted_state_counts": state_counts,
        "inference": summarise([float(row["inference_ms"]) for row in successful]),
        "udp_send_to_receive": summarise(
            [float(row["udp_send_to_receive_ms"]) for row in successful]
        ),
        "blender_apply": summarise(
            [float(row["blender_apply_ms"]) for row in successful]
        ),
        "udp_round_trip": summarise(
            [float(row["udp_round_trip_ms"]) for row in successful]
        ),
        "model_to_ack": summarise(
            [float(row["model_to_ack_ms"]) for row in successful]
        ),
        "scope": (
            "Includes model prediction, UDP transport, Blender difficulty "
            "property application and ACK. Excludes webcam acquisition, live "
            "feature extraction and viewport visual refresh."
        ),
    }
    (args.output_dir / "model_blender_latency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
